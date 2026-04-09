"""Cohere Transcribe audio transcription with sentence-level timestamps.

Transcribes audio files using Cohere Transcribe 03-2026, then generates
sentence-level timestamps via ctc-forced-aligner and spaCy.

Usage:
    python audio_transcriber.py <input>
    python audio_transcriber.py <input> --output-dir ./output
    python audio_transcriber.py <input> --config my_config.json

Input:
    Accepts a single file, a directory, or a glob pattern:
      - "Lecture 9.m4a"            (single file)
      - "./_audio/MSE403 S21/"     (all audio in directory)
      - "./_audio/**/*.m4a"        (glob pattern)

Output:
    [MM:SS.mmm --> MM:SS.mmm] Sentence text.

    One sentence per line. Timestamps derived from ctc-forced-aligner
    word timestamps grouped into sentences by spaCy.

Configuration:
    Settings are read from config.json (model, chunking, output, language).
    CLI flags override config values. See config.json for schema.
"""
import argparse
import json
import os
import re
import sys
import time
import warnings
from pathlib import Path

# Suppress known harmless warnings:
# - PySoundFile doesn't support M4A; librosa falls back to audioread (works fine)
# - librosa's audioread backend is deprecated (FutureWarning, multi-line message)
# - ctc-forced-aligner uses deprecated torch_dtype param (logged, not warned)
warnings.filterwarnings("ignore", message="PySoundFile failed")
warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")
warnings.filterwarnings("ignore", message=".*torch_dtype.*")

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

import spacy
import torch
from transformers import AutoProcessor, CohereAsrForConditionalGeneration

from _benchmark.verify_granite import (
    TARGET_SR,
    ALIGNER_REPO,
    ALIGNER_DIR,
    print_vram,
    ensure_model_downloaded,
    load_audio_16k_mono,
    chunk_audio,
)
from ctc_forced_aligner import (
    load_alignment_model,
    generate_emissions,
    preprocess_text,
    get_alignments,
    get_spans,
    postprocess_results,
)

AUDIO_EXTENSIONS = {".m4a", ".mp3", ".wav", ".flac", ".ogg", ".mp4", ".webm"}

_DIGIT_WORDS = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
}


# ── Config ───────────────────────────────────────────────────────────

def load_config(config_path: str = "config.json", **cli_overrides) -> dict:
    """Load config.json and merge CLI overrides."""
    config_file = Path(config_path)
    if config_file.exists():
        config = json.loads(config_file.read_text(encoding="utf-8"))
    else:
        config = {}

    if cli_overrides.get("output_dir"):
        config["output_dir"] = cli_overrides["output_dir"]

    config.setdefault("model", {})
    config["model"].setdefault("asr", "CohereLabs/cohere-transcribe-03-2026")
    config["model"].setdefault("dtype", "bfloat16")
    config.setdefault("chunking", {})
    config["chunking"].setdefault("max_duration_sec", 30)
    config["chunking"].setdefault("overlap_sec", 2)
    config.setdefault("language", "en")

    return config


# ── Input resolution ─────────────────────────────────────────────────

def resolve_input_files(input_path: str) -> list[Path]:
    """Resolve input to a list of audio files.

    Supports single file, directory, or glob pattern.
    """
    p = Path(input_path)
    if p.is_file():
        return [p] if p.suffix.lower() in AUDIO_EXTENSIONS else []
    if p.is_dir():
        return sorted(f for f in p.iterdir()
                       if f.suffix.lower() in AUDIO_EXTENSIONS)
    files = sorted(Path(".").glob(input_path))
    return [f for f in files
            if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS]


# ── Text utilities ───────────────────────────────────────────────────

def clean_text_for_aligner(text: str) -> str:
    """Convert digits to words and strip non-alpha chars for the aligner."""
    for digit, word in _DIGIT_WORDS.items():
        text = text.replace(digit, f" {word} ")
    text = re.sub(r"[^a-zA-Z\s']", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _format_timestamp(seconds: float) -> str:
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes:02}:{secs:06.3f}"


def words_to_sentences(
    full_text: str,
    word_timestamps: list[dict],
    nlp,
) -> list[dict]:
    """Group word timestamps into sentence-level timestamps using spaCy."""
    doc = nlp(full_text)
    sentences = []
    word_index = 0
    for sent in doc.sents:
        sent_text = sent.text.strip()
        sent_words = sent_text.split()
        if not sent_words:
            continue
        start_time = word_timestamps[word_index]["start"]
        end_idx = min(word_index + len(sent_words) - 1,
                      len(word_timestamps) - 1)
        end_time = word_timestamps[end_idx]["end"]
        sentences.append({
            "start": start_time, "end": end_time, "text": sent_text,
        })
        word_index += len(sent_words)
    return sentences


# ── ASR ──────────────────────────────────────────────────────────────

def transcribe_chunk(
    chunk_wav: torch.Tensor,
    processor,
    model,
    device: str,
    dtype: torch.dtype,
    language: str,
) -> str:
    """Transcribe a single audio chunk with Cohere."""
    chunk_np = chunk_wav.squeeze(0).numpy()
    inputs = processor(
        chunk_np, sampling_rate=TARGET_SR,
        return_tensors="pt", language=language,
    )
    inputs = inputs.to(device, dtype=dtype)

    chunk_duration = chunk_wav.shape[1] / TARGET_SR
    max_new_tokens = max(128, int(chunk_duration * 5) + 64)

    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    text = processor.decode(outputs[0], skip_special_tokens=True)
    return text.strip()


# ── Single file pipeline ────────────────────────────────────────────

def transcribe_file(
    audio_path: str,
    output_path: str,
    processor,
    model,
    alignment_model,
    alignment_tokenizer,
    nlp,
    config: dict,
    device: str,
    dtype: torch.dtype,
):
    """Transcribe a single audio file with Cohere + forced alignment."""
    chunk_sec = config["chunking"]["max_duration_sec"]
    overlap_sec = config["chunking"]["overlap_sec"]
    language = config.get("language", "en")

    # Load audio and chunk
    wav = load_audio_16k_mono(audio_path)
    audio_duration = wav.shape[1] / TARGET_SR
    chunks = chunk_audio(wav, chunk_sec, overlap_sec)
    print(f"  Duration: {audio_duration:.1f}s, {len(chunks)} chunks")

    # Transcribe each chunk
    t0 = time.time()
    chunk_texts = []
    for i, (start_sec, end_sec, chunk_wav) in enumerate(chunks):
        text = transcribe_chunk(
            chunk_wav, processor, model, device, dtype, language,
        )
        chunk_texts.append(text)
    full_text = " ".join(t for t in chunk_texts if t)
    asr_time = time.time() - t0
    print(f"  ASR: {asr_time:.1f}s (RTFx={audio_duration / asr_time:.1f}x), "
          f"{len(full_text.split())} words")

    # Forced alignment for word timestamps
    t0 = time.time()
    audio_tensor = wav.squeeze(0).to(device, dtype=dtype)
    align_text = clean_text_for_aligner(full_text)
    emissions, stride = generate_emissions(
        alignment_model, audio_tensor, batch_size=16,
    )
    tokens_starred, text_starred = preprocess_text(
        align_text, romanize=False, language="eng",
    )
    segments, scores, blank_token = get_alignments(
        emissions, tokens_starred, alignment_tokenizer,
    )
    spans = get_spans(tokens_starred, segments, blank_token)
    word_timestamps = postprocess_results(
        text_starred, spans, stride, scores,
    )
    align_time = time.time() - t0
    print(f"  Alignment: {align_time:.1f}s, "
          f"{len(word_timestamps)} aligned words")

    # Sentence segmentation with timestamps
    sentences = words_to_sentences(full_text, word_timestamps, nlp)

    # Write output
    with open(output_path, "w", encoding="utf-8") as f:
        for s in sentences:
            f.write(
                f"[{_format_timestamp(s['start'])} --> "
                f"{_format_timestamp(s['end'])}] {s['text']}\n"
            )
    print(f"  Saved: {output_path} ({len(sentences)} sentences)")


# ── Batch pipeline ──────────────────────────────────────────────────

def transcribe_files(files: list[Path], config: dict):
    """Transcribe a list of audio files."""
    output_dir = config.get("output_dir")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(config["model"]["dtype"], torch.bfloat16)

    asr_repo = config["model"]["asr"]
    asr_dir = Path(__file__).parent / "models" / asr_repo.split("/")[-1]

    print(f"ASR model: {asr_repo}")
    print(f"Aligner:   {ALIGNER_REPO}")
    print(f"Device:    {device}, dtype: {dtype}")
    if torch.cuda.is_available():
        print(f"GPU:       {torch.cuda.get_device_name()}")
    print_vram("start")

    # Download models to project-local ./models/
    asr_path = ensure_model_downloaded(asr_repo, asr_dir)
    aligner_path = ensure_model_downloaded(ALIGNER_REPO, ALIGNER_DIR)

    # Load ASR model
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(str(asr_path))
    model = CohereAsrForConditionalGeneration.from_pretrained(
        str(asr_path), dtype=dtype, device_map=device,
    )
    print(f"ASR loaded in {time.time() - t0:.1f}s")
    print_vram("after ASR load")

    # Load alignment model (both fit in 24GB VRAM simultaneously)
    t0 = time.time()
    alignment_model, alignment_tokenizer = load_alignment_model(
        device=device, dtype=dtype, model_path=str(aligner_path),
    )
    print(f"Aligner loaded in {time.time() - t0:.1f}s")
    print_vram("after aligner load")

    # Load spaCy once
    nlp = spacy.load("en_core_web_sm")

    # Process each file
    for i, file in enumerate(files):
        file_output_dir = output_dir or str(file.parent)
        os.makedirs(file_output_dir, exist_ok=True)
        output_path = os.path.join(file_output_dir, f"{file.stem} eng.txt")

        if os.path.exists(output_path):
            print(f"[{i+1}/{len(files)}] Skipping (exists): {file.name}")
            continue

        print(f"[{i+1}/{len(files)}] {file.name}")
        transcribe_file(
            str(file), output_path,
            processor, model,
            alignment_model, alignment_tokenizer,
            nlp, config, device, dtype,
        )

    print("\nAll files processed.")


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Cohere Transcribe audio transcription with timestamps",
    )
    parser.add_argument(
        "input",
        help="Audio file, directory, or glob pattern",
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory (default: same as input or config)",
    )
    parser.add_argument(
        "--config",
        default="config.json",
        help="Config file path (default: config.json)",
    )
    args = parser.parse_args()

    config = load_config(args.config, output_dir=args.output_dir)
    files = resolve_input_files(args.input)
    if not files:
        print(f"No audio files found for: {args.input}")
        sys.exit(1)

    print(f"Found {len(files)} audio file(s)")
    transcribe_files(files, config)


if __name__ == "__main__":
    main()
