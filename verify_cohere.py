"""Cohere Transcribe 03-2026 + ctc-forced-aligner verification script.

Minimal end-to-end test to validate:
  1. Cohere Transcribe loads and runs on local GPU
  2. Punctuation quality on real lecture audio (native PnC support)
  3. Long-form audio handling via manual chunking
  4. ctc-forced-aligner integration for word-level timestamps
  5. VRAM consumption and throughput (RTFx)

Usage:
    python verify_cohere.py <audio_file> [--chunk-sec 30] [--overlap-sec 2]

Chunking:
    Cohere Transcribe's internal ~35s chunking did not produce full
    transcripts for 7+ min audio in testing (only ~30s was transcribed).
    We therefore use manual chunking, matching verify_granite.py's approach.

Model storage:
    Both Cohere and the alignment model are stored in project-local
    ./models/ to make the project self-contained.
    No files are written to ~/.cache/huggingface.

Aligner choice:
    Uses the English-native wav2vec2 model from verify_granite.py
    (same rationale: avoids romanization issues with MMS default).

Notes:
    - Ampere GPUs (RTX 3090 Ti, SM 8.6) natively support bfloat16.
    - Requires transformers>=4.52.1 for CohereAsrForConditionalGeneration.
    - Audio loading uses librosa (via audioread/ffmpeg backend).
"""
import argparse
import re
import time
from pathlib import Path

import librosa
import numpy as np
import torch
from transformers import AutoProcessor, CohereAsrForConditionalGeneration

# Import stable primitives from the baseline Granite script.
from verify_granite import (
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

# ASR model
COHERE_REPO = "CohereLabs/cohere-transcribe-03-2026"
COHERE_DIR = Path(__file__).parent / "models" / "cohere-transcribe-03-2026"

_DIGIT_WORDS = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
}


def clean_text_for_aligner(text: str) -> str:
    """Convert digits to words and strip non-alpha chars for the aligner."""
    for digit, word in _DIGIT_WORDS.items():
        text = text.replace(digit, f" {word} ")
    text = re.sub(r"[^a-zA-Z\s']", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def transcribe_chunk_cohere(
    chunk_wav: torch.Tensor,
    processor,
    model,
    device: str,
    dtype: torch.dtype,
) -> str:
    """Transcribe a single audio chunk with Cohere."""
    chunk_np = chunk_wav.squeeze(0).numpy()
    inputs = processor(
        chunk_np, sampling_rate=TARGET_SR, return_tensors="pt", language="en"
    )
    inputs = inputs.to(device, dtype=dtype)

    chunk_duration = chunk_wav.shape[1] / TARGET_SR
    max_new_tokens = max(128, int(chunk_duration * 5) + 64)

    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    text = processor.decode(outputs[0], skip_special_tokens=True)
    return text.strip()


def main(
    audio_path: str,
    chunk_sec: float = 30.0,
    overlap_sec: float = 2.0,
    output: str | None = None,
) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print(f"Audio:     {audio_path}")
    print(f"ASR model: {COHERE_REPO}")
    print(f"Aligner:   {ALIGNER_REPO}")
    print(f"Device:    {device}, dtype: {dtype}")
    print(f"Chunking:  {chunk_sec}s chunks, {overlap_sec}s overlap")
    if torch.cuda.is_available():
        print(f"GPU:       {torch.cuda.get_device_name()}")
    print_vram("start")

    # === 0. Ensure both models are downloaded to project-local dirs ===
    cohere_path = ensure_model_downloaded(COHERE_REPO, COHERE_DIR)
    aligner_path = ensure_model_downloaded(ALIGNER_REPO, ALIGNER_DIR)

    # === 1. Load Cohere Transcribe ===
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(str(cohere_path))
    model = CohereAsrForConditionalGeneration.from_pretrained(
        str(cohere_path), torch_dtype=dtype, device_map=device,
    )
    print(f"\nCohere loaded in {time.time() - t0:.1f}s")
    print_vram("after Cohere load")

    # === 2. Load audio and chunk ===
    t0 = time.time()
    wav = load_audio_16k_mono(audio_path)
    audio_duration = wav.shape[1] / TARGET_SR
    print(f"\nAudio loaded in {time.time() - t0:.1f}s")
    print(f"Audio duration: {audio_duration:.1f}s ({audio_duration/60:.1f}min)")

    chunks = chunk_audio(wav, chunk_sec, overlap_sec)
    print(f"Split into {len(chunks)} chunks")

    # === 3. Transcribe each chunk ===
    t0 = time.time()
    chunk_texts = []
    for i, (start_sec, end_sec, chunk_wav) in enumerate(chunks):
        chunk_t0 = time.time()
        text = transcribe_chunk_cohere(
            chunk_wav, processor, model, device, dtype
        )
        chunk_time = time.time() - chunk_t0
        print(f"  [{i+1:2d}/{len(chunks)}] "
              f"{start_sec:6.1f}s-{end_sec:6.1f}s "
              f"({chunk_time:5.1f}s) "
              f"{len(text.split()):4d} words")
        chunk_texts.append(text)

    asr_time = time.time() - t0
    rtfx = audio_duration / asr_time
    print(f"\nASR done in {asr_time:.1f}s (RTFx={rtfx:.1f}x)")
    print_vram("after ASR")

    full_text = " ".join(t for t in chunk_texts if t)

    # === 4. Transcription diagnostics ===
    print("\n" + "=" * 60)
    print("Transcription (first 1000 chars)")
    print("=" * 60)
    print(full_text[:1000])
    print("=" * 60)
    word_count = len(full_text.split())
    print(f"Full length: {len(full_text)} chars, {word_count} words")
    print(f"Punctuation counts: "
          f". = {full_text.count('.')}, "
          f", = {full_text.count(',')}, "
          f"? = {full_text.count('?')}, "
          f"! = {full_text.count('!')}")

    if word_count < 10:
        print("WARNING: Output is suspiciously short — "
              "possible generation failure")
    if full_text.count(".") == 0 and audio_duration > 60:
        print("WARNING: No period in long audio — "
              "punctuation may have failed")

    # === Save full transcription if requested ===
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            f.write(full_text)
        print(f"Transcription saved to: {output}")

    # === 5. Free Cohere, load aligner ===
    del model
    torch.cuda.empty_cache()
    print_vram("after Cohere unload")

    t0 = time.time()
    alignment_model, alignment_tokenizer = load_alignment_model(
        device=device,
        dtype=dtype,
        model_path=str(aligner_path),
    )
    print(f"\nAligner loaded in {time.time() - t0:.1f}s")
    print_vram("after aligner load")

    # === 6. Run forced alignment ===
    t0 = time.time()
    audio_tensor = wav.squeeze(0).to(device, dtype=dtype)
    emissions, stride = generate_emissions(
        alignment_model, audio_tensor, batch_size=16
    )
    align_text = clean_text_for_aligner(full_text)
    tokens_starred, text_starred = preprocess_text(
        align_text, romanize=False, language="eng"
    )
    segments, scores, blank_token = get_alignments(
        emissions, tokens_starred, alignment_tokenizer
    )
    spans = get_spans(tokens_starred, segments, blank_token)
    word_timestamps = postprocess_results(
        text_starred, spans, stride, scores
    )
    align_time = time.time() - t0
    print(f"Alignment done in {align_time:.1f}s")

    # === 7. Sample word timestamps ===
    print("\n=== First 15 word timestamps ===")
    for w in word_timestamps[:15]:
        print(f"  [{w['start']:8.3f}s - {w['end']:8.3f}s] {w['text']}")
    if len(word_timestamps) > 15:
        print(f"\n=== Last 5 word timestamps ===")
        for w in word_timestamps[-5:]:
            print(f"  [{w['start']:8.3f}s - {w['end']:8.3f}s] {w['text']}")
    print(f"\nTotal aligned words: {len(word_timestamps)}")

    # === 8. Summary ===
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Audio duration: {audio_duration:7.1f}s "
          f"({audio_duration / 60:.1f}min)")
    print(f"Chunks:         {len(chunks)}")
    print(f"ASR time:       {asr_time:7.1f}s  (RTFx = {rtfx:.1f}x)")
    print(f"Alignment time: {align_time:7.1f}s")
    print(f"Total wall:     {asr_time + align_time:7.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Verify Cohere Transcribe + ctc-forced-aligner"
    )
    parser.add_argument("audio_path", help="Path to audio file")
    parser.add_argument(
        "--chunk-sec",
        type=float,
        default=30.0,
        help="Chunk duration in seconds (default: 30)",
    )
    parser.add_argument(
        "--overlap-sec",
        type=float,
        default=2.0,
        help="Overlap between chunks in seconds (default: 2)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Save full transcription text to this file",
    )
    args = parser.parse_args()
    main(args.audio_path, args.chunk_sec, args.overlap_sec, args.output)
