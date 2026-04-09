"""IBM Granite 4.0 1B Speech + ctc-forced-aligner verification script.

Minimal end-to-end test to validate:
  1. Granite 4.0 1B Speech loads and runs on local GPU
  2. Punctuation quality on real lecture audio (via chunking for long-form)
  3. Keyword biasing for technical vocabulary
  4. ctc-forced-aligner integration for word-level timestamps
  5. VRAM consumption and throughput (RTFx)

Usage:
    python verify_granite.py <audio_file> [--keywords "SEM, BSE, alumina"]
                                           [--chunk-sec 30] [--overlap-sec 2]

Chunking:
    Granite Speech is trained on short audio (~3 min max, per vLLM config).
    Long-form audio (>3 min) must be split into chunks to avoid degenerate
    repetition loops. This script uses fixed-duration chunks with a small
    overlap. The real implementation in audio_transcriber.py should use
    VAD-based chunking for cleaner sentence boundaries.

Model storage:
    Both Granite and the alignment model are stored in project-local
    ./models/ (flat directories) to make the project self-contained.
    No files are written to ~/.cache/huggingface.

Aligner choice:
    We use jonatasgrosman/wav2vec2-large-xlsr-53-english instead of the
    default multilingual MMS model because:
      - English-native vocabulary avoids romanization requirements
      - Avoids a ValueError in get_alignments() seen with MMS default
      - English-specific model is marginally more accurate for English

Notes:
    - Ampere GPUs (RTX 3090 Ti, SM 8.6) natively support bfloat16.
    - Requires transformers>=4.52.1 for native Granite Speech support.
    - Audio loading uses librosa (via audioread/ffmpeg backend).
"""
import argparse
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

from ctc_forced_aligner import (
    load_alignment_model,
    generate_emissions,
    preprocess_text,
    get_alignments,
    get_spans,
    postprocess_results,
)

# ASR model
GRANITE_REPO = "ibm-granite/granite-4.0-1b-speech"
GRANITE_DIR = Path(__file__).parent / "models" / "granite-4.0-1b-speech"

# Alignment model (English-native, avoids romanization issues)
ALIGNER_REPO = "jonatasgrosman/wav2vec2-large-xlsr-53-english"
ALIGNER_DIR = Path(__file__).parent / "models" / "wav2vec2-large-xlsr-53-english"

TARGET_SR = 16000


def print_vram(label: str) -> None:
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"[VRAM {label:22s}] allocated={allocated:5.2f}GB "
              f"reserved={reserved:5.2f}GB")


def ensure_model_downloaded(repo_id: str, local_dir: Path) -> Path:
    """Download a model to local_dir as a flat directory if not present."""
    if (local_dir / "config.json").exists():
        print(f"Model already present at: {local_dir}")
        return local_dir

    print(f"Downloading {repo_id} to: {local_dir}")
    local_dir.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
    )
    print("Download complete.")
    return local_dir


def load_audio_16k_mono(audio_path: str) -> torch.Tensor:
    """Load audio as 16kHz mono via librosa. Returns shape (1, num_samples)."""
    audio_np, _ = librosa.load(audio_path, sr=TARGET_SR, mono=True)
    wav = torch.from_numpy(audio_np).unsqueeze(0)  # (1, num_samples)
    return wav


def chunk_audio(
    wav: torch.Tensor, chunk_sec: float, overlap_sec: float
) -> list[tuple[float, float, torch.Tensor]]:
    """Split audio into fixed-duration chunks with overlap."""
    num_samples = wav.shape[1]
    chunk_samples = int(chunk_sec * TARGET_SR)
    overlap_samples = int(overlap_sec * TARGET_SR)
    stride = chunk_samples - overlap_samples

    chunks = []
    start = 0
    while start < num_samples:
        end = min(start + chunk_samples, num_samples)
        chunk_wav = wav[:, start:end]
        chunks.append((start / TARGET_SR, end / TARGET_SR, chunk_wav))
        if end == num_samples:
            break
        start += stride

    return chunks


def transcribe_chunk(
    chunk_wav: torch.Tensor,
    processor,
    tokenizer,
    model,
    device: str,
    keywords: str | None,
) -> str:
    """Transcribe a single audio chunk with Granite."""
    user_prompt = "<|audio|>can you transcribe the speech into a written format?"
    if keywords:
        user_prompt += f" Keywords: {keywords}"

    chat = [{"role": "user", "content": user_prompt}]
    prompt = tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True
    )

    model_inputs = processor(
        prompt, chunk_wav, device=device, return_tensors="pt"
    ).to(device)

    chunk_duration = chunk_wav.shape[1] / TARGET_SR
    max_new_tokens = max(128, int(chunk_duration * 5) + 64)

    with torch.no_grad():
        model_outputs = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            repetition_penalty=1.2,  # helps prevent loops
        )

    num_input_tokens = model_inputs["input_ids"].shape[-1]
    new_tokens = model_outputs[0, num_input_tokens:].unsqueeze(0)
    output_text = tokenizer.batch_decode(
        new_tokens, add_special_tokens=False, skip_special_tokens=True
    )
    return output_text[0].strip()


def main(
    audio_path: str,
    keywords: str | None = None,
    chunk_sec: float = 30.0,
    overlap_sec: float = 2.0,
    output: str | None = None,
) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print(f"Audio:     {audio_path}")
    print(f"ASR model: {GRANITE_REPO}")
    print(f"Aligner:   {ALIGNER_REPO}")
    print(f"Device:    {device}, dtype: {dtype}")
    print(f"Keywords:  {keywords if keywords else '(none)'}")
    print(f"Chunking:  {chunk_sec}s chunks, {overlap_sec}s overlap")
    if torch.cuda.is_available():
        print(f"GPU:       {torch.cuda.get_device_name()}")
    print_vram("start")

    # === 0. Ensure both models are downloaded to project-local dirs ===
    granite_path = ensure_model_downloaded(GRANITE_REPO, GRANITE_DIR)
    aligner_path = ensure_model_downloaded(ALIGNER_REPO, ALIGNER_DIR)

    # === 1. Load Granite ===
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(str(granite_path))
    tokenizer = processor.tokenizer
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        str(granite_path), device_map=device, dtype=dtype
    )
    model.eval()
    print(f"\nGranite loaded in {time.time() - t0:.1f}s")
    print_vram("after Granite load")

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
        text = transcribe_chunk(
            chunk_wav, processor, tokenizer, model, device, keywords
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

    full_text = " ".join(chunk_texts)

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

    # === Save full transcription if requested ===
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            f.write(full_text)
        print(f"Transcription saved to: {output}")

    # === 5. Free Granite, load aligner ===
    del model
    torch.cuda.empty_cache()
    print_vram("after Granite unload")

    t0 = time.time()
    alignment_model, alignment_tokenizer = load_alignment_model(
        device=device,
        dtype=dtype,
        model_path=str(aligner_path),  # local path instead of HF repo ID
    )
    print(f"\nAligner loaded in {time.time() - t0:.1f}s")
    print_vram("after aligner load")

    # === 6. Run forced alignment ===
    t0 = time.time()
    audio_tensor = wav.squeeze(0).to(device, dtype=dtype)
    emissions, stride = generate_emissions(
        alignment_model, audio_tensor, batch_size=16
    )
    # romanize=False because we use an English-native model with its own
    # English vocabulary (no need for unicode-to-latin romanization).
    tokens_starred, text_starred = preprocess_text(
        full_text, romanize=False, language="eng"
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
        description="Verify IBM Granite 4.0 1B Speech + ctc-forced-aligner"
    )
    parser.add_argument("audio_path", help="Path to audio file")
    parser.add_argument(
        "--keywords",
        default=None,
        help="Comma-separated keyword list for biasing",
    )
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
    main(args.audio_path, args.keywords, args.chunk_sec, args.overlap_sec,
         args.output)
