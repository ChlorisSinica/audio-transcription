"""NVIDIA Canary-Qwen-2.5B + ctc-forced-aligner verification script.

Minimal end-to-end test to validate:
  1. Canary-Qwen-2.5B loads via NeMo SALM API on local GPU
  2. Punctuation quality on real lecture audio (native PnC support!)
  3. ctc-forced-aligner integration for word-level timestamps
  4. VRAM consumption and throughput (RTFx)

Usage:
    python verify_canary.py <audio_file> [--chunk-sec 30] [--overlap-sec 2]

Key differences from verify_granite.py:
  - Uses NVIDIA NeMo toolkit (not transformers)
  - Canary expects audio as file paths, not tensors, so each chunk is
    written to a temporary .wav file for inference
  - Model loading is via SALM.from_pretrained() which uses NeMo's own
    caching mechanism (not snapshot_download). Model files end up in
    the NeMo cache, typically ~/.cache/huggingface/hub/. This is a
    known limitation; we accept it for the verify phase and will
    revisit in the main implementation.
  - bfloat16 inference (model was trained in bf16, 3090 Ti supports it)

Setup:
    This script requires a separate venv with NVIDIA NeMo installed.
    See the setup instructions in the top-level README or compare
    requirements.bat for the installation sequence. The NeMo install
    is large (~several GB) and Windows support is experimental.

License note:
    Canary-Qwen-2.5B is CC-BY-4.0 (commercial use allowed). The model
    uses Qwen3-1.7B as LLM decoder internally, but is packaged and
    distributed by NVIDIA as a single model.
"""
import argparse
import os
import re
import sys
import tempfile
import time
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

# Import stable primitives from the baseline Granite script.
# verify_granite.py is treated as read-only reference and provides:
# - TARGET_SR constant
# - print_vram helper
# - load_audio_16k_mono utility
# - chunk_audio utility
# - ensure_model_downloaded helper
# - ALIGNER_REPO, ALIGNER_DIR constants (English-native wav2vec2)
from _benchmark.verify_granite import (
    TARGET_SR,
    ALIGNER_REPO,
    ALIGNER_DIR,
    print_vram,
    load_audio_16k_mono,
    chunk_audio,
    ensure_model_downloaded,
)

# ctc-forced-aligner (same aligner as verify_granite)
from ctc_forced_aligner import (
    load_alignment_model,
    generate_emissions,
    preprocess_text,
    get_alignments,
    get_spans,
    postprocess_results,
)

# NeMo SALM (must be installed separately via nemo_toolkit[asr,tts])
try:
    from nemo.collections.speechlm2.models import SALM
except ImportError as e:
    print("ERROR: NVIDIA NeMo is not installed in this environment.")
    print("Install it with:")
    print('  pip install "nemo_toolkit[asr,tts] @ '
          'git+https://github.com/NVIDIA/NeMo.git"')
    print(f"\nOriginal error: {e}")
    sys.exit(1)

CANARY_REPO = "nvidia/canary-qwen-2.5b"

# Digit-to-word mapping for cleaning text before forced alignment.
# ctc-forced-aligner only handles alphabetic characters; digits cause
# AssertionError in get_spans().
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


def save_chunk_to_tempfile(
    chunk_wav: torch.Tensor, tmp_dir: Path, index: int
) -> Path:
    """Save an audio chunk tensor to a .wav file and return its path.

    NeMo SALM's generate() expects audio as file paths, not tensors,
    so we write each chunk to a temp file in the given directory.
    """
    audio_np = chunk_wav.squeeze(0).cpu().numpy()
    tmp_path = tmp_dir / f"chunk_{index:04d}.wav"
    sf.write(str(tmp_path), audio_np, TARGET_SR, subtype="PCM_16")
    return tmp_path


def transcribe_chunk_canary(
    chunk_wav_path: Path,
    model,
) -> str:
    """Transcribe a single audio chunk with Canary-Qwen SALM API."""
    prompts = [
        [
            {
                "role": "user",
                "content": (
                    f"Transcribe the following: {model.audio_locator_tag}"
                ),
                "audio": [str(chunk_wav_path)],
            }
        ]
    ]

    with torch.no_grad():
        answer_ids = model.generate(
            prompts=prompts,
            max_new_tokens=256,
        )

    text = model.tokenizer.ids_to_text(answer_ids[0].cpu())
    return text.strip()


def main(
    audio_path: str,
    chunk_sec: float = 30.0,
    overlap_sec: float = 2.0,
    output: str | None = None,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    print(f"Audio:     {audio_path}")
    print(f"ASR model: {CANARY_REPO}")
    print(f"Aligner:   {ALIGNER_REPO}")
    print(f"Device:    {device}, dtype: {dtype}")
    print(f"Chunking:  {chunk_sec}s chunks, {overlap_sec}s overlap")
    if torch.cuda.is_available():
        print(f"GPU:       {torch.cuda.get_device_name()}")
    print_vram("start")

    # === 1. Load Canary-Qwen via NeMo SALM ===
    print(f"\nLoading Canary-Qwen (this may take a while on first run)...")
    t0 = time.time()
    model = (
        SALM.from_pretrained(CANARY_REPO).bfloat16().eval().to(device)
    )
    print(f"Canary loaded in {time.time() - t0:.1f}s")
    print_vram("after Canary load")

    # === 2. Load audio and chunk ===
    t0 = time.time()
    wav = load_audio_16k_mono(audio_path)
    audio_duration = wav.shape[1] / TARGET_SR
    print(f"\nAudio loaded in {time.time() - t0:.1f}s")
    print(f"Audio duration: {audio_duration:.1f}s ({audio_duration/60:.1f}min)")

    chunks = chunk_audio(wav, chunk_sec, overlap_sec)
    print(f"Split into {len(chunks)} chunks")

    # === 3. Transcribe each chunk (via temp wav files) ===
    t0 = time.time()
    chunk_texts = []
    with tempfile.TemporaryDirectory(prefix="canary_chunks_") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        for i, (start_sec, end_sec, chunk_wav) in enumerate(chunks):
            if chunk_wav.shape[1] < TARGET_SR * 0.1:
                # Skip chunks shorter than 100ms (edge of file)
                print(f"  [{i+1:2d}/{len(chunks)}] "
                      f"{start_sec:6.1f}s-{end_sec:6.1f}s "
                      f"(skipped, too short)")
                chunk_texts.append("")
                continue

            chunk_t0 = time.time()
            chunk_path = save_chunk_to_tempfile(chunk_wav, tmp_dir, i)
            text = transcribe_chunk_canary(chunk_path, model)
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
    caps = sum(1 for c in full_text if c.isupper())
    print(f"Uppercase letters: {caps}")

    # === Save full transcription if requested ===
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            f.write(full_text)
        print(f"Transcription saved to: {output}")

    # === 5. Free Canary, load aligner ===
    del model
    torch.cuda.empty_cache()
    print_vram("after Canary unload")

    aligner_path = ensure_model_downloaded(ALIGNER_REPO, ALIGNER_DIR)
    t0 = time.time()
    alignment_model, alignment_tokenizer = load_alignment_model(
        device=str(device),
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
    # Clean text for aligner: convert digits to words, strip punctuation.
    # The aligner only handles alphabetic characters.
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
        description="Verify NVIDIA Canary-Qwen-2.5B + ctc-forced-aligner"
    )
    parser.add_argument("audio_path", help="Path to audio file")
    parser.add_argument(
        "--chunk-sec",
        type=float,
        default=30.0,
        help="Chunk duration in seconds (default: 30, max 40 for Canary)",
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

    # Canary trained max is 40s; clamp to stay safely within that
    if args.chunk_sec > 38:
        print(f"WARNING: chunk_sec={args.chunk_sec} exceeds Canary's "
              f"40s training limit. Clamping to 38s.")
        args.chunk_sec = 38.0

    main(args.audio_path, args.chunk_sec, args.overlap_sec, args.output)
