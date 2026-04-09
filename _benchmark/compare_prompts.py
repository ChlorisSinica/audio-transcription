"""Prompt comparison tool for exploring Granite transcription prompts.

This is an exploration helper, SEPARATE from verify_granite.py. The latter
serves as the clean baseline pipeline and should not be modified during
prompt exploration. This script imports the stable primitives from
verify_granite.py and adds prompt-comparison logic on top.

Purpose:
    Granite 4.0 1B Speech was trained on LibriSpeech-style unpunctuated
    data, so its default output has no punctuation. We hypothesize that
    explicit instructions in the prompt may activate the base LLM's
    punctuation capabilities. This script runs multiple candidate prompts
    on a short audio sample and compares punctuation quality.

Usage:
    python compare_prompts.py <audio_file> [--sample-sec 30]

Once a winning prompt is identified, delete this script and carry the
prompt forward to audio_transcriber.py (the main implementation).
"""
import argparse
import time

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

# Import stable primitives from the baseline verification script.
# verify_granite.py is treated as read-only reference.
from _benchmark.verify_granite import (
    GRANITE_REPO,
    GRANITE_DIR,
    TARGET_SR,
    ensure_model_downloaded,
    load_audio_16k_mono,
    print_vram,
)

# Candidate prompts to test. Each starts with <|audio|> which is the
# required placeholder for audio injection in the Granite chat template.
PROMPTS: dict[str, str] = {
    "A_baseline": (
        "<|audio|>can you transcribe the speech into a written format?"
    ),
    "B_explicit": (
        "<|audio|>Please transcribe this audio with proper punctuation, "
        "capitalization, and sentence boundaries."
    ),
    "C_formatted": (
        "<|audio|>Transcribe the speech as a written document. "
        "Use full punctuation (periods, commas, question marks) and "
        "capitalize proper nouns and sentence beginnings."
    ),
    "D_verbatim": (
        "<|audio|>Provide a verbatim transcription of this lecture audio, "
        "formatted with standard English punctuation and capitalization."
    ),
    "E_document": (
        "<|audio|>Write out what is said in this audio as a properly "
        "formatted English paragraph with correct punctuation and casing."
    ),
}


def run_prompt(
    prompt_template: str,
    chunk_wav: torch.Tensor,
    processor,
    tokenizer,
    model,
    device: str,
) -> str:
    """Run a single prompt on an audio sample and return the text output."""
    chat = [{"role": "user", "content": prompt_template}]
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
            repetition_penalty=1.2,
        )
    num_input_tokens = model_inputs["input_ids"].shape[-1]
    new_tokens = model_outputs[0, num_input_tokens:].unsqueeze(0)
    output_text = tokenizer.batch_decode(
        new_tokens, add_special_tokens=False, skip_special_tokens=True
    )
    return output_text[0].strip()


def count_punctuation(text: str) -> dict[str, int]:
    return {
        "period": text.count("."),
        "comma": text.count(","),
        "question": text.count("?"),
        "exclaim": text.count("!"),
        "semicolon": text.count(";"),
        "colon": text.count(":"),
    }


def count_caps(text: str) -> int:
    """Count uppercase letters (proxy for capitalization behavior)."""
    return sum(1 for c in text if c.isupper())


def main(audio_path: str, sample_sec: float = 30.0) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print(f"Audio:     {audio_path}")
    print(f"Sample:    first {sample_sec}s")
    print(f"Device:    {device}, dtype: {dtype}")
    print(f"Prompts:   {len(PROMPTS)} candidates")
    if torch.cuda.is_available():
        print(f"GPU:       {torch.cuda.get_device_name()}")
    print_vram("start")

    # === Load Granite ===
    granite_path = ensure_model_downloaded(GRANITE_REPO, GRANITE_DIR)
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(str(granite_path))
    tokenizer = processor.tokenizer
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        str(granite_path), device_map=device, dtype=dtype
    )
    model.eval()
    print(f"\nGranite loaded in {time.time() - t0:.1f}s")
    print_vram("after load")

    # === Load audio and trim to sample ===
    wav = load_audio_16k_mono(audio_path)
    sample_samples = int(sample_sec * TARGET_SR)
    if wav.shape[1] > sample_samples:
        wav = wav[:, :sample_samples]
    actual_duration = wav.shape[1] / TARGET_SR
    print(f"\nProcessing {actual_duration:.1f}s sample")

    # === Run each prompt ===
    print("\n" + "=" * 72)
    results: dict[str, str] = {}
    for name, prompt_template in PROMPTS.items():
        t0 = time.time()
        text = run_prompt(
            prompt_template, wav, processor, tokenizer, model, device
        )
        elapsed = time.time() - t0
        results[name] = text

        print(f"\n[{name}]  ({elapsed:.1f}s)")
        # Show prompt without the <|audio|> marker for readability
        readable_prompt = prompt_template.replace("<|audio|>", "")
        print(f"  Prompt : {readable_prompt}")
        print(f"  Output : {text[:400]}{'...' if len(text) > 400 else ''}")

    # === Summary table ===
    print("\n" + "=" * 72)
    print("Summary")
    print("=" * 72)
    header = f"{'Prompt':<14} {'Words':>6} {'.':>4} {',':>4} {'?':>4} {'!':>4} {'Caps':>6}"
    print(header)
    print("-" * 72)
    for name, text in results.items():
        words = len(text.split())
        p = count_punctuation(text)
        caps = count_caps(text)
        print(f"{name:<14} {words:>6} {p['period']:>4} {p['comma']:>4} "
              f"{p['question']:>4} {p['exclaim']:>4} {caps:>6}")

    # === Recommendation hint ===
    print("\n" + "=" * 72)
    # Find the prompt with the most total punctuation (simple heuristic)
    best_name = max(
        results.keys(),
        key=lambda n: sum(count_punctuation(results[n]).values())
    )
    best_punct = sum(count_punctuation(results[best_name]).values())
    if best_punct == 0:
        print("WARNING: No prompt produced any punctuation.")
        print("Granite may be fundamentally unable to output punctuation.")
        print("Consider switching to Qwen3-ASR or another model.")
    else:
        print(f"Best candidate by punctuation count: {best_name} "
              f"({best_punct} marks)")
        print("Inspect the output text quality before committing to it.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare Granite prompts for punctuation quality"
    )
    parser.add_argument("audio_path", help="Path to audio file")
    parser.add_argument(
        "--sample-sec",
        type=float,
        default=30.0,
        help="Duration of audio sample to use in seconds (default: 30)",
    )
    args = parser.parse_args()
    main(args.audio_path, args.sample_sec)
