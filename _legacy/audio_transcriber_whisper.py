"""Whisper-based audio transcription with punctuation restoration."""

import argparse
import os
import re
import sys
from pathlib import Path

import spacy
import torch
import whisper
from deepmultilingualpunctuation import PunctuationModel

AUDIO_EXTENSIONS = {".m4a", ".mp3", ".wav", ".flac", ".ogg", ".mp4", ".webm"}

# spaCy model for sentence segmentation
# (download: python -m spacy download en_core_web_sm)
nlp = spacy.load("en_core_web_sm")


def _format_timestamp(seconds: float) -> str:
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes:02}:{secs:06.3f}"


def _clean_spacing_and_capitalize(text: str) -> str:
    text = re.sub(r"\s+([?.!])", r"\1", text)
    text = re.sub(r"\s+'", "'", text)
    text = re.sub(r"'\s+", "'", text)
    text = re.sub(r"([.?!])\s+([a-z]\w*)",
                  lambda m: m.group(1) + " " + m.group(2).capitalize(), text)
    if text:
        text = text[0].upper() + text[1:]
    return text


def transcribe_file(path_audio: str, path_output: str,
                    whisper_model=None, punct_model=None):
    """Transcribe a single audio file with timestamps and punctuation."""
    if whisper_model is None:
        whisper_model = whisper.load_model("large-v3", device="cuda")
    if punct_model is None:
        punct_model = PunctuationModel()

    result = whisper_model.transcribe(
        path_audio, verbose=False, fp16=False, language="en",
        word_timestamps=True,
        initial_prompt=(
            "This is an English transcription. "
            "Use only English words with proper punctuation."
        ),
    )

    words = []
    for seg in result["segments"]:
        for w in seg["words"]:
            words.append({
                "start": w["start"], "end": w["end"], "text": w["word"],
            })

    full_text = "".join(w["text"] for w in words)
    punctuated = punct_model.restore_punctuation(full_text)
    punctuated = _clean_spacing_and_capitalize(punctuated)

    doc = nlp(punctuated)
    sentences = []
    word_index = 0
    for sent in doc.sents:
        sent_text = sent.text.strip()
        sent_words = sent_text.split()
        start_time = words[word_index]["start"]
        end_time = words[min(word_index + len(sent_words) - 1,
                             len(words) - 1)]["end"]
        sentences.append({
            "start": start_time, "end": end_time, "text": sent_text,
        })
        word_index += len(sent_words)

    with open(path_output, "w", encoding="utf-8") as f:
        for s in sentences:
            f.write(
                f"[{_format_timestamp(s['start'])} --> "
                f"{_format_timestamp(s['end'])}] {s['text']}\n"
            )

    print(f"Saved: {path_output}")


def transcribe_directory(input_dir: str, output_dir: str | None = None):
    """Transcribe all audio files in a directory."""
    if output_dir is None:
        output_dir = input_dir
    os.makedirs(output_dir, exist_ok=True)

    whisper_model = whisper.load_model("large-v3", device="cuda")
    punct_model = PunctuationModel()

    for file in sorted(Path(input_dir).iterdir()):
        if file.suffix.lower() in AUDIO_EXTENSIONS:
            path_output = os.path.join(output_dir, f"{file.stem} eng.txt")
            print(file)
            transcribe_file(str(file), path_output, whisper_model, punct_model)


def transcribe_mse403(m4a_dir: str, output_dir: str,
                      lec_start: int, lec_end: int):
    """Transcribe MSE403 lecture files (specialized naming convention)."""
    whisper_model = whisper.load_model("large-v3", device="cuda")
    punct_model = PunctuationModel()

    for i in range(lec_start, lec_end + 1):
        for m in range(0, 6):
            name = f"Lecture {i:02} - Module {m}"
            path_m4a = os.path.join(m4a_dir, f"{name} normalized.m4a")
            path_output = os.path.join(output_dir, f"{name} eng.txt")
            if os.path.exists(path_m4a) and not os.path.exists(path_output):
                print(path_m4a)
                transcribe_file(path_m4a, path_output,
                                whisper_model, punct_model)


def main():
    parser = argparse.ArgumentParser(
        description="Whisper audio transcription with punctuation restoration",
    )
    parser.add_argument("input_dir", help="Directory containing audio files")
    parser.add_argument("--output-dir",
                        help="Output directory (default: same as input)")
    args = parser.parse_args()

    transcribe_directory(args.input_dir, args.output_dir)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(sys.argv[0])))
    print("###############################")
    print("cpu core:", os.cpu_count())
    print("torch version:", torch.__version__)
    print("CUDA:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device count:", torch.cuda.device_count())
        print("current device:", torch.cuda.current_device())
        print("GPU name:", torch.cuda.get_device_name())
        print("device capacity:", torch.cuda.get_device_capability())
    print("###############################")
    main()
