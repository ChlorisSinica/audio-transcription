# audio-transcription

Audio transcription workspace for local model evaluation and batch processing.

## Structure

- `audio_transcriber.py`: main transcription pipeline
- `audio_converter.py`: FFmpeg-based audio conversion and normalization
- `_benchmark/`: model verification and prompt comparison scripts
- `_legacy/`: old scripts kept for reference only
- `models/`: locally cached model files
- `config.json`: default settings for the main transcription pipeline
- `requirements.bat`: Windows setup script for `.venv`

## Main usage

Run the main pipeline:

```powershell
python audio_transcriber.py <input>
python audio_transcriber.py <input> --output-dir .\out
python audio_transcriber.py <input> --config config.json
```

`<input>` can be a file, a directory, or a glob pattern.

Examples:

```powershell
python audio_transcriber.py .\_audio\lecture.m4a
python audio_transcriber.py .\_audio\course01\
python audio_transcriber.py ".\_audio\**\*.m4a"
```

## Benchmark scripts

Benchmark and verification helpers are grouped under `_benchmark/`.

Examples:

```powershell
python -m _benchmark.verify_granite <audio_file>
python -m _benchmark.verify_cohere <audio_file>
python -m _benchmark.verify_canary <audio_file>
python -m _benchmark.compare_prompts <audio_file>
```

These scripts are for model checks, prompt exploration, and performance comparison. They are not the production entrypoint.

## Setup

On Windows, install dependencies into the local virtual environment with:

```powershell
.\requirements.bat
```

Prerequisites:

- `.venv` already exists
- `ffmpeg` is installed and available on `PATH`
- GPU drivers and CUDA-compatible PyTorch environment are available if using GPU inference

## Notes

- `audio_transcriber.py` uses helpers from `_benchmark.verify_granite`.
- Downloaded model files are stored under `models/` in this repository.
- `_legacy/` is intentionally not part of the active workflow.
