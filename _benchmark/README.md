# _benchmark

This directory contains evaluation helpers, not the main application flow.

## Files

- `verify_granite.py`: Granite speech verification baseline
- `verify_cohere.py`: Cohere transcription verification
- `verify_canary.py`: Canary verification through NeMo
- `compare_prompts.py`: prompt comparison tool for Granite

## Execution

Run from the repository root:

```powershell
python -m _benchmark.verify_granite <audio_file>
python -m _benchmark.verify_cohere <audio_file>
python -m _benchmark.verify_canary <audio_file>
python -m _benchmark.compare_prompts <audio_file>
```

Using `python -m` keeps imports stable after moving these scripts into a package directory.
