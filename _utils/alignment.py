"""Forced alignment constants."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ALIGNER_REPO = "jonatasgrosman/wav2vec2-large-xlsr-53-english"
ALIGNER_DIR = PROJECT_ROOT / "models" / "wav2vec2-large-xlsr-53-english"
