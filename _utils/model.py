"""Model download and VRAM monitoring utilities."""
from pathlib import Path

import torch
from huggingface_hub import snapshot_download


def print_vram(label: str) -> None:
    """Print current GPU VRAM usage."""
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
