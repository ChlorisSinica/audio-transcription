"""Audio loading and chunking utilities."""
import librosa
import torch

TARGET_SR = 16000


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
