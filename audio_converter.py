"""Audio file converter: normalize loudness, MP4↔M4A conversion using FFmpeg."""

import argparse
import json
import pathlib
import re
import subprocess
import tempfile

DEFAULT_LUFS = -16.0
DEFAULT_PEAK = -1.0
DEFAULT_LRA = 11.0


# ── Utilities ─────────────────────────────────────────────────────────

def _win_path(p: pathlib.Path) -> str:
    """Windows long path prefix."""
    return f"\\\\?\\{p.resolve()}"


# ── Normalize ─────────────────────────────────────────────────────────

def _measure_loudness(input_path: pathlib.Path,
                      lufs: float, peak: float, lra: float) -> dict:
    cmd = [
        "ffmpeg", "-hide_banner", "-i", _win_path(input_path),
        "-map", "0:a:0",
        "-af", f"loudnorm=I={lufs}:TP={peak}:LRA={lra}:print_format=json",
        "-f", "null", "-",
    ]
    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8",
    )
    match = re.search(r"\{\s*\"input_i\".*?\}", result.stderr, re.S)
    if not match:
        raise ValueError(f"loudnorm JSON not found for {input_path}")
    return json.loads(match.group(0))


def _extract_album_art(input_path: pathlib.Path,
                       temp_image: pathlib.Path) -> bool:
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-i", _win_path(input_path),
        "-map", "0:v", "-c", "copy", _win_path(temp_image),
    ]
    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8",
    )
    return result.returncode == 0 and temp_image.exists()


def _apply_loudness(input_path: pathlib.Path, output_path: pathlib.Path,
                    params: dict, art_path: pathlib.Path | None,
                    lufs: float, peak: float, lra: float):
    af_filter = (
        f"loudnorm=I={lufs}:TP={peak}:LRA={lra}"
        f":measured_I={params['input_i']}"
        f":measured_TP={params['input_tp']}"
        f":measured_LRA={params['input_lra']}"
        f":measured_thresh={params['input_thresh']}"
        f":offset={params['target_offset']}"
        ":linear=true:print_format=summary"
    )
    base_cmd = ["ffmpeg", "-y", "-hide_banner", "-i", _win_path(input_path)]
    audio_opts = [
        "-af", af_filter,
        "-ar", "44100", "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart", "-map_metadata", "0", "-f", "mp4",
    ]

    if art_path and art_path.exists():
        cmd = base_cmd + [
            "-i", _win_path(art_path),
            "-map", "0:a:0", "-map", "1:v",
        ] + audio_opts + ["-c:v", "copy", _win_path(output_path)]
    else:
        cmd = base_cmd + ["-map", "0:a:0"] + audio_opts + [_win_path(output_path)]

    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def normalize(folder: str,
              lufs: float = DEFAULT_LUFS,
              peak: float = DEFAULT_PEAK,
              lra: float = DEFAULT_LRA):
    """Normalize M4A loudness (EBU R128, two-pass)."""
    folder = pathlib.Path(folder)
    out_dir = folder / "normalized"
    out_dir.mkdir(exist_ok=True)

    for file in folder.rglob("*.m4a"):
        out_file = out_dir / file.relative_to(folder)
        out_file = out_file.with_stem(file.stem + "_normalized")
        out_file.parent.mkdir(parents=True, exist_ok=True)

        print(f"Measuring: {file}")
        try:
            params = _measure_loudness(file, lufs, peak, lra)
        except Exception as e:
            print(f"  Skipping {file.name}: {e}")
            continue

        with tempfile.TemporaryDirectory() as tmpdir:
            art_path = pathlib.Path(tmpdir) / "cover.jpg"
            has_art = _extract_album_art(file, art_path)
            print(f"Normalizing: {file} -> {out_file}")
            _apply_loudness(file, out_file, params,
                            art_path if has_art else None, lufs, peak, lra)

    print("Normalize complete.")


# ── MP4 → M4A ─────────────────────────────────────────────────────────

def mp4_to_m4a(target_dir: str):
    """Extract audio from MP4 files to M4A (lossless copy)."""
    base = pathlib.Path(target_dir)
    out_dir = base / "m4a"
    out_dir.mkdir(exist_ok=True)

    files = list(base.glob("*.mp4"))
    if not files:
        print("No .mp4 files found.")
        return

    print(f"Extracting audio from {len(files)} file(s)...")
    for inp in files:
        outp = out_dir / (inp.stem + ".m4a")
        print(f"  {inp.name} ... ", end="", flush=True)
        try:
            cmd = [
                "ffmpeg", "-y", "-i", str(inp),
                "-vn", "-c:a", "copy", "-map_metadata", "0",
                str(outp),
            ]
            r = subprocess.run(cmd, capture_output=True, text=True,
                               encoding="utf-8", errors="ignore")
            if r.returncode != 0:
                raise Exception(r.stderr)
            print("done")
        except Exception as e:
            print(f"failed\n  [Error]: {e}")


# ── M4A → MP4 ─────────────────────────────────────────────────────────

def m4a_to_mp4(target_dir: str):
    """Convert M4A files to MP4 (with album art as video track)."""
    base = pathlib.Path(target_dir)
    out_dir = base / "mp4"
    out_dir.mkdir(exist_ok=True)

    files = list(base.glob("*.m4a"))
    if not files:
        print("No .m4a files found.")
        return

    print(f"Converting {len(files)} file(s)...")
    for inp in files:
        outp = out_dir / (inp.stem + ".mp4")
        print(f"  {inp.name} ... ", end="", flush=True)
        try:
            cmd = [
                "ffmpeg", "-y", "-i", str(inp),
                "-map", "0:a", "-map", "0:v?",
                "-c:a", "copy",
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                "-map_metadata", "0",
                str(outp),
            ]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                raise Exception(r.stderr)
            print("done")
        except Exception as e:
            print(f"failed\n  [Error]: {e}")


# ── CLI ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Audio file converter (FFmpeg wrapper)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_norm = sub.add_parser("normalize", help="Normalize M4A loudness (EBU R128)")
    p_norm.add_argument("directory", help="Directory containing .m4a files")
    p_norm.add_argument("--lufs", type=float, default=DEFAULT_LUFS)
    p_norm.add_argument("--peak", type=float, default=DEFAULT_PEAK)
    p_norm.add_argument("--lra", type=float, default=DEFAULT_LRA)

    p_m4a = sub.add_parser("mp4-to-m4a", help="Extract audio from MP4 to M4A")
    p_m4a.add_argument("directory", help="Directory containing .mp4 files")

    p_mp4 = sub.add_parser("m4a-to-mp4", help="Convert M4A to MP4")
    p_mp4.add_argument("directory", help="Directory containing .m4a files")

    args = parser.parse_args()

    if args.command == "normalize":
        normalize(args.directory, args.lufs, args.peak, args.lra)
    elif args.command == "mp4-to-m4a":
        mp4_to_m4a(args.directory)
    elif args.command == "m4a-to-mp4":
        m4a_to_mp4(args.directory)


if __name__ == "__main__":
    main()
