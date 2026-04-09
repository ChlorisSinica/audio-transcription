import subprocess
import pathlib
import json
import re
import tempfile

TARGET_DIR = r"C:\Users\CVSLab\Music\No Voice\nature sound\1 Hour"
TARGET_LUFS = -16.0
TARGET_PEAK = -1.0
TARGET_LRA = 11.0

def win_path(p: pathlib.Path):
    return f"\\\\?\\{p.resolve()}"

def measure_loudness(input_path: pathlib.Path):
    cmd = [
        "ffmpeg", "-hide_banner", "-i", win_path(input_path),
        "-map", "0:a:0",  # 音声ストリームのみ
        "-af", f"loudnorm=I={TARGET_LUFS}:TP={TARGET_PEAK}:LRA={TARGET_LRA}:print_format=json",
        "-f", "null", "-"
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8")
    match = re.search(r"\{\s*\"input_i\".*?\}", result.stderr, re.S)
    if not match:
        raise ValueError(f"loudnorm JSON data not found for {input_path}")
    return json.loads(match.group(0))

def extract_album_art(input_path: pathlib.Path, temp_image: pathlib.Path):
    """
    元ファイルからアルバムアートを抽出（存在しない場合はスキップ）
    """
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-i", win_path(input_path),
        "-map", "0:v", "-c", "copy", win_path(temp_image)
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8")
    # 画像がない場合はエラーになるので無視
    if result.returncode != 0:
        return False
    return temp_image.exists()

def apply_loudness_with_art(input_path: pathlib.Path, output_path: pathlib.Path, params: dict, art_path: pathlib.Path):
    """
    正規化後にアルバムアートを再埋め込み
    """
    if art_path and art_path.exists():
        cmd = [
            "ffmpeg", "-y", "-hide_banner",
            "-i", win_path(input_path),   # 元ファイル（メタデータ用）
            "-i", win_path(art_path),     # アルバムアート
            "-map", "0:a:0",              # 音声ストリームのみ
            "-map", "1:v",                # 画像ストリーム
            "-af", (
                f"loudnorm=I={TARGET_LUFS}:TP={TARGET_PEAK}:LRA={TARGET_LRA}"
                f":measured_I={params['input_i']}"
                f":measured_TP={params['input_tp']}"
                f":measured_LRA={params['input_lra']}"
                f":measured_thresh={params['input_thresh']}"
                f":offset={params['target_offset']}"
                ":linear=true:print_format=summary"
            ),
            "-ar", "44100",
            "-c:a", "aac", "-b:a", "192k",
            "-c:v", "copy",               # 画像は再エンコードせずコピー
            "-movflags", "+faststart",
            "-map_metadata", "0",         # メタデータコピー
            "-f", "mp4",
            win_path(output_path)
        ]
    else:
        # アルバムアートなし
        cmd = [
            "ffmpeg", "-y", "-hide_banner",
            "-i", win_path(input_path),
            "-map", "0:a:0",
            "-af", (
                f"loudnorm=I={TARGET_LUFS}:TP={TARGET_PEAK}:LRA={TARGET_LRA}"
                f":measured_I={params['input_i']}"
                f":measured_TP={params['input_tp']}"
                f":measured_LRA={params['input_lra']}"
                f":measured_thresh={params['input_thresh']}"
                f":offset={params['target_offset']}"
                ":linear=true:print_format=summary"
            ),
            "-ar", "44100",
            "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart",
            "-map_metadata", "0",
            "-f", "mp4",
            win_path(output_path)
        ]

    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

def batch_normalize(folder: pathlib.Path):
    normalized_dir = folder / "normalized"
    normalized_dir.mkdir(exist_ok=True)
    file_list = folder.rglob("*.m4a")

    for file in file_list:
        relative_path = file.relative_to(folder)
        output_file = normalized_dir / relative_path
        output_file = output_file.with_stem(file.stem + "_normalized")
        output_file.parent.mkdir(parents=True, exist_ok=True)

        print(f"Measuring: {file}")
        try:
            params = measure_loudness(file)
        except Exception as e:
            print(f"⚠ Skipping {file.name}: {e}")
            continue

        # 一時ファイルにアルバムアート抽出
        with tempfile.TemporaryDirectory() as tmpdir:
            art_path = pathlib.Path(tmpdir) / "cover.jpg"
            has_art = extract_album_art(file, art_path)

            print(f"Normalizing: {file} -> {output_file}")
            apply_loudness_with_art(file, output_file, params, art_path if has_art else None)

if __name__ == "__main__":
    batch_normalize(pathlib.Path(TARGET_DIR))
    print("✅ メタデータ＋アルバムアート保持で正規化完了（normalizedフォルダに保存）")
