import subprocess
from pathlib import Path


def run_ffmpeg_to_m4a(input_path, output_path):
    """
    UnicodeDecodeErrorを回避するように修正された変換関数
    """
    command = [
        'ffmpeg',
        '-y',
        '-i', str(input_path),
        '-vn',
        '-c:a', 'copy',
        '-map_metadata', '0',
        str(output_path)
    ]

    # encoding='utf-8' と errors='ignore' を追加
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='ignore'
    )

    if result.returncode != 0:
        raise Exception(result.stderr)


def process_mp4_directory(target_dir="."):
    """
    パスの整理・保存先の準備を行い、順次変換を実行する
    """
    base_dir = Path(target_dir)
    output_dir = base_dir / "m4a"

    # 保存先ディレクトリの作成
    output_dir.mkdir(exist_ok=True)

    # .mp4ファイルの検索
    mp4_files = list(base_dir.glob("*.mp4"))

    if not mp4_files:
        print("変換対象の .mp4 ファイルが見つかりませんでした。")
        return

    print(f"{len(mp4_files)}件の音声抽出を開始します...")

    for input_p in mp4_files:
        # 出力パスの生成（拡張子を .m4a に変更）
        output_p = output_dir / (input_p.stem + ".m4a")

        print(f"抽出中: {input_p.name} ... ", end="", flush=True)

        try:
            # 変換処理を呼び出し
            run_ffmpeg_to_m4a(input_p, output_p)
            print("完了")
        except Exception as e:
            print(f"失敗\n  [Error]: {e}")


def main():
    # スクリプトを実行するディレクトリを指定
    process_mp4_directory("C:/Users/CVSLab/Music/No Voice/Global Initiative of Academic Networks/Phase Field Modelling")


if __name__ == "__main__":
    main()