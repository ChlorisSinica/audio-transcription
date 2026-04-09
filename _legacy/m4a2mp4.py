import subprocess
from pathlib import Path


def run_ffmpeg_conversion(input_path, output_path):
    """
    FFmpegを実行して1つのファイルを変換する（コアロジック）
    """
    command = [
        'ffmpeg',
        '-y',
        '-i', str(input_path),
        '-map', '0:a',  # 音声
        '-map', '0:v?',  # アルバムアート（存在する場合のみ）
        '-c:a', 'copy',  # 音声は無劣化
        '-c:v', 'libx264',  # 画像を動画として変換
        '-pix_fmt', 'yuv420p',
        '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
        '-map_metadata', '0',  # メタデータを継承
        str(output_path)
    ]

    # capture_output=Trueでエラーログを取得可能にする
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(result.stderr)


def process_m4a_directory(target_dir="."):
    """
    パスの整理・保存先の準備を行い、順次変換を実行する
    """
    base_dir = Path(target_dir)
    output_dir = base_dir / "mp4"

    # 保存先ディレクトリの作成
    output_dir.mkdir(exist_ok=True)

    # .m4aファイルの検索
    m4a_files = list(base_dir.glob("*.m4a"))

    if not m4a_files:
        print("変換対象のファイルが見つかりませんでした。")
        return

    print(f"{len(m4a_files)}件の処理を開始します...")

    for input_p in m4a_files:
        # 出力パスの生成（ファイル名はそのままで拡張子のみ変更）
        output_p = output_dir / (input_p.stem + ".mp4")

        print(f"処理中: {input_p.name} ... ", end="", flush=True)

        try:
            # パスが整理された状態で変換処理を呼び出す
            run_ffmpeg_conversion(input_p, output_p)
            print("完了")
        except Exception as e:
            print(f"失敗\n  [Error]: {e}")


def main():
    # 実行
    process_m4a_directory("C:/Users/CVSLab/Music/No Voice/Global Initiative of Academic Networks/Phase Field Modelling")


if __name__ == "__main__":
    main()