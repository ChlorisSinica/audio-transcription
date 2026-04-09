# audio-transcription

音声文字起こしとモデル検証のためのローカル作業用リポジトリです。

## 構成

- `audio_transcriber.py`: 本番用の文字起こしパイプライン
- `audio_converter.py`: FFmpeg を使った音声変換・音量正規化
- `_benchmark/`: モデル検証、比較、プロンプト実験用スクリプト
- `_legacy/`: 旧スクリプトの退避先
- `models/`: ローカル保存するモデル本体
- `config.json`: `audio_transcriber.py` の既定設定
- `requirements.bat`: Windows 用セットアップスクリプト

## 主な使い方

文字起こしの実行例:

```powershell
python audio_transcriber.py <input>
python audio_transcriber.py <input> --output-dir .\out
python audio_transcriber.py <input> --config config.json
```

`<input>` には以下を指定できます。

- 単一ファイル
- ディレクトリ
- glob パターン

例:

```powershell
python audio_transcriber.py .\_audio\lecture.m4a
python audio_transcriber.py .\_audio\course01\
python audio_transcriber.py ".\_audio\**\*.m4a"
```

## ベンチマーク・検証

検証系スクリプトは `_benchmark/` にまとめています。

実行例:

```powershell
python -m _benchmark.verify_granite <audio_file>
python -m _benchmark.verify_cohere <audio_file>
python -m _benchmark.verify_canary <audio_file>
python -m _benchmark.compare_prompts <audio_file>
```

これらはモデル比較、動作確認、プロンプト探索用です。通常の文字起こし入口は `audio_transcriber.py` です。

## セットアップ

Windows では、ローカル仮想環境 `.venv` に依存関係を入れる前提です。

```powershell
.\requirements.bat
```

前提条件:

- `.venv` が事前に作成済みであること
- `ffmpeg` が `PATH` から実行できること
- GPU 推論を使う場合は、対応するドライバと PyTorch 環境があること

## 補足

- `audio_transcriber.py` は `_benchmark.verify_granite` の補助関数を利用します
- ダウンロードしたモデルはこのリポジトリ内の `models/` に保存します
- `_legacy/` は参照用であり、通常運用には含めません
