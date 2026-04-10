# audio-transcription

音声文字起こしとモデル検証のためのローカル作業用リポジトリです．

## 使用モデル

| モデル | 用途 | パラメータ | ライセンス |
|---|---|---|---|
| [Cohere Transcribe 03-2026](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026) | 本番 ASR (音声認識) | 2B | Apache 2.0 |
| [wav2vec2-large-xlsr-53-english](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-english) | 単語タイムスタンプ (forced alignment) | 300M | Apache 2.0 |
| [spaCy en_core_web_sm](https://spacy.io/models/en#en_core_web_sm) | 文分割 (sentence segmentation) | — | MIT |

### 検証済みモデル (verify スクリプト)

推論測度，句読点の精度の観点からCohereを選択しています．Whisperはハルシネーションを引き起こすため非推奨です．

| モデル | 平均 WER | RTFx | 句読点 | ソース                                                                                                      |
|---|---|---|---|----------------------------------------------------------------------------------------------------------|
| **Cohere Transcribe** | 5.42% | **35x** | ネイティブ | [Blog](https://cohere.com/blog/transcribe) / https://huggingface.co/CohereLabs/cohere-transcribe-03-2026 |
| IBM Granite 4.0 1B Speech | 5.52% | 8.5x | なし | https://huggingface.co/ibm-granite/granite-4.0-1b-speech                                                 |
| NVIDIA Canary Qwen 2.5B | 5.63% | 8.6x | ネイティブ | https://huggingface.co/nvidia/canary-qwen-2.5b                                                           |
| OpenAI Whisper large-v3 | 7.44% | ~1x | 不安定 | https://github.com/openai/whisper                                                                        |

WER は [HuggingFace Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) (英語 8 データセット平均)．RTFx は RTX 3090 Ti での実測値．

## 構成

- `audio_transcriber.py`: 本番用の文字起こしパイプライン (Cohere Transcribe)
- `audio_converter.py`: FFmpeg を使った音声変換・音量正規化
- `verify_granite.py`: Granite 検証 + 共通ユーティリティ
- `verify_cohere.py`: Cohere 検証
- `verify_canary.py`: Canary 検証 (NeMo 必須)
- `compare_prompts.py`: Granite プロンプト探索
- `_legacy/`: 旧スクリプトの退避先
- `models/`: ローカル保存するモデル本体 (git 管理外)
- `config.json`: `audio_transcriber.py` の既定設定
- `requirements.bat`: Windows 用セットアップスクリプト

## 主な使い方

```powershell
python audio_transcriber.py <input>
python audio_transcriber.py <input> --output-dir ./out
python audio_transcriber.py <input> --config config.json
```

`<input>` には以下を指定できます．

- 単一ファイル: `./lecture.m4a`
- ディレクトリ: `./_audio/MSE403 S21/`
- glob パターン: `./_audio/**/*.m4a`

出力形式:

```
[MM:SS.mmm --> MM:SS.mmm] Sentence text.
```

既に同名の出力ファイルが存在する場合はスキップします．

## セットアップ

Windows では，ローカル仮想環境 `.venv` に依存関係を入れる前提です．

```powershell
.\requirements.bat
```

前提条件:

- `.venv` が事前に作成済みであること
- `ffmpeg` が `PATH` から実行できること
- CUDA 対応 GPU + ドライバ (RTX 3090 Ti で検証済み)
- HuggingFace トークン (Cohere モデルは gated repo)

HuggingFace ログイン:

```powershell
.\.venv\Scripts\python.exe -c "from huggingface_hub import login; login(token='hf_...')"
```

## 補足

- `audio_transcriber.py` は `verify_granite.py` の補助関数を利用します
- ダウンロードしたモデルはこのリポジトリ内の `models/` に保存します (git 管理外)
- `_legacy/` は参照用であり，通常運用には含めません
