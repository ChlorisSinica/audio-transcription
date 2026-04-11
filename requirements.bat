@echo off
cd /d %~dp0

:: Check if virtual environment exists
if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] .venv not found.
    echo Please create the virtual environment using PyCharm before running this script.
    pause
    exit /b
)

echo Starting installation into the virtual environment (.venv)...
echo.

set PY_PATH=.venv\Scripts\python.exe

echo Upgrading pip...
"%PY_PATH%" -m pip install --upgrade pip

:: === PyTorch (CUDA 12.4) ===
:: Note: cu121 only has up to torch 2.5.x. Use cu124 for torch 2.7+.
echo Uninstalling old PyTorch (if any)...
"%PY_PATH%" -m pip uninstall torch torchaudio torchvision -y 2>nul
echo Installing PyTorch (GPU version, CUDA 12.4)...
"%PY_PATH%" -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124

:: === Core ASR dependencies ===
echo Installing core libraries...
"%PY_PATH%" -m pip install "transformers>=4.52.1"
"%PY_PATH%" -m pip install accelerate
"%PY_PATH%" -m pip install huggingface_hub
"%PY_PATH%" -m pip install openai-whisper
"%PY_PATH%" -m pip install librosa soundfile
"%PY_PATH%" -m pip install spacy
"%PY_PATH%" -m pip install deepmultilingualpunctuation
"%PY_PATH%" -m pip install tqdm
"%PY_PATH%" -m pip install git+https://github.com/MahmoudAshraf97/ctc-forced-aligner.git
"%PY_PATH%" -m spacy download en_core_web_sm

:: === Translation API + Testing ===
echo Installing translation and testing dependencies...
"%PY_PATH%" -m pip install openai
"%PY_PATH%" -m pip install pytest

:: === NeMo (for NVIDIA Canary Qwen 2.5B) ===
:: Note: [asr] only, not [asr,tts]. The tts extra pulls in pynini which
:: cannot be compiled on Windows (requires GCC flags unsupported by MSVC).
echo Installing NeMo toolkit (for Canary)...
"%PY_PATH%" -m pip install Cython packaging sacrebleu
"%PY_PATH%" -m pip install "nemo_toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git"

echo.
echo === Installation Complete ===
echo.
echo Note: ffmpeg must be installed separately and available on PATH.
echo       Download from https://www.gyan.dev/ffmpeg/builds/ (Windows)
echo       or install via: winget install ffmpeg
pause
