@echo off
setlocal
cd /d %~dp0

set "SOURCE_DIR=%~dp0models"
set "DEST_ROOT=%~1"

if "%DEST_ROOT%"=="" (
    echo Usage:
    echo   export_models_to_gdrive.bat "G:\My Drive\audio-transcription-backup"
    echo.
    echo Example:
    echo   export_models_to_gdrive.bat "G:\My Drive\audio-transcription"
    exit /b 1
)

if not exist "%SOURCE_DIR%" (
    echo [ERROR] Source models directory not found:
    echo   %SOURCE_DIR%
    exit /b 1
)

if not exist "%DEST_ROOT%" (
    echo [ERROR] Destination directory not found:
    echo   %DEST_ROOT%
    echo.
    echo Create the folder first in Google Drive for desktop, then rerun.
    exit /b 1
)

set "DEST_DIR=%DEST_ROOT%\models"

echo Source:
echo   %SOURCE_DIR%
echo Destination:
echo   %DEST_DIR%
echo.
echo Copying model files to Google Drive sync folder...
echo.

if not exist "%DEST_DIR%" mkdir "%DEST_DIR%"

robocopy "%SOURCE_DIR%" "%DEST_DIR%" /E /Z /FFT /R:2 /W:2 /XD ".cache" ".git" "__pycache__"
set "RC=%ERRORLEVEL%"

if %RC% GEQ 8 (
    echo.
    echo [ERROR] robocopy failed with exit code %RC%.
    exit /b %RC%
)

echo.
echo Done. Models were copied to:
echo   %DEST_DIR%
echo.
echo Notes:
echo - This script copies model files for personal backup / cross-PC sync.
echo - Canary model files are not fully covered here if NeMo cached them outside this repo.
echo - On the other PC, place the copied folders under that repo's models\ directory.
exit /b 0
