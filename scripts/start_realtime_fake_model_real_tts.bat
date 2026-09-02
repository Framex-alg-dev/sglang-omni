@echo off
setlocal

chcp 65001 >nul
cd /d "%~dp0\.."

set "PYTHON_EXE=.venv\Scripts\python.exe"
set "TTS_HOST=127.0.0.1"
set "TTS_PORT=51000"
set "TTS_URL=ws://127.0.0.1:51000/api-ws/v1/realtime"
set "TTS_VOICE=benchmark_qwen_cherry_zh"

if not exist "%PYTHON_EXE%" (
  echo [ERROR] Python virtual environment was not found: %PYTHON_EXE%
  echo Create or restore the project .venv before running this script.
  exit /b 1
)

echo [CHECK] Testing the SSH tunnel at %TTS_HOST%:%TTS_PORT% ...
powershell.exe -NoProfile -Command "$ok = Test-NetConnection '%TTS_HOST%' -Port %TTS_PORT% -InformationLevel Quiet -WarningAction SilentlyContinue; if ($ok) { exit 0 } else { exit 1 }"
if errorlevel 1 (
  echo [ERROR] No TTS listener is reachable at %TTS_HOST%:%TTS_PORT%.
  echo Start the SSH tunnel first:
  echo   ssh -N -L 51000:127.0.0.1:40001 -p 246 user@124.221.190.139
  exit /b 1
)

if /I "%~1"=="--check-only" (
  echo [OK] Local Python environment and TTS tunnel are ready.
  exit /b 0
)

set "SGLANG_OMNI_DEV_FAKE_MODEL_ENABLED=true"
set "SGLANG_OMNI_DEV_FAKE_MODEL_RESPONSE_TEXT=这是合并后服务返回的固定流式回复。"
set "SGLANG_OMNI_DEV_FAKE_MODEL_CHUNK_SIZE=2"
set "SGLANG_OMNI_DEV_FAKE_MODEL_CHUNK_INTERVAL_MS=30"

echo [START] Session Realtime fake model: ws://127.0.0.1:18080/v1/session/realtime
echo [START] Embedded TTS provider: %TTS_URL%
echo [INFO] Press Ctrl+C to stop the service.

"%PYTHON_EXE%" -m sglang_omni.serve.realtime.dev_server ^
  --host 127.0.0.1 ^
  --port 18080 ^
  --realtime-tts-url "%TTS_URL%" ^
  --realtime-tts-voice "%TTS_VOICE%" ^
  --log-level debug

set "SERVER_EXIT_CODE=%ERRORLEVEL%"
echo [STOP] Realtime development server exited with code %SERVER_EXIT_CODE%.
exit /b %SERVER_EXIT_CODE%
