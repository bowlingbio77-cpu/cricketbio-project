@echo off
REM Start the PaceAI Streamlit app detached (survives closing this window).
REM Idempotent: if the app is already running, this does NOT spawn a duplicate
REM server or open another browser tab -- it only brings the running app to front.
setlocal
cd /d "%~dp0"

REM --- Ensure Ollama (chat-assistant backend) is running on port 11434 ---
REM This is what actually breaks the chat assistant on restart: Ollama is not
REM auto-started by Windows, so start it here before the Streamlit app if it
REM is not already listening.
powershell -NoProfile -Command "$c = Get-NetTCPConnection -LocalPort 11434 -State Listen -ErrorAction SilentlyContinue; if ($c) { exit 0 } else { exit 1 }"
if %errorlevel% equ 1 (
    start "" "C:\Users\user\AppData\Local\Programs\Ollama\ollama.exe" serve
    powershell -NoProfile -Command "$i=0; while($i -lt 40 -and -not (Get-NetTCPConnection -LocalPort 11434 -State Listen -ErrorAction SilentlyContinue)){ Start-Sleep -Milliseconds 500; $i++ }; exit 0"
    echo Ollama started at http://localhost:11434
) else (
    echo Ollama already running at http://localhost:11434
)

REM Check whether something already listens on the app port (8501).
powershell -NoProfile -Command "$c = Get-NetTCPConnection -LocalPort 8501 -State Listen -ErrorAction SilentlyContinue; if ($c) { exit 0 } else { exit 1 }"
if %errorlevel% equ 0 goto up
if %errorlevel% equ 1 goto start

:start
start "" /b "C:\Users\user\AppData\Local\Programs\Python\Python314\python.exe" -m streamlit run app.py --server.headless true
echo Streamlit starting at http://localhost:8501
REM brief wait for the server to come up, then open the browser once.
powershell -NoProfile -Command "$i=0; while($i -lt 30 -and -not (Get-NetTCPConnection -LocalPort 8501 -State Listen -ErrorAction SilentlyContinue)){ Start-Sleep -Milliseconds 500; $i++ }; exit 0"
start "" http://localhost:8501
endlocal
exit /b 0

:up
echo PaceAI is already running at http://localhost:8501 -- not starting a duplicate.
endlocal
exit /b 0
