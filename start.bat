@echo off
TITLE AR Sign Language Communication System
cd /d "%~dp0"

echo ======================================================================
echo Launching AR Sign Language Communication System...
echo ======================================================================

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not found in your PATH!
    echo Please install Python 3.10+ and ensure it is added to your environment variables.
    pause
    exit /b 1
)

REM Run the interactive Python launcher
python start.py %*

if %errorlevel% neq 0 (
    echo.
    echo [INFO] Process exited with code %errorlevel%.
    pause
)
