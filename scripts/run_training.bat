@echo off
REM Batch script to run training with the venv
cd /d "d:\GitHub\aysan\class-projects\1"

REM Activate venv
call .venv\Scripts\activate.bat

REM Run the main training script from scripts directory
echo Starting ExU-Trans training...
echo Output will be saved to: %cd%\outputs\
echo.

python .\scripts\main.py

echo.
echo Training complete. Check outputs\ for results.
pause
