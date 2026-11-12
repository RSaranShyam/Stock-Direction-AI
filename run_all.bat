@echo off
setlocal
cd /d "%~dp0"
title Stock Direction AI - full pipeline
echo This runs the WHOLE pipeline. The first time it can take 10-30 minutes before the browser opens.
echo To open only the dashboard right now, close this window and double-click start_dashboard.bat
echo.
if not exist ".venv\Scripts\python.exe" goto novenv
".venv\Scripts\python.exe" run_all.py %*
goto done
:novenv
python run_all.py %*
:done
echo.
pause
