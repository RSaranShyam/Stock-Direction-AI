@echo off
setlocal
cd /d "%~dp0"
title Stock Direction AI - dashboard
echo Starting the dashboard. Your browser will open at http://localhost:8501
echo Keep this window open while you use it. Close it (or press Ctrl+C) to stop.
echo.
if not exist ".venv\Scripts\python.exe" goto novenv
".venv\Scripts\python.exe" -m streamlit run dashboard.py
goto done
:novenv
echo (no .venv folder found, using the system Python)
python -m streamlit run dashboard.py
:done
echo.
echo The dashboard has stopped. If you saw an error above, copy it and send it to me.
pause
