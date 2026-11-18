@echo off
cd /d "%~dp0.."
call .venv\Scripts\activate
streamlit run demo\app.py
pause
