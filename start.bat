@echo off
cd /d %~dp0
if not exist venv (
  echo [INFO] Creating venv...
  python -m venv venv
)
call .\venv\Scripts\activate
python -m pip install --upgrade pip
pip install Flask numpy soundfile pedalboard pyloudnorm
python app.py
