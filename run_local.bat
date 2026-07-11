@echo off
IF NOT EXIST .env copy .env.example .env
python -m venv .venv
call .venv\Scripts\activate.bat
pip install -r requirements.txt
uvicorn app.api:app --reload --host 0.0.0.0 --port 8000
