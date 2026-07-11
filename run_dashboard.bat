@echo off
IF NOT EXIST .env copy .env.example .env
IF NOT EXIST .venv python -m venv .venv
call .venv\Scripts\activate.bat
pip install -r requirements.txt
set API_BASE_URL=http://localhost:8000
streamlit run dashboard.py --server.port 8501
