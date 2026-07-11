if (!(Test-Path .env)) {
  Copy-Item .env.example .env
}
if (!(Test-Path .venv)) {
  python -m venv .venv
}
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:API_BASE_URL="http://localhost:8000"
streamlit run dashboard.py --server.port 8501
