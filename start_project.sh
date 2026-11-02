#!/bin/bash

set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

if [ ! -d ".venv" ]; then
  echo "Creating virtual environment..."
  python3 -m venv .venv
fi

source .venv/bin/activate

echo "Upgrading pip..."
python -m pip install --upgrade pip

echo "Installing requirements..."
pip install -r requirements.txt

echo "Training models..."
python train_models.py

echo "Starting Streamlit app..."
python -m streamlit run app.py --server.headless true --server.address 0.0.0.0 --server.port 8501 &
STREAMLIT_PID=$!

echo "Starting FastAPI backend..."
python -m uvicorn api:app --host 0.0.0.0 --port 8000 &
API_PID=$!

trap 'kill $STREAMLIT_PID $API_PID 2>/dev/null || true; exit 0' INT TERM EXIT

echo ""
echo "=================================================="
echo "Project is running"
echo "Frontend: http://localhost:8501"
echo "API Docs: http://localhost:8000/docs"
echo "Health Check: http://localhost:8000/health"
echo "Press Ctrl+C to stop"
echo "=================================================="
echo ""

wait
