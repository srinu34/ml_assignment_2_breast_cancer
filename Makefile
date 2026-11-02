.DEFAULT_GOAL := run

PYTHON ?= python3
VENV_DIR := .venv
VENV_PYTHON := $(VENV_DIR)/bin/python
PIP := $(VENV_PYTHON) -m pip

.PHONY: venv install train app api run

venv: $(VENV_PYTHON)

$(VENV_PYTHON):
	$(PYTHON) -m venv $(VENV_DIR)

install:
	. $(VENV_DIR)/bin/activate && $(PIP) install --upgrade pip
	. $(VENV_DIR)/bin/activate && $(PIP) install -r requirements.txt

train:
	. $(VENV_DIR)/bin/activate && python train_models.py

app:
	. $(VENV_DIR)/bin/activate && python -m streamlit run app.py --server.headless true --server.address 0.0.0.0 --server.port 8501

api:
	. $(VENV_DIR)/bin/activate && python -m uvicorn api:app --host 0.0.0.0 --port 8000

run:
	. $(VENV_DIR)/bin/activate && python run_all.py
