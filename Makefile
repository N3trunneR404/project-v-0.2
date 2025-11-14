# Minimal Makefile for DT controller

PY    ?= python3
PIP   ?= pip3
VENV  ?= .venv
ACT   ?= . $(VENV)/bin/activate
API_HOST ?= 127.0.0.1
API_PORT ?= 8080

.PHONY: help install venv deps run-api lint format clean freeze

help:
	@echo "Targets:"
	@echo "  install   - create venv and install requirements"
	@echo "  run-api   - start the DT API (http://$(API_HOST):$(API_PORT))"
	@echo "  lint      - run ruff"
	@echo "  format    - run black"
	@echo "  clean     - remove caches"
	@echo "  freeze    - export pip freeze to requirements.lock.txt"

venv:
	$(PY) -m venv $(VENV)

deps: requirements.txt | venv
	$(ACT); $(PIP) install -U pip
	$(ACT); $(PIP) install -r requirements.txt

install: deps
	@echo "✔ Environment ready."

run-api:
	$(ACT); FABRIC_API_HOST=$(API_HOST) FABRIC_API_PORT=$(API_PORT) $(PY) app.py

lint:
	$(ACT); ruff check dt chaos deploy experiments images jobs k8s_executor k8s_plugins sim tools app.py

format:
	$(ACT); black dt experiments tools app.py

clean:
	@find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	@find . -name '*.pyc' -delete
	@rm -rf .pytest_cache .mypy_cache .ruff_cache build dist *.egg-info
	@echo "✔ Cleaned."

freeze:
	$(ACT); $(PIP) freeze > requirements.lock.txt

