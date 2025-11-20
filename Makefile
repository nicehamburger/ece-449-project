# Makefile for Python project with virtual environment setup

VENV := .venv
PYTHON_EXEC := python3
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
REQ := requirements.txt

# Default target
.PHONY: all
all: install

# Create virtual environment if it doesn't exist
$(VENV)/bin/activate: 
	$(PYTHON_EXEC) -m venv $(VENV)
	@echo "Virtual environment created."

# Install dependencies
.PHONY: install
install: $(VENV)/bin/activate $(REQ)
	$(PIP) install --upgrade pip
	$(PIP) install -r $(REQ)
	@echo "Dependencies installed."

.PHONY: clean
clean:
	rm -rf $(VENV)
	find . -type d -name "__pycache__" -exec rm -rf {} +
	@echo "Environment and caches cleaned."

.PHONY: rebuild
rebuild: clean all