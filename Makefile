# Makefile for Python project with virtual environment setup (Windows-friendly)

VENV := .venv
PYTHON_EXEC := python3
PYTHON := $(VENV)/Scripts/python.exe
PIP := $(VENV)/Scripts/pip.exe
REQ := requirements.txt

# Default target
.PHONY: all
all: install

# Create virtual environment if it doesn't exist
$(VENV)/Scripts/activate:
	$(PYTHON_EXEC) -m venv $(VENV)
	@echo "Virtual environment created."

# Install dependencies
.PHONY: install
install: $(VENV)/Scripts/activate $(REQ)
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r $(REQ)
	@echo "Dependencies installed."

.PHONY: clean
clean:
	rm -rf $(VENV)
	find . -type d -name "__pycache__" -exec rm -rf {} +
	@echo "Environment and caches cleaned."

.PHONY: rebuild
rebuild: clean all
