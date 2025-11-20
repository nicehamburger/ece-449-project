VENV=".venv"
ENTRYPOINT="code/main.py"

if [ ! -f "$ENTRYPOINT" ]; then
    echo "Error: Python script '$ENTRYPOINT' not found in the current directory."
    exit 1
fi

if [ ! -f "$VENV/bin/python" ]; then
    echo "Error: Virtual environment interpreter not found at $VENV/bin/python."
    echo "Please run 'make' first to set up the environment."
    exit 1
fi

exec "$VENV/bin/python" "$ENTRYPOINT" "$@"