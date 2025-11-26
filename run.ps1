# PowerShell script to execute the main Python application using the virtual environment interpreter.

# --- Configuration Variables ---
$VENV = ".venv"
$ENTRYPOINT = "code/main.py"

function Get-VenvPythonExecutable {
    $WindowsPath = Join-Path -Path $VENV -ChildPath "Scripts\python.exe"
    if (Test-Path -Path $WindowsPath) {
        return $WindowsPath
    }

    $UnixPath = Join-Path -Path $VENV -ChildPath "bin\python"
    if (Test-Path -Path $UnixPath) {
        return $UnixPath
    }

    # If neither is found, return $null
    return $null
}

if (-not (Test-Path -Path $ENTRYPOINT)) {
    Write-Error "Error: Python script '$ENTRYPOINT' not found in the current directory."
    exit 1
}

$PYTHON = Get-VenvPythonExecutable

if (-not $PYTHON) {
    Write-Error "Error: Virtual environment interpreter not found in '$VENV'."
    Write-Error "Please run the setup script (like '.\build.ps1 install' or 'make') first to set up the environment."
    exit 1
}

Write-Host "Activating VENV and running $ENTRYPOINT..."

& $PYTHON $ENTRYPOINT $args