# PowerShell script for Python project with virtual environment setup

$VENV = ".venv"
$PYTHON_EXEC = "python"  # Use 'python' on Windows; often linked to 'python3'
$REQUIREMENTS = "requirements.txt"

function Get-VenvPythonExecutable {
    # Determine the correct path for the virtual environment's Python executable
    if (Test-Path -Path "$VENV\Scripts\python.exe") {
        return "$VENV\Scripts\python.exe"
    } else {
        # Fallback for Linux/macOS style venvs running in WSL/PowerShell
        return "$VENV\bin\python"
    }
}

function Get-VenvPipExecutable {
    # Determine the correct path for the virtual environment's Pip executable
    if (Test-Path -Path "$VENV\Scripts\pip.exe") {
        return "$VENV\Scripts\pip.exe"
    } else {
        # Fallback for Linux/macOS style venvs running in WSL/PowerShell
        return "$VENV\bin\pip"
    }
}

function Create-VirtualEnvironment {
    Write-Host "Checking for virtual environment..."
    if (-not (Test-Path -Path $VENV -PathType Container)) {
        Write-Host "Creating virtual environment: '$VENV'..."
        try {
            # Use 'python -m venv .venv' command
            & $PYTHON_EXEC -m venv $VENV
            Write-Host "Virtual environment created."
        } catch {
            Write-Error "Failed to create virtual environment. Ensure '$PYTHON_EXEC' is in your PATH."
            exit 1
        }
    } else {
        Write-Host "Virtual environment already exists."
    }
}

function Install-Dependencies {
    param(
        [Parameter(Mandatory=$false)]$CheckVenv = $true
    )

    if ($CheckVenv) {
        Create-VirtualEnvironment
    }

    $PIP = Get-VenvPipExecutable

    if (-not (Test-Path -Path $PIP)) {
        Write-Error "Could not find pip executable in virtual environment. Did venv creation fail?"
        exit 1
    }

    Write-Host "Installing/Updating dependencies..."
    
    # Upgrade pip
    & $PIP install --upgrade pip

    # Install requirements from file
    if (Test-Path -Path $REQUIREMENTS) {
        & $PIP install -r $REQUIREMENTS
        Write-Host "Dependencies installed."
    } else {
        Write-Warning "Requirements file '$REQUIREMENTS' not found."
    }
}

function Clean-Project {
    Write-Host "Cleaning environment and caches..."
    
    # Remove virtual environment directory
    if (Test-Path -Path $VENV -PathType Container) {
        Remove-Item -Path $VENV -Recurse -Force
        Write-Host "Removed virtual environment ($VENV)."
    }
    
    # Remove __pycache__ directories
    Get-ChildItem -Path . -Directory -Recurse -ErrorAction SilentlyContinue | Where-Object { $_.Name -eq '__pycache__' } | ForEach-Object {
        Remove-Item -Path $_.FullName -Recurse -Force
    }
    Write-Host "Removed all __pycache__ directories."
    Write-Host "Environment and caches cleaned."
}

function Rebuild-Project {
    Clean-Project
    Install-Dependencies -CheckVenv $true
}

# --- Main execution block to handle targets ---

# Get the target argument, default to 'install' if none provided
$Target = $args[0]
if (-not $Target) {
    $Target = "install"
}

Write-Host "Target: $Target"
Write-Host "---"

switch ($Target) {
    "all" {
        Install-Dependencies
    }
    "install" {
        Install-Dependencies
    }
    "clean" {
        Clean-Project
    }
    "rebuild" {
        Rebuild-Project
    }
    default {
        Write-Error "Unknown target: '$Target'. Available targets are: all, install, clean, rebuild."
        exit 1
    }
}