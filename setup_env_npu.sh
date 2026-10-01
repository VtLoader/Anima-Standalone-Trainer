#!/bin/bash
# Setup script for Ascend NPU (torch_npu) environments.
# Creates ./venv, installs requirements.txt (torch + torch-npu + training deps)
# and installs the Web UI npm dependencies.

set -e

cd "$(dirname "$0")"

echo "----------------------------------------------------------------------"
echo "Anima Trainer - Ascend NPU environment setup"
echo "----------------------------------------------------------------------"

# 1. Load the CANN environment if available (required for torch_npu at runtime)
for CANN_ENV in \
    /usr/local/Ascend/ascend-toolkit/set_env.sh \
    /usr/local/Ascend/ascend-toolkit/latest/set_env.sh \
    "$ASCEND_TOOLKIT_HOME/set_env.sh"; do
    if [ -f "$CANN_ENV" ]; then
        echo "Sourcing CANN environment: $CANN_ENV"
        # shellcheck disable=SC1090
        source "$CANN_ENV"
        break
    fi
done

if ! command -v npu-smi >/dev/null 2>&1; then
    echo "[WARN] npu-smi not found. Make sure the Ascend driver and CANN toolkit are installed."
fi

# 2. Detect Python (3.10 - 3.13)
if command -v python3 >/dev/null 2>&1; then
    PYTHON_CMD=python3
elif command -v python >/dev/null 2>&1; then
    PYTHON_CMD=python
else
    echo "[ERROR] Python is not installed. Install Python 3.10 - 3.13."
    exit 1
fi

PY_MINOR=$($PYTHON_CMD -c "import sys; print(sys.version_info.minor)")
if [ "$PY_MINOR" -lt 10 ] || [ "$PY_MINOR" -ge 14 ]; then
    echo "[ERROR] Unsupported Python version. Use Python 3.10 - 3.13."
    exit 1
fi

echo "Using $PYTHON_CMD ($($PYTHON_CMD --version 2>&1))"

# 3. Create the virtual environment
if [ ! -d "venv" ]; then
    echo "Creating venv..."
    $PYTHON_CMD -m venv venv
else
    echo "Venv already exists."
fi

# shellcheck disable=SC1091
source venv/bin/activate

# 4. Install Python dependencies (torch, torch-npu, training stack)
echo "Installing Python requirements..."
pip install --upgrade pip
pip install -r requirements.txt

# 5. Install the Web UI dependencies
echo "Installing Web UI dependencies (npm install)..."
cd training-ui
npm install
cd ..

echo ""
echo "----------------------------------------------------------------------"
echo "Installation complete."
echo "Activate the environment with:"
echo "  source /usr/local/Ascend/ascend-toolkit/set_env.sh && source venv/bin/activate"
echo "Then start the UI with:"
echo "  ./training-ui/start_linux.sh"
echo "----------------------------------------------------------------------"
