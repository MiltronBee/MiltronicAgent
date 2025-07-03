#!/bin/bash

# ==============================================================================
# Miltronic Agent Environment Setup
# ==============================================================================
#
# Author: Gemini
#
# This script prepares the complete environment for the Miltronic Agent.
# It performs the following steps:
#   1. Checks for Python 3 and the 'venv' module.
#   2. Creates a Python virtual environment named 'venv'.
#   3. Activates the virtual environment.
#   4. Installs all required Python packages from 'requirements.txt'.
#   5. Creates a '.env' file for environment variables (e.g., WANDB_API_KEY).
#   6. Provides clear instructions for the user to proceed.
#
# ==============================================================================

# --- Colors and Logging ---
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

log() {
    echo -e "${BLUE}INFO:${NC} $1"
}

success() {
    echo -e "${GREEN}SUCCESS:${NC} $1"
}

warn() {
    echo -e "${YELLOW}WARNING:${NC} $1"
}

error() {
    echo -e "${RED}ERROR:${NC} $1"
    exit 1
}

# --- Banner ---
echo -e "${GREEN}"
cat << "EOF"
 __  __ _ _ __  _ __ ___ _ __  _ __ ___   ___  _ __
|  \/  | | '_ \| '__/ _ \ '_ \| '_ ` _ \ / _ \| '_ \
| |\/| | | | | | | |  __/ | | | | | | | | (_) | | | |
|_|  |_|_|_| |_|_|  \___|_| |_|_| |_| |_|\___/|_| |_|

EOF
echo -e "          ${YELLOW}Environment Setup Utility${NC}"
echo -e "${NC}"

# --- Pre-flight Checks ---
log "Performing pre-flight checks..."
if ! command -v python3 &> /dev/null; then
    error "Python 3 is not installed. Please install Python 3 to continue."
fi
log "Python 3 found."

if ! python3 -m venv -h &> /dev/null; then
    error "The 'venv' module is not available. Please install it (e.g., 'sudo apt-get install python3-venv')."
fi
log "Python 'venv' module found."
echo ""

# --- Virtual Environment Setup ---
VENV_DIR="venv"
if [ -d "$VENV_DIR" ]; then
    warn "Virtual environment '$VENV_DIR' already exists. Skipping creation."
else
    log "Creating Python virtual environment in './$VENV_DIR'..."
    python3 -m venv $VENV_DIR
    success "Virtual environment created."
fi
echo ""

# --- Activate and Install Dependencies ---
log "Activating virtual environment..."
source $VENV_DIR/bin/activate
success "Virtual environment activated."
echo ""

log "Creating 'requirements.txt'..."
cat << EOF > requirements.txt
stable-baselines3[extra]
wandb
ale-py
gymnasium[atari]
torch
EOF
success "'requirements.txt' created."
echo ""

log "Installing dependencies from 'requirements.txt'. This may take a few minutes..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    error "Failed to install dependencies. Please check the output above."
fi
success "All dependencies installed successfully."
echo ""

# --- Environment File Setup ---
ENV_FILE="miltronic_agent/.env"
if [ -f "$ENV_FILE" ]; then
    warn "Environment file '$ENV_FILE' already exists. Skipping creation."
else
    log "Creating environment file in '$ENV_FILE'..."
    touch $ENV_FILE
    echo "WANDB_API_KEY='YOUR_API_KEY_HERE'" >> $ENV_FILE
    success "Created '$ENV_FILE'. Please edit it to add your W&B API key."
fi
echo ""

# --- Final Instructions ---
success "Setup is complete!"
echo ""
echo -e "${YELLOW}Next Steps:${NC}"
echo "  1. Edit the '${GREEN}$ENV_FILE${NC}' file and replace 'YOUR_API_KEY_HERE' with your actual Weights & Biases API key."
echo "  2. Activate the environment in your shell by running: ${GREEN}source $VENV_DIR/bin/activate${NC}"
echo "  3. Run the agent using the './run.sh' script."
echo ""