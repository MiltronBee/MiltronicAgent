#!/bin/bash

# ==============================================================================
# Miltronic Agent Comparative Analysis Runner
# ==============================================================================
#
# Author: Gemini
#
# This script orchestrates a head-to-head comparison between the Miltronic
# agent and a baseline PPO agent. It runs two full training sequences back-to-back
# on the same environment and seed to ensure a fair comparison.
#
# Usage:
#   ./run.sh [environment] [seed]
#
# Parameters:
#   [environment]: The Atari environment to use (e.g., 'ALE/MsPacman-v5').
#   [seed]:        An integer for the random seed (default: 42).
#
# ==============================================================================

# --- Configuration ---
ENV=${1:-"ALE/MsPacman-v5"}
SEED=${2:-42}
VENV_DIR="venv"

# --- Colors and Logging ---
BLUE='\033[1;34m'
GREEN='\033[1;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[1;36m'
NC='\033[0m' # No Color

log() {
    echo -e "${BLUE}>>${NC} $1"
}

warn() {
    echo -e "${YELLOW}!!${NC} $1"
}

error() {
    echo -e "${RED}XX ERROR:${NC} $1"
    exit 1
}

section_break() {
    echo -e "${CYAN}"
    echo "=============================================================================="
    echo "$1"
    echo "=============================================================================="
    echo -e "${NC}"
}

# --- Banner ---
echo -e "${BLUE}"
cat << "EOF"
 __  __ _ _ __  _ __ ___ _ __  _ __ ___   ___  _ __
|  \/  | | '_ \| '__/ _ \ '_ \| '_ ` _ \ / _ \| '_ \
| |\/| | | | | | | |  __/ | | | | | | | | (_) | | | |
|_|  |_|_|_|_| |_|_|  \___|_| |_|_| |_| |_|\___/|_| |_|
EOF
echo -e "             ${YELLOW}Comparative Analysis Runner${NC}"
echo -e "${NC}"

# --- Pre-flight Checks ---
log "Performing pre-flight checks..."
if [ ! -d "$VENV_DIR" ]; then
    error "Virtual environment '$VENV_DIR' not found. Please run './setup.sh' first."
fi
log "Virtual environment found."

ENV_FILE="miltronic_agent/.env"
if [ -f "$ENV_FILE" ]; then
    export $(grep -v '^#' $ENV_FILE | xargs)
    if grep -q "YOUR_API_KEY_HERE" "$ENV_FILE" || [ -z "$WANDB_API_KEY" ]; then
        warn "WANDB_API_KEY in '$ENV_FILE' is not set. Wandb logging will fail for both runs."
    else
        log "WANDB_API_KEY loaded."
    fi
else
    warn "Environment file '$ENV_FILE' not found. Wandb logging will be disabled."
fi
log "Environment checks complete."
echo ""

# --- Activate Environment ---
log "Activating Python virtual environment..."
source $VENV_DIR/bin/activate
echo ""

# --- Run 1: Miltronic Agent ---
section_break "PHASE 1: Training Miltronic Agent (A Vessel for Coherence)"
log "Initializing the training sequence for 'miltronic_k_prime'..."
echo -e "--------------------------------------------------"
echo -e "  ${YELLOW}Mode:${NC}         ${GREEN}miltronic_k_prime${NC}"
echo -e "  ${YELLOW}Environment:${NC}  ${GREEN}${ENV}${NC}"
echo -e "  ${YELLOW}Seed:${NC}         ${GREEN}${SEED}${NC}"
echo -e "--------------------------------------------------"
echo ""

python3 -m miltronic_agent.train --mode "miltronic_k_prime" --env "$ENV" --seed "$SEED"
if [ $? -ne 0 ]; then
    error "Miltronic training script failed."
fi
log "Miltronic agent training complete."
echo ""


# --- Run 2: Baseline PPO Agent ---
section_break "PHASE 2: Training Baseline PPO Agent (A Reward Maximizer)"
log "Initializing the training sequence for 'baseline'..."
echo -e "--------------------------------------------------"
echo -e "  ${YELLOW}Mode:${NC}         ${GREEN}baseline${NC}"
echo -e "  ${YELLOW}Environment:${NC}  ${GREEN}${ENV}${NC}"
echo -e "  ${YELLOW}Seed:${NC}         ${GREEN}${SEED}${NC}"
echo -e "--------------------------------------------------"
echo ""

python3 -m miltronic_agent.train --mode "baseline" --env "$ENV" --seed "$SEED"
if [ $? -ne 0 ]; then
    error "Baseline training script failed."
fi
log "Baseline PPO agent training complete."
echo ""

# --- Conclusion ---
section_break "ANALYSIS COMPLETE"
log "Both training runs have concluded."
echo -e "You can now compare the performance of the two agents in your"
echo -e "Weights & Biases project: ${YELLOW}miltronic-collapse${NC}"
echo ""
echo -e "Look for the two runs with the names:"
echo -e "  - ${GREEN}collapse_${ENV}_seed_${SEED}${NC}"
echo -e "  - ${GREEN}baseline_${ENV}_seed_${SEED}${NC}"
echo ""
log "The ritual is complete. The data has been recorded."
echo ""