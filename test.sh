#!/bin/bash
set -euo pipefail

# --- Miltronic Harmonic Agent Bootstrap Script ---
# Date: June 23, 2025 – Final Version

echo "=== [Miltronic Execution Initiated] ==="
echo "Timestamp: $(date)"
echo "Working Directory: $(pwd)"

# STEP 0: System packages
echo "-> Installing system dependencies..."
sudo apt-get update -y
sudo apt-get install -y python3 python3-pip python3-venv ffmpeg curl git

# STEP 1: Virtual environment
ENV="miltronic_env"
if [ ! -d "$ENV" ]; then
  echo "-> Creating virtualenv $ENV"
  python3 -m venv "$ENV"
fi

echo "-> Activating virtualenv"
source "$ENV/bin/activate"

# STEP 2: Upgrade pip
pip install --upgrade pip

# STEP 3: Install dependencies
echo "-> Installing Python dependencies"
pip install \
  torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install \
  stable-baselines3[extra] \
  "gymnasium[atari]" \
  autorom \
  ale-py \
  wandb \
  mpmath \
  matplotlib \
  seaborn \
  pandas

# STEP 4: Install ROMs
echo "-> Installing Atari ROMs"
AutoROM --accept-license

# STEP 5: Verify ALE env
echo "-> Verifying ALE/MsPacman-v5"
python - <<EOF
import gymnasium
import ale_py  # ensures ALE environments register
env = gymnasium.make("ALE/MsPacman-v5")
print("✔ Environment loaded:", env)
EOF

# STEP 6: WandB config (optional)
export WANDB_PROJECT="miltronic-pacman-v1-final"
export WANDB_MODE="online"
export WANDB_API_KEY="8a00b1cbdd3ad81aa24736b9e741380001d9de3b"

# STEP 7: Prepare directories
mkdir -p models logs runs

# STEP 8: Train
echo "-> Starting training"
python3 train.py | tee "logs/train_$(date +%Y%m%d_%H%M).log"

# STEP 9: Analyze
echo "-> Running analysis"
python3 analyze.py | tee "logs/analyze_$(date +%Y%m%d_%H%M).log"

echo "=== [Miltronic Execution Complete] ==="
