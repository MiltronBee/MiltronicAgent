#!/bin/bash
set -euo pipefail

# --- Miltronic Harmonic Agent Bootstrap Script ---
# Enhanced CLI Experience with Visual Flair
# Date: June 27, 2025 – Enhanced Version

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
GRAY='\033[0;90m'
NC='\033[0m' # No Color

# Progress spinner
spinner() {
    local pid=$1
    local delay=0.1
    local spinstr='|/-\'
    while [ "$(ps a | awk '{print $1}' | grep $pid)" ]; do
        local temp=${spinstr#?}
        printf " [%c]  " "$spinstr"
        local spinstr=$temp${spinstr%"$temp"}
        sleep $delay
        printf "\b\b\b\b\b\b"
    done
    printf "    \b\b\b\b"
}

# Enhanced logging
log_step() {
    echo -e "${CYAN}▶${NC} ${WHITE}$1${NC}"
}

log_success() {
    echo -e "${GREEN}✓${NC} $1"
}

log_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

log_error() {
    echo -e "${RED}✗${NC} $1"
}

# ASCII Art Header
print_header() {
    echo -e "${PURPLE}"
    cat << "EOF"
    ███╗   ███╗██╗██╗  ████████╗██████╗  ██████╗ ███╗   ██╗██╗ ██████╗
    ████╗ ████║██║██║  ╚══██╔══╝██╔══██╗██╔═══██╗████╗  ██║██║██╔════╝
    ██╔████╔██║██║██║     ██║   ██████╔╝██║   ██║██╔██╗ ██║██║██║     
    ██║╚██╔╝██║██║██║     ██║   ██╔══██╗██║   ██║██║╚██╗██║██║██║     
    ██║ ╚═╝ ██║██║███████╗██║   ██║  ██║╚██████╔╝██║ ╚████║██║╚██████╗
    ╚═╝     ╚═╝╚═╝╚══════╝╚═╝   ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═══╝╚═╝ ╚═════╝
                     Meta-Adaptive Gating Training Pipeline                    
EOF
    echo -e "${NC}"
}

# Progress bar
progress_bar() {
    local current=$1
    local total=$2
    local width=50
    local percentage=$((current * 100 / total))
    local completed=$((current * width / total))
    local remaining=$((width - completed))
    
    printf "\r${CYAN}["
    printf "%${completed}s" | tr ' ' '█'
    printf "%${remaining}s" | tr ' ' '░'
    printf "] ${percentage}%% ${NC}"
}

# System info display
show_system_info() {
    echo -e "\n${GRAY}┌─ System Information ────────────────────────────────────┐${NC}"
    echo -e "${GRAY}│${NC} ${BLUE}Timestamp:${NC}       $(date)"
    echo -e "${GRAY}│${NC} ${BLUE}Working Dir:${NC}     $(pwd)"
    echo -e "${GRAY}│${NC} ${BLUE}User:${NC}            $(whoami)"
    echo -e "${GRAY}│${NC} ${BLUE}Python:${NC}          $(python3 --version 2>/dev/null || echo 'Not installed')"
    echo -e "${GRAY}│${NC} ${BLUE}CUDA Available:${NC}  $(python3 -c 'import torch; print("Yes" if torch.cuda.is_available() else "No")' 2>/dev/null || echo 'Unknown')"
    echo -e "${GRAY}└──────────────────────────────────────────────────────────┘${NC}\n"
}

# Main execution
main() {
    clear
    print_header
    show_system_info
    
    # STEP 0: System packages
    log_step "Installing system dependencies..."
    if command -v apt-get &> /dev/null; then
        sudo apt-get update -y > /dev/null 2>&1 &
        spinner $!
        sudo apt-get install -y python3 python3-pip python3-venv ffmpeg curl git swig > /dev/null 2>&1 &
        spinner $!
        log_success "System dependencies installed"
    else
        log_warning "apt-get not available, skipping system packages"
    fi

    # STEP 1: Virtual environment
    ENV="miltronic_env"
    log_step "Setting up Python environment..."
    if [ ! -d "$ENV" ]; then
        log_info "Creating virtual environment: $ENV"
        python3 -m venv "$ENV" > /dev/null 2>&1 &
        spinner $!
        log_success "Virtual environment created"
    else
        log_info "Virtual environment already exists"
    fi

    log_info "Activating virtual environment"
    source "$ENV/bin/activate"
    log_success "Environment activated: ${GREEN}$ENV${NC}"

    # STEP 2: Upgrade pip
    log_step "Upgrading pip..."
    pip install --upgrade pip > /dev/null 2>&1 &
    spinner $!
    log_success "pip upgraded"

    # STEP 3: Install dependencies  
    log_step "Installing Python dependencies..."
    log_info "Installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 > /dev/null 2>&1 &
    spinner $!
    
    log_info "Installing RL and analysis libraries..."
    pip install stable-baselines3[extra] "gymnasium[atari,box2d]" autorom ale-py wandb mpmath matplotlib seaborn pandas box2d-py > /dev/null 2>&1 &
    spinner $!
    log_success "All dependencies installed"

    # STEP 4: Install ROMs
    log_step "Installing Atari ROMs..."
    AutoROM --accept-license > /dev/null 2>&1 &
    spinner $!
    log_success "Atari ROMs installed"

    # STEP 5: Verify environment
    log_step "Verifying ALE/MsPacman-v5 environment..."
    if python3 -c "import gymnasium; import ale_py; env = gymnasium.make('ALE/MsPacman-v5'); print('Environment verified')" > /dev/null 2>&1; then
        log_success "Environment verification passed"
    else
        log_error "Environment verification failed"
        exit 1
    fi

    # STEP 6: WandB config
    log_step "Configuring Weights & Biases..."
    export WANDB_PROJECT="miltronic-pacman-v1-final"
    export WANDB_MODE="online"
    export WANDB_API_KEY="8a00b1cbdd3ad81aa24736b9e741380001d9de3b"
    log_success "W&B configured: ${CYAN}$WANDB_PROJECT${NC}"

    # STEP 7: Prepare directories
    log_step "Preparing directories..."
    mkdir -p models logs runs
    log_success "Directories created: ${CYAN}models/, logs/, runs/${NC}"

    # STEP 8: Training phase
    echo -e "\n${YELLOW}╔══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${YELLOW}║                    TRAINING PHASE                        ║${NC}"
    echo -e "${YELLOW}╚══════════════════════════════════════════════════════════╝${NC}"
    
    log_step "Starting comparative training (MAG + Baseline)..."
    log_info "This will run two 5M timestep training sessions"
    log_info "Expected duration: ${YELLOW}~8-12 hours${NC}"
    
    TRAIN_LOG="logs/train_$(date +%Y%m%d_%H%M).log"
    echo -e "${GRAY}Logging to: $TRAIN_LOG${NC}"
    
    python3 train.py | tee "$TRAIN_LOG"
    
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        log_success "Training completed successfully!"
    else
        log_error "Training failed!"
        exit 1
    fi

    # STEP 9: Analysis phase
    echo -e "\n${GREEN}╔══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║                   ANALYSIS PHASE                         ║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════════════════════╝${NC}"
    
    log_step "Running comparative analysis..."
    ANALYSIS_LOG="logs/analyze_$(date +%Y%m%d_%H%M).log"
    echo -e "${GRAY}Logging to: $ANALYSIS_LOG${NC}"
    
    python3 analyze.py | tee "$ANALYSIS_LOG"
    
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        log_success "Analysis completed successfully!"
    else
        log_warning "Analysis completed with warnings"
    fi

    # Completion banner
    echo -e "\n${GREEN}╔══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║                 🎉 EXECUTION COMPLETE 🎉                 ║${NC}"
    echo -e "${GREEN}║                                                          ║${NC}"
    echo -e "${GREEN}║  Check the analysis_results/ directory for plots        ║${NC}"
    echo -e "${GREEN}║  Review logs/ directory for detailed training logs      ║${NC}"
    echo -e "${GREEN}║  Models saved in models/ directory                      ║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════════════════════╝${NC}"
    
    log_info "Results summary:"
    echo -e "  ${CYAN}•${NC} Training logs: ${TRAIN_LOG}"
    echo -e "  ${CYAN}•${NC} Analysis logs: ${ANALYSIS_LOG}"
    echo -e "  ${CYAN}•${NC} Models: ${YELLOW}models/${NC}"
    echo -e "  ${CYAN}•${NC} Plots: ${YELLOW}analysis_results/${NC}"
}

# Handle interrupts gracefully
trap 'echo -e "\n${RED}Execution interrupted by user${NC}"; exit 130' INT

# Run main function
main "$@"
