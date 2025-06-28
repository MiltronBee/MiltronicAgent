#!/bin/bash
set -euo pipefail

# --- Miltronic Harmonic Agent Bootstrap Script ---
# Enhanced CLI Experience with SSD-Aware Setup
# Date: June 28, 2025 – Ground Zero + Training Pipeline

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

# SSD Configuration
PRIMARY_DISK="/dev/nvme1n1"
PRIMARY_MOUNT="/mnt/data0"
USER_HOME="/home/hl"
DEV_SYMLINK="$USER_HOME/dev"

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
                     k'-Prime Entropy Control System                    
EOF
    echo -e "${NC}"
}

# Check if disk exists
check_disk_exists() {
    local disk=$1
    if [[ ! -b "$disk" ]]; then
        log_error "Disk $disk not found!"
        echo -e "${YELLOW}Available NVMe disks:${NC}"
        lsblk | grep nvme || echo "No NVMe disks detected"
        return 1
    fi
    return 0
}

# Mount primary workspace disk
setup_primary_disk() {
    log_step "Setting up primary workspace disk..."
    
    # Check if disk exists
    if ! check_disk_exists "$PRIMARY_DISK"; then
        log_warning "Primary disk $PRIMARY_DISK not available, using current directory"
        return 1
    fi
    
    # Create mount point
    log_info "Creating mount point: $PRIMARY_MOUNT"
    sudo mkdir -p "$PRIMARY_MOUNT"
    
    # Check if already mounted
    if mountpoint -q "$PRIMARY_MOUNT" 2>/dev/null; then
        log_info "Disk already mounted at $PRIMARY_MOUNT"
    else
        # Check if disk is already formatted
        if sudo blkid "$PRIMARY_DISK" >/dev/null 2>&1; then
            log_info "Disk already formatted, mounting..."
        else
            log_info "Formatting disk $PRIMARY_DISK with ext4..."
            sudo mkfs.ext4 -F "$PRIMARY_DISK" >/dev/null 2>&1
            log_success "Disk formatted"
        fi
        
        # Mount the disk
        log_info "Mounting $PRIMARY_DISK to $PRIMARY_MOUNT"
        sudo mount "$PRIMARY_DISK" "$PRIMARY_MOUNT"
        log_success "Disk mounted"
    fi
    
    # Set permissions
    sudo chown -R $(whoami):$(whoami) "$PRIMARY_MOUNT" 2>/dev/null || true
    log_success "Permissions set for user access"
    
    # Add to fstab if not already present
    if ! grep -q "$PRIMARY_DISK" /etc/fstab 2>/dev/null; then
        log_info "Adding mount to /etc/fstab for persistence"
        echo "$PRIMARY_DISK $PRIMARY_MOUNT ext4 defaults,nofail 0 2" | sudo tee -a /etc/fstab >/dev/null
        log_success "Mount added to fstab"
    else
        log_info "Mount already in fstab"
    fi
    
    return 0
}

# Setup development directory symlink
setup_dev_symlink() {
    log_step "Setting up development directory symlink..."
    
    # Only proceed if SSD mount was successful
    if [[ ! -d "$PRIMARY_MOUNT" ]]; then
        log_info "SSD not available, skipping symlink setup"
        return 0
    fi
    
    # Create dev directory on SSD
    mkdir -p "$PRIMARY_MOUNT/dev"
    log_success "Created $PRIMARY_MOUNT/dev"
    
    # Remove existing symlink or directory if it exists
    if [[ -L "$DEV_SYMLINK" ]]; then
        log_info "Removing existing symlink: $DEV_SYMLINK"
        rm "$DEV_SYMLINK"
    elif [[ -d "$DEV_SYMLINK" ]]; then
        log_warning "Moving existing directory $DEV_SYMLINK to backup"
        mv "$DEV_SYMLINK" "${DEV_SYMLINK}.backup.$(date +%Y%m%d_%H%M%S)"
    fi
    
    # Create symlink
    ln -sfn "$PRIMARY_MOUNT/dev" "$DEV_SYMLINK"
    log_success "Symlink created: $DEV_SYMLINK → $PRIMARY_MOUNT/dev"
    
    return 0
}

# Optional: Setup additional SSDs
setup_additional_ssds() {
    log_step "Checking for additional SSDs..."
    
    local disk_count=0
    for i in {2..16}; do
        local disk="/dev/nvme${i}n1"
        local mountpoint="/mnt/data${i}"
        
        if [[ -b "$disk" ]]; then
            log_info "Setting up $disk → $mountpoint"
            
            # Create mount point
            sudo mkdir -p "$mountpoint"
            
            # Check if already mounted
            if mountpoint -q "$mountpoint" 2>/dev/null; then
                log_info "$disk already mounted"
            else
                # Format and mount
                sudo mkfs.ext4 -F "$disk" >/dev/null 2>&1
                sudo mount "$disk" "$mountpoint"
                
                # Set permissions
                sudo chown -R $(whoami):$(whoami) "$mountpoint" 2>/dev/null || true
                
                # Add to fstab
                if ! grep -q "$disk" /etc/fstab 2>/dev/null; then
                    echo "$disk $mountpoint ext4 defaults,nofail 0 2" | sudo tee -a /etc/fstab >/dev/null
                fi
                
                log_success "$disk mounted and configured"
            fi
            
            ((disk_count++))
        fi
    done
    
    if [[ $disk_count -gt 0 ]]; then
        log_success "Configured $disk_count additional SSDs"
    else
        log_info "No additional SSDs found"
    fi
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
    echo -e "${GRAY}│${NC} ${BLUE}NVMe SSDs:${NC}       $(lsblk | grep nvme | wc -l) detected"
    echo -e "${GRAY}│${NC} ${BLUE}Primary Target:${NC}  $PRIMARY_DISK → $PRIMARY_MOUNT"
    echo -e "${GRAY}└──────────────────────────────────────────────────────────┘${NC}\n"
}

# Verify SSD setup
verify_ssd_setup() {
    log_step "Verifying SSD setup..."
    
    # Check mount
    if mountpoint -q "$PRIMARY_MOUNT" 2>/dev/null; then
        log_success "Primary mount verified: $PRIMARY_MOUNT"
        
        # Show space usage
        echo -e "\n${CYAN}━━━ Storage Information ━━━${NC}"
        df -h "$PRIMARY_MOUNT" | tail -1 | while read filesystem size used avail use_percent mountpoint; do
            echo -e "${BLUE}Filesystem:${NC} $filesystem"
            echo -e "${BLUE}Size:${NC}       $size"
            echo -e "${BLUE}Used:${NC}       $used"
            echo -e "${BLUE}Available:${NC}  $avail"
            echo -e "${BLUE}Usage:${NC}      $use_percent"
            echo -e "${BLUE}Mount:${NC}      $mountpoint"
        done
        echo ""
    else
        log_info "SSD not mounted, using current filesystem"
    fi
    
    # Check symlink
    if [[ -L "$DEV_SYMLINK" ]] && [[ -d "$DEV_SYMLINK" ]]; then
        log_success "Dev symlink verified: $DEV_SYMLINK"
    else
        log_info "Dev symlink not created (SSD setup skipped)"
    fi
}

# Main execution
main() {
    clear
    print_header
    show_system_info
    
    # Prompt for SSD setup mode
    echo -e "${YELLOW}Setup Mode:${NC}"
    echo -e "  ${CYAN}1.${NC} Standard setup (current directory)"
    echo -e "  ${CYAN}2.${NC} SSD-aware setup (mount and symlink)"
    echo -e "  ${CYAN}3.${NC} Full SSD array setup (all available SSDs)"
    echo -e ""
    read -p "Select mode [1-3] (default: 1): " setup_mode
    setup_mode=${setup_mode:-1}
    
    # SSD Setup Phase
    if [[ "$setup_mode" == "2" ]] || [[ "$setup_mode" == "3" ]]; then
        echo -e "\n${PURPLE}╔══════════════════════════════════════════════════════════╗${NC}"
        echo -e "${PURPLE}║                    SSD SETUP PHASE                       ║${NC}"
        echo -e "${PURPLE}╚══════════════════════════════════════════════════════════╝${NC}"
        
        # Check sudo access
        if ! sudo -n true 2>/dev/null; then
            log_warning "SSD setup requires sudo access for disk operations"
            read -p "Continue without SSD setup? [y/N]: " continue_anyway
            if [[ "${continue_anyway,,}" != "y" ]]; then
                echo -e "${RED}Exiting. Please run with sudo or configure sudo access.${NC}"
                exit 1
            fi
            setup_mode=1
        else
            # Setup primary SSD
            if setup_primary_disk; then
                setup_dev_symlink
                
                # Setup additional SSDs if requested
                if [[ "$setup_mode" == "3" ]]; then
                    setup_additional_ssds
                fi
                
                verify_ssd_setup
            else
                log_warning "SSD setup failed, continuing with standard setup"
                setup_mode=1
            fi
        fi
    fi
    
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
    ENV="$(pwd)/miltronic_env"
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
    "$ENV/bin/pip" install --upgrade pip > /dev/null 2>&1 &
    spinner $!
    log_success "pip upgraded"

    # STEP 3: Install dependencies  
    log_step "Installing Python dependencies..."
    log_info "Installing PyTorch with CUDA support..."
    "$ENV/bin/pip" install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 > /dev/null 2>&1 &
    spinner $!
    
    log_info "Installing RL and analysis libraries..."
    "$ENV/bin/pip" install stable-baselines3[extra] "gymnasium[atari,box2d]" autorom ale-py wandb mpmath matplotlib seaborn pandas box2d-py > /dev/null 2>&1 &
    spinner $!
    log_success "All dependencies installed"

    # STEP 4: Install ROMs
    log_step "Installing Atari ROMs..."
    if "$ENV/bin/python" -m AutoROM --accept-license > /dev/null 2>&1; then
        log_success "Atari ROMs installed"
    else
        log_warning "ROM installation may have failed, but continuing..."
    fi

    # STEP 5: Verify environment
    log_step "Verifying ALE/MsPacman-v5 environment..."
    if "$ENV/bin/python" -c "
import gymnasium
import ale_py
try:
    env = gymnasium.make('ALE/MsPacman-v5')
    obs = env.reset()
    print('Environment verified successfully')
    env.close()
except Exception as e:
    print(f'Verification failed: {str(e)}')
    exit(1)
" 2>&1; then
        log_success "Environment verification passed"
    else
        log_warning "Environment verification failed"
        log_info "This might be due to missing ROMs or other setup issues"
        log_info "Training may still work - continuing with execution..."
    fi

    # STEP 6: WandB config
    log_step "Configuring Weights & Biases..."
    export WANDB_PROJECT="miltronic-k-prime-release"
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
    
    log_step "Starting comparative training (k'-Prime + Baseline)..."
    log_info "This will run two 5M timestep training sessions"
    log_info "Expected duration: ${YELLOW}~8-12 hours${NC}"
    
    TRAIN_LOG="logs/train_$(date +%Y%m%d_%H%M).log"
    echo -e "${GRAY}Logging to: $TRAIN_LOG${NC}"
    
    # Ensure we're in the virtual environment for training
    "$ENV/bin/python" train.py | tee "$TRAIN_LOG"
    
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
    
    # Ensure we're in the virtual environment for analysis
    "$ENV/bin/python" analyze.py | tee "$ANALYSIS_LOG"
    
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
