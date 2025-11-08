#!/bin/bash

# ResolveAI Universal Platform Installation Script
# This script installs ResolveAI and all its dependencies

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print functions
print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️ $1${NC}"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Detect OS
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    elif [[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
        echo "windows"
    else
        echo "unknown"
    fi
}

# Install system dependencies
install_system_deps() {
    print_header "Installing System Dependencies"
    
    OS=$(detect_os)
    
    case $OS in
        "linux")
            print_info "Detected Linux. Installing system packages..."
            
            # Check for package manager
            if command_exists apt-get; then
                sudo apt-get update
                sudo apt-get install -y \
                    python3 \
                    python3-pip \
                    python3-venv \
                    git \
                    curl \
                    wget \
                    build-essential \
                    cmake \
                    pkg-config \
                    libjpeg-dev \
                    libpng-dev \
                    libtiff-dev \
                    libavcodec-dev \
                    libavformat-dev \
                    libswscale-dev \
                    libv4l-dev \
                    libxvidcore-dev \
                    libx264-dev \
                    libgtk-3-dev \
                    libatlas-base-dev \
                    gfortran
            elif command_exists yum; then
                sudo yum update -y
                sudo yum install -y \
                    python3 \
                    python3-pip \
                    git \
                    curl \
                    wget \
                    gcc \
                    gcc-c++ \
                    cmake \
                    pkgconfig \
                    libjpeg-turbo-devel \
                    libpng-devel \
                    libtiff-devel \
                    libavcodec-devel \
                    libavformat-devel \
                    libswscale-devel \
                    gtk3-devel \
                    atlas-devel \
                    gcc-gfortran
            else
                print_warning "Could not detect package manager. Please install Python 3.9+ and build tools manually."
            fi
            ;;
            
        "macos")
            print_info "Detected macOS. Installing with Homebrew..."
            
            if ! command_exists brew; then
                print_info "Installing Homebrew..."
                /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
            fi
            
            brew update
            brew install \
                python@3.11 \
                git \
                cmake \
                pkg-config \
                jpeg \
                libpng \
                libtiff \
                ffmpeg \
                gtk+3
            ;;
            
        "windows")
            print_error "Windows installation requires manual setup. Please see the manual installation guide."
            exit 1
            ;;
            
        *)
            print_error "Unsupported operating system: $OS"
            exit 1
            ;;
    esac
    
    print_success "System dependencies installed"
}

# Create Python virtual environment
create_venv() {
    print_header "Setting up Python Virtual Environment"
    
    if [ -d "venv" ]; then
        print_warning "Virtual environment already exists. Removing it..."
        rm -rf venv
    fi
    
    print_info "Creating virtual environment..."
    python3 -m venv venv
    
    print_info "Activating virtual environment..."
    source venv/bin/activate
    
    print_success "Virtual environment created and activated"
}

# Upgrade pip and install wheel
upgrade_pip() {
    print_header "Upgrading pip and installing wheel"
    
    source venv/bin/activate
    pip install --upgrade pip setuptools wheel
    
    print_success "pip upgraded successfully"
}

# Install Python dependencies
install_python_deps() {
    print_header "Installing Python Dependencies"
    
    source venv/bin/activate
    
    # Install requirements in batches to avoid memory issues
    print_info "Installing core dependencies..."
    pip install -r requirements.txt
    
    print_success "Python dependencies installed"
}

# Install ResolveAI in development mode
install_resolveai() {
    print_header "Installing ResolveAI"
    
    source venv/bin/activate
    
    pip install -e .
    
    print_success "ResolveAI installed successfully"
}

# Initialize configuration
init_config() {
    print_header "Initializing Configuration"
    
    source venv/bin/activate
    
    # Create config directory
    mkdir -p ~/.resolveai/logs
    
    # Initialize configuration
    resolveai config --init
    
    print_success "Configuration initialized"
}

# Setup screen capture permissions (macOS)
setup_screen_permissions() {
    OS=$(detect_os)
    
    if [ "$OS" = "macos" ]; then
        print_header "Screen Capture Permissions (macOS)"
        print_warning "On macOS, you must grant screen recording permissions to Terminal:"
        print_info "1. Go to System Preferences > Security & Privacy > Privacy"
        print_info "2. Click on 'Screen Recording'"
        print_info "3. Add Terminal to the list of allowed applications"
        print_info "4. Restart Terminal after granting permissions"
        echo ""
        print_info "Press Enter to continue..."
        read -r
    fi
}

# Test installation
test_installation() {
    print_header "Testing Installation"
    
    source venv/bin/activate
    
    print_info "Testing ResolveAI CLI..."
    if resolveai --help > /dev/null 2>&1; then
        print_success "ResolveAI CLI is working"
    else
        print_error "ResolveAI CLI test failed"
        exit 1
    fi
    
    print_info "Testing system status..."
    resolveai status
    
    print_success "Installation test completed"
}

# Create desktop shortcut (optional)
create_shortcut() {
    print_header "Creating Desktop Shortcut"
    
    OS=$(detect_os)
    
    case $OS in
        "linux")
            DESKTOP_DIR="$HOME/Desktop"
            if [ -d "$DESKTOP_DIR" ]; then
                cat > "$DESKTOP_DIR/ResolveAI.desktop" << EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=ResolveAI
Comment=Universal AI Assistant Platform
Exec=$(pwd)/venv/bin/resolveai start
Icon=$(pwd)/assets/icon.png
Terminal=true
Categories=Development;Utility;
EOF
                chmod +x "$DESKTOP_DIR/ResolveAI.desktop"
                print_success "Desktop shortcut created"
            fi
            ;;
        "macos")
            print_info "To create a macOS app, run: ./scripts/create-macos-app.sh"
            ;;
    esac
}

# Main installation function
main() {
    print_header "ResolveAI Universal Platform Installation"
    print_info "This script will install ResolveAI and all dependencies"
    echo ""
    
    # Check if we're in the right directory
    if [ ! -f "setup.py" ]; then
        print_error "Please run this script from the ResolveAI root directory"
        exit 1
    fi
    
    # Installation steps
    install_system_deps
    create_venv
    upgrade_pip
    install_python_deps
    install_resolveai
    init_config
    setup_screen_permissions
    test_installation
    
    # Optional steps
    read -p "Do you want to create a desktop shortcut? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        create_shortcut
    fi
    
    print_header "Installation Complete!"
    print_success "ResolveAI has been successfully installed!"
    echo ""
    print_info "To get started:"
    print_info "1. Activate the virtual environment: source venv/bin/activate"
    print_info "2. Set your API keys in ~/.resolveai/config.yaml"
    print_info "3. Start the assistant: resolveai start"
    print_info "4. Open your browser to: http://localhost:8000"
    echo ""
    print_info "For help: resolveai --help"
    print_info "Documentation: https://docs.resolveai.ai"
    print_info "GitHub: https://github.com/resolveai/resolveai-universal"
}

# Run the installation
main "$@"