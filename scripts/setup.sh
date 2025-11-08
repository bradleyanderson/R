#!/bin/bash

# ResolveAI Setup Script
# This script sets up the development environment for ResolveAI

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   print_error "This script should not be run as root"
   exit 1
fi

# Detect OS
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="linux"
        if [ -f /etc/debian_version ]; then
            DISTRO="debian"
        elif [ -f /etc/redhat-release ]; then
            DISTRO="redhat"
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
        OS="windows"
    else
        print_error "Unsupported OS: $OSTYPE"
        exit 1
    fi
    print_status "Detected OS: $OS"
}

# Check Python version
check_python() {
    print_status "Checking Python installation..."
    
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        REQUIRED_VERSION="3.8"
        
        if python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
            print_success "Python $PYTHON_VERSION found"
        else
            print_error "Python 3.8+ is required. Found: $PYTHON_VERSION"
            exit 1
        fi
    else
        print_error "Python 3 is not installed"
        exit 1
    fi
}

# Check system dependencies
check_dependencies() {
    print_status "Checking system dependencies..."
    
    if [[ "$OS" == "linux" ]]; then
        DEPS=("git" "curl" "wget" "ffmpeg")
        for dep in "${DEPS[@]}"; do
            if ! command -v $dep &> /dev/null; then
                print_warning "$dep is not installed"
                MISSING_DEPS+=($dep)
            fi
        done
        
        if [ ${#MISSING_DEPS[@]} -ne 0 ]; then
            print_status "Installing missing dependencies..."
            if [[ "$DISTRO" == "debian" ]]; then
                sudo apt-get update
                sudo apt-get install -y "${MISSING_DEPS[@]}"
            elif [[ "$DISTRO" == "redhat" ]]; then
                sudo yum install -y "${MISSING_DEPS[@]}"
            fi
        fi
    elif [[ "$OS" == "macos" ]]; then
        if ! command -v brew &> /dev/null; then
            print_error "Homebrew is not installed. Please install it first."
            exit 1
        fi
        
        DEPS=("git" "curl" "wget" "ffmpeg")
        for dep in "${DEPS[@]}"; do
            if ! brew list $dep &> /dev/null; then
                print_status "Installing $dep..."
                brew install $dep
            fi
        done
    fi
    
    print_success "System dependencies check completed"
}

# Create virtual environment
create_venv() {
    print_status "Creating Python virtual environment..."
    
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        print_success "Virtual environment created"
    else
        print_warning "Virtual environment already exists"
    fi
    
    # Activate virtual environment
    if [[ "$OS" == "windows" ]]; then
        source venv/Scripts/activate
    else
        source venv/bin/activate
    fi
    
    # Upgrade pip
    pip install --upgrade pip setuptools wheel
    
    print_success "Virtual environment activated"
}

# Install Python dependencies
install_python_deps() {
    print_status "Installing Python dependencies..."
    
    # Install requirements
    pip install -r requirements.txt
    
    # Install development dependencies
    if [ -f "requirements-dev.txt" ]; then
        pip install -r requirements-dev.txt
    fi
    
    # Install the package in development mode
    pip install -e .
    
    print_success "Python dependencies installed"
}

# Setup configuration files
setup_config() {
    print_status "Setting up configuration..."
    
    # Create .env file if it doesn't exist
    if [ ! -f ".env" ]; then
        cp .env.example .env
        print_warning "Please edit .env file with your configuration"
    else
        print_warning ".env file already exists"
    fi
    
    # Create necessary directories
    mkdir -p data logs config static
    
    # Generate encryption key if not set
    if ! grep -q "ENCRYPTION_KEY=" .env || grep -q "your-32-character-encryption-key-here" .env; then
        ENCRYPTION_KEY=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
        sed -i "s/ENCRYPTION_KEY=.*/ENCRYPTION_KEY=$ENCRYPTION_KEY/" .env
        print_success "Generated encryption key"
    fi
    
    print_success "Configuration setup completed"
}

# Setup pre-commit hooks
setup_precommit() {
    print_status "Setting up pre-commit hooks..."
    
    if command -v pre-commit &> /dev/null; then
        pre-commit install
        print_success "Pre-commit hooks installed"
    else
        print_warning "pre-commit not available. Install with: pip install pre-commit"
    fi
}

# Initialize database (if needed)
init_database() {
    print_status "Checking database setup..."
    
    # Check if PostgreSQL is available
    if command -v psql &> /dev/null; then
        print_success "PostgreSQL found"
    else
        print_warning "PostgreSQL not found. Using Docker for database..."
    fi
    
    # Check if Redis is available
    if command -v redis-cli &> /dev/null; then
        print_success "Redis found"
    else
        print_warning "Redis not found. Using Docker for Redis..."
    fi
}

# Setup DaVinci Resolve integration
setup_davinci() {
    print_status "Setting up DaVinci Resolve integration..."
    
    # Check if DaVinci Resolve is installed
    if [[ "$OS" == "macos" ]]; then
        DAVINCI_PATH="/Applications/DaVinci Resolve/DaVinci Resolve.app"
    elif [[ "$OS" == "linux" ]]; then
        DAVINCI_PATH="/opt/resolve/bin/resolve"
    elif [[ "$OS" == "windows" ]]; then
        DAVINCI_PATH="C:\\Program Files\\Blackmagic Design\\DaVinci Resolve\\DaVinciResolve.exe"
    fi
    
    if [ -f "$DAVINCI_PATH" ] || [ -d "$DAVINCI_PATH" ]; then
        print_success "DaVinci Resolve installation found"
    else
        print_warning "DaVinci Resolve not found at $DAVINCI_PATH"
        print_warning "Please install DaVinci Resolve and update the path in configuration"
    fi
}

# Run tests
run_tests() {
    print_status "Running tests..."
    
    if command -v pytest &> /dev/null; then
        pytest tests/ -v --tb=short
        print_success "Tests completed"
    else
        print_warning "pytest not found. Install with: pip install pytest"
    fi
}

# Build Docker images
build_docker() {
    print_status "Building Docker images..."
    
    if command -v docker &> /dev/null; then
        docker-compose build
        print_success "Docker images built successfully"
    else
        print_warning "Docker not found. Skipping Docker build."
    fi
}

# Main setup function
main() {
    print_status "Starting ResolveAI setup..."
    
    detect_os
    check_python
    check_dependencies
    create_venv
    install_python_deps
    setup_config
    setup_precommit
    init_database
    setup_davinci
    
    # Optional steps
    read -p "Run tests? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        run_tests
    fi
    
    read -p "Build Docker images? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        build_docker
    fi
    
    print_success "ResolveAI setup completed!"
    print_status "Next steps:"
    echo "  1. Edit .env file with your configuration"
    echo "  2. Start the development server: python -m resolveai.server.api"
    echo "  3. Or use Docker: docker-compose up"
    echo "  4. Open http://localhost:8000/docs for API documentation"
}

# Handle script arguments
case "${1:-}" in
    --help|-h)
        echo "ResolveAI Setup Script"
        echo "Usage: $0 [options]"
        echo ""
        echo "Options:"
        echo "  --help, -h     Show this help message"
        echo "  --test         Run tests after setup"
        echo "  --docker       Build Docker images"
        echo "  --no-deps      Skip system dependency check"
        exit 0
        ;;
    --test)
        RUN_TESTS=true
        ;;
    --docker)
        BUILD_DOCKER=true
        ;;
    --no-deps)
        SKIP_DEPS=true
        ;;
esac

# Run main function
main