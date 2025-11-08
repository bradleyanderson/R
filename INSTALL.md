# 🚀 ResolveAI Universal Platform - Installation Guide

## 📋 Prerequisites

Before installing ResolveAI, ensure your system meets these requirements:

### System Requirements
- **Operating System**: Linux (Ubuntu 18.04+), macOS (10.15+), or Windows 10+
- **Python**: 3.9, 3.10, 3.11, or 3.12
- **RAM**: Minimum 8GB, recommended 16GB+
- **Storage**: At least 10GB free space
- **GPU**: Optional but recommended for AI processing (NVIDIA CUDA or Apple Silicon)

### Required Software
- Git
- Python 3.9+ with pip
- Build tools (gcc, make, cmake)
- Screen capture permissions

---

## 🎯 Quick Installation (Recommended)

### Automated Installation Script

The easiest way to install ResolveAI is using our automated installation script:

```bash
# Clone the repository
git clone https://github.com/bradleyanderson/R.git resolveai-universal
cd resolveai-universal

# Run the installation script
./scripts/install.sh
```

The script will:
- ✅ Install all system dependencies
- ✅ Create a Python virtual environment
- ✅ Install all Python packages
- ✅ Initialize configuration
- ✅ Set up permissions
- ✅ Test the installation

---

## 🔧 Manual Installation

If you prefer to install manually, follow these steps:

### 1. System Dependencies

#### Linux (Ubuntu/Debian)
```bash
sudo apt-get update
sudo apt-get install -y \
    python3 python3-pip python3-venv \
    git curl wget build-essential cmake pkg-config \
    libjpeg-dev libpng-dev libtiff-dev \
    libavcodec-dev libavformat-dev libswscale-dev \
    libv4l-dev libxvidcore-dev libx264-dev \
    libgtk-3-dev libatlas-base-dev gfortran
```

#### macOS
```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew update
brew install python@3.11 git cmake pkgconfig \
    jpeg libpng libtiff ffmpeg gtk+3
```

#### Windows
```powershell
# Install Python 3.9+ from https://python.org
# Install Git from https://git-scm.com/
# Install Visual Studio Build Tools from https://visualstudio.microsoft.com/downloads/
```

### 2. Clone Repository
```bash
git clone https://github.com/bradleyanderson/R.git resolveai-universal
cd resolveai-universal
```

### 3. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 4. Upgrade pip
```bash
pip install --upgrade pip setuptools wheel
```

### 5. Install Dependencies
```bash
pip install -r requirements.txt
```

### 6. Install ResolveAI
```bash
pip install -e .
```

### 7. Initialize Configuration
```bash
resolveai config --init
```

---

## 🔑 Configuration Setup

After installation, configure your AI providers:

### 1. Edit Configuration File
```bash
# Open the configuration file
nano ~/.resolveai/config.yaml
```

### 2. Add API Keys
```yaml
ai_providers:
  openai:
    api_key: "sk-your-openai-api-key-here"
    model: "gpt-4-vision-preview"
  anthropic:
    api_key: "sk-ant-your-anthropic-api-key-here"
    model: "claude-3-opus-20240229"

security:
  encryption_key: "your-encryption-key-here"
```

### 3. Set Environment Variables (Optional)
```bash
export OPENAI_API_KEY="sk-your-openai-api-key"
export ANTHROPIC_API_KEY="sk-ant-your-anthropic-api-key"
export ENCRYPTION_KEY="your-encryption-key"
```

---

## 🖥️ Screen Capture Permissions

### macOS
Screen recording permissions are required for screen intelligence:

1. **System Preferences** → **Security & Privacy** → **Privacy**
2. Click **Screen Recording**
3. Add **Terminal** to the allowed applications
4. Restart Terminal after granting permissions

### Linux
Ensure your user is in the appropriate groups:
```bash
sudo usermod -a -G video,input $USER
# Logout and login again for changes to take effect
```

### Windows
Run the terminal as Administrator when using screen capture features.

---

## 🐳 Docker Installation (Alternative)

### 1. Install Docker
Follow the official Docker installation guide for your operating system.

### 2. Clone and Setup
```bash
git clone https://github.com/bradleyanderson/R.git resolveai-universal
cd resolveai-universal
```

### 3. Configure Environment
```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

### 4. Start with Docker Compose
```bash
docker-compose up -d
```

The services will be available at:
- **API Server**: http://localhost:8000
- **Redis**: localhost:6379
- **PostgreSQL**: localhost:5432
- **Grafana**: http://localhost:3000
- **Prometheus**: http://localhost:9090

---

## ✅ Verification

### 1. Test CLI Installation
```bash
resolveai --help
```

### 2. Check System Status
```bash
resolveai status
```

### 3. Test Screen Intelligence
```bash
resolveai screen --analyze
```

### 4. Start the Assistant
```bash
resolveai start
```

### 5. Access Web Interface
Open your browser to: http://localhost:8000

---

## 🛠️ Troubleshooting

### Common Issues

#### 1. Import Errors
```bash
# Reinstall in development mode
pip install -e .

# Check virtual environment is active
which python
```

#### 2. Screen Capture Issues
```bash
# Check permissions (macOS)
tccutil reset ScreenCapture

# Check display server (Linux)
echo $DISPLAY
```

#### 3. Memory Issues
```bash
# Increase swap space (Linux)
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

#### 4. GPU Support
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Install CUDA toolkit if needed
# https://developer.nvidia.com/cuda-downloads
```

### Getting Help

- **Documentation**: https://docs.resolveai.ai
- **GitHub Issues**: https://github.com/bradleyanderson/R/issues
- **Discord Community**: https://discord.gg/resolveai
- **Email Support**: support@resolveai.ai

---

## 🔄 Updates

### Update to Latest Version
```bash
cd resolveai-universal
git pull origin main
source venv/bin/activate
pip install -r requirements.txt --upgrade
```

### Update Specific Components
```bash
# Update core packages
pip install --upgrade opencv-python torch transformers

# Update ResolveAI
pip install --upgrade resolveai-universal
```

---

## 🚦 Next Steps

After successful installation:

1. **Configure AI providers** with your API keys
2. **Start the assistant**: `resolveai start`
3. **Open web interface**: http://localhost:8000
4. **Try examples**: Visit the documentation for usage examples
5. **Join community**: Get help and share your experiences

## 📚 Additional Resources

- **User Guide**: [docs/user-guide.md](docs/user-guide.md)
- **Developer Documentation**: [docs/developer-guide.md](docs/developer-guide.md)
- **Plugin Development**: [docs/plugin-development.md](docs/plugin-development.md)
- **API Reference**: https://api.resolveai.ai
- **Video Tutorials**: https://youtube.com/@resolveai

---

**🎉 Congratulations! You've successfully installed ResolveAI Universal Platform!**

The future of human-computer interaction is now at your fingertips. 🤖✨