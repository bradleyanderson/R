# ⚡ ResolveAI Universal Platform - Quick Start Guide

Get up and running with ResolveAI in just 5 minutes! 🚀

---

## 🎯 One-Line Installation

```bash
curl -fsSL https://raw.githubusercontent.com/bradleyanderson/R/main/scripts/install.sh | bash
```

---

## ⚡ Quick Start Commands

### 1. Install & Setup
```bash
# Clone and install
git clone https://github.com/bradleyanderson/R.git resolveai-universal
cd resolveai-universal
./scripts/install.sh
```

### 2. Configure API Keys
```bash
# Edit configuration
nano ~/.resolveai/config.yaml

# Add your keys:
# OPENAI_API_KEY=sk-your-key
# ANTHROPIC_API_KEY=sk-ant-your-key
```

### 3. Start the Assistant
```bash
# Activate virtual environment
source venv/bin/activate

# Start ResolveAI
resolveai start
```

### 4. Access Web Interface
Open your browser to: **http://localhost:8000**

---

## 🤖 Try These Commands

### Screen Intelligence
```bash
# Analyze current screen
resolveai screen --analyze

# Take screenshot
resolveai screen --screenshot my_screen.png

# Start monitoring
resolveai screen --monitor
```

### Automation
```bash
# List available workflows
resolveai automate --list

# Run a workflow
resolveai automate --workflow examples/hello_world.yaml

# Record new workflow
resolveai automate --record
```

### Plugin Management
```bash
# List installed plugins
resolveai plugin --list

# Install new plugin
resolveai plugin --install https://github.com/user/plugin.git
```

---

## 🎮 First Time Examples

### Example 1: Control Any Application
```python
# In the web interface or Python console:
from resolveai import UniversalAssistant

assistant = UniversalAssistant()
await assistant.start()

# Control software with natural language
response = await assistant.process_request(
    "Open Chrome and navigate to youtube.com"
)
```

### Example 2: Screen Analysis
```python
# Analyze what's on screen
analysis = await assistant.analyze_screen()
print(f"Found {len(analysis['ui_elements'])} UI elements")
print(f"Interface type: {analysis['interface_type']}")
```

### Example 3: Workflow Automation
```python
# Create cross-app workflow
workflow = {
    "name": "Daily Report",
    "steps": [
        {"action": "open_app", "app": "Excel"},
        {"action": "create_chart", "data": "sales_data"},
        {"action": "save_file", "format": "pdf"}
    ]
}

await assistant.execute_workflow(workflow)
```

---

## 🔧 Configuration Templates

### Minimal Config
```yaml
# ~/.resolveai/config.yaml
assistant:
  enable_screen_intelligence: true
  enable_automation: true

ai_providers:
  openai:
    api_key: "${OPENAI_API_KEY}"
    model: "gpt-4-vision-preview"

server:
  port: 8000
  debug: false
```

### Full Production Config
```yaml
assistant:
  name: "Production Assistant"
  enable_screen_intelligence: true
  enable_automation: true
  enable_conversational_interface: true
  enable_learning: true

ai_providers:
  openai:
    api_key: "${OPENAI_API_KEY}"
    model: "gpt-4-vision-preview"
    max_tokens: 4096
  anthropic:
    api_key: "${ANTHROPIC_API_KEY}"
    model: "claude-3-opus-20240229"

server:
  host: "0.0.0.0"
  port: 8000
  workers: 4
  debug: false

security:
  encryption_key: "${ENCRYPTION_KEY}"
  enable_audit_logging: true

cloud:
  provider: "aws"
  region: "us-west-2"
  bucket: "my-resolveai-data"

logging:
  level: "INFO"
  file: "~/.resolveai/logs/resolveai.log"
```

---

## 📱 Web Interface Features

Visit **http://localhost:8000** to access:

- 🎥 **Live Screen Preview** - See what the AI sees
- 💬 **Chat Interface** - Natural language control
- 🤖 **Workflow Builder** - Visual automation creation
- 📊 **Analytics Dashboard** - Usage and performance metrics
- 🔌 **Plugin Manager** - Install and manage extensions
- ⚙️ **Settings Panel** - Configure all features

---

## 🎯 Common Use Cases

### 1. Developer Assistant
```bash
# Automate coding tasks
resolveai start
# "Run tests in VS Code and open the failed test file"
# "Create a new React component with TypeScript"
# "Deploy the current branch to staging"
```

### 2. Designer Assistant
```bash
# Design workflow automation
# "Create a 1080x1080 Instagram post in Photoshop"
# "Export all artboards as PNG in Illustrator"
# "Apply this color palette to all layers"
```

### 3. Data Analyst
```bash
# Data analysis automation
# "Open the sales CSV in Excel and create a pivot table"
# "Generate a Python notebook for this dataset"
# "Create a dashboard in Google Sheets"
```

### 4. Content Creator
```bash
# Video and content automation
# "Edit this video in DaVinci Resolve: add transitions and color grade"
# "Generate a thumbnail for this YouTube video"
# "Post this content to all social media platforms"
```

---

## 🔍 Troubleshooting Quick Fixes

### Permission Issues (macOS)
```bash
# Reset screen permissions
tccutil reset ScreenCapture
# Then grant Terminal permission in System Preferences
```

### Memory Issues
```bash
# Close other applications
# Use smaller AI models
# Enable GPU acceleration
```

### Network Issues
```bash
# Check firewall settings
# Verify API keys are valid
# Test internet connection
```

---

## 📚 Next Steps

1. **Read the full documentation**: [INSTALL.md](INSTALL.md)
2. **Explore examples**: `examples/` directory
3. **Join the community**: Discord and GitHub
4. **Build plugins**: Create custom integrations
5. **Deploy to cloud**: Scale your assistant

---

## 🆘 Need Help?

- **📖 Full Documentation**: [INSTALL.md](INSTALL.md)
- **💬 Discord**: https://discord.gg/resolveai
- **🐛 Issues**: https://github.com/bradleyanderson/R/issues
- **📧 Email**: support@resolveai.ai

---

## 🎉 You're Ready!

**Welcome to the future of human-computer interaction!** 🤖✨

Start with: `resolveai start` and visit http://localhost:8000

The universal AI assistant is now ready to help you with any software, any workflow, any task.