# Contributing to ResolveAI

Thank you for your interest in contributing to ResolveAI! This document provides guidelines and information for contributors.

## 🤝 How to Contribute

### Reporting Bugs

- Use the [GitHub issue tracker](https://github.com/resolveai/resolveai-assistant/issues) to report bugs
- Provide detailed information including:
  - Steps to reproduce the issue
  - Expected vs actual behavior
  - System information (OS, DaVinci Resolve version, etc.)
  - Screenshots if applicable

### Suggesting Features

- Open an issue with the "enhancement" label
- Describe the feature and its use case
- Explain how it would benefit the community

### Code Contributions

1. **Fork the Repository**
   ```bash
   git clone https://github.com/your-username/resolveai-assistant.git
   cd resolveai-assistant
   ```

2. **Create a Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

4. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

5. **Make Your Changes**
   - Follow the coding standards below
   - Write tests for new functionality
   - Update documentation as needed

6. **Run Tests**
   ```bash
   pytest
   ```

7. **Submit a Pull Request**
   - Push to your fork
   - Create a pull request with a clear description
   - Link any relevant issues

## 📝 Coding Standards

### Python Code Style

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) and use the following tools:

- **Black** for code formatting
- **flake8** for linting
- **mypy** for type checking

Run the pre-commit hooks before submitting:

```bash
pre-commit install
pre-commit run --all-files
```

### Code Organization

```
resolveai/
├── core/           # Core functionality
├── security/       # Security and encryption
├── cloud/          # Cloud provider integrations
├── davinci/        # DaVinci Resolve API
├── video/          # Video processing
├── audio/          # Audio processing
├── models/         # AI models
├── server/         # API server
├── config/         # Configuration management
├── utils/          # Utility functions
└── plugins/        # DaVinci Resolve plugins
```

### Documentation

- Use docstrings for all public functions and classes
- Follow the Google style for docstrings
- Include type hints for function signatures
- Update README.md and user documentation for new features

Example:

```python
def process_video(file_path: str, options: Dict[str, Any]) -> Dict[str, Any]:
    """Process a video file with AI analysis.
    
    Args:
        file_path: Path to the video file to process
        options: Dictionary of processing options
        
    Returns:
        Dictionary containing analysis results and metadata
        
    Raises:
        FileNotFoundError: If the video file doesn't exist
        ProcessingError: If video processing fails
    """
    pass
```

## 🧪 Testing

### Test Structure

```
tests/
├── unit/           # Unit tests
├── integration/    # Integration tests
├── e2e/           # End-to-end tests
└── fixtures/      # Test data and fixtures
```

### Writing Tests

- Use pytest for all tests
- Write descriptive test names
- Mock external dependencies
- Test both success and failure cases
- Aim for high code coverage

Example:

```python
import pytest
from resolveai.core.assistant import ResolveAIAssistant, AssistantConfig

class TestResolveAIAssistant:
    def test_initialization(self):
        """Test assistant initialization with default config."""
        assistant = ResolveAIAssistant()
        assert assistant.config.enable_cloud_processing is True
    
    @pytest.mark.asyncio
    async def test_start_stop(self):
        """Test starting and stopping the assistant."""
        assistant = ResolveAIAssistant()
        await assistant.start()
        assert assistant._is_running is True
        
        await assistant.stop()
        assert assistant._is_running is False
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=resolveai

# Run specific test file
pytest tests/unit/test_assistant.py

# Run with verbose output
pytest -v
```

## 🔧 Development Setup

### Prerequisites

- Python 3.8+
- DaVinci Resolve 18.5+ (for testing integration)
- Docker and Docker Compose (for containerized development)
- Git

### Local Development

1. **Clone and Setup**
   ```bash
   git clone https://github.com/resolveai/resolveai-assistant.git
   cd resolveai-assistant
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements-dev.txt
   ```

2. **Configuration**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Database Setup**
   ```bash
   # Using Docker Compose
   docker-compose up -d postgres redis
   
   # Or setup local PostgreSQL/Redis
   ```

4. **Run Development Server**
   ```bash
   python -m resolveai.server.api
   ```

### Docker Development

```bash
# Build and run all services
docker-compose up --build

# Run specific service
docker-compose up resolveai-api

# Run tests in container
docker-compose run --rm resolveai-api pytest
```

## 📋 Pull Request Process

### Before Submitting

- [ ] Code follows style guidelines
- [ ] Tests pass locally
- [ ] Documentation is updated
- [ ] Commit messages are clear
- [ ] No merge conflicts

### Pull Request Template

```markdown
## Description
Brief description of changes made.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests written/passed
- [ ] Integration tests written/passed
- [ ] Manual testing completed

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or clearly documented)
```

### Review Process

1. Automated checks must pass
2. At least one maintainer approval required
3. Address all review comments
4. Maintain clean commit history

## 🏗️ Architecture Guidelines

### Modularity

- Keep components loosely coupled
- Use dependency injection
- Implement clear interfaces
- Follow SOLID principles

### Security

- Never commit sensitive data
- Use environment variables for secrets
- Validate all inputs
- Follow secure coding practices

### Performance

- Profile critical paths
- Optimize database queries
- Use caching appropriately
- Consider async/await for I/O operations

## 📚 Documentation Contributions

### User Documentation

- Update `docs/user-guide.md` for user-facing changes
- Add examples for new features
- Include screenshots/video tutorials when helpful

### Developer Documentation

- Update API docs for new endpoints
- Document new configuration options
- Add architecture decision records (ADRs)

### README Updates

- Keep installation instructions current
- Update feature list
- Add new contributor guidelines

## 🌟 Recognition

Contributors are recognized in several ways:

- Contributors section in README
- Release notes mentioning contributors
- Special badges for significant contributions
- Invitation to core team for consistent contributors

## 📞 Getting Help

- **Discord**: [Join our community](https://discord.gg/resolveai)
- **Discussions**: [GitHub Discussions](https://github.com/resolveai/resolveai-assistant/discussions)
- **Issues**: [GitHub Issues](https://github.com/resolveai/resolveai-assistant/issues)
- **Email**: contributors@resolveai.ai

## 📄 License

By contributing to ResolveAI, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to ResolveAI! Your contributions help make video editing more accessible and efficient for everyone.