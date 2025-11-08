# ResolveAI User Guide

Welcome to ResolveAI, your AI-powered assistant for DaVinci Resolve video editing!

## 🚀 Getting Started

### System Requirements

- **Operating System**: Windows 10+, macOS 10.15+, or Linux (Ubuntu 20.04+)
- **DaVinci Resolve**: Version 18.5 or later
- **Python**: 3.8 or later (for local development)
- **Memory**: 16GB RAM recommended (8GB minimum)
- **Storage**: 10GB free space for application and models
- **GPU**: NVIDIA GPU with CUDA support recommended for AI processing

### Quick Installation

1. **Download the latest release**
   ```bash
   git clone https://github.com/resolveai/resolveai-assistant.git
   cd resolveai-assistant
   ```

2. **Run the setup script**
   ```bash
   ./scripts/setup.sh
   ```

3. **Configure your environment**
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

4. **Start ResolveAI**
   ```bash
   docker-compose up
   ```

5. **Access the web interface**
   - Open http://localhost:8000 in your browser
   - Navigate to http://localhost:8000/docs for API documentation

## 🔧 Configuration

### Environment Variables

Key configuration options in your `.env` file:

```bash
# Cloud Processing
RESOLVEAI_CLOUD_PROVIDER=aws
RESOLVEAI_CLOUD_REGION=us-west-2
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key

# Security
ENCRYPTION_KEY=your-32-character-key
LOCAL_PROCESSING_ONLY=false

# Video Processing
RESOLVEAI_MAX_RESOLUTION=4K
RESOLVEAI_MAX_FILE_SIZE_MB=2048
GPU_ACCELERATION=true

# Audio Processing
RESOLVEAI_SAMPLE_RATE=48000
WHISPER_MODEL=base
```

### Cloud Setup

#### AWS Configuration

1. Create an AWS account if you don't have one
2. Create an S3 bucket for video storage
3. Create an IAM user with these permissions:
   - S3: Full access to your bucket
   - Rekognition: Video analysis permissions
   - Lambda: Function execution permissions

4. Add your credentials to `.env`:
   ```bash
   AWS_ACCESS_KEY_ID=your-access-key-id
   AWS_SECRET_ACCESS_KEY=your-secret-access-key
   AWS_BUCKET_NAME=your-bucket-name
   ```

#### Google Cloud Configuration

1. Create a Google Cloud project
2. Enable Cloud Storage and Video Intelligence APIs
3. Create a service account with necessary permissions
4. Download the service account key file

5. Update `.env`:
   ```bash
   GCP_PROJECT_ID=your-project-id
   GCP_SERVICE_ACCOUNT_KEY=path/to/key.json
   GCP_BUCKET_NAME=your-bucket-name
   ```

## 🎯 Core Features

### 1. Timeline Analysis

ResolveAI can analyze your DaVinci Resolve timeline to provide intelligent suggestions:

**How to use:**
1. Open your project in DaVinci Resolve
2. Start ResolveAI assistant
3. Click "Analyze Timeline" in the web interface
4. Review the analysis and suggestions

**What it analyzes:**
- Clip transitions and pacing
- Audio levels and synchronization
- Color consistency across shots
- Edit points and potential improvements
- Timeline structure and organization

### 2. Video File Processing

Process individual video files for content analysis:

**Supported formats:**
- MP4, MOV, AVI, MKV, WebM, MXF
- Up to 4K resolution
 Various frame rates and codecs

**Processing options:**
- Scene detection
- Object recognition
- Content moderation
- Color analysis
- Audio transcription

### 3. Screen Analysis

Get real-time analysis of your DaVinci Resolve interface:

**Features:**
- Interface element recognition
- Tool and mode detection
- Timeline position awareness
- Context-aware suggestions
- Real-time guidance

### 4. AI-Powered Suggestions

Receive intelligent editing suggestions based on your content:

**Types of suggestions:**
- **Edit recommendations**: Cut points, transitions, timing
- **Color grading**: Color correction, matching, grading suggestions
- **Audio improvements**: Level adjustments, noise reduction, EQ
- **Effects and transitions**: Creative effects and transition ideas

## 📱 Web Interface

### Dashboard

The main dashboard provides:
- Assistant status and health
- Current project information
- Recent analysis results
- Quick action buttons

### Timeline Analysis View

- Visual timeline representation
- Clip-by-clip analysis
- Suggestion cards with details
- Apply/edit/reject options

### Video Processing View

- Upload interface for video files
- Processing progress tracking
- Result visualization
- Export options

### Settings Panel

Configure:
- Cloud provider settings
- Processing preferences
- Security options
- API credentials

## 🔌 DaVinci Resolve Integration

### Plugin Installation

1. **Download the ResolveAI plugin** from the releases page
2. **Install in DaVinci Resolve**:
   - Go to `DaVinci Resolve > Preferences > Plugins`
   - Click "Install Plugin" and select the downloaded file
   - Restart DaVinci Resolve

### Connection Setup

1. **Enable the plugin** in the Workspace menu
2. **Configure connection settings**:
   - Server URL: `http://localhost:8000`
   - API Key: Generated from the web interface
   - Auto-connect: Enable for seamless integration

### Real-time Features

- **Timeline synchronization**: Live timeline updates
- **Context awareness**: Knows what you're working on
- **Instant suggestions**: Real-time editing recommendations
- **Screen sharing**: Interface analysis for guidance

## 🎨 Workflows

### Beginner Workflow

1. **Import footage** into DaVinci Resolve
2. **Start ResolveAI** assistant
3. **Analyze timeline** for basic suggestions
4. **Apply AI recommendations** with one click
5. **Review and fine-tune** edits manually

### Professional Workflow

1. **Set up project** with organized media
2. **Configure AI preferences** for your style
3. **Process footage** in the cloud for deep analysis
4. **Review AI insights** alongside your expertise
5. **Apply selective suggestions** that match your vision
6. **Use screen analysis** for real-time guidance

### Color Grading Workflow

1. **Perform basic edit** first
2. **Analyze timeline** for color consistency
3. **Review color suggestions** from AI
4. **Apply base corrections** automatically
5. **Fine-tune creatively** on the Color page

### Audio Post-Production Workflow

1. **Sync and organize** audio tracks
2. **Run audio analysis** for issues
3. **Apply AI suggestions** for levels and cleanup
4. **Review transcription** for content timing
5. **Finalize mix** with manual adjustments

## 🛠️ Advanced Features

### Custom AI Models

Train custom models for your specific needs:

1. **Prepare training data** from your projects
2. **Configure training** in the settings
3. **Upload dataset** to cloud storage
4. **Monitor training** progress
5. **Deploy model** for use in suggestions

### Batch Processing

Process multiple files simultaneously:

1. **Select files** in the interface
2. **Configure processing options**
3. **Start batch job**
4. **Monitor progress** in real-time
5. **Download results** when complete

### API Integration

Use ResolveAI programmatically:

```python
import requests

# Start assistant
response = requests.post('http://localhost:8000/assistant/start', 
                        json={'enable_cloud_processing': True})
session_id = response.json()['session_id']

# Analyze timeline
response = requests.post('http://localhost:8000/analyze/timeline',
                        headers={'Authorization': 'Bearer your-token'})
analysis = response.json()['data']
```

### Collaboration Features

- **Real-time updates**: Share analysis with team members
- **Comment system**: Add notes to suggestions
- **Version control**: Track changes and decisions
- **Export reports**: Share analysis results

## 🔒 Security and Privacy

### Data Protection

- **End-to-end encryption**: All data encrypted in transit and at rest
- **Local processing option**: Keep sensitive data on your machine
- **User control**: Choose what gets processed in the cloud
- **Data retention**: Automatic cleanup after specified period

### Privacy Settings

```bash
# Enable local-only processing
LOCAL_PROCESSING_ONLY=true

# Set data retention period
DATA_RETENTION_DAYS=7

# Disable analytics
ANALYTICS_ENABLED=false
```

### Authentication

- **JWT tokens**: Secure API authentication
- **Two-factor auth**: Optional 2FA for additional security
- **Session management**: Automatic timeout and logout
- **Audit logging**: Track all actions for compliance

## 🚨 Troubleshooting

### Common Issues

#### Connection Problems

**Issue**: Cannot connect to DaVinci Resolve
**Solution**:
1. Verify DaVinci Resolve is running
2. Check plugin is installed and enabled
3. Confirm firewall allows port 8080
4. Restart both applications

#### Cloud Processing Errors

**Issue**: Cloud processing fails
**Solution**:
1. Check internet connection
2. Verify cloud credentials in .env
3. Check cloud service permissions
4. Review cloud service status

#### Performance Issues

**Issue**: Slow processing
**Solution**:
1. Enable GPU acceleration
2. Increase processing threads
3. Use local processing for small files
4. Close other applications

#### Memory Errors

**Issue**: Out of memory errors
**Solution**:
1. Reduce video resolution
2. Limit concurrent processes
3. Increase system RAM
4. Use cloud processing

### Debug Mode

Enable debug logging for troubleshooting:

```bash
# Add to .env
RESOLVEAI_LOG_LEVEL=DEBUG
DEBUG=true
```

Check logs in the `logs/` directory or Docker logs.

### Getting Help

- **Documentation**: Visit docs.resolveai.ai
- **Community**: Join our Discord server
- **Issues**: Report bugs on GitHub
- **Email**: support@resolveai.ai

## 📈 Performance Tips

### Optimization

1. **Use GPU acceleration** for video processing
2. **Enable cloud processing** for large files
3. **Configure appropriate resolution** limits
4. **Optimize timeline structure** before analysis

### Best Practices

1. **Organize media** before analysis
2. **Use consistent naming** conventions
3. **Regularly clean up** temp files
4. **Monitor cloud costs** and usage

### Resource Monitoring

Use the built-in monitoring tools:
- CPU and memory usage
- Cloud processing costs
- API request limits
- Storage usage

## 🎓 Learning Resources

### Tutorials

- **Getting Started**: Basic setup and first analysis
- **Advanced Workflows**: Professional editing techniques
- **Color Grading**: AI-assisted color correction
- **Audio Post**: Audio enhancement and cleanup

### Video Guides

- [Setup and Installation](https://www.youtube.com/watch?v=example)
- [Timeline Analysis](https://www.youtube.com/watch?v=example)
- [Cloud Processing](https://www.youtube.com/watch?v=example)
- [API Integration](https://www.youtube.com/watch?v=example)

### Community

- **Discord**: Real-time chat with other users
- **Forum**: Discussion and help threads
- **GitHub**: Contribute to development
- **Blog**: Tips and best practices

---

Thank you for using ResolveAI! We're here to help you create amazing videos with the power of AI.