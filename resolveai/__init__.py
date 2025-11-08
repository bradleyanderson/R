"""
ResolveAI - Open Source AI Assistant for DaVinci Resolve

A secure, cloud-enabled AI assistant specifically designed for DaVinci Resolve 
video editing workflows.
"""

__version__ = "1.0.0"
__author__ = "ResolveAI Contributors"
__email__ = "contributors@resolveai.ai"
__license__ = "MIT"

from .core.assistant import ResolveAIAssistant
from .core.universal_assistant import UniversalAssistant, UniversalAssistantConfig
from .core.video_analyzer import VideoAnalyzer
from .core.screen_capture import ScreenCapture
from .core.security import SecurityManager
from .config.settings import Settings

# Import universal components
from .universal import (
    UniversalScreenIntelligence,
    CrossApplicationAutomationEngine,
    ConversationalInterface,
    AdaptiveLearningEngine,
    PluginManager,
    MultiModalProcessor,
    RealTimeCollaborationEngine,
    AIOrchestrator,
    universal_platform
)

__all__ = [
    # Core components
    "ResolveAIAssistant",
    "UniversalAssistant",
    "UniversalAssistantConfig",
    "VideoAnalyzer", 
    "ScreenCapture",
    "SecurityManager",
    "Settings",
    
    # Universal components
    "UniversalScreenIntelligence",
    "CrossApplicationAutomationEngine",
    "ConversationalInterface",
    "AdaptiveLearningEngine",
    "PluginManager",
    "MultiModalProcessor",
    "RealTimeCollaborationEngine",
    "AIOrchestrator",
    "universal_platform"
]