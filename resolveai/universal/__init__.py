"""
ResolveAI Universal Components

Universal AI assistant components that work across any software, platform,
or digital workflow.

This module provides the core infrastructure for:
- Universal screen intelligence and interface understanding
- Cross-application automation and workflow orchestration
- Conversational interface control with natural language
- Adaptive learning and personalization
- Plugin ecosystem for extensibility
- Multi-modal content processing
- Real-time collaboration frameworks
- Multi-AI integration and orchestration

The universal components transform ResolveAI from a video editing assistant
into a comprehensive AI platform that can understand and interact with
any digital technology.
"""

from .screen_intelligence import UniversalScreenIntelligence, ScreenAnalysis, UIElement
from .automation_engine import (
    CrossApplicationAutomationEngine, 
    Workflow, 
    AutomationStep, 
    ActionType
)
from .conversational_interface import (
    ConversationalInterface, 
    Command, 
    ConversationContext
)
from .learning_engine import (
    AdaptiveLearningEngine, 
    UserProfile, 
    ApplicationProfile
)
from .plugin_system import (
    PluginManager, 
    Plugin, 
    ApplicationIntegrationPlugin,
    AIModelPlugin
)
from .multi_modal_processor import (
    MultiModalProcessor, 
    MediaContent, 
    MediaType
)
from .collaboration_engine import (
    RealTimeCollaborationEngine,
    Workspace,
    User,
    CollaborationRole
)
from .ai_orchestrator import (
    AIOrchestrator,
    AIProvider,
    AIModel,
    AIRequest,
    AIResponse
)

__version__ = "1.0.0"
__author__ = "ResolveAI Contributors"

__all__ = [
    # Screen Intelligence
    "UniversalScreenIntelligence",
    "ScreenAnalysis", 
    "UIElement",
    
    # Automation Engine
    "CrossApplicationAutomationEngine",
    "Workflow",
    "AutomationStep", 
    "ActionType",
    
    # Conversational Interface
    "ConversationalInterface",
    "Command",
    "ConversationContext",
    
    # Learning Engine
    "AdaptiveLearningEngine",
    "UserProfile",
    "ApplicationProfile",
    
    # Plugin System
    "PluginManager",
    "Plugin",
    "ApplicationIntegrationPlugin", 
    "AIModelPlugin",
    
    # Multi-Modal Processing
    "MultiModalProcessor",
    "MediaContent",
    "MediaType",
    
    # Collaboration Engine
    "RealTimeCollaborationEngine",
    "Workspace",
    "User",
    "CollaborationRole",
    
    # AI Orchestrator
    "AIOrchestrator",
    "AIProvider",
    "AIModel",
    "AIRequest", 
    "AIResponse"
]

# Component metadata
COMPONENT_INFO = {
    "universal_screen_intelligence": {
        "description": "Advanced computer vision for any interface",
        "capabilities": ["ui_detection", "text_extraction", "layout_analysis"],
        "dependencies": ["opencv-python", "torch", "transformers"]
    },
    "automation_engine": {
        "description": "Cross-application workflow orchestration",
        "capabilities": ["workflow_automation", "task_scheduling", "error_handling"],
        "dependencies": ["asyncio", "json", "uuid"]
    },
    "conversational_interface": {
        "description": "Natural language control of any software",
        "capabilities": ["command_parsing", "context_understanding", "response_generation"],
        "dependencies": ["nltk", "spacy", "transformers"]
    },
    "learning_engine": {
        "description": "Adaptive personalization and learning",
        "capabilities": ["user_profiling", "pattern_learning", "preference_adaptation"],
        "dependencies": ["scikit-learn", "pandas", "numpy"]
    },
    "plugin_system": {
        "description": "Extensible plugin ecosystem",
        "capabilities": ["plugin_loading", "dependency_management", "security"],
        "dependencies": ["importlib", "inspect", "hashlib"]
    },
    "multi_modal_processor": {
        "description": "Universal media processing",
        "capabilities": ["text_processing", "image_analysis", "audio_processing", "video_processing"],
        "dependencies": ["pillow", "librosa", "moviepy", "pandas"]
    },
    "collaboration_engine": {
        "description": "Real-time multi-user collaboration",
        "capabilities": ["workspace_management", "real_time_sync", "activity_tracking"],
        "dependencies": ["websockets", "redis", "asyncio"]
    },
    "ai_orchestrator": {
        "description": "Multi-AI provider integration",
        "capabilities": ["provider_management", "model_selection", "load_balancing"],
        "dependencies": ["openai", "anthropic", "transformers"]
    }
}


def get_component_info(component_name: str) -> dict:
    """Get information about a specific component."""
    return COMPONENT_INFO.get(component_name, {})


def list_all_components() -> list:
    """List all available universal components."""
    return list(COMPONENT_INFO.keys())


def get_component_dependencies(component_name: str) -> list:
    """Get dependencies for a specific component."""
    info = COMPONENT_INFO.get(component_name, {})
    return info.get("dependencies", [])


def get_component_capabilities(component_name: str) -> list:
    """Get capabilities for a specific component."""
    info = COMPONENT_INFO.get(component_name, {})
    return info.get("capabilities", [])


# Universal platform integration helper
class UniversalPlatform:
    """
    Helper class for integrating all universal components
    into a cohesive platform.
    """
    
    def __init__(self):
        self.components = {}
        self.initialized = False
    
    async def initialize_all(self, config: dict) -> bool:
        """Initialize all universal components."""
        try:
            # Initialize components in dependency order
            init_order = [
                "screen_intelligence",
                "multi_modal_processor", 
                "ai_orchestrator",
                "learning_engine",
                "automation_engine",
                "conversational_interface",
                "plugin_system",
                "collaboration_engine"
            ]
            
            for component_name in init_order:
                component_config = config.get(component_name, {})
                success = await self._initialize_component(component_name, component_config)
                if not success:
                    print(f"Failed to initialize {component_name}")
                    return False
            
            self.initialized = True
            print("Universal platform initialized successfully")
            return True
            
        except Exception as e:
            print(f"Failed to initialize universal platform: {e}")
            return False
    
    async def _initialize_component(self, component_name: str, config: dict) -> bool:
        """Initialize a specific component."""
        try:
            if component_name == "screen_intelligence":
                self.components[component_name] = UniversalScreenIntelligence()
            elif component_name == "automation_engine":
                screen_intel = self.components.get("screen_intelligence")
                self.components[component_name] = CrossApplicationAutomationEngine(screen_intel)
            elif component_name == "conversational_interface":
                screen_intel = self.components.get("screen_intelligence")
                auto_engine = self.components.get("automation_engine")
                self.components[component_name] = ConversationalInterface(screen_intel, auto_engine)
            elif component_name == "learning_engine":
                self.components[component_name] = AdaptiveLearningEngine()
            elif component_name == "plugin_system":
                self.components[component_name] = PluginManager()
            elif component_name == "multi_modal_processor":
                self.components[component_name] = MultiModalProcessor()
            elif component_name == "collaboration_engine":
                self.components[component_name] = RealTimeCollaborationEngine()
            elif component_name == "ai_orchestrator":
                self.components[component_name] = AIOrchestrator()
            
            # Initialize the component
            if hasattr(self.components[component_name], 'initialize'):
                await self.components[component_name].initialize()
            
            return True
            
        except Exception as e:
            print(f"Error initializing {component_name}: {e}")
            return False
    
    def get_component(self, component_name: str):
        """Get an initialized component."""
        return self.components.get(component_name)
    
    def is_ready(self) -> bool:
        """Check if the universal platform is ready."""
        return self.initialized and len(self.components) == 8


# Create global platform instance
universal_platform = UniversalPlatform()