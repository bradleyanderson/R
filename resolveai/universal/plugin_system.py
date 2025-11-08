"""
Plugin Ecosystem Framework

Extensible architecture allowing third-party developers to create specialized
integrations for niche software and custom applications.
"""

import asyncio
import logging
import json
import importlib
import inspect
import os
import sys
from typing import Dict, Any, List, Optional, Callable, Type
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import uuid
import time
import hashlib


class PluginType(Enum):
    """Types of plugins supported by the system."""
    APPLICATION_INTEGRATION = "application_integration"
    AI_MODEL = "ai_model"
    AUTOMATION_ACTION = "automation_action"
    DATA_PROCESSOR = "data_processor"
    UI_ENHANCEMENT = "ui_enhancement"
    SECURITY_PROVIDER = "security_provider"
    CLOUD_CONNECTOR = "cloud_connector"
    ANALYTICS = "analytics"


class PluginStatus(Enum):
    """Status of plugins in the system."""
    LOADED = "loaded"
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    DISABLED = "disabled"


@dataclass
class PluginMetadata:
    """Metadata for a plugin."""
    plugin_id: str
    name: str
    version: str
    description: str
    author: str
    plugin_type: PluginType
    dependencies: List[str]
    supported_platforms: List[str]
    min_resolveai_version: str
    max_resolveai_version: Optional[str]
    permissions: List[str]
    configuration_schema: Dict[str, Any]
    created_at: float
    updated_at: float
    file_hash: str


@dataclass
class PluginInstance:
    """Instance of a loaded plugin."""
    metadata: PluginMetadata
    plugin_class: Type
    instance: Any
    status: PluginStatus
    loaded_at: float
    error_message: Optional[str] = None
    usage_count: int = 0
    last_used: float = field(default_factory=time.time)


class Plugin(ABC):
    """Abstract base class for all plugins."""
    
    @property
    @abstractmethod
    def metadata(self) -> PluginMetadata:
        """Plugin metadata."""
        pass
    
    @abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the plugin with configuration."""
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up plugin resources."""
        pass
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the plugin."""
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "version": self.metadata.version
        }
    
    async def get_capabilities(self) -> List[str]:
        """Get list of plugin capabilities."""
        return []


class ApplicationIntegrationPlugin(Plugin):
    """Base class for application integration plugins."""
    
    @abstractmethod
    async def connect_to_application(self, app_info: Dict[str, Any]) -> bool:
        """Connect to the target application."""
        pass
    
    @abstractmethod
    async def get_ui_elements(self) -> List[Dict[str, Any]]:
        """Get UI elements from the application."""
        pass
    
    @abstractmethod
    async def execute_action(self, action: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an action in the application."""
        pass


class AIModelPlugin(Plugin):
    """Base class for AI model plugins."""
    
    @abstractmethod
    async def load_model(self, model_path: str) -> bool:
        """Load the AI model."""
        pass
    
    @abstractmethod
    async def predict(self, input_data: Any) -> Any:
        """Make predictions with the model."""
        pass
    
    @abstractmethod
    async def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        pass


class PluginManager:
    """
    Plugin management system for ResolveAI.
    
    Features:
    - Dynamic plugin loading and unloading
    - Plugin dependency management
    - Version compatibility checking
    - Security and permission management
    - Plugin marketplace integration
    - Hot-reloading support
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Plugin storage
        self.loaded_plugins: Dict[str, PluginInstance] = {}
        self.plugin_configs: Dict[str, Dict[str, Any]] = {}
        
        # Plugin directories
        self.plugin_directories = [
            "./plugins",
            "~/.resolveai/plugins",
            "/usr/local/share/resolveai/plugins"
        ]
        
        # Security and permissions
        self.permission_manager = PluginPermissionManager()
        
        # Plugin registry
        self.registry = PluginRegistry()
        
        # Configuration
        self.auto_load_enabled = True
        self.security_strict_mode = True
        
        self.logger.info("Plugin Manager initialized")
    
    async def initialize(self) -> None:
        """Initialize the plugin manager."""
        try:
            self.logger.info("Initializing plugin manager...")
            
            # Create plugin directories
            await self._ensure_plugin_directories()
            
            # Load existing plugin configurations
            await self._load_plugin_configurations()
            
            # Auto-load plugins if enabled
            if self.auto_load_enabled:
                await self._auto_load_plugins()
            
            self.logger.info("Plugin manager ready")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize plugin manager: {e}")
            raise
    
    async def load_plugin(self, plugin_path: str, config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Load a plugin from file path.
        
        Args:
            plugin_path: Path to plugin file or directory
            config: Optional configuration for the plugin
            
        Returns:
            True if plugin loaded successfully
        """
        try:
            self.logger.info(f"Loading plugin from: {plugin_path}")
            
            # Validate plugin file
            if not await self._validate_plugin_file(plugin_path):
                return False
            
            # Extract plugin metadata
            metadata = await self._extract_plugin_metadata(plugin_path)
            if not metadata:
                self.logger.error("Could not extract plugin metadata")
                return False
            
            # Check compatibility
            if not await self._check_compatibility(metadata):
                return False
            
            # Check permissions
            if not await self._check_permissions(metadata):
                return False
            
            # Load plugin module
            plugin_class = await self._load_plugin_module(plugin_path)
            if not plugin_class:
                return False
            
            # Create plugin instance
            plugin_instance = plugin_class()
            
            # Initialize plugin
            config = config or {}
            if not await plugin_instance.initialize(config):
                self.logger.error("Plugin initialization failed")
                return False
            
            # Register plugin
            instance = PluginInstance(
                metadata=metadata,
                plugin_class=plugin_class,
                instance=plugin_instance,
                status=PluginStatus.ACTIVE,
                loaded_at=time.time()
            )
            
            self.loaded_plugins[metadata.plugin_id] = instance
            self.plugin_configs[metadata.plugin_id] = config
            
            # Update registry
            await self.registry.register_plugin(metadata)
            
            self.logger.info(f"Successfully loaded plugin: {metadata.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load plugin: {e}")
            return False
    
    async def unload_plugin(self, plugin_id: str) -> bool:
        """
        Unload a plugin.
        
        Args:
            plugin_id: ID of plugin to unload
            
        Returns:
            True if plugin unloaded successfully
        """
        try:
            if plugin_id not in self.loaded_plugins:
                self.logger.warning(f"Plugin not found: {plugin_id}")
                return False
            
            instance = self.loaded_plugins[plugin_id]
            
            # Cleanup plugin
            await instance.instance.cleanup()
            
            # Remove from loaded plugins
            del self.loaded_plugins[plugin_id]
            del self.plugin_configs[plugin_id]
            
            # Update registry
            await self.registry.unregister_plugin(plugin_id)
            
            self.logger.info(f"Successfully unloaded plugin: {plugin_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to unload plugin {plugin_id}: {e}")
            return False
    
    async def get_plugin(self, plugin_id: str) -> Optional[Plugin]:
        """
        Get a loaded plugin by ID.
        
        Args:
            plugin_id: ID of the plugin
            
        Returns:
            Plugin instance or None if not found
        """
        instance = self.loaded_plugins.get(plugin_id)
        if instance and instance.status == PluginStatus.ACTIVE:
            instance.usage_count += 1
            instance.last_used = time.time()
            return instance.instance
        
        return None
    
    async def get_plugins_by_type(self, plugin_type: PluginType) -> List[Plugin]:
        """
        Get all loaded plugins of a specific type.
        
        Args:
            plugin_type: Type of plugins to get
            
        Returns:
            List of plugin instances
        """
        plugins = []
        
        for instance in self.loaded_plugins.values():
            if (instance.metadata.plugin_type == plugin_type and 
                instance.status == PluginStatus.ACTIVE):
                plugins.append(instance.instance)
        
        return plugins
    
    async def install_plugin_from_marketplace(self, plugin_id: str, 
                                            version: str = "latest") -> bool:
        """
        Install a plugin from the marketplace.
        
        Args:
            plugin_id: ID of plugin in marketplace
            version: Version to install
            
        Returns:
            True if installation successful
        """
        try:
            self.logger.info(f"Installing plugin {plugin_id} from marketplace...")
            
            # Download plugin from marketplace
            plugin_path = await self.registry.download_plugin(plugin_id, version)
            if not plugin_path:
                return False
            
            # Verify plugin integrity
            if not await self._verify_plugin_integrity(plugin_path):
                self.logger.error("Plugin integrity verification failed")
                return False
            
            # Load plugin
            return await self.load_plugin(plugin_path)
            
        except Exception as e:
            self.logger.error(f"Failed to install plugin from marketplace: {e}")
            return False
    
    async def create_plugin_template(self, plugin_type: PluginType, 
                                   output_path: str, name: str) -> bool:
        """
        Create a plugin template for development.
        
        Args:
            plugin_type: Type of plugin to create
            output_path: Path where template should be created
            name: Name of the plugin
            
        Returns:
            True if template created successfully
        """
        try:
            self.logger.info(f"Creating {plugin_type.value} plugin template: {name}")
            
            # Generate template based on type
            template_code = await self._generate_plugin_template(plugin_type, name)
            
            # Write template files
            await self._write_plugin_template(output_path, name, template_code)
            
            self.logger.info(f"Plugin template created at: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create plugin template: {e}")
            return False
    
    async def get_plugin_status(self, plugin_id: str) -> Dict[str, Any]:
        """Get status information for a plugin."""
        instance = self.loaded_plugins.get(plugin_id)
        if not instance:
            return {"status": "not_found"}
        
        try:
            health_info = await instance.instance.health_check()
            
            return {
                "plugin_id": plugin_id,
                "status": instance.status.value,
                "loaded_at": instance.loaded_at,
                "usage_count": instance.usage_count,
                "last_used": instance.last_used,
                "health": health_info,
                "metadata": instance.metadata.__dict__
            }
            
        except Exception as e:
            return {
                "plugin_id": plugin_id,
                "status": "error",
                "error": str(e)
            }
    
    async def list_plugins(self, include_inactive: bool = False) -> List[Dict[str, Any]]:
        """List all plugins."""
        plugins = []
        
        for plugin_id, instance in self.loaded_plugins.items():
            if include_inactive or instance.status == PluginStatus.ACTIVE:
                plugin_info = {
                    "plugin_id": plugin_id,
                    "name": instance.metadata.name,
                    "version": instance.metadata.version,
                    "type": instance.metadata.plugin_type.value,
                    "status": instance.status.value,
                    "author": instance.metadata.author
                }
                plugins.append(plugin_info)
        
        return plugins
    
    async def _ensure_plugin_directories(self) -> None:
        """Ensure plugin directories exist."""
        for directory in self.plugin_directories:
            # Expand ~ to home directory
            expanded_dir = os.path.expanduser(directory)
            os.makedirs(expanded_dir, exist_ok=True)
    
    async def _load_plugin_configurations(self) -> None:
        """Load existing plugin configurations."""
        config_file = "./config/plugins.json"
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    self.plugin_configs = json.load(f)
            except Exception as e:
                self.logger.warning(f"Could not load plugin configurations: {e}")
    
    async def _auto_load_plugins(self) -> None:
        """Auto-load plugins from directories."""
        for directory in self.plugin_directories:
            expanded_dir = os.path.expanduser(directory)
            if os.path.exists(expanded_dir):
                for item in os.listdir(expanded_dir):
                    item_path = os.path.join(expanded_dir, item)
                    if os.path.isdir(item_path) or item.endswith('.py'):
                        config = self.plugin_configs.get(item, {})
                        await self.load_plugin(item_path, config)
    
    async def _validate_plugin_file(self, plugin_path: str) -> bool:
        """Validate plugin file structure."""
        if os.path.isdir(plugin_path):
            # Check for main.py or __init__.py
            main_file = os.path.join(plugin_path, "main.py")
            init_file = os.path.join(plugin_path, "__init__.py")
            
            if not (os.path.exists(main_file) or os.path.exists(init_file)):
                self.logger.error("Plugin directory must contain main.py or __init__.py")
                return False
        
        elif plugin_path.endswith('.py'):
            # Single file plugin
            if not os.path.exists(plugin_path):
                self.logger.error("Plugin file not found")
                return False
        
        else:
            self.logger.error("Invalid plugin format")
            return False
        
        return True
    
    async def _extract_plugin_metadata(self, plugin_path: str) -> Optional[PluginMetadata]:
        """Extract metadata from plugin."""
        try:
            # Load plugin module temporarily to get metadata
            spec = importlib.util.spec_from_file_location("temp_plugin", plugin_path)
            if not spec or not spec.loader:
                return None
            
            temp_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(temp_module)
            
            # Find plugin class
            plugin_class = None
            for name, obj in inspect.getmembers(temp_module):
                if (inspect.isclass(obj) and 
                    issubclass(obj, Plugin) and 
                    obj != Plugin):
                    plugin_class = obj
                    break
            
            if not plugin_class:
                return None
            
            # Get metadata from plugin instance
            temp_instance = plugin_class()
            metadata = temp_instance.metadata
            
            # Calculate file hash
            file_hash = await self._calculate_file_hash(plugin_path)
            metadata.file_hash = file_hash
            
            return metadata
            
        except Exception as e:
            self.logger.error(f"Failed to extract plugin metadata: {e}")
            return None
    
    async def _load_plugin_module(self, plugin_path: str) -> Optional[Type]:
        """Load plugin module and return plugin class."""
        try:
            # Add plugin directory to Python path if needed
            if os.path.isdir(plugin_path):
                if plugin_path not in sys.path:
                    sys.path.insert(0, plugin_path)
                module_name = "main"
            else:
                module_name = os.path.basename(plugin_path)[:-3]
            
            # Import the module
            spec = importlib.util.spec_from_file_location(module_name, plugin_path)
            if not spec or not spec.loader:
                return None
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Find plugin class
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    issubclass(obj, Plugin) and 
                    obj != Plugin):
                    return obj
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to load plugin module: {e}")
            return None
    
    async def _check_compatibility(self, metadata: PluginMetadata) -> bool:
        """Check plugin compatibility with current ResolveAI version."""
        # Simple version check (would be more sophisticated in production)
        current_version = "1.0.0"
        
        # Parse versions
        from packaging import version as pkg_version
        
        try:
            current = pkg_version.parse(current_version)
            min_required = pkg_version.parse(metadata.min_resolveai_version)
            
            if current < min_required:
                self.logger.error(f"Plugin requires ResolveAI {metadata.min_resolveai_version} or higher")
                return False
            
            if metadata.max_resolveai_version:
                max_supported = pkg_version.parse(metadata.max_resolveai_version)
                if current > max_supported:
                    self.logger.error(f"Plugin only supports up to ResolveAI {metadata.max_resolveai_version}")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Version compatibility check failed: {e}")
            return False
    
    async def _check_permissions(self, metadata: PluginMetadata) -> bool:
        """Check if plugin has required permissions."""
        return self.permission_manager.check_permissions(metadata.permissions)
    
    async def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of plugin file."""
        hash_sha256 = hashlib.sha256()
        
        if os.path.isdir(file_path):
            # Hash all files in directory
            for root, dirs, files in os.walk(file_path):
                for file in files:
                    file_path_full = os.path.join(root, file)
                    with open(file_path_full, "rb") as f:
                        for chunk in iter(lambda: f.read(4096), b""):
                            hash_sha256.update(chunk)
        else:
            # Hash single file
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
        
        return hash_sha256.hexdigest()
    
    async def _verify_plugin_integrity(self, plugin_path: str) -> bool:
        """Verify plugin integrity using digital signature."""
        # Implementation would verify digital signatures
        # For now, just check if file exists
        return os.path.exists(plugin_path)
    
    async def _generate_plugin_template(self, plugin_type: PluginType, name: str) -> Dict[str, str]:
        """Generate plugin template code."""
        templates = {
            PluginType.APPLICATION_INTEGRATION: self._generate_app_integration_template(name),
            PluginType.AI_MODEL: self._generate_ai_model_template(name),
            PluginType.AUTOMATION_ACTION: self._generate_automation_action_template(name),
        }
        
        return templates.get(plugin_type, self._generate_basic_template(name))
    
    def _generate_app_integration_template(self, name: str) -> Dict[str, str]:
        """Generate application integration plugin template."""
        class_name = "".join(word.capitalize() for word in name.split("_"))
        
        return {
            "__init__.py": f"""
&quot;&quot;&quot;
{name} Application Integration Plugin
&quot;&quot;&quot;

from .main import {class_name}Plugin
""",
            
            "main.py": f"""
&quot;&quot;&quot;
{name} Application Integration Plugin
&quot;&quot;&quot;

import asyncio
from typing import Dict, Any, List
from resolveai.universal.plugin_system import (
    ApplicationIntegrationPlugin, PluginMetadata, PluginType
)


class {class_name}Plugin(ApplicationIntegrationPlugin):
    &quot;&quot;&quot;Plugin for integrating with {name} application.&quot;&quot;&quot;
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            plugin_id="{name}_integration",
            name="{name.title()} Integration",
            version="1.0.0",
            description="Integration plugin for {name} application",
            author="Your Name",
            plugin_type=PluginType.APPLICATION_INTEGRATION,
            dependencies=[],
            supported_platforms=["windows", "macos", "linux"],
            min_resolveai_version="1.0.0",
            max_resolveai_version=None,
            permissions=["screen_capture", "keyboard_input"],
            configuration_schema={{}},
            created_at=0.0,
            updated_at=0.0,
            file_hash=""
        )
    
    async def initialize(self, config: Dict[str, Any]) -> bool:
        &quot;&quot;&quot;Initialize the plugin.&quot;&quot;&quot;
        # TODO: Implement initialization logic
        return True
    
    async def cleanup(self) -> None:
        &quot;&quot;&quot;Clean up plugin resources.&quot;&quot;&quot;
        # TODO: Implement cleanup logic
        pass
    
    async def connect_to_application(self, app_info: Dict[str, Any]) -> bool:
        &quot;&quot;&quot;Connect to the target application.&quot;&quot;&quot;
        # TODO: Implement connection logic
        return True
    
    async def get_ui_elements(self) -> List[Dict[str, Any]]:
        &quot;&quot;&quot;Get UI elements from the application.&quot;&quot;&quot;
        # TODO: Implement UI element detection
        return []
    
    async def execute_action(self, action: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        &quot;&quot;&quot;Execute an action in the application.&quot;&quot;&quot;
        # TODO: Implement action execution
        return {{"success": False, "error": "Not implemented"}}
"""
        }
    
    def _generate_ai_model_template(self, name: str) -> Dict[str, str]:
        """Generate AI model plugin template."""
        class_name = "".join(word.capitalize() for word in name.split("_"))
        
        return {
            "__init__.py": f"""
&quot;&quot;&quot;
{name} AI Model Plugin
&quot;&quot;&quot;

from .main import {class_name}Plugin
""",
            
            "main.py": f"""
&quot;&quot;&quot;
{name} AI Model Plugin
&quot;&quot;&quot;

import asyncio
from typing import Dict, Any
from resolveai.universal.plugin_system import (
    AIModelPlugin, PluginMetadata, PluginType
)


class {class_name}Plugin(AIModelPlugin):
    &quot;&quot;&quot;Plugin for {name} AI model.&quot;&quot;&quot;
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            plugin_id="{name}_model",
            name="{name.title()} Model",
            version="1.0.0",
            description="AI model plugin for {name}",
            author="Your Name",
            plugin_type=PluginType.AI_MODEL,
            dependencies=[],
            supported_platforms=["windows", "macos", "linux"],
            min_resolveai_version="1.0.0",
            max_resolveai_version=None,
            permissions=["model_inference"],
            configuration_schema={{}},
            created_at=0.0,
            updated_at=0.0,
            file_hash=""
        )
    
    async def initialize(self, config: Dict[str, Any]) -> bool:
        &quot;&quot;&quot;Initialize the plugin.&quot;&quot;&quot;
        # TODO: Implement initialization logic
        return True
    
    async def cleanup(self) -> None:
        &quot;&quot;&quot;Clean up plugin resources.&quot;&quot;&quot;
        # TODO: Implement cleanup logic
        pass
    
    async def load_model(self, model_path: str) -> bool:
        &quot;&quot;&quot;Load the AI model.&quot;&quot;&quot;
        # TODO: Implement model loading
        return True
    
    async def predict(self, input_data: Any) -> Any:
        &quot;&quot;&quot;Make predictions with the model.&quot;&quot;&quot;
        # TODO: Implement prediction logic
        return None
    
    async def get_model_info(self) -> Dict[str, Any]:
        &quot;&quot;&quot;Get information about the loaded model.&quot;&quot;&quot;
        # TODO: Return model information
        return {{"name": "{name}", "version": "1.0.0"}}
"""
        }
    
    def _generate_basic_template(self, name: str) -> Dict[str, str]:
        """Generate basic plugin template."""
        class_name = "".join(word.capitalize() for word in name.split("_"))
        
        return {
            "__init__.py": f"""
&quot;&quot;&quot;
{name} Plugin
&quot;&quot;&quot;

from .main import {class_name}Plugin
""",
            
            "main.py": f"""
&quot;&quot;&quot;
{name} Plugin
&quot;&quot;&quot;

import asyncio
from typing import Dict, Any, List
from resolveai.universal.plugin_system import (
    Plugin, PluginMetadata, PluginType
)


class {class_name}Plugin(Plugin):
    &quot;&quot;&quot;Basic {name} plugin.&quot;&quot;&quot;
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            plugin_id="{name}",
            name="{name.title()}",
            version="1.0.0",
            description="A {name} plugin for ResolveAI",
            author="Your Name",
            plugin_type=PluginType.DATA_PROCESSOR,
            dependencies=[],
            supported_platforms=["windows", "macos", "linux"],
            min_resolveai_version="1.0.0",
            max_resolveai_version=None,
            permissions=[],
            configuration_schema={{}},
            created_at=0.0,
            updated_at=0.0,
            file_hash=""
        )
    
    async def initialize(self, config: Dict[str, Any]) -> bool:
        &quot;&quot;&quot;Initialize the plugin.&quot;&quot;&quot;
        # TODO: Implement initialization logic
        return True
    
    async def cleanup(self) -> None:
        &quot;&quot;&quot;Clean up plugin resources.&quot;&quot;&quot;
        # TODO: Implement cleanup logic
        pass
    
    async def get_capabilities(self) -> List[str]:
        &quot;&quot;&quot;Get list of plugin capabilities.&quot;&quot;&quot;
        return ["capability_1", "capability_2"]
"""
        }
    
    async def _write_plugin_template(self, output_path: str, name: str, 
                                   template_code: Dict[str, str]) -> None:
        """Write plugin template files."""
        os.makedirs(output_path, exist_ok=True)
        
        for filename, code in template_code.items():
            file_path = os.path.join(output_path, filename)
            with open(file_path, 'w') as f:
                f.write(code)


class PluginPermissionManager:
    """Manages plugin permissions and security."""
    
    def __init__(self):
        self.allowed_permissions = [
            "screen_capture", "keyboard_input", "mouse_input",
            "file_access", "network_access", "model_inference",
            "system_commands", "clipboard_access"
        ]
    
    def check_permissions(self, required_permissions: List[str]) -> bool:
        """Check if required permissions are allowed."""
        for permission in required_permissions:
            if permission not in self.allowed_permissions:
                return False
        return True


class PluginRegistry:
    """Plugin registry for marketplace integration."""
    
    def __init__(self):
        self.registered_plugins: Dict[str, PluginMetadata] = {}
    
    async def register_plugin(self, metadata: PluginMetadata) -> None:
        """Register a plugin."""
        self.registered_plugins[metadata.plugin_id] = metadata
    
    async def unregister_plugin(self, plugin_id: str) -> None:
        """Unregister a plugin."""
        if plugin_id in self.registered_plugins:
            del self.registered_plugins[plugin_id]
    
    async def download_plugin(self, plugin_id: str, version: str) -> Optional[str]:
        """Download plugin from marketplace."""
        # Implementation would download from actual marketplace
        return f"/tmp/{plugin_id}_{version}.zip"