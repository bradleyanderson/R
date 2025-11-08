"""
Universal ResolveAI Assistant

Main orchestrator that brings together all universal components into a cohesive
AI assistant capable of understanding and interacting with any software platform.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
import json
import time
import uuid

from ..config.settings import Settings
from ..security.encryption import EncryptionManager
from ..universal import (
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


@dataclass
class UniversalAssistantConfig:
    """Configuration for the universal assistant."""
    enable_screen_intelligence: bool = True
    enable_automation: bool = True
    enable_conversational_interface: bool = True
    enable_learning: bool = True
    enable_plugins: bool = True
    enable_multi_modal: bool = True
    enable_collaboration: bool = True
    enable_ai_orchestration: bool = True
    
    # AI provider configuration
    ai_providers: Dict[str, Dict[str, Any]] = None
    
    # Security and privacy
    encryption_enabled: bool = True
    local_processing_only: bool = False
    data_retention_days: int = 30
    
    # Performance settings
    max_concurrent_requests: int = 10
    cache_enabled: bool = True
    performance_monitoring: bool = True
    
    def __post_init__(self):
        if self.ai_providers is None:
            self.ai_providers = {
                "openai": {
                    "api_key": "your-openai-key",
                    "models": ["gpt-4", "gpt-3.5-turbo"]
                },
                "anthropic": {
                    "api_key": "your-anthropic-key", 
                    "models": ["claude-3-opus", "claude-3-sonnet"]
                }
            }


class UniversalAssistant:
    """
    Universal AI assistant that can understand and interact with any software.
    
    This is the main orchestrator that combines all universal components into
    a single, cohesive AI platform capable of:
    
    - Understanding any interface through screen intelligence
    - Automating workflows across multiple applications
    - Responding to natural language commands
    - Learning and adapting to user preferences
    - Extending through plugins
    - Processing all types of media content
    - Enabling real-time collaboration
    - Orchestrating multiple AI providers
    """
    
    def __init__(self, config: Optional[UniversalAssistantConfig] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or UniversalAssistantConfig()
        self.settings = Settings()
        self.encryption_manager = EncryptionManager() if self.config.encryption_enabled else None
        
        # Component instances (will be initialized)
        self.screen_intelligence: Optional[UniversalScreenIntelligence] = None
        self.automation_engine: Optional[CrossApplicationAutomationEngine] = None
        self.conversational_interface: Optional[ConversationalInterface] = None
        self.learning_engine: Optional[AdaptiveLearningEngine] = None
        self.plugin_manager: Optional[PluginManager] = None
        self.multi_modal_processor: Optional[MultiModalProcessor] = None
        self.collaboration_engine: Optional[RealTimeCollaborationEngine] = None
        self.ai_orchestrator: Optional[AIOrchestrator] = None
        
        # State management
        self._is_running = False
        self._initialized = False
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        
        # Performance metrics
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "average_response_time": 0.0,
            "active_users": 0,
            "uptime": time.time()
        }
        
        self.logger.info("Universal Assistant initialized")
    
    async def start(self) -> bool:
        """
        Start the universal assistant and initialize all components.
        
        Returns:
            True if started successfully
        """
        if self._is_running:
            self.logger.warning("Universal Assistant is already running")
            return True
        
        try:
            self.logger.info("Starting Universal Assistant...")
            
            # Initialize components in dependency order
            await self._initialize_components()
            
            # Start background services
            await self._start_background_services()
            
            # Register signal handlers
            self._register_signal_handlers()
            
            self._is_running = True
            self._initialized = True
            
            self.logger.info("Universal Assistant started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start Universal Assistant: {e}")
            return False
    
    async def stop(self) -> None:
        """Stop the universal assistant and clean up resources."""
        if not self._is_running:
            return
        
        try:
            self.logger.info("Stopping Universal Assistant...")
            
            self._is_running = False
            
            # Stop components in reverse order
            await self._stop_components()
            
            # Clean up resources
            await self._cleanup_resources()
            
            self.logger.info("Universal Assistant stopped successfully")
            
        except Exception as e:
            self.logger.error(f"Error stopping Universal Assistant: {e}")
    
    async def process_user_request(self, user_id: str, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a user request through the universal assistant.
        
        Args:
            user_id: ID of the user making the request
            request: User request containing type and data
            
        Returns:
            Response from the assistant
        """
        if not self._is_running:
            return {"success": False, "error": "Assistant not running"}
        
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        try:
            # Update metrics
            self.metrics["total_requests"] += 1
            
            # Get user session
            session = await self._get_or_create_session(user_id)
            
            # Route request based on type
            request_type = request.get("type", "unknown")
            
            if request_type == "conversational":
                result = await self._handle_conversational_request(user_id, request, session)
            elif request_type == "automation":
                result = await self._handle_automation_request(user_id, request, session)
            elif request_type == "analysis":
                result = await self._handle_analysis_request(user_id, request, session)
            elif request_type == "collaboration":
                result = await self._handle_collaboration_request(user_id, request, session)
            elif request_type == "media_processing":
                result = await self._handle_media_request(user_id, request, session)
            else:
                result = {"success": False, "error": f"Unknown request type: {request_type}"}
            
            # Update success metrics
            if result.get("success", False):
                self.metrics["successful_requests"] += 1
            
            # Update response time metrics
            response_time = (time.time() - start_time) * 1000
            self._update_response_time_metrics(response_time)
            
            # Log interaction for learning
            if self.learning_engine:
                await self.learning_engine.record_interaction(
                    user_id, request, session.get("context", {})
                )
            
            return {
                "request_id": request_id,
                "success": result.get("success", False),
                "data": result.get("data"),
                "error": result.get("error"),
                "response_time_ms": response_time
            }
            
        except Exception as e:
            self.logger.error(f"Error processing user request: {e}")
            return {
                "request_id": request_id,
                "success": False,
                "error": str(e),
                "response_time_ms": (time.time() - start_time) * 1000
            }
    
    async def get_capabilities(self) -> Dict[str, Any]:
        """
        Get capabilities of the universal assistant.
        
        Returns:
            Dictionary describing all available capabilities
        """
        capabilities = {
            "universal_capabilities": {
                "any_software_control": True,
                "natural_language_interface": True,
                "cross_application_automation": True,
                "real_time_collaboration": True,
                "adaptive_learning": True,
                "multi_modal_processing": True,
                "extensible_plugins": True,
                "multi_ai_orchestration": True
            },
            "supported_platforms": [
                "Windows", "macOS", "Linux", "Web", "Mobile"
            ],
            "supported_applications": "Any software with visual interface",
            "ai_providers": list(self.config.ai_providers.keys()) if self.ai_orchestrator else [],
            "components": {}
        }
        
        # Add component-specific capabilities
        if self.screen_intelligence:
            capabilities["components"]["screen_intelligence"] = {
                "ui_detection": True,
                "text_extraction": True,
                "layout_analysis": True,
                "application_identification": True
            }
        
        if self.automation_engine:
            capabilities["components"]["automation"] = {
                "workflow_automation": True,
                "cross_app_integration": True,
                "conditional_logic": True,
                "error_handling": True
            }
        
        if self.conversational_interface:
            capabilities["components"]["conversational"] = {
                "natural_language_processing": True,
                "context_understanding": True,
                "multi_turn_conversation": True,
                "voice_input_support": True
            }
        
        if self.multi_modal_processor:
            capabilities["components"]["multi_modal"] = {
                "text_processing": True,
                "image_analysis": True,
                "audio_transcription": True,
                "video_analysis": True
            }
        
        return capabilities
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        status = {
            "assistant": {
                "is_running": self._is_running,
                "initialized": self._initialized,
                "uptime_seconds": time.time() - self.metrics["uptime"],
                "version": "1.0.0"
            },
            "components": {},
            "metrics": self.metrics.copy(),
            "active_sessions": len(self.active_sessions)
        }
        
        # Get component status
        if self.screen_intelligence:
            status["components"]["screen_intelligence"] = {"status": "active"}
        
        if self.automation_engine:
            status["components"]["automation_engine"] = {"status": "active"}
        
        if self.conversational_interface:
            status["components"]["conversational_interface"] = {"status": "active"}
        
        if self.learning_engine:
            status["components"]["learning_engine"] = self.learning_engine.get_learning_statistics()
        
        if self.plugin_manager:
            status["components"]["plugin_manager"] = {
                "loaded_plugins": len(await self.plugin_manager.list_plugins())
            }
        
        if self.ai_orchestrator:
            status["components"]["ai_orchestrator"] = await self.ai_orchestrator.get_provider_status()
        
        return status
    
    async def _initialize_components(self) -> None:
        """Initialize all universal components."""
        self.logger.info("Initializing universal components...")
        
        # Initialize universal platform
        platform_config = {
            "screen_intelligence": {},
            "multi_modal_processor": {},
            "ai_orchestrator": self.config.ai_providers,
            "learning_engine": {},
            "automation_engine": {},
            "conversational_interface": {},
            "plugin_system": {},
            "collaboration_engine": {}
        }
        
        success = await universal_platform.initialize_all(platform_config)
        if not success:
            raise RuntimeError("Failed to initialize universal platform")
        
        # Get component instances
        self.screen_intelligence = universal_platform.get_component("screen_intelligence")
        self.multi_modal_processor = universal_platform.get_component("multi_modal_processor")
        self.ai_orchestrator = universal_platform.get_component("ai_orchestrator")
        self.learning_engine = universal_platform.get_component("learning_engine")
        self.automation_engine = universal_platform.get_component("automation_engine")
        self.conversational_interface = universal_platform.get_component("conversational_interface")
        self.plugin_manager = universal_platform.get_component("plugin_system")
        self.collaboration_engine = universal_platform.get_component("collaboration_engine")
        
        self.logger.info("All universal components initialized")
    
    async def _start_background_services(self) -> None:
        """Start background services."""
        # Start performance monitoring
        if self.config.performance_monitoring:
            asyncio.create_task(self._performance_monitor())
        
        # Start session cleanup
        asyncio.create_task(self._session_cleanup_worker())
        
        self.logger.info("Background services started")
    
    async def _stop_components(self) -> None:
        """Stop all components."""
        if self.collaboration_engine:
            # Clean up all active sessions
            for session_id in list(self.active_sessions.keys()):
                await self.collaboration_engine.leave_workspace(session_id)
        
        # Component-specific cleanup would be handled by universal_platform
        self.logger.info("Components stopped")
    
    async def _cleanup_resources(self) -> None:
        """Clean up system resources."""
        self.active_sessions.clear()
        self.logger.info("Resources cleaned up")
    
    async def _handle_conversational_request(self, user_id: str, request: Dict[str, Any], 
                                           session: Dict[str, Any]) -> Dict[str, Any]:
        """Handle conversational requests."""
        if not self.conversational_interface:
            return {"success": False, "error": "Conversational interface not available"}
        
        conversation_id = session.get("conversation_id")
        if not conversation_id:
            conversation_id = await self.conversational_interface.start_conversation(user_id)
            session["conversation_id"] = conversation_id
        
        user_input = request.get("input", "")
        result = await self.conversational_interface.process_command(
            conversation_id, user_input
        )
        
        return {"success": True, "data": result}
    
    async def _handle_automation_request(self, user_id: str, request: Dict[str, Any], 
                                        session: Dict[str, Any]) -> Dict[str, Any]:
        """Handle automation requests."""
        if not self.automation_engine:
            return {"success": False, "error": "Automation engine not available"}
        
        automation_type = request.get("automation_type", "workflow")
        
        if automation_type == "workflow":
            workflow_def = request.get("workflow", {})
            workflow = await self.automation_engine.create_workflow(
                workflow_def.get("name", "Unnamed workflow"),
                workflow_def.get("description", ""),
                workflow_def.get("steps", [])
            )
            
            execution_id = await self.automation_engine.execute_workflow(
                workflow.workflow_id, request.get("variables", {})
            )
            
            return {"success": True, "data": {"workflow_id": workflow.workflow_id, "execution_id": execution_id}}
        
        return {"success": False, "error": "Unknown automation type"}
    
    async def _handle_analysis_request(self, user_id: str, request: Dict[str, Any], 
                                     session: Dict[str, Any]) -> Dict[str, Any]:
        """Handle analysis requests."""
        if not self.screen_intelligence:
            return {"success": False, "error": "Screen intelligence not available"}
        
        screenshot_path = request.get("screenshot_path")
        if not screenshot_path:
            return {"success": False, "error": "Screenshot path required"}
        
        analysis = await self.screen_intelligence.analyze_screen(
            screenshot_path, {"user_id": user_id}
        )
        
        return {"success": True, "data": analysis.__dict__}
    
    async def _handle_collaboration_request(self, user_id: str, request: Dict[str, Any], 
                                          session: Dict[str, Any]) -> Dict[str, Any]:
        """Handle collaboration requests."""
        if not self.collaboration_engine:
            return {"success": False, "error": "Collaboration engine not available"}
        
        action = request.get("action", "join")
        
        if action == "create_workspace":
            workspace_id = await self.collaboration_engine.create_workspace(
                user_id,
                request.get("name", "New Workspace"),
                request.get("description", "")
            )
            return {"success": True, "data": {"workspace_id": workspace_id}}
        
        elif action == "join_workspace":
            workspace_id = request.get("workspace_id")
            success = await self.collaboration_engine.join_workspace(
                workspace_id, user_id, 
                request.get("username", f"user_{user_id[:8]}"),
                request.get("email", f"{user_id}@example.com")
            )
            return {"success": success, "data": {"workspace_id": workspace_id}}
        
        return {"success": False, "error": "Unknown collaboration action"}
    
    async def _handle_media_request(self, user_id: str, request: Dict[str, Any], 
                                  session: Dict[str, Any]) -> Dict[str, Any]:
        """Handle media processing requests."""
        if not self.multi_modal_processor:
            return {"success": False, "error": "Multi-modal processor not available"}
        
        file_path = request.get("file_path")
        if not file_path:
            return {"success": False, "error": "File path required"}
        
        processing_tasks = request.get("tasks", [])
        
        content = await self.multi_modal_processor.process_media(file_path, processing_tasks)
        
        return {"success": True, "data": content.__dict__}
    
    async def _get_or_create_session(self, user_id: str) -> Dict[str, Any]:
        """Get or create user session."""
        if user_id not in self.active_sessions:
            self.active_sessions[user_id] = {
                "user_id": user_id,
                "created_at": time.time(),
                "last_activity": time.time(),
                "context": {},
                "conversation_id": None
            }
        
        session = self.active_sessions[user_id]
        session["last_activity"] = time.time()
        return session
    
    def _update_response_time_metrics(self, response_time: float) -> None:
        """Update response time metrics."""
        current_avg = self.metrics["average_response_time"]
        self.metrics["average_response_time"] = (current_avg + response_time) / 2
    
    def _register_signal_handlers(self) -> None:
        """Register signal handlers for graceful shutdown."""
        import signal
        
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, shutting down...")
            asyncio.create_task(self.stop())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def _performance_monitor(self) -> None:
        """Background performance monitoring."""
        while self._is_running:
            try:
                # Update active users count
                current_time = time.time()
                active_cutoff = current_time - 300  # 5 minutes
                
                active_users = sum(
                    1 for session in self.active_sessions.values()
                    if session["last_activity"] > active_cutoff
                )
                
                self.metrics["active_users"] = active_users
                
                # Log metrics periodically
                if int(current_time) % 60 == 0:  # Every minute
                    self.logger.info(f"Metrics: {self.metrics}")
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Error in performance monitor: {e}")
                await asyncio.sleep(30)
    
    async def _session_cleanup_worker(self) -> None:
        """Background session cleanup."""
        while self._is_running:
            try:
                current_time = time.time()
                cutoff_time = current_time - 3600  # 1 hour
                
                inactive_sessions = [
                    user_id for user_id, session in self.active_sessions.items()
                    if session["last_activity"] < cutoff_time
                ]
                
                for user_id in inactive_sessions:
                    del self.active_sessions[user_id]
                    self.logger.info(f"Cleaned up inactive session: {user_id}")
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Error in session cleanup: {e}")
                await asyncio.sleep(60)