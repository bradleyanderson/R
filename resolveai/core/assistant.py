"""
ResolveAI Core Assistant Module

Main AI assistant class that coordinates all functionality for DaVinci Resolve integration.
"""

import asyncio
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import threading
import queue

from ..config.settings import Settings
from ..security.encryption import EncryptionManager
from ..cloud.cloud_manager import CloudManager
from ..video.analyzer import VideoAnalyzer
from ..screen.capture import ScreenCapture
from ..audio.transcriber import AudioTranscriber
from ..davinci.api import DaVinciAPI
from ..models.ai_models import AIModelManager


@dataclass
class AssistantConfig:
    """Configuration for the ResolveAI assistant."""
    enable_cloud_processing: bool = True
    enable_screen_capture: bool = True
    enable_audio_analysis: bool = True
    encryption_key: Optional[str] = None
    cloud_provider: str = "aws"
    processing_region: str = "us-west-2"
    auto_backup: bool = True
    local_processing_only: bool = False


class ResolveAIAssistant:
    """
    Main ResolveAI Assistant class that orchestrates all AI functionality
    for DaVinci Resolve video editing workflows.
    """
    
    def __init__(self, config: Optional[AssistantConfig] = None):
        """
        Initialize the ResolveAI Assistant.
        
        Args:
            config: Optional configuration object
        """
        self.config = config or AssistantConfig()
        self.settings = Settings()
        self.logger = logging.getLogger(__name__)
        
        # Core components
        self.encryption_manager = EncryptionManager(
            key=self.config.encryption_key
        )
        self.cloud_manager = CloudManager(
            provider=self.config.cloud_provider,
            region=self.config.processing_region,
            encryption_manager=self.encryption_manager
        )
        self.video_analyzer = VideoAnalyzer()
        self.screen_capture = ScreenCapture() if self.config.enable_screen_capture else None
        self.audio_transcriber = AudioTranscriber() if self.config.enable_audio_analysis else None
        self.davinci_api = DaVinciAPI()
        self.ai_models = AIModelManager()
        
        # State management
        self._is_running = False
        self._analysis_queue = queue.Queue()
        self._results_queue = queue.Queue()
        self._worker_threads = []
        
        self.logger.info("ResolveAI Assistant initialized")
    
    async def start(self) -> None:
        """Start the assistant and all its components."""
        if self._is_running:
            self.logger.warning("Assistant is already running")
            return
        
        try:
            # Initialize components
            await self._initialize_components()
            
            # Start worker threads
            self._start_worker_threads()
            
            # Connect to DaVinci Resolve
            await self.davinci_api.connect()
            
            self._is_running = True
            self.logger.info("ResolveAI Assistant started successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to start assistant: {e}")
            raise
    
    async def stop(self) -> None:
        """Stop the assistant and clean up resources."""
        if not self._is_running:
            return
        
        try:
            self._is_running = False
            
            # Stop worker threads
            self._stop_worker_threads()
            
            # Disconnect from DaVinci Resolve
            await self.davinci_api.disconnect()
            
            # Cleanup components
            await self._cleanup_components()
            
            self.logger.info("ResolveAI Assistant stopped successfully")
            
        except Exception as e:
            self.logger.error(f"Error stopping assistant: {e}")
    
    async def analyze_timeline(self) -> Dict[str, Any]:
        """
        Analyze the current DaVinci Resolve timeline.
        
        Returns:
            Dictionary containing analysis results
        """
        if not self._is_running:
            raise RuntimeError("Assistant is not running")
        
        try:
            # Get timeline data from DaVinci Resolve
            timeline_data = await self.davinci_api.get_timeline_data()
            
            # Analyze video content
            video_analysis = await self.video_analyzer.analyze_timeline(timeline_data)
            
            # Analyze audio content if enabled
            audio_analysis = None
            if self.audio_transcriber:
                audio_analysis = await self.audio_transcriber.analyze_timeline_audio(timeline_data)
            
            # Generate AI suggestions
            suggestions = await self.ai_models.generate_editing_suggestions(
                video_analysis, audio_analysis, timeline_data
            )
            
            result = {
                "video_analysis": video_analysis,
                "audio_analysis": audio_analysis,
                "suggestions": suggestions,
                "timeline_info": timeline_data.get("timeline_info", {}),
                "timestamp": self._get_current_timestamp()
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error analyzing timeline: {e}")
            raise
    
    async def capture_and_analyze_screen(self) -> Dict[str, Any]:
        """
        Capture current screen and analyze DaVinci Resolve interface.
        
        Returns:
            Dictionary containing screen analysis results
        """
        if not self.screen_capture:
            raise RuntimeError("Screen capture is not enabled")
        
        try:
            # Capture screen
            screenshot = await self.screen_capture.capture_davinci_window()
            
            # Analyze interface
            interface_analysis = await self.ai_models.analyze_interface(screenshot)
            
            # Get contextual information
            context = await self.davinci_api.get_current_context()
            
            result = {
                "screenshot_path": screenshot,
                "interface_analysis": interface_analysis,
                "context": context,
                "timestamp": self._get_current_timestamp()
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error capturing and analyzing screen: {e}")
            raise
    
    async def process_video_file(self, file_path: str) -> Dict[str, Any]:
        """
        Process a video file for analysis and suggestions.
        
        Args:
            file_path: Path to the video file
            
        Returns:
            Dictionary containing processing results
        """
        try:
            # Determine processing location
            if self.config.local_processing_only or not self.config.enable_cloud_processing:
                # Local processing
                result = await self.video_analyzer.analyze_file_locally(file_path)
            else:
                # Cloud processing
                result = await self.cloud_manager.process_video_file(file_path)
            
            # Generate suggestions based on analysis
            suggestions = await self.ai_models.generate_content_suggestions(result)
            
            return {
                "analysis": result,
                "suggestions": suggestions,
                "file_path": file_path,
                "processing_location": "local" if self.config.local_processing_only else "cloud",
                "timestamp": self._get_current_timestamp()
            }
            
        except Exception as e:
            self.logger.error(f"Error processing video file: {e}")
            raise
    
    async def apply_ai_suggestion(self, suggestion_id: str) -> bool:
        """
        Apply an AI-generated suggestion to the timeline.
        
        Args:
            suggestion_id: ID of the suggestion to apply
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get suggestion details
            suggestion = await self.ai_models.get_suggestion(suggestion_id)
            
            # Apply suggestion through DaVinci API
            result = await self.davinci_api.apply_edit(suggestion)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error applying suggestion {suggestion_id}: {e}")
            return False
    
    def _get_current_timestamp(self) -> str:
        """Get current timestamp for logging."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    async def _initialize_components(self) -> None:
        """Initialize all components."""
        await self.cloud_manager.initialize()
        await self.video_analyzer.initialize()
        if self.screen_capture:
            await self.screen_capture.initialize()
        if self.audio_transcriber:
            await self.audio_transcriber.initialize()
        await self.ai_models.initialize()
    
    async def _cleanup_components(self) -> None:
        """Clean up all components."""
        await self.cloud_manager.cleanup()
        await self.video_analyzer.cleanup()
        if self.screen_capture:
            await self.screen_capture.cleanup()
        if self.audio_transcriber:
            await self.audio_transcriber.cleanup()
        await self.ai_models.cleanup()
    
    def _start_worker_threads(self) -> None:
        """Start background worker threads."""
        # Video analysis worker
        video_worker = threading.Thread(
            target=self._video_analysis_worker,
            daemon=True
        )
        video_worker.start()
        self._worker_threads.append(video_worker)
        
        # Cloud processing worker
        if self.config.enable_cloud_processing:
            cloud_worker = threading.Thread(
                target=self._cloud_processing_worker,
                daemon=True
            )
            cloud_worker.start()
            self._worker_threads.append(cloud_worker)
    
    def _stop_worker_threads(self) -> None:
        """Stop all worker threads."""
        for thread in self._worker_threads:
            if thread.is_alive():
                thread.join(timeout=5.0)
        self._worker_threads.clear()
    
    def _video_analysis_worker(self) -> None:
        """Background worker for video analysis tasks."""
        while self._is_running:
            try:
                # Get task from queue
                task = self._analysis_queue.get(timeout=1.0)
                
                # Process task
                result = asyncio.run(self._process_analysis_task(task))
                
                # Put result in results queue
                self._results_queue.put(result)
                
                self._analysis_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error in video analysis worker: {e}")
    
    def _cloud_processing_worker(self) -> None:
        """Background worker for cloud processing tasks."""
        while self._is_running:
            try:
                # Get cloud task from queue
                task = self._analysis_queue.get(timeout=1.0)
                
                if task.get("type") == "cloud_processing":
                    # Process cloud task
                    result = asyncio.run(self.cloud_manager.process_task(task))
                    self._results_queue.put(result)
                
                self._analysis_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error in cloud processing worker: {e}")
    
    async def _process_analysis_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process an analysis task."""
        task_type = task.get("type")
        
        if task_type == "timeline_analysis":
            return await self.analyze_timeline()
        elif task_type == "screen_analysis":
            return await self.capture_and_analyze_screen()
        elif task_type == "video_file":
            return await self.process_video_file(task["file_path"])
        else:
            raise ValueError(f"Unknown task type: {task_type}")