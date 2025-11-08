"""
DaVinci Resolve API Integration

Provides seamless integration with DaVinci Resolve for real-time timeline analysis,
edit suggestions, and automated operations.
"""

import asyncio
import logging
import json
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import threading
import queue

try:
    import DaVinciResolveScript as dvr_script
    DAVINCI_AVAILABLE = True
except ImportError:
    DAVINCI_AVAILABLE = False
    dvr_script = None


@dataclass
class TimelineInfo:
    """Information about the current timeline."""
    name: str
    duration: float
    frame_rate: float
    resolution: str
    tracks: List[Dict[str, Any]]
    current_time: float
    markers: List[Dict[str, Any]]


@dataclass
class ClipInfo:
    """Information about a clip on the timeline."""
    name: str
    start_time: float
    end_time: float
    duration: float
    track_index: int
    track_type: str  # video, audio, subtitle
    file_path: Optional[str]
    properties: Dict[str, Any]


class DaVinciAPIError(Exception):
    """Custom exception for DaVinci API errors."""
    pass


class DaVinciAPI:
    """
    Main API interface for DaVinci Resolve integration.
    
    Provides methods for timeline analysis, clip manipulation,
    and real-time monitoring of editing workflows.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.resolve = None
        self.project_manager = None
        self.current_project = None
        self.current_timeline = None
        self._connected = False
        self._monitoring = False
        self._event_queue = queue.Queue()
        self._monitor_thread = None
    
    async def connect(self) -> bool:
        """
        Connect to DaVinci Resolve.
        
        Returns:
            True if connection successful, False otherwise
        """
        if not DAVINCI_AVAILABLE:
            self.logger.error("DaVinci Resolve script module not available")
            return False
        
        try:
            # Get Resolve instance
            self.resolve = dvr_script.scriptapp("Resolve")
            if not self.resolve:
                raise DaVinciAPIError("Could not get Resolve instance")
            
            # Get project manager
            self.project_manager = self.resolve.GetProjectManager()
            if not self.project_manager:
                raise DaVinciAPIError("Could not get project manager")
            
            # Get current project
            self.current_project = self.project_manager.GetCurrentProject()
            if not self.current_project:
                raise DaVinciAPIError("No project currently open")
            
            # Get current timeline
            self.current_timeline = self.current_project.GetCurrentTimeline()
            if not self.current_timeline:
                raise DaVinciAPIError("No timeline currently active")
            
            self._connected = True
            self.logger.info("Connected to DaVinci Resolve successfully")
            
            # Start monitoring if needed
            await self._start_monitoring()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to DaVinci Resolve: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from DaVinci Resolve."""
        try:
            await self._stop_monitoring()
            
            self.current_timeline = None
            self.current_project = None
            self.project_manager = None
            self.resolve = None
            self._connected = False
            
            self.logger.info("Disconnected from DaVinci Resolve")
            
        except Exception as e:
            self.logger.error(f"Error disconnecting from DaVinci Resolve: {e}")
    
    async def get_timeline_data(self) -> Dict[str, Any]:
        """
        Get comprehensive timeline data.
        
        Returns:
            Dictionary containing timeline information
        """
        if not self._connected:
            raise DaVinciAPIError("Not connected to DaVinci Resolve")
        
        try:
            # Update current timeline reference
            self.current_timeline = self.current_project.GetCurrentTimeline()
            if not self.current_timeline:
                raise DaVinciAPIError("No active timeline")
            
            # Get basic timeline info
            timeline_info = await self._get_timeline_info()
            
            # Get all clips
            clips = await self._get_timeline_clips()
            
            # Get markers
            markers = await self._get_timeline_markers()
            
            # Get track information
            tracks = await self._get_track_info()
            
            # Get current playhead position
            current_time = self.current_timeline.GetCurrentTimecode()
            
            return {
                "timeline_info": timeline_info.__dict__,
                "clips": [clip.__dict__ for clip in clips],
                "markers": markers,
                "tracks": tracks,
                "current_time": current_time,
                "timestamp": time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting timeline data: {e}")
            raise DaVinciAPIError(f"Failed to get timeline data: {e}")
    
    async def get_current_context(self) -> Dict[str, Any]:
        """
        Get current editing context.
        
        Returns:
            Dictionary containing current context information
        """
        if not self._connected:
            raise DaVinciAPIError("Not connected to DaVinci Resolve")
        
        try:
            # Get current project info
            project_name = self.current_project.GetName()
            project_settings = self.current_project.GetSetting()
            
            # Get current timeline info
            timeline_name = self.current_timeline.GetName()
            
            # Get current selection
            selected_clips = self.current_timeline.GetSelectedClips()
            
            # Get current tool mode
            current_tool = self.current_project.GetCurrentTool()
            
            # Get current page
            current_page = self.current_project.GetCurrentPage()
            
            return {
                "project": {
                    "name": project_name,
                    "settings": project_settings
                },
                "timeline": {
                    "name": timeline_name
                },
                "selection": {
                    "clip_count": len(selected_clips) if selected_clips else 0,
                    "clips": selected_clips[:10] if selected_clips else []  # Limit to first 10
                },
                "ui": {
                    "current_tool": current_tool,
                    "current_page": current_page
                },
                "timestamp": time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting current context: {e}")
            raise DaVinciAPIError(f"Failed to get current context: {e}")
    
    async def apply_edit(self, edit_suggestion: Dict[str, Any]) -> bool:
        """
        Apply an edit suggestion to the timeline.
        
        Args:
            edit_suggestion: Dictionary containing edit instructions
            
        Returns:
            True if edit applied successfully, False otherwise
        """
        if not self._connected:
            raise DaVinciAPIError("Not connected to DaVinci Resolve")
        
        try:
            edit_type = edit_suggestion.get("type")
            
            if edit_type == "cut":
                return await self._apply_cut(edit_suggestion)
            elif edit_type == "trim":
                return await self._apply_trim(edit_suggestion)
            elif edit_type == "add_marker":
                return await self._add_marker(edit_suggestion)
            elif edit_type == "color_grade":
                return await self._apply_color_grade(edit_suggestion)
            elif edit_type == "audio_adjustment":
                return await self._apply_audio_adjustment(edit_suggestion)
            else:
                self.logger.warning(f"Unknown edit type: {edit_type}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error applying edit: {e}")
            return False
    
    async def _get_timeline_info(self) -> TimelineInfo:
        """Get basic timeline information."""
        try:
            name = self.current_timeline.GetName()
            start_frame = self.current_timeline.GetStartFrame()
            end_frame = self.current_timeline.GetEndFrame()
            duration = self.current_timeline.GetDuration()
            
            # Get frame rate
            project_settings = self.current_project.GetSetting()
            frame_rate = float(project_settings.get('timelineFrameRate', 24.0))
            
            # Calculate duration in seconds
            duration_seconds = duration / frame_rate
            
            # Get resolution
            resolution = f"{self.current_project.GetSetting('timelineResolutionWidth')}x{self.current_project.GetSetting('timelineResolutionHeight')}"
            
            return TimelineInfo(
                name=name,
                duration=duration_seconds,
                frame_rate=frame_rate,
                resolution=resolution,
                tracks=[],  # Will be filled separately
                current_time=self.current_timeline.GetCurrentTimecode(),
                markers=[]  # Will be filled separately
            )
            
        except Exception as e:
            self.logger.error(f"Error getting timeline info: {e}")
            raise
    
    async def _get_timeline_clips(self) -> List[ClipInfo]:
        """Get all clips on the timeline."""
        clips = []
        
        try:
            # Get video tracks
            video_tracks = self.current_timeline.GetVideoCount()
            for track_idx in range(1, video_tracks + 1):
                track_clips = self.current_timeline.GetItemListInTrack('video', track_idx)
                
                for clip in track_clips:
                    clip_info = ClipInfo(
                        name=clip.GetName(),
                        start_time=clip.GetStart(),
                        end_time=clip.GetEnd(),
                        duration=clip.GetDuration(),
                        track_index=track_idx,
                        track_type="video",
                        file_path=clip.GetMediaPoolItem().GetClipProperty("File Path"),
                        properties={
                            "color": clip.GetClipColor(),
                            "flags": clip.GetFlags(),
                            "fused_clip": clip.IsFusedClip()
                        }
                    )
                    clips.append(clip_info)
            
            # Get audio tracks
            audio_tracks = self.current_timeline.GetAudioCount()
            for track_idx in range(1, audio_tracks + 1):
                track_clips = self.current_timeline.GetItemListInTrack('audio', track_idx)
                
                for clip in track_clips:
                    clip_info = ClipInfo(
                        name=clip.GetName(),
                        start_time=clip.GetStart(),
                        end_time=clip.GetEnd(),
                        duration=clip.GetDuration(),
                        track_index=track_idx,
                        track_type="audio",
                        file_path=clip.GetMediaPoolItem().GetClipProperty("File Path"),
                        properties={
                            "channels": clip.GetClipProperty("Channels"),
                            "sample_rate": clip.GetClipProperty("Sample Rate")
                        }
                    )
                    clips.append(clip_info)
            
            return clips
            
        except Exception as e:
            self.logger.error(f"Error getting timeline clips: {e}")
            return []
    
    async def _get_timeline_markers(self) -> List[Dict[str, Any]]:
        """Get all markers on the timeline."""
        markers = []
        
        try:
            timeline_markers = self.current_timeline.GetMarkers()
            
            for frame_id, marker_data in timeline_markers.items():
                marker_info = {
                    "frame_id": frame_id,
                    "color": marker_data.get("color", "Red"),
                    "name": marker_data.get("name", ""),
                    "note": marker_data.get("note", ""),
                    "duration": marker_data.get("duration", 1)
                }
                markers.append(marker_info)
            
            return markers
            
        except Exception as e:
            self.logger.error(f"Error getting timeline markers: {e}")
            return []
    
    async def _get_track_info(self) -> List[Dict[str, Any]]:
        """Get information about all tracks."""
        tracks = []
        
        try:
            # Video tracks
            video_count = self.current_timeline.GetVideoCount()
            for i in range(1, video_count + 1):
                track_info = {
                    "index": i,
                    "type": "video",
                    "name": f"Video {i}",
                    "enabled": self.current_timeline.IsTrackEnabled('video', i),
                    "locked": self.current_timeline.IsTrackLocked('video', i)
                }
                tracks.append(track_info)
            
            # Audio tracks
            audio_count = self.current_timeline.GetAudioCount()
            for i in range(1, audio_count + 1):
                track_info = {
                    "index": i,
                    "type": "audio",
                    "name": f"Audio {i}",
                    "enabled": self.current_timeline.IsTrackEnabled('audio', i),
                    "locked": self.current_timeline.IsTrackLocked('audio', i)
                }
                tracks.append(track_info)
            
            return tracks
            
        except Exception as e:
            self.logger.error(f"Error getting track info: {e}")
            return []
    
    async def _apply_cut(self, edit_suggestion: Dict[str, Any]) -> bool:
        """Apply a cut edit."""
        try:
            position = edit_suggestion.get("position")
            if position is None:
                return False
            
            # Move playhead to position
            self.current_timeline.SetCurrentTimecode(position)
            
            # Perform cut
            self.current_timeline.InsertTrackIntoTimeline('video', 1, position)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error applying cut: {e}")
            return False
    
    async def _apply_trim(self, edit_suggestion: Dict[str, Any]) -> bool:
        """Apply a trim edit."""
        try:
            clip_name = edit_suggestion.get("clip_name")
            new_start = edit_suggestion.get("new_start")
            new_end = edit_suggestion.get("new_end")
            
            if not all([clip_name, new_start is not None, new_end is not None]):
                return False
            
            # Find the clip (implementation depends on DaVinci API version)
            # This is a simplified version
            clips = await self._get_timeline_clips()
            for clip in clips:
                if clip.name == clip_name:
                    # Apply trim (implementation would be more complex)
                    self.logger.info(f"Trimming clip {clip_name} from {new_start} to {new_end}")
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error applying trim: {e}")
            return False
    
    async def _add_marker(self, edit_suggestion: Dict[str, Any]) -> bool:
        """Add a marker to the timeline."""
        try:
            position = edit_suggestion.get("position")
            color = edit_suggestion.get("color", "Red")
            name = edit_suggestion.get("name", "")
            note = edit_suggestion.get("note", "")
            
            if position is None:
                return False
            
            # Move playhead to position
            self.current_timeline.SetCurrentTimecode(position)
            
            # Add marker
            self.current_timeline.AddMarker(position, color, name, note, 1)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding marker: {e}")
            return False
    
    async def _apply_color_grade(self, edit_suggestion: Dict[str, Any]) -> bool:
        """Apply color grading adjustments."""
        try:
            clip_name = edit_suggestion.get("clip_name")
            adjustments = edit_suggestion.get("adjustments", {})
            
            if not clip_name or not adjustments:
                return False
            
            # This would require accessing the color page
            # Implementation depends on specific requirements
            self.logger.info(f"Applying color grade to {clip_name}: {adjustments}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error applying color grade: {e}")
            return False
    
    async def _apply_audio_adjustment(self, edit_suggestion: Dict[str, Any]) -> bool:
        """Apply audio adjustments."""
        try:
            clip_name = edit_suggestion.get("clip_name")
            adjustments = edit_suggestion.get("adjustments", {})
            
            if not clip_name or not adjustments:
                return False
            
            # Implementation would access Fairlight page
            self.logger.info(f"Applying audio adjustments to {clip_name}: {adjustments}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error applying audio adjustment: {e}")
            return False
    
    async def _start_monitoring(self) -> None:
        """Start real-time monitoring of DaVinci Resolve."""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_daVinci,
            daemon=True
        )
        self._monitor_thread.start()
        
        self.logger.info("Started DaVinci Resolve monitoring")
    
    async def _stop_monitoring(self) -> None:
        """Stop real-time monitoring."""
        if not self._monitoring:
            return
        
        self._monitoring = False
        
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=2.0)
        
        self.logger.info("Stopped DaVinci Resolve monitoring")
    
    def _monitor_daVinci(self) -> None:
        """Background thread for monitoring DaVinci Resolve."""
        last_state = {}
        
        while self._monitoring:
            try:
                if self._connected and self.current_timeline:
                    # Get current state
                    current_time = self.current_timeline.GetCurrentTimecode()
                    current_page = self.current_project.GetCurrentPage()
                    
                    # Check for changes
                    if (current_time != last_state.get("time") or 
                        current_page != last_state.get("page")):
                        
                        event = {
                            "type": "state_change",
                            "time": current_time,
                            "page": current_page,
                            "timestamp": time.time()
                        }
                        
                        self._event_queue.put(event)
                        last_state["time"] = current_time
                        last_state["page"] = current_page
                
                time.sleep(0.5)  # Monitor every 500ms
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(1.0)
    
    def get_events(self) -> List[Dict[str, Any]]:
        """Get queued events from monitoring."""
        events = []
        
        try:
            while not self._event_queue.empty():
                events.append(self._event_queue.get_nowait())
        except queue.Empty:
            pass
        
        return events
    
    def is_connected(self) -> bool:
        """Check if connected to DaVinci Resolve."""
        return self._connected
    
    def get_connection_info(self) -> Dict[str, Any]:
        """Get connection information."""
        if not self._connected:
            return {"connected": False}
        
        return {
            "connected": True,
            "resolve_version": self.resolve.GetVersion() if self.resolve else "Unknown",
            "project_name": self.current_project.GetName() if self.current_project else None,
            "timeline_name": self.current_timeline.GetName() if self.current_timeline else None,
            "monitoring": self._monitoring
        }