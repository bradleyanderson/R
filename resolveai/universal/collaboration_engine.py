"""
Real-Time Collaboration Framework

Shared AI workspaces where teams can leverage the same assistant across different
tools and platforms simultaneously.
"""

import asyncio
import logging
import json
import time
from typing import Dict, Any, List, Optional, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import uuid
import websockets
from websockets.server import WebSocketServerProtocol

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


class CollaborationRole(Enum):
    """Roles in collaborative workspace."""
    OWNER = "owner"
    ADMIN = "admin"
    EDITOR = "editor"
    VIEWER = "viewer"
    GUEST = "guest"


class PermissionLevel(Enum):
    """Permission levels for actions."""
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    MANAGE = "manage"
    ADMIN = "admin"


class ActivityType(Enum):
    """Types of collaborative activities."""
    USER_JOIN = "user_join"
    USER_LEAVE = "user_leave"
    SCREEN_SHARE = "screen_share"
    COMMAND_EXECUTE = "command_execute"
    WORKFLOW_RUN = "workflow_run"
    MESSAGE_SEND = "message_send"
    FILE_UPLOAD = "file_upload"
    CURSOR_MOVE = "cursor_move"
    SELECTION_CHANGE = "selection_change"


@dataclass
class User:
    """User in collaborative workspace."""
    user_id: str
    username: str
    email: str
    role: CollaborationRole
    permissions: Set[PermissionLevel]
    current_application: Optional[str] = None
    cursor_position: Optional[Dict[str, Any]] = None
    selection: Optional[Dict[str, Any]] = None
    last_activity: float = field(default_factory=time.time)
    is_online: bool = True
    websocket: Optional[WebSocketServerProtocol] = None


@dataclass
class Workspace:
    """Collaborative workspace."""
    workspace_id: str
    name: str
    description: str
    owner_id: str
    users: Dict[str, User]
    shared_state: Dict[str, Any]
    activity_log: List[Dict[str, Any]]
    created_at: float
    is_active: bool = True
    max_users: int = 50
    settings: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Activity:
    """Activity in collaborative workspace."""
    activity_id: str
    workspace_id: str
    user_id: str
    activity_type: ActivityType
    data: Dict[str, Any]
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class RealTimeCollaborationEngine:
    """
    Real-time collaboration engine for shared AI workspaces.
    
    Features:
    - Multi-user shared workspaces
    - Real-time synchronization
    - Screen sharing and co-viewing
    - Collaborative command execution
    - Activity tracking and history
    - Permission management
    - Cross-platform synchronization
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Workspace management
        self.workspaces: Dict[str, Workspace] = {}
        self.user_sessions: Dict[str, str] = {}  # user_id -> workspace_id
        
        # Real-time communication
        self.websocket_connections: Dict[str, WebSocketServerProtocol] = {}
        self.broadcast_queue: List[Dict[str, Any]] = []
        
        # State synchronization
        self.state_lock = asyncio.Lock()
        self.activity_buffer: List[Activity] = []
        
        # External services
        self.redis_client = None
        if REDIS_AVAILABLE:
            self._initialize_redis()
        
        # Configuration
        self.max_workspaces = 1000
        self.activity_retention_days = 30
        self.sync_frequency = 0.1  # 100ms
        
        self.logger.info("Real-Time Collaboration Engine initialized")
    
    async def initialize(self) -> None:
        """Initialize the collaboration engine."""
        try:
            self.logger.info("Initializing collaboration engine...")
            
            # Start background tasks
            asyncio.create_task(self._broadcast_worker())
            asyncio.create_task(self._cleanup_worker())
            asyncio.create_task(self._state_synchronization_worker())
            
            self.logger.info("Collaboration engine ready")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize collaboration engine: {e}")
            raise
    
    async def create_workspace(self, owner_id: str, name: str, 
                             description: str = "",
                             settings: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a new collaborative workspace.
        
        Args:
            owner_id: ID of the workspace owner
            name: Workspace name
            description: Workspace description
            settings: Optional workspace settings
            
        Returns:
            Workspace ID
        """
        try:
            if len(self.workspaces) >= self.max_workspaces:
                raise ValueError("Maximum number of workspaces reached")
            
            workspace_id = str(uuid.uuid4())
            
            # Create owner user
            owner_user = User(
                user_id=owner_id,
                username=f"user_{owner_id[:8]}",
                email=f"{owner_id}@example.com",
                role=CollaborationRole.OWNER,
                permissions={PermissionLevel.READ, PermissionLevel.WRITE, 
                           PermissionLevel.EXECUTE, PermissionLevel.MANAGE, 
                           PermissionLevel.ADMIN}
            )
            
            # Create workspace
            workspace = Workspace(
                workspace_id=workspace_id,
                name=name,
                description=description,
                owner_id=owner_id,
                users={owner_id: owner_user},
                shared_state={},
                activity_log=[],
                created_at=time.time(),
                settings=settings or {}
            )
            
            # Store workspace
            self.workspaces[workspace_id] = workspace
            self.user_sessions[owner_id] = workspace_id
            
            # Log activity
            await self._log_activity(workspace_id, owner_id, ActivityType.USER_JOIN, 
                                   {"workspace_created": True})
            
            self.logger.info(f"Created workspace {workspace_id} by {owner_id}")
            return workspace_id
            
        except Exception as e:
            self.logger.error(f"Failed to create workspace: {e}")
            raise
    
    async def join_workspace(self, workspace_id: str, user_id: str, 
                           username: str, email: str) -> bool:
        """
        Join a collaborative workspace.
        
        Args:
            workspace_id: ID of workspace to join
            user_id: ID of user joining
            username: Username of joining user
            email: Email of joining user
            
        Returns:
            True if successfully joined
        """
        try:
            if workspace_id not in self.workspaces:
                return False
            
            workspace = self.workspaces[workspace_id]
            
            # Check if user already in workspace
            if user_id in workspace.users:
                return True
            
            # Check capacity
            if len(workspace.users) >= workspace.max_users:
                return False
            
            # Create user with default role
            new_user = User(
                user_id=user_id,
                username=username,
                email=email,
                role=CollaborationRole.GUEST,
                permissions={PermissionLevel.READ}
            )
            
            # Add to workspace
            async with self.state_lock:
                workspace.users[user_id] = new_user
                self.user_sessions[user_id] = workspace_id
            
            # Log activity
            await self._log_activity(workspace_id, user_id, ActivityType.USER_JOIN,
                                   {"username": username})
            
            # Broadcast to other users
            await self._broadcast_to_workspace(workspace_id, {
                "type": "user_joined",
                "user_id": user_id,
                "username": username,
                "timestamp": time.time()
            }, exclude_user=user_id)
            
            self.logger.info(f"User {user_id} joined workspace {workspace_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to join workspace: {e}")
            return False
    
    async def leave_workspace(self, user_id: str) -> bool:
        """
        Leave current workspace.
        
        Args:
            user_id: ID of user leaving
            
        Returns:
            True if successfully left
        """
        try:
            if user_id not in self.user_sessions:
                return False
            
            workspace_id = self.user_sessions[user_id]
            workspace = self.workspaces.get(workspace_id)
            
            if not workspace:
                return False
            
            # Remove user from workspace
            async with self.state_lock:
                if user_id in workspace.users:
                    del workspace.users[user_id]
                del self.user_sessions[user_id]
            
            # Log activity
            await self._log_activity(workspace_id, user_id, ActivityType.USER_LEAVE,
                                   {"username": workspace.users.get(user_id, {}).get("username", "unknown")})
            
            # Broadcast to other users
            await self._broadcast_to_workspace(workspace_id, {
                "type": "user_left",
                "user_id": user_id,
                "timestamp": time.time()
            })
            
            # Clean up workspace if empty (except owner)
            if len(workspace.users) == 0:
                await self._cleanup_workspace(workspace_id)
            
            self.logger.info(f"User {user_id} left workspace {workspace_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to leave workspace: {e}")
            return False
    
    async def share_screen(self, user_id: str, screenshot_data: bytes, 
                         application_info: Dict[str, Any]) -> bool:
        """
        Share screen with workspace.
        
        Args:
            user_id: ID of user sharing screen
            screenshot_data: Screenshot image data
            application_info: Information about current application
            
        Returns:
            True if successfully shared
        """
        try:
            workspace_id = self.user_sessions.get(user_id)
            if not workspace_id:
                return False
            
            workspace = self.workspaces.get(workspace_id)
            if not workspace:
                return False
            
            # Check permissions
            user = workspace.users.get(user_id)
            if not user or PermissionLevel.READ not in user.permissions:
                return False
            
            # Update shared state
            async with self.state_lock:
                workspace.shared_state["screen_share"] = {
                    "user_id": user_id,
                    "screenshot_data": screenshot_data,
                    "application_info": application_info,
                    "timestamp": time.time()
                }
                
                # Update user's current application
                user.current_application = application_info.get("name", "unknown")
            
            # Log activity
            await self._log_activity(workspace_id, user_id, ActivityType.SCREEN_SHARE,
                                   application_info)
            
            # Broadcast to workspace
            await self._broadcast_to_workspace(workspace_id, {
                "type": "screen_share_update",
                "user_id": user_id,
                "application_info": application_info,
                "timestamp": time.time()
            }, exclude_user=user_id)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to share screen: {e}")
            return False
    
    async def execute_collaborative_command(self, user_id: str, 
                                          command: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a command in collaborative workspace.
        
        Args:
            user_id: ID of user executing command
            command: Command to execute
            
        Returns:
            Command execution result
        """
        try:
            workspace_id = self.user_sessions.get(user_id)
            if not workspace_id:
                return {"success": False, "error": "Not in workspace"}
            
            workspace = self.workspaces.get(workspace_id)
            if not workspace:
                return {"success": False, "error": "Workspace not found"}
            
            # Check permissions
            user = workspace.users.get(user_id)
            if not user or PermissionLevel.EXECUTE not in user.permissions:
                return {"success": False, "error": "Insufficient permissions"}
            
            # Execute command (would integrate with automation engine)
            result = {
                "success": True,
                "command_id": str(uuid.uuid4()),
                "executed_by": user_id,
                "timestamp": time.time()
            }
            
            # Log activity
            await self._log_activity(workspace_id, user_id, ActivityType.COMMAND_EXECUTE,
                                   {"command": command, "result": result})
            
            # Broadcast to workspace
            await self._broadcast_to_workspace(workspace_id, {
                "type": "command_executed",
                "user_id": user_id,
                "command": command,
                "result": result,
                "timestamp": time.time()
            }, exclude_user=user_id)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to execute collaborative command: {e}")
            return {"success": False, "error": str(e)}
    
    async def send_message(self, user_id: str, message: str, 
                         message_type: str = "text") -> bool:
        """
        Send message to workspace.
        
        Args:
            user_id: ID of user sending message
            message: Message content
            message_type: Type of message (text, file, etc.)
            
        Returns:
            True if message sent successfully
        """
        try:
            workspace_id = self.user_sessions.get(user_id)
            if not workspace_id:
                return False
            
            workspace = self.workspaces.get(workspace_id)
            if not workspace:
                return False
            
            # Check permissions
            user = workspace.users.get(user_id)
            if not user or PermissionLevel.WRITE not in user.permissions:
                return False
            
            # Create message
            message_data = {
                "message_id": str(uuid.uuid4()),
                "user_id": user_id,
                "username": user.username,
                "content": message,
                "type": message_type,
                "timestamp": time.time()
            }
            
            # Log activity
            await self._log_activity(workspace_id, user_id, ActivityType.MESSAGE_SEND,
                                   {"message": message_data})
            
            # Broadcast to workspace
            await self._broadcast_to_workspace(workspace_id, {
                "type": "message",
                "data": message_data
            })
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send message: {e}")
            return False
    
    async def update_cursor_position(self, user_id: str, 
                                   x: int, y: int, application: str) -> bool:
        """
        Update user's cursor position in workspace.
        
        Args:
            user_id: ID of user
            x: X coordinate
            y: Y coordinate
            application: Current application
            
        Returns:
            True if position updated successfully
        """
        try:
            workspace_id = self.user_sessions.get(user_id)
            if not workspace_id:
                return False
            
            workspace = self.workspaces.get(workspace_id)
            if not workspace:
                return False
            
            user = workspace.users.get(user_id)
            if not user:
                return False
            
            # Update cursor position
            async with self.state_lock:
                user.cursor_position = {"x": x, "y": y, "application": application}
                user.last_activity = time.time()
            
            # Log activity
            await self._log_activity(workspace_id, user_id, ActivityType.CURSOR_MOVE,
                                   {"x": x, "y": y, "application": application})
            
            # Broadcast to workspace (throttled)
            await self._broadcast_to_workspace(workspace_id, {
                "type": "cursor_update",
                "user_id": user_id,
                "position": {"x": x, "y": y, "application": application},
                "timestamp": time.time()
            }, exclude_user=user_id)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update cursor position: {e}")
            return False
    
    async def get_workspace_state(self, workspace_id: str, 
                                user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get current workspace state for user.
        
        Args:
            workspace_id: ID of workspace
            user_id: ID of requesting user
            
        Returns:
            Workspace state or None if not found
        """
        try:
            workspace = self.workspaces.get(workspace_id)
            if not workspace:
                return None
            
            user = workspace.users.get(user_id)
            if not user:
                return None
            
            # Check permissions
            if PermissionLevel.READ not in user.permissions:
                return None
            
            # Prepare state for user
            state = {
                "workspace_id": workspace_id,
                "name": workspace.name,
                "description": workspace.description,
                "users": {
                    uid: {
                        "username": u.username,
                        "role": u.role.value,
                        "current_application": u.current_application,
                        "cursor_position": u.cursor_position,
                        "is_online": u.is_online,
                        "last_activity": u.last_activity
                    }
                    for uid, u in workspace.users.items()
                },
                "shared_state": workspace.shared_state,
                "activity_log": workspace.activity_log[-50:],  # Last 50 activities
                "user_permissions": list(user.permissions),
                "user_role": user.role.value
            }
            
            return state
            
        except Exception as e:
            self.logger.error(f"Failed to get workspace state: {e}")
            return None
    
    async def get_workspace_list(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Get list of workspaces accessible to user.
        
        Args:
            user_id: ID of user
            
        Returns:
            List of accessible workspaces
        """
        workspaces = []
        
        for workspace_id, workspace in self.workspaces.items():
            if workspace_id == self.user_sessions.get(user_id):
                # User is currently in this workspace
                user = workspace.users.get(user_id)
                if user:
                    workspaces.append({
                        "workspace_id": workspace_id,
                        "name": workspace.name,
                        "description": workspace.description,
                        "user_count": len(workspace.users),
                        "user_role": user.role.value,
                        "is_active": True
                    })
            else:
                # Check if user can join (public workspaces, etc.)
                if workspace.settings.get("public", False):
                    workspaces.append({
                        "workspace_id": workspace_id,
                        "name": workspace.name,
                        "description": workspace.description,
                        "user_count": len(workspace.users),
                        "user_role": "none",
                        "is_active": True
                    })
        
        return workspaces
    
    async def _log_activity(self, workspace_id: str, user_id: str, 
                          activity_type: ActivityType, data: Dict[str, Any]) -> None:
        """Log activity in workspace."""
        activity = Activity(
            activity_id=str(uuid.uuid4()),
            workspace_id=workspace_id,
            user_id=user_id,
            activity_type=activity_type,
            data=data,
            timestamp=time.time()
        )
        
        if workspace_id in self.workspaces:
            self.workspaces[workspace_id].activity_log.append(activity.__dict__)
        
        # Add to buffer for external storage
        self.activity_buffer.append(activity)
    
    async def _broadcast_to_workspace(self, workspace_id: str, 
                                    message: Dict[str, Any],
                                    exclude_user: Optional[str] = None) -> None:
        """Broadcast message to all users in workspace."""
        if workspace_id not in self.workspaces:
            return
        
        workspace = self.workspaces[workspace_id]
        
        for user_id, user in workspace.users.items():
            if user_id != exclude_user and user.websocket:
                try:
                    await self._send_websocket_message(user.websocket, message)
                except Exception as e:
                    self.logger.warning(f"Failed to send message to user {user_id}: {e}")
    
    async def _send_websocket_message(self, websocket: WebSocketServerProtocol, 
                                    message: Dict[str, Any]) -> None:
        """Send message via WebSocket."""
        try:
            await websocket.send(json.dumps(message))
        except Exception as e:
            self.logger.error(f"WebSocket send failed: {e}")
    
    async def _broadcast_worker(self) -> None:
        """Background worker for broadcasting messages."""
        while True:
            try:
                if self.broadcast_queue:
                    message = self.broadcast_queue.pop(0)
                    await self._broadcast_to_workspace(
                        message["workspace_id"], 
                        message["data"],
                        message.get("exclude_user")
                    )
                
                await asyncio.sleep(0.01)
                
            except Exception as e:
                self.logger.error(f"Error in broadcast worker: {e}")
                await asyncio.sleep(0.1)
    
    async def _cleanup_worker(self) -> None:
        """Background worker for cleaning up old data."""
        while True:
            try:
                current_time = time.time()
                cutoff_time = current_time - (self.activity_retention_days * 24 * 3600)
                
                # Clean up old activity logs
                for workspace in self.workspaces.values():
                    workspace.activity_log = [
                        activity for activity in workspace.activity_log
                        if activity.get("timestamp", 0) > cutoff_time
                    ]
                
                # Clean up inactive users
                inactive_users = []
                for workspace in self.workspaces.values():
                    for user_id, user in workspace.users.items():
                        if current_time - user.last_activity > 300:  # 5 minutes
                            user.is_online = False
                            if current_time - user.last_activity > 3600:  # 1 hour
                                inactive_users.append((workspace.workspace_id, user_id))
                
                # Remove inactive users
                for workspace_id, user_id in inactive_users:
                    await self.leave_workspace(user_id)
                
                await asyncio.sleep(300)  # Run every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Error in cleanup worker: {e}")
                await asyncio.sleep(60)
    
    async def _state_synchronization_worker(self) -> None:
        """Background worker for synchronizing state."""
        while True:
            try:
                # Synchronize state with external services
                if self.redis_client:
                    await self._sync_with_redis()
                
                await asyncio.sleep(self.sync_frequency)
                
            except Exception as e:
                self.logger.error(f"Error in state sync worker: {e}")
                await asyncio.sleep(1)
    
    async def _sync_with_redis(self) -> None:
        """Synchronize state with Redis."""
        try:
            # Sync workspace states to Redis
            for workspace_id, workspace in self.workspaces.items():
                state_key = f"workspace:{workspace_id}:state"
                await self.redis_client.set(
                    state_key,
                    json.dumps(workspace.shared_state),
                    ex=3600  # Expire in 1 hour
                )
        except Exception as e:
            self.logger.warning(f"Redis sync failed: {e}")
    
    async def _cleanup_workspace(self, workspace_id: str) -> None:
        """Clean up empty workspace."""
        if workspace_id in self.workspaces:
            del self.workspaces[workspace_id]
            self.logger.info(f"Cleaned up empty workspace {workspace_id}")
    
    def _initialize_redis(self) -> None:
        """Initialize Redis client."""
        try:
            self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
            self.redis_client.ping()
            self.logger.info("Redis client initialized")
        except Exception as e:
            self.logger.warning(f"Redis initialization failed: {e}")
            self.redis_client = None
    
    def get_workspace_statistics(self, workspace_id: str) -> Dict[str, Any]:
        """Get statistics for a workspace."""
        workspace = self.workspaces.get(workspace_id)
        if not workspace:
            return {}
        
        return {
            "workspace_id": workspace_id,
            "name": workspace.name,
            "user_count": len(workspace.users),
            "online_users": sum(1 for u in workspace.users.values() if u.is_online),
            "total_activities": len(workspace.activity_log),
            "created_at": workspace.created_at,
            "is_active": workspace.is_active
        }
    
    def get_engine_statistics(self) -> Dict[str, Any]:
        """Get overall engine statistics."""
        return {
            "total_workspaces": len(self.workspaces),
            "total_users": len(self.user_sessions),
            "active_workspaces": sum(1 for w in self.workspaces.values() if w.is_active),
            "total_activities": sum(len(w.activity_log) for w in self.workspaces.values()),
            "websocket_connections": len(self.websocket_connections),
            "redis_connected": self.redis_client is not None
        }