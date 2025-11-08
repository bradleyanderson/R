"""
FastAPI Server for ResolveAI

Provides REST API endpoints for the ResolveAI assistant, including
authentication, video processing, and real-time collaboration features.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import uuid
import json
import base64

from fastapi import FastAPI, HTTPException, Depends, status, WebSocket, WebSocketDisconnect
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import jwt

from ..core.assistant import ResolveAIAssistant, AssistantConfig
from ..security.encryption import EncryptionManager
from ..config.settings import Settings


# Pydantic models for API
class UserCredentials(BaseModel):
    username: str
    password: str


class AssistantConfigModel(BaseModel):
    enable_cloud_processing: bool = True
    enable_screen_capture: bool = True
    enable_audio_analysis: bool = True
    cloud_provider: str = "aws"
    processing_region: str = "us-west-2"
    local_processing_only: bool = False


class VideoProcessingRequest(BaseModel):
    file_path: str
    processing_options: Optional[Dict[str, Any]] = None


class TimelineAnalysisRequest(BaseModel):
    include_suggestions: bool = True
    analysis_depth: str = "standard"  # basic, standard, deep


class EditSuggestionRequest(BaseModel):
    suggestion_id: str
    confirm: bool = False


class ScreenAnalysisRequest(BaseModel):
    include_interface_analysis: bool = True
    include_context: bool = True


# Authentication
security = HTTPBearer()
SECRET_KEY = "your-secret-key-here"  # In production, use environment variable
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30


class AuthenticationError(Exception):
    """Custom authentication error."""
    pass


class ConnectionManager:
    """Manages WebSocket connections for real-time features."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.logger = logging.getLogger(__name__)
    
    async def connect(self, websocket: WebSocket):
        """Accept a WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
        self.logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            self.logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        """Send a message to a specific WebSocket."""
        try:
            await websocket.send_text(message)
        except Exception as e:
            self.logger.error(f"Error sending WebSocket message: {e}")
    
    async def broadcast(self, message: str):
        """Broadcast a message to all connected WebSockets."""
        disconnected = []
        
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                self.logger.error(f"Error broadcasting to WebSocket: {e}")
                disconnected.append(connection)
        
        # Remove disconnected clients
        for connection in disconnected:
            self.disconnect(connection)
    
    def get_connection_count(self) -> int:
        """Get the number of active connections."""
        return len(self.active_connections)


class ResolveAPIServer:
    """
    Main FastAPI server for ResolveAI.
    
    Provides REST API endpoints and WebSocket connections for
    real-time collaboration and monitoring.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.settings = Settings()
        self.encryption_manager = EncryptionManager()
        self.assistant: Optional[ResolveAIAssistant] = None
        self.connection_manager = ConnectionManager()
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        
        # Initialize FastAPI app
        self.app = FastAPI(
            title="ResolveAI API",
            description="AI Assistant for DaVinci Resolve Video Editing",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # In production, specify allowed origins
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Setup routes
        self._setup_routes()
        
        self.logger.info("ResolveAI API Server initialized")
    
    def _setup_routes(self):
        """Setup all API routes."""
        
        @self.app.post("/auth/login")
        async def login(credentials: UserCredentials):
            """Authenticate user and return access token."""
            try:
                # Simple authentication (in production, use proper user database)
                if credentials.username == "admin" and credentials.password == "password":
                    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
                    access_token = self._create_access_token(
                        data={"sub": credentials.username}, 
                        expires_delta=access_token_expires
                    )
                    
                    return {
                        "access_token": access_token,
                        "token_type": "bearer",
                        "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60
                    }
                else:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Invalid credentials"
                    )
            
            except Exception as e:
                self.logger.error(f"Login error: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Authentication failed"
                )
        
        @self.app.post("/assistant/start")
        async def start_assistant(
            config: AssistantConfigModel,
            credentials: HTTPAuthorizationCredentials = Depends(security)
        ):
            """Start the ResolveAI assistant."""
            try:
                # Verify token
                self._verify_token(credentials.credentials)
                
                # Create assistant config
                assistant_config = AssistantConfig(
                    enable_cloud_processing=config.enable_cloud_processing,
                    enable_screen_capture=config.enable_screen_capture,
                    enable_audio_analysis=config.enable_audio_analysis,
                    encryption_key=self.settings.security.encryption_key,
                    cloud_provider=config.cloud_provider,
                    processing_region=config.processing_region,
                    local_processing_only=config.local_processing_only
                )
                
                # Initialize and start assistant
                self.assistant = ResolveAIAssistant(assistant_config)
                await self.assistant.start()
                
                # Create session
                session_id = str(uuid.uuid4())
                self.active_sessions[session_id] = {
                    "assistant": self.assistant,
                    "config": assistant_config,
                    "started_at": datetime.now(),
                    "user": self._get_token_user(credentials.credentials)
                }
                
                return {
                    "session_id": session_id,
                    "status": "started",
                    "config": assistant_config.__dict__
                }
            
            except Exception as e:
                self.logger.error(f"Error starting assistant: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to start assistant: {str(e)}"
                )
        
        @self.app.post("/assistant/stop")
        async def stop_assistant(
            credentials: HTTPAuthorizationCredentials = Depends(security)
        ):
            """Stop the ResolveAI assistant."""
            try:
                self._verify_token(credentials.credentials)
                
                if self.assistant:
                    await self.assistant.stop()
                    self.assistant = None
                    
                    # Clear sessions
                    self.active_sessions.clear()
                
                return {"status": "stopped"}
            
            except Exception as e:
                self.logger.error(f"Error stopping assistant: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to stop assistant: {str(e)}"
                )
        
        @self.app.post("/analyze/timeline")
        async def analyze_timeline(
            request: TimelineAnalysisRequest,
            credentials: HTTPAuthorizationCredentials = Depends(security)
        ):
            """Analyze the current DaVinci Resolve timeline."""
            try:
                self._verify_token(credentials.credentials)
                
                if not self.assistant:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Assistant not started"
                    )
                
                result = await self.assistant.analyze_timeline()
                
                return {
                    "success": True,
                    "data": result,
                    "timestamp": datetime.now().isoformat()
                }
            
            except Exception as e:
                self.logger.error(f"Error analyzing timeline: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Timeline analysis failed: {str(e)}"
                )
        
        @self.app.post("/process/video")
        async def process_video(
            request: VideoProcessingRequest,
            credentials: HTTPAuthorizationCredentials = Depends(security)
        ):
            """Process a video file."""
            try:
                self._verify_token(credentials.credentials)
                
                if not self.assistant:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Assistant not started"
                    )
                
                result = await self.assistant.process_video_file(
                    request.file_path,
                    request.processing_options
                )
                
                return {
                    "success": True,
                    "data": result,
                    "timestamp": datetime.now().isoformat()
                }
            
            except Exception as e:
                self.logger.error(f"Error processing video: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Video processing failed: {str(e)}"
                )
        
        @self.app.post("/analyze/screen")
        async def analyze_screen(
            request: ScreenAnalysisRequest,
            credentials: HTTPAuthorizationCredentials = Depends(security)
        ):
            """Capture and analyze the current screen."""
            try:
                self._verify_token(credentials.credentials)
                
                if not self.assistant:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Assistant not started"
                    )
                
                result = await self.assistant.capture_and_analyze_screen()
                
                return {
                    "success": True,
                    "data": result,
                    "timestamp": datetime.now().isoformat()
                }
            
            except Exception as e:
                self.logger.error(f"Error analyzing screen: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Screen analysis failed: {str(e)}"
                )
        
        @self.app.post("/suggestions/apply")
        async def apply_suggestion(
            request: EditSuggestionRequest,
            credentials: HTTPAuthorizationCredentials = Depends(security)
        ):
            """Apply an AI-generated suggestion."""
            try:
                self._verify_token(credentials.credentials)
                
                if not self.assistant:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Assistant not started"
                    )
                
                if not request.confirm:
                    return {
                        "success": False,
                        "message": "Confirmation required to apply suggestion",
                        "requires_confirmation": True
                    }
                
                result = await self.assistant.apply_ai_suggestion(request.suggestion_id)
                
                return {
                    "success": result,
                    "suggestion_id": request.suggestion_id,
                    "timestamp": datetime.now().isoformat()
                }
            
            except Exception as e:
                self.logger.error(f"Error applying suggestion: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to apply suggestion: {str(e)}"
                )
        
        @self.app.get("/status")
        async def get_status(
            credentials: HTTPAuthorizationCredentials = Depends(security)
        ):
            """Get current assistant status."""
            try:
                self._verify_token(credentials.credentials)
                
                status = {
                    "assistant_running": self.assistant is not None,
                    "active_sessions": len(self.active_sessions),
                    "websocket_connections": self.connection_manager.get_connection_count(),
                    "cloud_configured": self.settings.is_cloud_configured(),
                    "timestamp": datetime.now().isoformat()
                }
                
                if self.assistant:
                    status["assistant_config"] = self.assistant.config.__dict__
                
                return status
            
            except Exception as e:
                self.logger.error(f"Error getting status: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to get status: {str(e)}"
                )
        
        @self.app.websocket("/ws/realtime")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time updates."""
            await self.connection_manager.connect(websocket)
            
            try:
                while True:
                    # Receive message from client
                    data = await websocket.receive_text()
                    message = json.loads(data)
                    
                    # Process message
                    if message.get("type") == "subscribe":
                        # Client wants to subscribe to updates
                        await self.connection_manager.send_personal_message(
                            json.dumps({
                                "type": "subscribed",
                                "timestamp": datetime.now().isoformat()
                            }),
                            websocket
                        )
                    elif message.get("type") == "ping":
                        # Keep-alive ping
                        await self.connection_manager.send_personal_message(
                            json.dumps({
                                "type": "pong",
                                "timestamp": datetime.now().isoformat()
                            }),
                            websocket
                        )
            
            except WebSocketDisconnect:
                self.connection_manager.disconnect(websocket)
            except Exception as e:
                self.logger.error(f"WebSocket error: {e}")
                self.connection_manager.disconnect(websocket)
    
    def _create_access_token(self, data: dict, expires_delta: timedelta):
        """Create a JWT access token."""
        to_encode = data.copy()
        expire = datetime.utcnow() + expires_delta
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt
    
    def _verify_token(self, token: str) -> str:
        """Verify JWT token and return username."""
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            username: str = payload.get("sub")
            if username is None:
                raise AuthenticationError("Invalid token")
            return username
        except jwt.PyJWTError:
            raise AuthenticationError("Invalid token")
    
    def _get_token_user(self, token: str) -> str:
        """Get username from token."""
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            return payload.get("sub", "unknown")
        except jwt.PyJWTError:
            return "unknown"
    
    async def broadcast_update(self, update_type: str, data: Dict[str, Any]):
        """Broadcast real-time updates to all connected clients."""
        message = json.dumps({
            "type": update_type,
            "data": data,
            "timestamp": datetime.now().isoformat()
        })
        
        await self.connection_manager.broadcast(message)
    
    def get_app(self) -> FastAPI:
        """Get the FastAPI application instance."""
        return self.app


# Create global server instance
server = ResolveAPIServer()
app = server.get_app()


# Event handlers
@app.on_event("startup")
async def startup_event():
    """Handle application startup."""
    logging.info("ResolveAI API Server starting up...")


@app.on_event("shutdown")
async def shutdown_event():
    """Handle application shutdown."""
    logging.info("ResolveAI API Server shutting down...")
    
    # Stop assistant if running
    if server.assistant:
        await server.assistant.stop()