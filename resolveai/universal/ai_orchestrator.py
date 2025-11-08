"""
Multi-AI Integration System

Connect and orchestrate multiple AI providers including OpenAI, Claude, local models,
and specialized AI services for comprehensive intelligence capabilities.
"""

import asyncio
import logging
import json
import time
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import uuid
import aiohttp

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    LOCAL_MODELS_AVAILABLE = True
except ImportError:
    LOCAL_MODELS_AVAILABLE = False


class AIProvider(Enum):
    """Available AI providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"
    HUGGINGFACE = "huggingface"
    AZURE_OPENAI = "azure_openai"
    GOOGLE_AI = "google_ai"
    CUSTOM = "custom"


class ModelCapability(Enum):
    """Capabilities of AI models."""
    TEXT_GENERATION = "text_generation"
    TEXT_UNDERSTANDING = "text_understanding"
    CODE_GENERATION = "code_generation"
    IMAGE_ANALYSIS = "image_analysis"
    IMAGE_GENERATION = "image_generation"
    AUDIO_TRANSCRIPTION = "audio_transcription"
    AUDIO_GENERATION = "audio_generation"
    VIDEO_ANALYSIS = "video_analysis"
    TRANSLATION = "translation"
    SUMMARIZATION = "summarization"
    CLASSIFICATION = "classification"
    EMBEDDING = "embedding"


@dataclass
class AIModel:
    """AI model configuration."""
    model_id: str
    provider: AIProvider
    name: str
    capabilities: List[ModelCapability]
    max_tokens: int
    cost_per_token: float
    latency_ms: float
    accuracy_score: float
    is_available: bool = True
    requires_api_key: bool = True
    api_endpoint: Optional[str] = None
    model_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AIRequest:
    """Request to AI provider."""
    request_id: str
    model_id: str
    task_type: ModelCapability
    input_data: Dict[str, Any]
    parameters: Dict[str, Any]
    priority: int = 1
    timeout: float = 30.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AIResponse:
    """Response from AI provider."""
    request_id: str
    model_id: str
    provider: AIProvider
    result: Any
    confidence: float
    tokens_used: int
    processing_time_ms: float
    cost: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class AIProviderInterface(ABC):
    """Abstract interface for AI providers."""
    
    @abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the AI provider."""
        pass
    
    @abstractmethod
    async def get_available_models(self) -> List[AIModel]:
        """Get list of available models."""
        pass
    
    @abstractmethod
    async def process_request(self, request: AIRequest) -> AIResponse:
        """Process an AI request."""
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Check provider health."""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> List[ModelCapability]:
        """Get provider capabilities."""
        pass


class OpenAIProvider(AIProviderInterface):
    """OpenAI AI provider."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.client = None
        self.models = []
        self.config = {}
    
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize OpenAI provider."""
        try:
            if not OPENAI_AVAILABLE:
                raise ImportError("OpenAI library not available")
            
            self.config = config
            api_key = config.get("api_key")
            
            if not api_key:
                self.logger.error("OpenAI API key not provided")
                return False
            
            self.client = openai.AsyncOpenAI(api_key=api_key)
            
            # Load available models
            await self._load_models()
            
            self.logger.info("OpenAI provider initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAI provider: {e}")
            return False
    
    async def get_available_models(self) -> List[AIModel]:
        """Get available OpenAI models."""
        return self.models
    
    async def process_request(self, request: AIRequest) -> AIResponse:
        """Process request with OpenAI."""
        start_time = time.time()
        
        try:
            model = self._get_model_by_id(request.model_id)
            if not model:
                return AIResponse(
                    request_id=request.request_id,
                    model_id=request.model_id,
                    provider=AIProvider.OPENAI,
                    result=None,
                    confidence=0.0,
                    tokens_used=0,
                    processing_time_ms=0,
                    cost=0.0,
                    error="Model not found"
                )
            
            # Process based on task type
            if request.task_type == ModelCapability.TEXT_GENERATION:
                result = await self._generate_text(request)
            elif request.task_type == ModelCapability.CODE_GENERATION:
                result = await self._generate_code(request)
            elif request.task_type == ModelCapability.IMAGE_ANALYSIS:
                result = await self._analyze_image(request)
            else:
                result = await self._generic_process(request)
            
            processing_time = (time.time() - start_time) * 1000
            
            return AIResponse(
                request_id=request.request_id,
                model_id=request.model_id,
                provider=AIProvider.OPENAI,
                result=result.get("content"),
                confidence=result.get("confidence", 0.8),
                tokens_used=result.get("tokens_used", 0),
                processing_time_ms=processing_time,
                cost=result.get("tokens_used", 0) * model.cost_per_token,
                metadata=result.get("metadata", {})
            )
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            
            return AIResponse(
                request_id=request.request_id,
                model_id=request.model_id,
                provider=AIProvider.OPENAI,
                result=None,
                confidence=0.0,
                tokens_used=0,
                processing_time_ms=processing_time,
                cost=0.0,
                error=str(e)
            )
    
    async def health_check(self) -> Dict[str, Any]:
        """Check OpenAI provider health."""
        try:
            # Simple health check - list models
            models = await self.client.models.list()
            return {
                "status": "healthy",
                "available_models": len(models.data),
                "provider": "openai"
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "provider": "openai"
            }
    
    def get_capabilities(self) -> List[ModelCapability]:
        """Get OpenAI capabilities."""
        return [
            ModelCapability.TEXT_GENERATION,
            ModelCapability.TEXT_UNDERSTANDING,
            ModelCapability.CODE_GENERATION,
            ModelCapability.IMAGE_ANALYSIS,
            ModelCapability.IMAGE_GENERATION,
            ModelCapability.AUDIO_TRANSCRIPTION,
            ModelCapability.TRANSLATION,
            ModelCapability.SUMMARIZATION,
            ModelCapability.CLASSIFICATION,
            ModelCapability.EMBEDDING
        ]
    
    async def _load_models(self) -> None:
        """Load OpenAI models."""
        self.models = [
            AIModel(
                model_id="gpt-4",
                provider=AIProvider.OPENAI,
                name="GPT-4",
                capabilities=[
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.CODE_GENERATION,
                    ModelCapability.TEXT_UNDERSTANDING
                ],
                max_tokens=8192,
                cost_per_token=0.00003,
                latency_ms=2000,
                accuracy_score=0.95
            ),
            AIModel(
                model_id="gpt-3.5-turbo",
                provider=AIProvider.OPENAI,
                name="GPT-3.5 Turbo",
                capabilities=[
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.CODE_GENERATION,
                    ModelCapability.TEXT_UNDERSTANDING
                ],
                max_tokens=4096,
                cost_per_token=0.000002,
                latency_ms=1000,
                accuracy_score=0.90
            ),
            AIModel(
                model_id="gpt-4-vision-preview",
                provider=AIProvider.OPENAI,
                name="GPT-4 Vision",
                capabilities=[
                    ModelCapability.IMAGE_ANALYSIS,
                    ModelCapability.TEXT_GENERATION
                ],
                max_tokens=4096,
                cost_per_token=0.00003,
                latency_ms=3000,
                accuracy_score=0.92
            )
        ]
    
    def _get_model_by_id(self, model_id: str) -> Optional[AIModel]:
        """Get model by ID."""
        for model in self.models:
            if model.model_id == model_id:
                return model
        return None
    
    async def _generate_text(self, request: AIRequest) -> Dict[str, Any]:
        """Generate text using OpenAI."""
        prompt = request.input_data.get("prompt", "")
        max_tokens = request.parameters.get("max_tokens", 1000)
        
        response = await self.client.chat.completions.create(
            model=request.model_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens
        )
        
        return {
            "content": response.choices[0].message.content,
            "tokens_used": response.usage.total_tokens,
            "confidence": 0.9
        }
    
    async def _generate_code(self, request: AIRequest) -> Dict[str, Any]:
        """Generate code using OpenAI."""
        prompt = request.input_data.get("prompt", "")
        language = request.input_data.get("language", "python")
        
        enhanced_prompt = f"Generate {language} code for: {prompt}"
        
        response = await self.client.chat.completions.create(
            model=request.model_id,
            messages=[{"role": "user", "content": enhanced_prompt}],
            max_tokens=request.parameters.get("max_tokens", 1000)
        )
        
        return {
            "content": response.choices[0].message.content,
            "tokens_used": response.usage.total_tokens,
            "confidence": 0.85
        }
    
    async def _analyze_image(self, request: AIRequest) -> Dict[str, Any]:
        """Analyze image using OpenAI Vision API."""
        image_url = request.input_data.get("image_url", "")
        prompt = request.input_data.get("prompt", "Describe this image")
        
        response = await self.client.chat.completions.create(
            model=request.model_id,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ]
                }
            ],
            max_tokens=request.parameters.get("max_tokens", 500)
        )
        
        return {
            "content": response.choices[0].message.content,
            "tokens_used": response.usage.total_tokens,
            "confidence": 0.88
        }
    
    async def _generic_process(self, request: AIRequest) -> Dict[str, Any]:
        """Generic processing for other tasks."""
        return {
            "content": "Generic processing not implemented",
            "tokens_used": 0,
            "confidence": 0.5
        }


class AnthropicProvider(AIProviderInterface):
    """Anthropic (Claude) AI provider."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.client = None
        self.models = []
        self.config = {}
    
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize Anthropic provider."""
        try:
            if not ANTHROPIC_AVAILABLE:
                raise ImportError("Anthropic library not available")
            
            self.config = config
            api_key = config.get("api_key")
            
            if not api_key:
                self.logger.error("Anthropic API key not provided")
                return False
            
            self.client = anthropic.AsyncAnthropic(api_key=api_key)
            
            # Load available models
            await self._load_models()
            
            self.logger.info("Anthropic provider initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Anthropic provider: {e}")
            return False
    
    async def get_available_models(self) -> List[AIModel]:
        """Get available Anthropic models."""
        return self.models
    
    async def process_request(self, request: AIRequest) -> AIResponse:
        """Process request with Anthropic."""
        start_time = time.time()
        
        try:
            model = self._get_model_by_id(request.model_id)
            if not model:
                return AIResponse(
                    request_id=request.request_id,
                    model_id=request.model_id,
                    provider=AIProvider.ANTHROPIC,
                    result=None,
                    confidence=0.0,
                    tokens_used=0,
                    processing_time_ms=0,
                    cost=0.0,
                    error="Model not found"
                )
            
            # Process request
            result = await self._generate_text(request)
            processing_time = (time.time() - start_time) * 1000
            
            return AIResponse(
                request_id=request.request_id,
                model_id=request.model_id,
                provider=AIProvider.ANTHROPIC,
                result=result.get("content"),
                confidence=result.get("confidence", 0.8),
                tokens_used=result.get("tokens_used", 0),
                processing_time_ms=processing_time,
                cost=result.get("tokens_used", 0) * model.cost_per_token,
                metadata=result.get("metadata", {})
            )
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            
            return AIResponse(
                request_id=request.request_id,
                model_id=request.model_id,
                provider=AIProvider.ANTHROPIC,
                result=None,
                confidence=0.0,
                tokens_used=0,
                processing_time_ms=processing_time,
                cost=0.0,
                error=str(e)
            )
    
    async def health_check(self) -> Dict[str, Any]:
        """Check Anthropic provider health."""
        try:
            # Simple health check
            return {
                "status": "healthy",
                "provider": "anthropic"
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "provider": "anthropic"
            }
    
    def get_capabilities(self) -> List[ModelCapability]:
        """Get Anthropic capabilities."""
        return [
            ModelCapability.TEXT_GENERATION,
            ModelCapability.TEXT_UNDERSTANDING,
            ModelCapability.CODE_GENERATION,
            ModelCapability.SUMMARIZATION,
            ModelCapability.CLASSIFICATION
        ]
    
    async def _load_models(self) -> None:
        """Load Anthropic models."""
        self.models = [
            AIModel(
                model_id="claude-3-opus-20240229",
                provider=AIProvider.ANTHROPIC,
                name="Claude 3 Opus",
                capabilities=[
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.CODE_GENERATION,
                    ModelCapability.TEXT_UNDERSTANDING
                ],
                max_tokens=4096,
                cost_per_token=0.000075,
                latency_ms=2500,
                accuracy_score=0.96
            ),
            AIModel(
                model_id="claude-3-sonnet-20240229",
                provider=AIProvider.ANTHROPIC,
                name="Claude 3 Sonnet",
                capabilities=[
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.CODE_GENERATION,
                    ModelCapability.TEXT_UNDERSTANDING
                ],
                max_tokens=4096,
                cost_per_token=0.000015,
                latency_ms=1500,
                accuracy_score=0.94
            )
        ]
    
    def _get_model_by_id(self, model_id: str) -> Optional[AIModel]:
        """Get model by ID."""
        for model in self.models:
            if model.model_id == model_id:
                return model
        return None
    
    async def _generate_text(self, request: AIRequest) -> Dict[str, Any]:
        """Generate text using Claude."""
        prompt = request.input_data.get("prompt", "")
        max_tokens = request.parameters.get("max_tokens", 1000)
        
        response = await self.client.messages.create(
            model=request.model_id,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return {
            "content": response.content[0].text,
            "tokens_used": response.usage.input_tokens + response.usage.output_tokens,
            "confidence": 0.92
        }


class LocalModelProvider(AIProviderInterface):
    """Local AI model provider."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.models = []
        self.loaded_models: Dict[str, Any] = {}
        self.config = {}
    
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize local model provider."""
        try:
            if not LOCAL_MODELS_AVAILABLE:
                raise ImportError("Transformers library not available")
            
            self.config = config
            
            # Load available models
            await self._load_models()
            
            self.logger.info("Local model provider initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize local model provider: {e}")
            return False
    
    async def get_available_models(self) -> List[AIModel]:
        """Get available local models."""
        return self.models
    
    async def process_request(self, request: AIRequest) -> AIResponse:
        """Process request with local model."""
        start_time = time.time()
        
        try:
            model = self._get_model_by_id(request.model_id)
            if not model:
                return AIResponse(
                    request_id=request.request_id,
                    model_id=request.model_id,
                    provider=AIProvider.LOCAL,
                    result=None,
                    confidence=0.0,
                    tokens_used=0,
                    processing_time_ms=0,
                    cost=0.0,
                    error="Model not found"
                )
            
            # Load model if not already loaded
            if request.model_id not in self.loaded_models:
                await self._load_model(request.model_id)
            
            # Process request
            result = await self._generate_text_local(request, self.loaded_models[request.model_id])
            processing_time = (time.time() - start_time) * 1000
            
            return AIResponse(
                request_id=request.request_id,
                model_id=request.model_id,
                provider=AIProvider.LOCAL,
                result=result.get("content"),
                confidence=result.get("confidence", 0.8),
                tokens_used=result.get("tokens_used", 0),
                processing_time_ms=processing_time,
                cost=0.0,  # Local models are free
                metadata=result.get("metadata", {})
            )
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            
            return AIResponse(
                request_id=request.request_id,
                model_id=request.model_id,
                provider=AIProvider.LOCAL,
                result=None,
                confidence=0.0,
                tokens_used=0,
                processing_time_ms=processing_time,
                cost=0.0,
                error=str(e)
            )
    
    async def health_check(self) -> Dict[str, Any]:
        """Check local model provider health."""
        try:
            return {
                "status": "healthy",
                "loaded_models": len(self.loaded_models),
                "provider": "local"
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "provider": "local"
            }
    
    def get_capabilities(self) -> List[ModelCapability]:
        """Get local model capabilities."""
        return [
            ModelCapability.TEXT_GENERATION,
            ModelCapability.TEXT_UNDERSTANDING,
            ModelCapability.CLASSIFICATION
        ]
    
    async def _load_models(self) -> None:
        """Load local model configurations."""
        self.models = [
            AIModel(
                model_id="llama-2-7b-chat",
                provider=AIProvider.LOCAL,
                name="Llama 2 7B Chat",
                capabilities=[
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.TEXT_UNDERSTANDING
                ],
                max_tokens=2048,
                cost_per_token=0.0,  # Free
                latency_ms=5000,
                accuracy_score=0.85
            ),
            AIModel(
                model_id="mistral-7b-instruct",
                provider=AIProvider.LOCAL,
                name="Mistral 7B Instruct",
                capabilities=[
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.TEXT_UNDERSTANDING
                ],
                max_tokens=2048,
                cost_per_token=0.0,  # Free
                latency_ms=3000,
                accuracy_score=0.88
            )
        ]
    
    def _get_model_by_id(self, model_id: str) -> Optional[AIModel]:
        """Get model by ID."""
        for model in self.models:
            if model.model_id == model_id:
                return model
        return None
    
    async def _load_model(self, model_id: str) -> None:
        """Load a local model."""
        try:
            # This is a simplified version - in production would handle model loading properly
            self.logger.info(f"Loading local model: {model_id}")
            self.loaded_models[model_id] = {"loaded": True}
        except Exception as e:
            self.logger.error(f"Failed to load model {model_id}: {e}")
            raise
    
    async def _generate_text_local(self, request: AIRequest, 
                                 model: Dict[str, Any]) -> Dict[str, Any]:
        """Generate text using local model."""
        # Simplified local inference
        prompt = request.input_data.get("prompt", "")
        
        # In production, would use actual model inference
        return {
            "content": f"Local model response for: {prompt[:100]}...",
            "tokens_used": len(prompt.split()),
            "confidence": 0.80
        }


class AIOrchestrator:
    """
    Multi-AI orchestration system for managing and coordinating multiple AI providers.
    
    Features:
    - Multi-provider support (OpenAI, Claude, local models)
    - Intelligent model selection
    - Load balancing and failover
    - Cost optimization
    - Performance monitoring
    - Capability matching
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Provider management
        self.providers: Dict[AIProvider, AIProviderInterface] = {}
        self.available_models: Dict[str, AIModel] = {}
        
        # Request processing
        self.request_queue: List[AIRequest] = []
        self.active_requests: Dict[str, AIRequest] = {}
        
        # Performance tracking
        self.performance_metrics: Dict[str, Any] = {}
        self.cost_tracking: Dict[str, float] = {}
        
        # Configuration
        self.default_provider = AIProvider.OPENAI
        self.max_concurrent_requests = 10
        self.auto_failover = True
        
        self.logger.info("AI Orchestrator initialized")
    
    async def initialize(self, provider_configs: Dict[str, Dict[str, Any]]) -> bool:
        """
        Initialize AI orchestrator with provider configurations.
        
        Args:
            provider_configs: Configuration for each provider
            
        Returns:
            True if initialization successful
        """
        try:
            self.logger.info("Initializing AI Orchestrator...")
            
            # Initialize providers
            for provider_name, config in provider_configs.items():
                try:
                    provider = self._create_provider(AIProvider(provider_name))
                    if await provider.initialize(config):
                        self.providers[AIProvider(provider_name)] = provider
                        
                        # Load available models
                        models = await provider.get_available_models()
                        for model in models:
                            self.available_models[model.model_id] = model
                        
                        self.logger.info(f"Initialized {provider_name} provider")
                    else:
                        self.logger.warning(f"Failed to initialize {provider_name} provider")
                        
                except Exception as e:
                    self.logger.error(f"Error initializing {provider_name}: {e}")
            
            # Start background workers
            asyncio.create_task(self._request_processor())
            asyncio.create_task(self._health_monitor())
            
            self.logger.info(f"AI Orchestrator ready with {len(self.providers)} providers")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize AI Orchestrator: {e}")
            return False
    
    async def process_request(self, task_type: ModelCapability, 
                            input_data: Dict[str, Any],
                            preferred_provider: Optional[AIProvider] = None,
                            model_id: Optional[str] = None) -> AIResponse:
        """
        Process an AI request with automatic provider selection.
        
        Args:
            task_type: Type of AI task
            input_data: Input data for the task
            preferred_provider: Preferred AI provider
            model_id: Specific model to use
            
        Returns:
            AI response
        """
        try:
            # Select best model
            selected_model = await self._select_model(task_type, preferred_provider, model_id)
            if not selected_model:
                return AIResponse(
                    request_id=str(uuid.uuid4()),
                    model_id="",
                    provider=AIProvider.OPENAI,
                    result=None,
                    confidence=0.0,
                    tokens_used=0,
                    processing_time_ms=0,
                    cost=0.0,
                    error="No suitable model available"
                )
            
            # Create request
            request = AIRequest(
                request_id=str(uuid.uuid4()),
                model_id=selected_model.model_id,
                task_type=task_type,
                input_data=input_data,
                parameters={}
            )
            
            # Process request
            response = await self._execute_request(request)
            
            # Track metrics
            await self._track_metrics(response)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Failed to process AI request: {e}")
            return AIResponse(
                request_id=str(uuid.uuid4()),
                model_id="",
                provider=AIProvider.OPENAI,
                result=None,
                confidence=0.0,
                tokens_used=0,
                processing_time_ms=0,
                cost=0.0,
                error=str(e)
            )
    
    async def get_available_models(self, capability: Optional[ModelCapability] = None) -> List[AIModel]:
        """
        Get available models, optionally filtered by capability.
        
        Args:
            capability: Optional capability filter
            
        Returns:
            List of available models
        """
        if capability:
            return [model for model in self.available_models.values() 
                   if capability in model.capabilities]
        return list(self.available_models.values())
    
    async def get_provider_status(self) -> Dict[str, Any]:
        """Get status of all providers."""
        status = {}
        
        for provider, interface in self.providers.items():
            try:
                health = await interface.health_check()
                models = await interface.get_available_models()
                
                status[provider.value] = {
                    "status": health.get("status", "unknown"),
                    "available_models": len(models),
                    "capabilities": [cap.value for cap in interface.get_capabilities()]
                }
            except Exception as e:
                status[provider.value] = {
                    "status": "error",
                    "error": str(e),
                    "available_models": 0
                }
        
        return status
    
    async def _create_provider(self, provider: AIProvider) -> AIProviderInterface:
        """Create provider instance."""
        if provider == AIProvider.OPENAI:
            return OpenAIProvider()
        elif provider == AIProvider.ANTHROPIC:
            return AnthropicProvider()
        elif provider == AIProvider.LOCAL:
            return LocalModelProvider()
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    async def _select_model(self, task_type: ModelCapability, 
                           preferred_provider: Optional[AIProvider],
                           model_id: Optional[str]) -> Optional[AIModel]:
        """Select best model for the task."""
        # If specific model requested
        if model_id and model_id in self.available_models:
            model = self.available_models[model_id]
            if task_type in model.capabilities:
                return model
        
        # Filter models by capability
        capable_models = [
            model for model in self.available_models.values()
            if task_type in model.capabilities and model.is_available
        ]
        
        if not capable_models:
            return None
        
        # Filter by preferred provider
        if preferred_provider:
            provider_models = [
                model for model in capable_models
                if model.provider == preferred_provider
            ]
            if provider_models:
                capable_models = provider_models
        
        # Sort by accuracy and cost (best balance)
        capable_models.sort(key=lambda m: (m.accuracy_score, -m.cost_per_token), reverse=True)
        
        return capable_models[0]
    
    async def _execute_request(self, request: AIRequest) -> AIResponse:
        """Execute AI request with failover."""
        model = self.available_models.get(request.model_id)
        if not model:
            raise ValueError(f"Model not found: {request.model_id}")
        
        provider = self.providers.get(model.provider)
        if not provider:
            raise ValueError(f"Provider not available: {model.provider}")
        
        try:
            return await provider.process_request(request)
        except Exception as e:
            self.logger.warning(f"Primary provider failed: {e}")
            
            # Try failover if enabled
            if self.auto_failover:
                return await self._failover_request(request, task_type=request.task_type)
            
            raise
    
    async def _failover_request(self, request: AIRequest, 
                              task_type: ModelCapability) -> AIResponse:
        """Attempt failover to alternative provider."""
        # Find alternative model
        alt_model = await self._select_model(task_type, None, None)
        if not alt_model or alt_model.model_id == request.model_id:
            raise ValueError("No failover options available")
        
        # Create new request with alternative model
        alt_request = AIRequest(
            request_id=request.request_id,
            model_id=alt_model.model_id,
            task_type=request.task_type,
            input_data=request.input_data,
            parameters=request.parameters
        )
        
        provider = self.providers.get(alt_model.provider)
        if not provider:
            raise ValueError(f"Failover provider not available: {alt_model.provider}")
        
        return await provider.process_request(alt_request)
    
    async def _track_metrics(self, response: AIResponse) -> None:
        """Track performance and cost metrics."""
        provider = response.provider.value
        model_id = response.model_id
        
        # Initialize metrics if not exists
        if provider not in self.performance_metrics:
            self.performance_metrics[provider] = {}
        
        if model_id not in self.performance_metrics[provider]:
            self.performance_metrics[provider][model_id] = {
                "total_requests": 0,
                "successful_requests": 0,
                "average_latency": 0.0,
                "total_cost": 0.0
            }
        
        # Update metrics
        metrics = self.performance_metrics[provider][model_id]
        metrics["total_requests"] += 1
        
        if response.error is None:
            metrics["successful_requests"] += 1
        
        # Update average latency
        current_avg = metrics["average_latency"]
        new_latency = response.processing_time_ms
        metrics["average_latency"] = (current_avg + new_latency) / 2
        
        # Update cost
        metrics["total_cost"] += response.cost
    
    async def _request_processor(self) -> None:
        """Background worker for processing requests."""
        while True:
            try:
                if self.request_queue:
                    request = self.request_queue.pop(0)
                    
                    if len(self.active_requests) < self.max_concurrent_requests:
                        self.active_requests[request.request_id] = request
                        
                        # Process request asynchronously
                        asyncio.create_task(self._process_and_complete(request))
                
                await asyncio.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Error in request processor: {e}")
                await asyncio.sleep(0.5)
    
    async def _process_and_complete(self, request: AIRequest) -> None:
        """Process request and clean up."""
        try:
            await self._execute_request(request)
        finally:
            if request.request_id in self.active_requests:
                del self.active_requests[request.request_id]
    
    async def _health_monitor(self) -> None:
        """Background worker for monitoring provider health."""
        while True:
            try:
                for provider, interface in self.providers.items():
                    try:
                        health = await interface.health_check()
                        if health.get("status") != "healthy":
                            self.logger.warning(f"Provider {provider.value} unhealthy: {health}")
                    except Exception as e:
                        self.logger.error(f"Health check failed for {provider.value}: {e}")
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Error in health monitor: {e}")
                await asyncio.sleep(60)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        return self.performance_metrics
    
    def get_cost_summary(self) -> Dict[str, float]:
        """Get cost summary by provider."""
        summary = {}
        
        for provider, models in self.performance_metrics.items():
            total_cost = sum(model["total_cost"] for model in models.values())
            summary[provider] = total_cost
        
        return summary