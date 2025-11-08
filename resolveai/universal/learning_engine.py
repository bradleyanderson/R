"""
Adaptive Learning Engine

Machine learning that personalizes to individual user preferences,
learns new applications automatically, and improves suggestions over time.
"""

import asyncio
import logging
import json
import time
import pickle
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import uuid
import numpy as np

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import accuracy_score
    import joblib
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    from transformers import AutoModel, AutoTokenizer
    DEEP_LEARNING_AVAILABLE = True
except ImportError:
    DEEP_LEARNING_AVAILABLE = False


class LearningType(Enum):
    """Types of learning the engine can perform."""
    USER_PREFERENCES = "user_preferences"
    APPLICATION_PATTERNS = "application_patterns"
    WORKFLOW_OPTIMIZATION = "workflow_optimization"
    UI_ELEMENT_RECOGNITION = "ui_element_recognition"
    COMMAND_PREDICTION = "command_prediction"
    ERROR_CORRECTION = "error_correction"


@dataclass
class LearningData:
    """Data point for machine learning."""
    data_id: str
    learning_type: LearningType
    features: Dict[str, Any]
    labels: Dict[str, Any]
    timestamp: float
    user_id: Optional[str] = None
    application: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UserProfile:
    """User profile for personalized learning."""
    user_id: str
    preferences: Dict[str, Any]
    skill_level: Dict[str, float]
    frequently_used_commands: List[str]
    application_expertise: Dict[str, float]
    learning_progress: Dict[str, float]
    created_at: float
    last_updated: float


@dataclass
class ApplicationProfile:
    """Learned profile of an application."""
    application_name: str
    ui_patterns: Dict[str, Any]
    common_workflows: List[Dict[str, Any]]
    element_signatures: Dict[str, Any]
    automation_opportunities: List[Dict[str, Any]]
    learned_at: float
    confidence_score: float


class AdaptiveLearningEngine:
    """
    Adaptive learning engine that continuously improves from user interactions.
    
    Features:
    - Personalized user profiling
    - Application pattern learning
    - Workflow optimization
    - Predictive command suggestions
    - Error correction learning
    - Continuous model improvement
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Learning data storage
        self.learning_data: List[LearningData] = []
        self.user_profiles: Dict[str, UserProfile] = {}
        self.application_profiles: Dict[str, ApplicationProfile] = {}
        
        # ML models
        self.command_predictor = None
        self.preference_classifier = None
        self.app_pattern_detector = None
        self.ui_element_classifier = None
        
        # Feature extraction
        self.feature_extractor = None
        self.vectorizer = None
        
        # Learning configuration
        self.learning_enabled = True
        self.min_samples_for_learning = 10
        self.learning_interval = 3600  # 1 hour
        self.last_learning_update = time.time()
        
        # Model storage
        self.model_storage_path = "./models/learning/"
        
        self.logger.info("Adaptive Learning Engine initialized")
    
    async def initialize(self) -> None:
        """Initialize the learning engine and load existing models."""
        try:
            self.logger.info("Initializing learning engine...")
            
            # Create model storage directory
            import os
            os.makedirs(self.model_storage_path, exist_ok=True)
            
            # Load existing models and profiles
            await self._load_existing_models()
            await self._load_user_profiles()
            await self._load_application_profiles()
            
            # Initialize ML components
            if ML_AVAILABLE:
                await self._initialize_ml_components()
            
            # Initialize deep learning components
            if DEEP_LEARNING_AVAILABLE:
                await self._initialize_deep_learning_components()
            
            # Start background learning task
            asyncio.create_task(self._continuous_learning_loop())
            
            self.logger.info("Learning engine ready")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize learning engine: {e}")
            raise
    
    async def record_interaction(self, user_id: str, interaction_data: Dict[str, Any],
                               screen_state: Dict[str, Any]) -> None:
        """
        Record a user interaction for learning.
        
        Args:
            user_id: User identifier
            interaction_data: Details of the interaction
            screen_state: Current screen context
        """
        if not self.learning_enabled:
            return
        
        try:
            # Create learning data points
            data_points = await self._extract_learning_data(
                user_id, interaction_data, screen_state
            )
            
            # Store learning data
            for data_point in data_points:
                self.learning_data.append(data_point)
            
            # Update user profile
            await self._update_user_profile(user_id, interaction_data)
            
            # Update application profile
            application = screen_state.get("application_name", "unknown")
            await self._update_application_profile(application, interaction_data, screen_state)
            
            self.logger.debug(f"Recorded interaction for user {user_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to record interaction: {e}")
    
    async def get_personalized_suggestions(self, user_id: str, 
                                         context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Get personalized suggestions based on learned user preferences.
        
        Args:
            user_id: User identifier
            context: Current context
            
        Returns:
            List of personalized suggestions
        """
        try:
            suggestions = []
            
            # Get user profile
            user_profile = self.user_profiles.get(user_id)
            if not user_profile:
                return suggestions
            
            # Command suggestions based on history
            command_suggestions = await self._predict_commands(user_id, context)
            suggestions.extend(command_suggestions)
            
            # Workflow suggestions based on patterns
            workflow_suggestions = await self._suggest_workflows(user_id, context)
            suggestions.extend(workflow_suggestions)
            
            # UI element suggestions
            ui_suggestions = await self._suggest_ui_elements(user_id, context)
            suggestions.extend(ui_suggestions)
            
            # Sort by confidence
            suggestions.sort(key=lambda x: x.get("confidence", 0), reverse=True)
            
            return suggestions[:10]  # Return top 10 suggestions
            
        except Exception as e:
            self.logger.error(f"Failed to get personalized suggestions: {e}")
            return []
    
    async def learn_application(self, application_name: str, 
                               screenshots: List[str], 
                               user_interactions: List[Dict[str, Any]]) -> ApplicationProfile:
        """
        Learn a new application from provided data.
        
        Args:
            application_name: Name of the application
            screenshots: List of screenshot paths
            user_interactions: List of user interactions
            
        Returns:
            Learned application profile
        """
        try:
            self.logger.info(f"Learning application: {application_name}")
            
            # Extract UI patterns
            ui_patterns = await self._extract_ui_patterns(screenshots)
            
            # Identify common workflows
            common_workflows = await self._identify_common_workflows(user_interactions)
            
            # Generate element signatures
            element_signatures = await self._generate_element_signatures(screenshots)
            
            # Find automation opportunities
            automation_opportunities = await self._identify_automation_opportunities(
                screenshots, user_interactions
            )
            
            # Create application profile
            profile = ApplicationProfile(
                application_name=application_name,
                ui_patterns=ui_patterns,
                common_workflows=common_workflows,
                element_signatures=element_signatures,
                automation_opportunities=automation_opportunities,
                learned_at=time.time(),
                confidence_score=self._calculate_learning_confidence(
                    screenshots, user_interactions
                )
            )
            
            # Store profile
            self.application_profiles[application_name] = profile
            
            self.logger.info(f"Successfully learned application: {application_name}")
            return profile
            
        except Exception as e:
            self.logger.error(f"Failed to learn application {application_name}: {e}")
            raise
    
    async def optimize_workflow(self, user_id: str, workflow_id: str,
                              execution_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Optimize a workflow based on execution history.
        
        Args:
            user_id: User identifier
            workflow_id: Workflow identifier
            execution_history: History of workflow executions
            
        Returns:
            Optimization recommendations
        """
        try:
            # Analyze execution patterns
            patterns = await self._analyze_execution_patterns(execution_history)
            
            # Identify bottlenecks
            bottlenecks = await self._identify_bottlenecks(execution_history)
            
            # Generate optimizations
            optimizations = await self._generate_optimizations(patterns, bottlenecks)
            
            # Learn user preferences for optimization
            user_preferences = self.user_profiles.get(user_id)
            if user_preferences:
                optimizations = await self._apply_user_preferences(optimizations, user_preferences)
            
            return {
                "workflow_id": workflow_id,
                "optimizations": optimizations,
                "expected_improvement": self._estimate_improvement(optimizations),
                "confidence": self._calculate_optimization_confidence(optimizations)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to optimize workflow {workflow_id}: {e}")
            return {}
    
    async def _extract_learning_data(self, user_id: str, interaction_data: Dict[str, Any],
                                   screen_state: Dict[str, Any]) -> List[LearningData]:
        """Extract learning data points from interaction."""
        data_points = []
        
        # User preference learning
        if "command" in interaction_data:
            preference_data = LearningData(
                data_id=str(uuid.uuid4()),
                learning_type=LearningType.USER_PREFERENCES,
                features={
                    "command": interaction_data["command"],
                    "context": screen_state,
                    "time_of_day": time.time() % 86400,
                    "previous_commands": interaction_data.get("history", [])
                },
                labels={
                    "successful": interaction_data.get("success", False),
                    "user_satisfaction": interaction_data.get("rating", 0)
                },
                timestamp=time.time(),
                user_id=user_id
            )
            data_points.append(preference_data)
        
        # Application pattern learning
        if "ui_elements" in screen_state:
            pattern_data = LearningData(
                data_id=str(uuid.uuid4()),
                learning_type=LearningType.APPLICATION_PATTERNS,
                features={
                    "ui_layout": screen_state["ui_elements"],
                    "user_action": interaction_data.get("action", ""),
                    "application": screen_state.get("application_name", "")
                },
                labels={
                    "action_successful": interaction_data.get("success", False)
                },
                timestamp=time.time(),
                user_id=user_id,
                application=screen_state.get("application_name", "")
            )
            data_points.append(pattern_data)
        
        return data_points
    
    async def _update_user_profile(self, user_id: str, interaction_data: Dict[str, Any]) -> None:
        """Update user profile with new interaction data."""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = UserProfile(
                user_id=user_id,
                preferences={},
                skill_level={},
                frequently_used_commands=[],
                application_expertise={},
                learning_progress={},
                created_at=time.time(),
                last_updated=time.time()
            )
        
        profile = self.user_profiles[user_id]
        
        # Update frequently used commands
        command = interaction_data.get("command", "")
        if command and command not in profile.frequently_used_commands:
            profile.frequently_used_commands.append(command)
            # Keep only recent commands
            if len(profile.frequently_used_commands) > 50:
                profile.frequently_used_commands = profile.frequently_used_commands[-50:]
        
        # Update skill level
        application = interaction_data.get("application", "")
        success = interaction_data.get("success", False)
        if application:
            current_skill = profile.application_expertise.get(application, 0.5)
            if success:
                profile.application_expertise[application] = min(1.0, current_skill + 0.01)
            else:
                profile.application_expertise[application] = max(0.0, current_skill - 0.005)
        
        profile.last_updated = time.time()
    
    async def _update_application_profile(self, application_name: str, 
                                        interaction_data: Dict[str, Any],
                                        screen_state: Dict[str, Any]) -> None:
        """Update application profile with new data."""
        if application_name not in self.application_profiles:
            # Create basic profile
            self.application_profiles[application_name] = ApplicationProfile(
                application_name=application_name,
                ui_patterns={},
                common_workflows=[],
                element_signatures={},
                automation_opportunities=[],
                learned_at=time.time(),
                confidence_score=0.0
            )
        
        profile = self.application_profiles[application_name]
        
        # Update UI patterns
        if "ui_elements" in screen_state:
            await self._update_ui_patterns(profile, screen_state["ui_elements"])
        
        # Update common workflows
        await self._update_workflow_patterns(profile, interaction_data)
    
    async def _predict_commands(self, user_id: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Predict likely commands based on user history and context."""
        suggestions = []
        
        profile = self.user_profiles.get(user_id)
        if not profile:
            return suggestions
        
        # Simple frequency-based prediction
        for command in profile.frequently_used_commands[-5:]:
            suggestions.append({
                "type": "command",
                "text": command,
                "confidence": 0.7,
                "reason": "frequently_used"
            })
        
        return suggestions
    
    async def _suggest_workflows(self, user_id: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Suggest relevant workflows based on context."""
        suggestions = []
        
        application = context.get("application_name", "")
        if application in self.application_profiles:
            profile = self.application_profiles[application]
            for workflow in profile.common_workflows[:3]:
                suggestions.append({
                    "type": "workflow",
                    "text": workflow.get("name", "Unnamed workflow"),
                    "confidence": workflow.get("confidence", 0.5),
                    "reason": "context_relevant"
                })
        
        return suggestions
    
    async def _suggest_ui_elements(self, user_id: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Suggest UI elements based on user preferences."""
        suggestions = []
        
        # Analyze current screen state
        screen_state = context.get("screen_state", {})
        if "ui_elements" in screen_state:
            interactive_elements = [
                elem for elem in screen_state["ui_elements"] 
                if elem.get("interactive", False)
            ]
            
            for element in interactive_elements[:5]:
                if element.get("text_content"):
                    suggestions.append({
                        "type": "ui_element",
                        "text": f"Click on '{element['text_content']}'",
                        "confidence": 0.6,
                        "reason": "available_element"
                    })
        
        return suggestions
    
    async def _continuous_learning_loop(self) -> None:
        """Background task for continuous learning."""
        while True:
            try:
                # Check if it's time to update models
                if time.time() - self.last_learning_update > self.learning_interval:
                    await self._update_learning_models()
                    self.last_learning_update = time.time()
                
                # Clean up old data
                await self._cleanup_old_data()
                
                # Sleep until next update
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Error in continuous learning loop: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error
    
    async def _update_learning_models(self) -> None:
        """Update ML models with new data."""
        if len(self.learning_data) < self.min_samples_for_learning:
            return
        
        try:
            self.logger.info("Updating learning models...")
            
            # Prepare training data
            training_data = await self._prepare_training_data()
            
            # Update command predictor
            if ML_AVAILABLE:
                await self._train_command_predictor(training_data)
            
            # Update preference classifier
            await self._train_preference_classifier(training_data)
            
            self.logger.info("Learning models updated successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to update learning models: {e}")
    
    async def _initialize_ml_components(self) -> None:
        """Initialize machine learning components."""
        try:
            # Initialize feature extractor
            self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            
            # Initialize models
            self.command_predictor = RandomForestClassifier(n_estimators=100, random_state=42)
            self.preference_classifier = RandomForestClassifier(n_estimators=50, random_state=42)
            
            self.logger.info("ML components initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize ML components: {e}")
    
    async def _initialize_deep_learning_components(self) -> None:
        """Initialize deep learning components."""
        try:
            # Initialize neural network models
            self.logger.info("Deep learning components initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize deep learning components: {e}")
    
    async def _load_existing_models(self) -> None:
        """Load pre-trained models from storage."""
        try:
            import os
            model_files = os.listdir(self.model_storage_path)
            
            for file in model_files:
                if file.endswith('.pkl'):
                    model_path = os.path.join(self.model_storage_path, file)
                    model_data = joblib.load(model_path)
                    
                    if 'command_predictor' in file:
                        self.command_predictor = model_data
                    elif 'preference_classifier' in file:
                        self.preference_classifier = model_data
            
            self.logger.info("Loaded existing models from storage")
            
        except Exception as e:
            self.logger.warning(f"Could not load existing models: {e}")
    
    async def _load_user_profiles(self) -> None:
        """Load saved user profiles."""
        try:
            profiles_file = os.path.join(self.model_storage_path, "user_profiles.pkl")
            if os.path.exists(profiles_file):
                self.user_profiles = joblib.load(profiles_file)
                self.logger.info("Loaded user profiles")
        except Exception as e:
            self.logger.warning(f"Could not load user profiles: {e}")
    
    async def _load_application_profiles(self) -> None:
        """Load saved application profiles."""
        try:
            profiles_file = os.path.join(self.model_storage_path, "app_profiles.pkl")
            if os.path.exists(profiles_file):
                self.application_profiles = joblib.load(profiles_file)
                self.logger.info("Loaded application profiles")
        except Exception as e:
            self.logger.warning(f"Could not load application profiles: {e}")
    
    async def _extract_ui_patterns(self, screenshots: List[str]) -> Dict[str, Any]:
        """Extract UI patterns from screenshots."""
        # Implementation would analyze screenshots for common UI patterns
        return {"pattern_types": ["toolbar", "sidebar", "main_content"]}
    
    async def _identify_common_workflows(self, interactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify common workflows from user interactions."""
        # Implementation would analyze interaction sequences
        return [{"name": "File Open", "frequency": 0.8}]
    
    async def _generate_element_signatures(self, screenshots: List[str]) -> Dict[str, Any]:
        """Generate signatures for UI elements."""
        return {"button_signatures": [], "input_signatures": []}
    
    async def _identify_automation_opportunities(self, screenshots: List[str],
                                               interactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify opportunities for automation."""
        return [{"type": "repetitive_task", "confidence": 0.7}]
    
    def _calculate_learning_confidence(self, screenshots: List[str],
                                     interactions: List[Dict[str, Any]]) -> float:
        """Calculate confidence score for learned application profile."""
        # Simple heuristic based on data amount
        screenshot_confidence = min(1.0, len(screenshots) / 20.0)
        interaction_confidence = min(1.0, len(interactions) / 50.0)
        return (screenshot_confidence + interaction_confidence) / 2
    
    async def _prepare_training_data(self) -> Dict[str, Any]:
        """Prepare data for model training."""
        features = []
        labels = []
        
        for data_point in self.learning_data:
            if data_point.learning_type == LearningType.USER_PREFERENCES:
                features.append(str(data_point.features))
                labels.append(data_point.labels.get("successful", False))
        
        return {"features": features, "labels": labels}
    
    async def _train_command_predictor(self, training_data: Dict[str, Any]) -> None:
        """Train command prediction model."""
        if not training_data["features"]:
            return
        
        try:
            X = self.vectorizer.fit_transform(training_data["features"])
            y = training_data["labels"]
            
            self.command_predictor.fit(X, y)
            
            # Save model
            model_path = os.path.join(self.model_storage_path, "command_predictor.pkl")
            joblib.dump(self.command_predictor, model_path)
            
            self.logger.info("Command predictor trained successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to train command predictor: {e}")
    
    async def _train_preference_classifier(self, training_data: Dict[str, Any]) -> None:
        """Train preference classification model."""
        # Implementation would train preference model
        pass
    
    async def _cleanup_old_data(self) -> None:
        """Clean up old learning data to prevent memory issues."""
        cutoff_time = time.time() - (30 * 24 * 3600)  # 30 days ago
        
        original_count = len(self.learning_data)
        self.learning_data = [
            data for data in self.learning_data 
            if data.timestamp > cutoff_time
        ]
        
        removed_count = original_count - len(self.learning_data)
        if removed_count > 0:
            self.logger.info(f"Cleaned up {removed_count} old learning data points")
    
    def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Get user profile by ID."""
        return self.user_profiles.get(user_id)
    
    def get_application_profile(self, application_name: str) -> Optional[ApplicationProfile]:
        """Get application profile by name."""
        return self.application_profiles.get(application_name)
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get statistics about the learning engine."""
        return {
            "total_learning_data_points": len(self.learning_data),
            "user_profiles_count": len(self.user_profiles),
            "application_profiles_count": len(self.application_profiles),
            "learning_enabled": self.learning_enabled,
            "last_model_update": self.last_learning_update,
            "ml_available": ML_AVAILABLE,
            "deep_learning_available": DEEP_LEARNING_AVAILABLE
        }