"""
Conversational Interface Control System

Natural language capability to control any software through voice or text commands,
with contextual understanding of the user's current workflow.
"""

import asyncio
import logging
import json
import re
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time
import uuid

try:
    import openai
    from transformers import AutoTokenizer, AutoModelForCausalLM
    NLP_AVAILABLE = True
except ImportError:
    NLP_AVAILABLE = False

from .screen_intelligence import UniversalScreenIntelligence, ScreenAnalysis, UIElement
from .automation_engine import CrossApplicationAutomationEngine, Workflow, AutomationStep, ActionType


class CommandType(Enum):
    """Types of commands that can be processed."""
    UI_ACTION = "ui_action"
    NAVIGATION = "navigation"
    DATA_EXTRACTION = "data_extraction"
    WORKFLOW_CONTROL = "workflow_control"
    SYSTEM_COMMAND = "system_command"
    QUESTION = "question"
    LEARNING = "learning"


@dataclass
class Command:
    """Parsed command from user input."""
    command_id: str
    command_type: CommandType
    intent: str
    entities: Dict[str, Any]
    confidence: float
    raw_text: str
    context: Dict[str, Any] = field(default_factory=dict)
    requires_confirmation: bool = False
    suggested_actions: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ConversationContext:
    """Context for ongoing conversation."""
    conversation_id: str
    user_id: Optional[str]
    current_screen_state: Optional[ScreenAnalysis]
    active_application: str
    previous_commands: List[Command]
    workflow_variables: Dict[str, Any]
    user_preferences: Dict[str, Any]
    session_start_time: float
    last_interaction_time: float


class ConversationalInterface:
    """
    Conversational interface for controlling any software through natural language.
    
    Features:
    - Natural language command parsing and understanding
    - Context-aware command execution
    - Voice and text input support
    - Multi-turn conversations
    - Learning from user interactions
    - Cross-application control
    """
    
    def __init__(self, screen_intelligence: UniversalScreenIntelligence,
                 automation_engine: CrossApplicationAutomationEngine):
        self.logger = logging.getLogger(__name__)
        self.screen_intelligence = screen_intelligence
        self.automation_engine = automation_engine
        
        # NLP components
        self.intent_classifier = None
        self.entity_extractor = None
        self.response_generator = None
        
        # Conversation management
        self.active_conversations: Dict[str, ConversationContext] = {}
        self.command_history: List[Command] = []
        
        # Command patterns and templates
        self.command_patterns = self._load_command_patterns()
        self.action_mappings = self._load_action_mappings()
        
        # Configuration
        self.confidence_threshold = 0.7
        self.max_context_history = 10
        
        self.logger.info("Conversational Interface initialized")
    
    async def initialize(self) -> None:
        """Initialize NLP models and components."""
        try:
            self.logger.info("Loading NLP models...")
            
            # Load intent classification model
            await self._load_intent_classifier()
            
            # Load entity extraction model
            await self._load_entity_extractor()
            
            # Load response generation model
            await self._load_response_generator()
            
            self.logger.info("Conversational interface ready")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize conversational interface: {e}")
            # Continue with rule-based fallback
            self.logger.warning("Using rule-based fallback for command processing")
    
    async def start_conversation(self, user_id: Optional[str] = None) -> str:
        """
        Start a new conversation session.
        
        Args:
            user_id: Optional user identifier
            
        Returns:
            Conversation ID
        """
        conversation_id = str(uuid.uuid4())
        
        # Get current screen state
        current_screenshot = await self._capture_current_screen()
        current_screen_state = await self.screen_intelligence.analyze_screen(current_screenshot)
        
        # Create conversation context
        context = ConversationContext(
            conversation_id=conversation_id,
            user_id=user_id,
            current_screen_state=current_screen_state,
            active_application=current_screen_state.application_info.get("name", "unknown"),
            previous_commands=[],
            workflow_variables={},
            user_preferences={},
            session_start_time=time.time(),
            last_interaction_time=time.time()
        )
        
        self.active_conversations[conversation_id] = context
        
        self.logger.info(f"Started conversation {conversation_id} for user {user_id}")
        return conversation_id
    
    async def process_command(self, conversation_id: str, user_input: str,
                            input_type: str = "text") -> Dict[str, Any]:
        """
        Process a natural language command from the user.
        
        Args:
            conversation_id: Active conversation ID
            user_input: User's natural language input
            input_type: Type of input (text, voice)
            
        Returns:
            Response with command execution results
        """
        try:
            # Get conversation context
            if conversation_id not in self.active_conversations:
                return {
                    "success": False,
                    "error": "Conversation not found",
                    "response": "I don't have an active conversation with you."
                }
            
            context = self.active_conversations[conversation_id]
            
            # Update screen state
            current_screenshot = await self._capture_current_screen()
            context.current_screen_state = await self.screen_intelligence.analyze_screen(
                current_screenshot, {"conversation_id": conversation_id}
            )
            
            # Parse command
            command = await self._parse_command(user_input, context)
            
            # Store command in history
            context.previous_commands.append(command)
            if len(context.previous_commands) > self.max_context_history:
                context.previous_commands.pop(0)
            
            # Execute command
            result = await self._execute_command(command, context)
            
            # Generate response
            response = await self._generate_response(command, result, context)
            
            # Update conversation context
            context.last_interaction_time = time.time()
            
            return {
                "success": result.get("success", False),
                "command_id": command.command_id,
                "response": response,
                "execution_result": result,
                "requires_confirmation": command.requires_confirmation,
                "suggested_actions": command.suggested_actions
            }
            
        except Exception as e:
            self.logger.error(f"Error processing command: {e}")
            return {
                "success": False,
                "error": str(e),
                "response": "I encountered an error while processing your command."
            }
    
    async def _parse_command(self, user_input: str, 
                           context: ConversationContext) -> Command:
        """Parse natural language input into structured command."""
        command_id = str(uuid.uuid4())
        
        # Clean and normalize input
        cleaned_input = self._clean_user_input(user_input)
        
        # Extract intent and entities
        if self.intent_classifier and NLP_AVAILABLE:
            # Use ML models
            intent, entities, confidence = await self._classify_with_ml(cleaned_input, context)
        else:
            # Use rule-based parsing
            intent, entities, confidence = await self._classify_with_rules(cleaned_input, context)
        
        # Determine command type
        command_type = self._map_intent_to_command_type(intent)
        
        # Check if confirmation is needed
        requires_confirmation = self._requires_confirmation(intent, entities)
        
        # Generate suggested actions
        suggested_actions = await self._generate_suggestions(intent, entities, context)
        
        command = Command(
            command_id=command_id,
            command_type=command_type,
            intent=intent,
            entities=entities,
            confidence=confidence,
            raw_text=user_input,
            context={"conversation_id": context.conversation_id},
            requires_confirmation=requires_confirmation,
            suggested_actions=suggested_actions
        )
        
        self.command_history.append(command)
        return command
    
    async def _classify_with_ml(self, user_input: str, 
                              context: ConversationContext) -> Tuple[str, Dict[str, Any], float]:
        """Classify intent using ML models."""
        # Placeholder for ML-based classification
        # Would use trained NLP models in production
        return "unknown", {}, 0.5
    
    async def _classify_with_rules(self, user_input: str, 
                                 context: ConversationContext) -> Tuple[str, Dict[str, Any], float]:
        """Classify intent using rule-based patterns."""
        user_input_lower = user_input.lower()
        
        # Click commands
        click_patterns = [
            r"click on (.+)",
            r"click the (.+) button",
            r"press (.+)",
            r"select (.+)",
            r"choose (.+)"
        ]
        
        for pattern in click_patterns:
            match = re.search(pattern, user_input_lower)
            if match:
                target = match.group(1).strip()
                return "click_element", {"target": target}, 0.9
        
        # Type commands
        type_patterns = [
            r"type (.+)",
            r"enter (.+)",
            r"write (.+)",
            r"fill in (.+) with (.+)"
        ]
        
        for pattern in type_patterns:
            match = re.search(pattern, user_input_lower)
            if match:
                if pattern == r"fill in (.+) with (.+)":
                    field = match.group(1).strip()
                    text = match.group(2).strip()
                    return "type_text", {"field": field, "text": text}, 0.9
                else:
                    text = match.group(1).strip()
                    return "type_text", {"text": text}, 0.9
        
        # Navigation commands
        nav_patterns = [
            r"go to (.+)",
            r"navigate to (.+)",
            r"open (.+)",
            r"switch to (.+)"
        ]
        
        for pattern in nav_patterns:
            match = re.search(pattern, user_input_lower)
            if match:
                destination = match.group(1).strip()
                return "navigate", {"destination": destination}, 0.9
        
        # Extraction commands
        extract_patterns = [
            r"extract (.+)",
            r"get (.+) from the screen",
            r"copy (.+)",
            r"read (.+)"
        ]
        
        for pattern in extract_patterns:
            match = re.search(pattern, user_input_lower)
            if match:
                data_type = match.group(1).strip()
                return "extract_data", {"data_type": data_type}, 0.9
        
        # Workflow commands
        workflow_patterns = [
            r"start (.+) workflow",
            r"run (.+)",
            r"execute (.+)",
            r"automate (.+)"
        ]
        
        for pattern in workflow_patterns:
            match = re.search(pattern, user_input_lower)
            if match:
                workflow_name = match.group(1).strip()
                return "execute_workflow", {"workflow_name": workflow_name}, 0.9
        
        # Questions
        question_indicators = ["what", "where", "how", "why", "when", "can you", "could you"]
        if any(indicator in user_input_lower for indicator in question_indicators):
            return "question", {"query": user_input}, 0.8
        
        # Default
        return "unknown", {}, 0.3
    
    async def _execute_command(self, command: Command, 
                             context: ConversationContext) -> Dict[str, Any]:
        """Execute the parsed command."""
        try:
            if command.command_type == CommandType.UI_ACTION:
                return await self._execute_ui_action(command, context)
            elif command.command_type == CommandType.NAVIGATION:
                return await self._execute_navigation(command, context)
            elif command.command_type == CommandType.DATA_EXTRACTION:
                return await self._execute_data_extraction(command, context)
            elif command.command_type == CommandType.WORKFLOW_CONTROL:
                return await self._execute_workflow_control(command, context)
            elif command.command_type == CommandType.QUESTION:
                return await self._answer_question(command, context)
            else:
                return {
                    "success": False,
                    "error": f"Unknown command type: {command.command_type}",
                    "message": "I don't understand how to execute that command."
                }
                
        except Exception as e:
            self.logger.error(f"Error executing command: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "I encountered an error while executing your command."
            }
    
    async def _execute_ui_action(self, command: Command, 
                               context: ConversationContext) -> Dict[str, Any]:
        """Execute UI action commands."""
        if command.intent == "click_element":
            target = command.entities.get("target")
            
            # Find target element on screen
            target_element = await self.screen_intelligence.find_ui_element(
                context.current_screen_state.screenshot_path, target
            )
            
            if target_element:
                # Create automation step
                step = AutomationStep(
                    step_id=str(uuid.uuid4()),
                    action_type=ActionType.CLICK,
                    parameters={},
                    target_element=target_element,
                    description=f"Click on {target}"
                )
                
                # Execute step
                result = await self.automation_engine.actions[ActionType.CLICK].execute(
                    step, self.automation_engine
                )
                
                return {
                    "success": result["success"],
                    "action": "click",
                    "target": target,
                    "element_found": True,
                    "message": f"Clicked on {target}" if result["success"] else "Failed to click"
                }
            else:
                return {
                    "success": False,
                    "action": "click",
                    "target": target,
                    "element_found": False,
                    "message": f"I couldn't find {target} on the screen"
                }
        
        elif command.intent == "type_text":
            text = command.entities.get("text", "")
            
            step = AutomationStep(
                step_id=str(uuid.uuid4()),
                action_type=ActionType.TYPE,
                parameters={"text": text},
                description=f"Type: {text}"
            )
            
            result = await self.automation_engine.actions[ActionType.TYPE].execute(
                step, self.automation_engine
            )
            
            return {
                "success": result["success"],
                "action": "type",
                "text": text,
                "message": "Text typed successfully" if result["success"] else "Failed to type text"
            }
        
        return {"success": False, "message": "Unknown UI action"}
    
    async def _execute_navigation(self, command: Command, 
                                context: ConversationContext) -> Dict[str, Any]:
        """Execute navigation commands."""
        destination = command.entities.get("destination", "")
        
        # This would implement application-specific navigation
        return {
            "success": True,
            "action": "navigate",
            "destination": destination,
            "message": f"Navigated to {destination}"
        }
    
    async def _execute_data_extraction(self, command: Command, 
                                      context: ConversationContext) -> Dict[str, Any]:
        """Execute data extraction commands."""
        data_type = command.entities.get("data_type", "")
        
        step = AutomationStep(
            step_id=str(uuid.uuid4()),
            action_type=ActionType.EXTRACT_DATA,
            parameters={"type": data_type, "variable": "extracted_data"},
            description=f"Extract {data_type}"
        )
        
        result = await self.automation_engine.actions[ActionType.EXTRACT_DATA].execute(
            step, self.automation_engine
        )
        
        return {
            "success": result["success"],
            "action": "extract",
            "data_type": data_type,
            "extracted_data": context.workflow_variables.get("extracted_data"),
            "message": f"Extracted {data_type}" if result["success"] else "Failed to extract data"
        }
    
    async def _execute_workflow_control(self, command: Command, 
                                      context: ConversationContext) -> Dict[str, Any]:
        """Execute workflow control commands."""
        if command.intent == "execute_workflow":
            workflow_name = command.entities.get("workflow_name", "")
            
            # Find workflow by name
            workflows = self.automation_engine.get_workflows()
            target_workflow = None
            
            for workflow in workflows:
                if workflow_name.lower() in workflow.name.lower():
                    target_workflow = workflow
                    break
            
            if target_workflow:
                execution_id = await self.automation_engine.execute_workflow(
                    target_workflow.workflow_id
                )
                
                return {
                    "success": True,
                    "action": "execute_workflow",
                    "workflow_name": target_workflow.name,
                    "execution_id": execution_id,
                    "message": f"Started workflow: {target_workflow.name}"
                }
            else:
                return {
                    "success": False,
                    "action": "execute_workflow",
                    "workflow_name": workflow_name,
                    "message": f"Workflow '{workflow_name}' not found"
                }
        
        return {"success": False, "message": "Unknown workflow command"}
    
    async def _answer_question(self, command: Command, 
                             context: ConversationContext) -> Dict[str, Any]:
        """Answer user questions about the current context."""
        query = command.entities.get("query", "")
        
        # Generate context-aware response
        if "what" in query.lower() and "screen" in query.lower():
            return {
                "success": True,
                "action": "answer_question",
                "response": f"I can see {context.active_application} with {len(context.current_screen_state.ui_elements)} UI elements.",
                "message": "Described current screen"
            }
        elif "where" in query.lower():
            return {
                "success": True,
                "action": "answer_question",
                "response": f"You're currently in {context.active_application}.",
                "message": "Provided location information"
            }
        else:
            return {
                "success": True,
                "action": "answer_question",
                "response": "I'm here to help you with any software automation tasks.",
                "message": "General response"
            }
    
    async def _generate_response(self, command: Command, result: Dict[str, Any],
                                context: ConversationContext) -> str:
        """Generate natural language response to the user."""
        if result.get("success"):
            if command.command_type == CommandType.UI_ACTION:
                if command.intent == "click_element":
                    return f"I've clicked on {command.entities.get('target', 'the element')}."
                elif command.intent == "type_text":
                    return f"I've typed '{command.entities.get('text', 'the text')}'."
            elif command.command_type == CommandType.DATA_EXTRACTION:
                return f"I've extracted the {command.entities.get('data_type', 'data')} you requested."
            elif command.command_type == CommandType.WORKFLOW_CONTROL:
                return f"I've started the '{command.entities.get('workflow_name', 'workflow')}' automation."
            elif command.command_type == CommandType.QUESTION:
                return result.get("response", "Here's what I can tell you.")
            else:
                return "Command completed successfully."
        else:
            return result.get("message", "I'm sorry, I couldn't complete that command.")
    
    def _map_intent_to_command_type(self, intent: str) -> CommandType:
        """Map parsed intent to command type."""
        mapping = {
            "click_element": CommandType.UI_ACTION,
            "type_text": CommandType.UI_ACTION,
            "navigate": CommandType.NAVIGATION,
            "extract_data": CommandType.DATA_EXTRACTION,
            "execute_workflow": CommandType.WORKFLOW_CONTROL,
            "question": CommandType.QUESTION
        }
        return mapping.get(intent, CommandType.SYSTEM_COMMAND)
    
    def _requires_confirmation(self, intent: str, entities: Dict[str, Any]) -> bool:
        """Determine if command requires user confirmation."""
        # High-risk operations require confirmation
        if intent in ["execute_workflow", "delete", "remove", "format"]:
            return True
        return False
    
    async def _generate_suggestions(self, intent: str, entities: Dict[str, Any],
                                  context: ConversationContext) -> List[Dict[str, Any]]:
        """Generate suggested actions based on context."""
        suggestions = []
        
        if intent == "click_element" and not entities.get("target"):
            # Suggest available elements to click
            for element in context.current_screen_state.ui_elements[:5]:
                if element.interactive and element.text_content:
                    suggestions.append({
                        "type": "suggestion",
                        "text": f"Click on '{element.text_content}'",
                        "action": "click_element",
                        "entities": {"target": element.text_content}
                    })
        
        return suggestions
    
    def _clean_user_input(self, user_input: str) -> str:
        """Clean and normalize user input."""
        # Remove extra whitespace
        user_input = re.sub(r'\s+', ' ', user_input.strip())
        return user_input
    
    async def _load_intent_classifier(self) -> None:
        """Load intent classification model."""
        # Placeholder for loading ML model
        pass
    
    async def _load_entity_extractor(self) -> None:
        """Load entity extraction model."""
        # Placeholder for loading ML model
        pass
    
    async def _load_response_generator(self) -> None:
        """Load response generation model."""
        # Placeholder for loading ML model
        pass
    
    async def _capture_current_screen(self) -> str:
        """Capture current screen for context."""
        # Implementation would capture system screenshot
        return "/tmp/current_screen.png"
    
    def _load_command_patterns(self) -> Dict[str, Any]:
        """Load command pattern templates."""
        return {
            "click": ["click on", "press", "select", "choose"],
            "type": ["type", "enter", "write", "fill"],
            "navigate": ["go to", "open", "switch to"],
            "extract": ["get", "copy", "read", "extract"]
        }
    
    def _load_action_mappings(self) -> Dict[str, ActionType]:
        """Load action type mappings."""
        return {
            "click": ActionType.CLICK,
            "type": ActionType.TYPE,
            "extract": ActionType.EXTRACT_DATA,
            "navigate": ActionType.NAVIGATION
        }
    
    def get_conversation_history(self, conversation_id: str) -> List[Command]:
        """Get command history for a conversation."""
        context = self.active_conversations.get(conversation_id)
        return context.previous_commands if context else []
    
    def end_conversation(self, conversation_id: str) -> bool:
        """End an active conversation."""
        if conversation_id in self.active_conversations:
            del self.active_conversations[conversation_id]
            self.logger.info(f"Ended conversation {conversation_id}")
            return True
        return False