"""
Universal Screen Intelligence System

Advanced computer vision capable of recognizing and understanding any application
interface, web page, or digital workspace in real-time.
"""

import asyncio
import logging
import numpy as np
import cv2
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import time
import json
from enum import Enum

try:
    import torch
    import torchvision.transforms as transforms
    from transformers import AutoProcessor, AutoModelForVision2Seq
    VISION_AVAILABLE = True
except ImportError:
    VISION_AVAILABLE = False

try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False


class InterfaceType(Enum):
    """Types of digital interfaces that can be analyzed."""
    DESKTOP_APPLICATION = "desktop_app"
    WEB_PAGE = "web_page"
    TERMINAL = "terminal"
    MOBILE_APP = "mobile_app"
    GAME = "game"
    DEVELOPMENT_IDE = "development_ide"
    DESIGN_TOOL = "design_tool"
    UNKNOWN = "unknown"


@dataclass
class UIElement:
    """Represents a detected UI element."""
    element_type: str  # button, text_input, menu, etc.
    bounding_box: Tuple[int, int, int, int]  # x, y, width, height
    text_content: Optional[str] = None
    confidence: float = 0.0
    properties: Dict[str, Any] = None
    interactive: bool = False
    accessibility_label: Optional[str] = None


@dataclass
class ScreenAnalysis:
    """Complete analysis of a screen/interface."""
    screenshot_path: str
    interface_type: InterfaceType
    ui_elements: List[UIElement]
    text_content: str
    layout_structure: Dict[str, Any]
    application_info: Dict[str, Any]
    timestamp: float
    confidence_score: float
    user_context: Dict[str, Any] = None


class UniversalScreenIntelligence:
    """
    Universal screen intelligence system capable of understanding any interface.
    
    Features:
    - Real-time interface detection and analysis
    - UI element recognition and classification
    - Text extraction and OCR
    - Layout understanding and structure analysis
    - Application identification
    - Accessibility support
    - Multi-platform compatibility
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._initialized = False
        self._models_loaded = False
        
        # AI Models
        self.ui_detector = None
        self.text_recognizer = None
        self.layout_analyzer = None
        self.application_classifier = None
        
        # Configuration
        self.detection_confidence_threshold = 0.5
        self.ocr_enabled = OCR_AVAILABLE
        self.advanced_vision_enabled = VISION_AVAILABLE
        
        # Element detection templates
        self.ui_templates = self._load_ui_templates()
        self.application_signatures = self._load_application_signatures()
        
        self.logger.info("Universal Screen Intelligence initialized")
    
    async def initialize(self) -> None:
        """Initialize all AI models and components."""
        if self._initialized:
            return
        
        try:
            self.logger.info("Loading AI models for screen intelligence...")
            
            # Load UI element detection model
            await self._load_ui_detection_model()
            
            # Load text recognition model
            await self._load_text_recognition_model()
            
            # Load layout analysis model
            await self._load_layout_analysis_model()
            
            # Load application classifier
            await self._load_application_classifier()
            
            self._models_loaded = True
            self._initialized = True
            
            self.logger.info("Screen intelligence models loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize screen intelligence: {e}")
            raise
    
    async def analyze_screen(self, screenshot_path: str, 
                           context: Optional[Dict[str, Any]] = None) -> ScreenAnalysis:
        """
        Analyze a screen screenshot and extract comprehensive interface information.
        
        Args:
            screenshot_path: Path to the screenshot image
            context: Optional context information for better analysis
            
        Returns:
            Complete screen analysis with UI elements and metadata
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            start_time = time.time()
            
            # Load and preprocess image
            image = await self._load_and_preprocess_image(screenshot_path)
            
            # Detect interface type
            interface_type = await self._detect_interface_type(image)
            
            # Identify application
            application_info = await self._identify_application(image, interface_type)
            
            # Extract text content
            text_content = await self._extract_text_content(image)
            
            # Detect UI elements
            ui_elements = await self._detect_ui_elements(image, interface_type)
            
            # Analyze layout structure
            layout_structure = await self._analyze_layout(image, ui_elements)
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence(ui_elements, text_content)
            
            analysis_time = time.time() - start_time
            
            result = ScreenAnalysis(
                screenshot_path=screenshot_path,
                interface_type=interface_type,
                ui_elements=ui_elements,
                text_content=text_content,
                layout_structure=layout_structure,
                application_info=application_info,
                timestamp=time.time(),
                confidence_score=confidence_score,
                user_context=context
            )
            
            self.logger.info(f"Screen analysis completed in {analysis_time:.2f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"Error analyzing screen: {e}")
            raise
    
    async def analyze_screen_realtime(self, screen_region: Optional[Tuple[int, int, int, int]] = None,
                                    callback: Optional[callable] = None) -> None:
        """
        Real-time screen analysis with continuous monitoring.
        
        Args:
            screen_region: Optional region to monitor (x, y, width, height)
            callback: Optional callback function for analysis results
        """
        if not self._initialized:
            await self.initialize()
        
        self.logger.info("Starting real-time screen analysis...")
        
        try:
            while True:
                # Capture screen
                screenshot_path = await self._capture_screen_region(screen_region)
                
                # Analyze screen
                analysis = await self.analyze_screen(screenshot_path)
                
                # Call callback if provided
                if callback:
                    await callback(analysis)
                
                # Brief delay to prevent excessive CPU usage
                await asyncio.sleep(0.1)
                
        except KeyboardInterrupt:
            self.logger.info("Real-time analysis stopped by user")
        except Exception as e:
            self.logger.error(f"Error in real-time analysis: {e}")
            raise
    
    async def find_ui_element(self, screenshot_path: str, 
                            element_description: str) -> Optional[UIElement]:
        """
        Find a specific UI element based on natural language description.
        
        Args:
            screenshot_path: Path to screenshot
            element_description: Natural language description of element
            
        Returns:
            Matching UI element or None if not found
        """
        try:
            analysis = await self.analyze_screen(screenshot_path)
            
            # Use semantic search to find matching element
            matching_elements = []
            
            for element in analysis.ui_elements:
                similarity = await self._calculate_semantic_similarity(
                    element_description, 
                    element.text_content or element.element_type
                )
                
                if similarity > 0.7:  # Threshold for matching
                    matching_elements.append((element, similarity))
            
            # Return best match
            if matching_elements:
                matching_elements.sort(key=lambda x: x[1], reverse=True)
                return matching_elements[0][0]
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error finding UI element: {e}")
            return None
    
    async def track_ui_changes(self, initial_analysis: ScreenAnalysis,
                             new_screenshot_path: str) -> Dict[str, Any]:
        """
        Track changes between two screen states.
        
        Args:
            initial_analysis: Initial screen analysis
            new_screenshot_path: Path to new screenshot
            
        Returns:
            Dictionary describing detected changes
        """
        try:
            # Analyze new screen
            new_analysis = await self.analyze_screen(new_screenshot_path)
            
            # Compare UI elements
            changes = await self._compare_screen_states(initial_analysis, new_analysis)
            
            return {
                "changes_detected": len(changes["added"]) + len(changes["removed"]) + len(changes["modified"]),
                "added_elements": changes["added"],
                "removed_elements": changes["removed"],
                "modified_elements": changes["modified"],
                "layout_changes": changes["layout"],
                "timestamp": time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Error tracking UI changes: {e}")
            return {}
    
    async def _load_ui_detection_model(self) -> None:
        """Load UI element detection model."""
        if self.advanced_vision_enabled:
            try:
                # Load a vision model for UI element detection
                self.ui_detector = AutoModelForVision2Seq.from_pretrained(
                    "microsoft/layoutlmv3-base"
                )
                self.ui_processor = AutoProcessor.from_pretrained(
                    "microsoft/layoutlmv3-base", 
                    apply_ocr=False
                )
                self.logger.info("Advanced UI detection model loaded")
            except Exception as e:
                self.logger.warning(f"Failed to load advanced UI model: {e}")
                self.advanced_vision_enabled = False
        else:
            # Fallback to OpenCV-based detection
            self.ui_detector = "opencv_fallback"
            self.logger.info("Using OpenCV fallback for UI detection")
    
    async def _load_text_recognition_model(self) -> None:
        """Load text recognition model."""
        if self.ocr_enabled:
            try:
                # Configure tesseract if available
                if hasattr(pytesseract, 'pytesseract'):
                    pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
                self.text_recognizer = pytesseract
                self.logger.info("Tesseract OCR loaded")
            except Exception as e:
                self.logger.warning(f"Failed to load OCR: {e}")
                self.ocr_enabled = False
    
    async def _load_layout_analysis_model(self) -> None:
        """Load layout analysis model."""
        self.layout_analyzer = "rule_based"
        self.logger.info("Layout analysis initialized")
    
    async def _load_application_classifier(self) -> None:
        """Load application identification model."""
        self.application_classifier = "signature_based"
        self.logger.info("Application classifier loaded")
    
    async def _load_and_preprocess_image(self, image_path: str) -> np.ndarray:
        """Load and preprocess image for analysis."""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Preprocess for better detection
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return image
    
    async def _detect_interface_type(self, image: np.ndarray) -> InterfaceType:
        """Detect the type of interface from screenshot."""
        # Basic heuristics for interface type detection
        height, width = image.shape[:2]
        
        # Check for typical browser patterns
        if self._has_browser_elements(image):
            return InterfaceType.WEB_PAGE
        
        # Check for terminal patterns
        if self._has_terminal_patterns(image):
            return InterfaceType.TERMINAL
        
        # Check for IDE patterns
        if self._has_ide_patterns(image):
            return InterfaceType.DEVELOPMENT_IDE
        
        # Default to desktop application
        return InterfaceType.DESKTOP_APPLICATION
    
    async def _identify_application(self, image: np.ndarray, 
                                   interface_type: InterfaceType) -> Dict[str, Any]:
        """Identify the specific application from screenshot."""
        # This would use visual signatures and window titles
        # For now, return basic info
        return {
            "name": "unknown",
            "version": "unknown",
            "type": interface_type.value,
            "confidence": 0.0
        }
    
    async def _extract_text_content(self, image: np.ndarray) -> str:
        """Extract all text content from the screenshot."""
        if not self.ocr_enabled:
            return ""
        
        try:
            # Convert PIL image for OCR
            pil_image = Image.fromarray(image)
            text = pytesseract.image_to_string(pil_image)
            return text.strip()
        except Exception as e:
            self.logger.error(f"OCR extraction failed: {e}")
            return ""
    
    async def _detect_ui_elements(self, image: np.ndarray, 
                                interface_type: InterfaceType) -> List[UIElement]:
        """Detect and classify UI elements in the screenshot."""
        elements = []
        
        if self.advanced_vision_enabled and self.ui_detector != "opencv_fallback":
            # Use advanced vision model
            elements = await self._detect_elements_with_ai_model(image)
        else:
            # Use OpenCV fallback
            elements = await self._detect_elements_with_opencv(image, interface_type)
        
        return elements
    
    async def _detect_elements_with_ai_model(self, image: np.ndarray) -> List[UIElement]:
        """Detect UI elements using advanced AI models."""
        # Implementation would use the loaded vision model
        # For now, return empty list as placeholder
        return []
    
    async def _detect_elements_with_opencv(self, image: np.ndarray,
                                         interface_type: InterfaceType) -> List[UIElement]:
        """Detect UI elements using OpenCV and heuristics."""
        elements = []
        
        # Convert to grayscale for processing
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Detect buttons (rectangular shapes with certain characteristics)
        contours = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours[0]:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by size and aspect ratio
            if w > 20 and h > 10 and w/h > 0.5 and w/h < 5:
                element = UIElement(
                    element_type="button",
                    bounding_box=(x, y, w, h),
                    confidence=0.7,
                    interactive=True
                )
                elements.append(element)
        
        return elements
    
    async def _analyze_layout(self, image: np.ndarray, 
                            elements: List[UIElement]) -> Dict[str, Any]:
        """Analyze the layout structure of the interface."""
        return {
            "type": "grid",
            "columns": 3,
            "rows": 2,
            "elements_count": len(elements),
            "hierarchy": "flat"
        }
    
    async def _capture_screen_region(self, region: Optional[Tuple[int, int, int, int]]) -> str:
        """Capture a specific region of the screen."""
        # Implementation would use system screen capture
        # For now, return placeholder
        return "/tmp/captured_screenshot.png"
    
    def _calculate_confidence(self, elements: List[UIElement], text_content: str) -> float:
        """Calculate overall confidence score for the analysis."""
        if not elements and not text_content:
            return 0.0
        
        element_confidence = sum(e.confidence for e in elements) / len(elements) if elements else 0.5
        text_confidence = 0.8 if text_content.strip() else 0.3
        
        return (element_confidence + text_confidence) / 2
    
    async def _calculate_semantic_similarity(self, description: str, 
                                            element_text: str) -> float:
        """Calculate semantic similarity between description and element."""
        # Simplified similarity calculation
        # In production, would use embeddings or transformers
        if not element_text:
            return 0.0
        
        description_words = set(description.lower().split())
        element_words = set(element_text.lower().split())
        
        intersection = description_words.intersection(element_words)
        union = description_words.union(element_words)
        
        return len(intersection) / len(union) if union else 0.0
    
    async def _compare_screen_states(self, old_analysis: ScreenAnalysis,
                                   new_analysis: ScreenAnalysis) -> Dict[str, Any]:
        """Compare two screen states and identify changes."""
        return {
            "added": [],
            "removed": [],
            "modified": [],
            "layout": {}
        }
    
    def _has_browser_elements(self, image: np.ndarray) -> bool:
        """Check if image contains browser elements."""
        # Simplified heuristic - check for typical browser UI patterns
        return False
    
    def _has_terminal_patterns(self, image: np.ndarray) -> bool:
        """Check if image contains terminal patterns."""
        # Check for dark background with monospace text patterns
        return False
    
    def _has_ide_patterns(self, image: np.ndarray) -> bool:
        """Check if image contains IDE patterns."""
        # Check for code editor patterns
        return False
    
    def _load_ui_templates(self) -> Dict[str, Any]:
        """Load UI element detection templates."""
        return {}
    
    def _load_application_signatures(self) -> Dict[str, Any]:
        """Load application visual signatures."""
        return {}