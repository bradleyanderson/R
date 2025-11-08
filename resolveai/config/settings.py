"""
Configuration Settings Module

Manages all configuration settings for ResolveAI including security,
cloud providers, and application preferences.
"""

import os
import yaml
import json
from typing import Dict, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass, field
import logging


@dataclass
class CloudConfig:
    """Configuration for cloud services."""
    provider: str = "aws"
    region: str = "us-west-2"
    access_key: Optional[str] = None
    secret_key: Optional[str] = None
    project_id: Optional[str] = None
    bucket_name: Optional[str] = None
    enable_caching: bool = True
    cache_ttl: int = 3600


@dataclass
class SecurityConfig:
    """Security configuration settings."""
    encryption_enabled: bool = True
    encryption_key: Optional[str] = None
    two_factor_auth: bool = False
    session_timeout: int = 3600
    max_login_attempts: int = 5
    audit_logging: bool = True
    local_processing_only: bool = False
    data_retention_days: int = 30


@dataclass
class VideoConfig:
    """Video processing configuration."""
    max_resolution: str = "4K"
    supported_formats: list = field(default_factory=lambda: [
        ".mp4", ".mov", ".avi", ".mkv", ".webm", ".mxf"
    ])
    max_file_size_mb: int = 2048
    processing_threads: int = 4
    gpu_acceleration: bool = True
    temp_dir: str = "/tmp/resolveai"


@dataclass
class AudioConfig:
    """Audio processing configuration."""
    sample_rate: int = 48000
    bit_depth: int = 24
    channels: int = 2
    enable_noise_reduction: bool = True
    enable_transcription: bool = True
    transcription_language: str = "en"
    whisper_model: str = "base"


@dataclass
class DaVinciConfig:
    """DaVinci Resolve integration configuration."""
    auto_connect: bool = True
    sync_frequency: float = 1.0  # seconds
    enable_real_time_analysis: bool = True
    plugin_port: int = 8080
    api_timeout: int = 30


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    file_path: Optional[str] = None
    max_file_size_mb: int = 10
    backup_count: int = 5
    enable_console: bool = True
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


class Settings:
    """
    Central settings manager for ResolveAI.
    
    Handles loading, saving, and validating all configuration settings
    from various sources (files, environment variables, etc.).
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize settings manager.
        
        Args:
            config_file: Optional path to configuration file
        """
        self.logger = logging.getLogger(__name__)
        
        # Default configuration paths
        self.config_paths = [
            os.path.expanduser("~/.resolveai/config.yaml"),
            os.path.expanduser("~/.resolveai/config.yml"),
            "./config.yaml",
            "./config.yml",
            "/etc/resolveai/config.yaml"
        ]
        
        if config_file:
            self.config_paths.insert(0, config_file)
        
        # Initialize configuration objects
        self.cloud = CloudConfig()
        self.security = SecurityConfig()
        self.video = VideoConfig()
        self.audio = AudioConfig()
        self.davinci = DaVinciConfig()
        self.logging = LoggingConfig()
        
        # Load configuration
        self._load_configuration()
        
        # Override with environment variables
        self._load_environment_variables()
        
        # Validate configuration
        self._validate_configuration()
    
    def _load_configuration(self) -> None:
        """Load configuration from file."""
        config_data = {}
        
        # Try each config path
        for config_path in self.config_paths:
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r') as f:
                        if config_path.endswith(('.yaml', '.yml')):
                            config_data = yaml.safe_load(f) or {}
                        else:
                            config_data = json.load(f)
                    
                    self.logger.info(f"Loaded configuration from {config_path}")
                    break
                    
                except Exception as e:
                    self.logger.warning(f"Failed to load config from {config_path}: {e}")
        
        # Update configuration objects
        if config_data:
            self._update_from_dict(config_data)
    
    def _load_environment_variables(self) -> None:
        """Load configuration from environment variables."""
        env_mappings = {
            # Cloud settings
            "RESOLVEAI_CLOUD_PROVIDER": ("cloud", "provider"),
            "RESOLVEAI_CLOUD_REGION": ("cloud", "region"),
            "RESOLVEAI_ACCESS_KEY": ("cloud", "access_key"),
            "RESOLVEAI_SECRET_KEY": ("cloud", "secret_key"),
            "RESOLVEAI_PROJECT_ID": ("cloud", "project_id"),
            "RESOLVEAI_BUCKET_NAME": ("cloud", "bucket_name"),
            
            # Security settings
            "RESOLVEAI_ENCRYPTION_KEY": ("security", "encryption_key"),
            "RESOLVEAI_LOCAL_PROCESSING_ONLY": ("security", "local_processing_only"),
            "RESOLVEAI_DATA_RETENTION_DAYS": ("security", "data_retention_days"),
            
            # Video settings
            "RESOLVEAI_MAX_RESOLUTION": ("video", "max_resolution"),
            "RESOLVEAI_MAX_FILE_SIZE_MB": ("video", "max_file_size_mb"),
            "RESOLVEAI_GPU_ACCELERATION": ("video", "gpu_acceleration"),
            
            # Audio settings
            "RESOLVEAI_SAMPLE_RATE": ("audio", "sample_rate"),
            "RESOLVEAI_WHISPER_MODEL": ("audio", "whisper_model"),
            
            # DaVinci settings
            "RESOLVEAI_PLUGIN_PORT": ("davinci", "plugin_port"),
            "RESOLVEAI_AUTO_CONNECT": ("davinci", "auto_connect"),
            
            # Logging settings
            "RESOLVEAI_LOG_LEVEL": ("logging", "level"),
            "RESOLVEAI_LOG_FILE": ("logging", "file_path"),
        }
        
        for env_var, (section, key) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                section_obj = getattr(self, section)
                
                # Type conversion
                if hasattr(section_obj, key):
                    current_value = getattr(section_obj, key)
                    if isinstance(current_value, bool):
                        value = value.lower() in ('true', '1', 'yes', 'on')
                    elif isinstance(current_value, int):
                        value = int(value)
                    elif isinstance(current_value, float):
                        value = float(value)
                
                setattr(section_obj, key, value)
                self.logger.debug(f"Set {section}.{key} from environment variable {env_var}")
    
    def _update_from_dict(self, config_data: Dict[str, Any]) -> None:
        """Update configuration from dictionary."""
        sections = {
            "cloud": self.cloud,
            "security": self.security,
            "video": self.video,
            "audio": self.audio,
            "davinci": self.davinci,
            "logging": self.logging,
        }
        
        for section_name, section_data in config_data.items():
            if section_name in sections and isinstance(section_data, dict):
                section_obj = sections[section_name]
                
                for key, value in section_data.items():
                    if hasattr(section_obj, key):
                        setattr(section_obj, key, value)
                        self.logger.debug(f"Updated {section_name}.{key} = {value}")
    
    def _validate_configuration(self) -> None:
        """Validate configuration settings."""
        errors = []
        
        # Validate cloud configuration
        if self.cloud.provider not in ["aws", "gcp", "azure"]:
            errors.append(f"Invalid cloud provider: {self.cloud.provider}")
        
        # Validate security configuration
        if self.security.session_timeout <= 0:
            errors.append("Session timeout must be positive")
        
        if self.security.max_login_attempts <= 0:
            errors.append("Max login attempts must be positive")
        
        # Validate video configuration
        if self.video.max_resolution not in ["720p", "1080p", "4K", "8K"]:
            errors.append(f"Invalid max resolution: {self.video.max_resolution}")
        
        if self.video.max_file_size_mb <= 0:
            errors.append("Max file size must be positive")
        
        if self.video.processing_threads <= 0:
            errors.append("Processing threads must be positive")
        
        # Validate audio configuration
        if self.audio.sample_rate not in [44100, 48000, 96000]:
            errors.append(f"Invalid sample rate: {self.audio.sample_rate}")
        
        if self.audio.bit_depth not in [16, 24, 32]:
            errors.append(f"Invalid bit depth: {self.audio.bit_depth}")
        
        if self.audio.channels not in [1, 2, 6, 8]:
            errors.append(f"Invalid channel count: {self.audio.channels}")
        
        # Validate DaVinci configuration
        if not (1024 <= self.davinci.plugin_port <= 65535):
            errors.append(f"Invalid plugin port: {self.davinci.plugin_port}")
        
        # Validate logging configuration
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.logging.level not in valid_log_levels:
            errors.append(f"Invalid log level: {self.logging.level}")
        
        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {error}" for error in errors)
            raise ValueError(error_msg)
        
        self.logger.info("Configuration validation passed")
    
    def save_configuration(self, config_path: Optional[str] = None) -> None:
        """
        Save current configuration to file.
        
        Args:
            config_path: Optional path to save configuration. If None, uses first config path.
        """
        if config_path is None:
            config_path = self.config_paths[0]
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        # Build configuration dictionary
        config_data = {
            "cloud": self.cloud.__dict__,
            "security": self.security.__dict__,
            "video": self.video.__dict__,
            "audio": self.audio.__dict__,
            "davinci": self.davinci.__dict__,
            "logging": self.logging.__dict__,
        }
        
        try:
            with open(config_path, 'w') as f:
                if config_path.endswith(('.yaml', '.yml')):
                    yaml.dump(config_data, f, default_flow_style=False, indent=2)
                else:
                    json.dump(config_data, f, indent=2)
            
            self.logger.info(f"Configuration saved to {config_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}")
            raise
    
    def get_cloud_credentials(self) -> Dict[str, Any]:
        """Get cloud provider credentials."""
        if self.cloud.provider == "aws":
            return {
                "aws_access_key_id": self.cloud.access_key,
                "aws_secret_access_key": self.cloud.secret_key,
                "region_name": self.cloud.region
            }
        elif self.cloud.provider == "gcp":
            return {
                "project_id": self.cloud.project_id,
                "credentials": self.cloud.access_key
            }
        elif self.cloud.provider == "azure":
            return {
                "account_name": self.cloud.access_key,
                "account_key": self.cloud.secret_key
            }
        else:
            raise ValueError(f"Unsupported cloud provider: {self.cloud.provider}")
    
    def is_cloud_configured(self) -> bool:
        """Check if cloud services are properly configured."""
        if self.security.local_processing_only:
            return False
        
        if self.cloud.provider == "aws":
            return bool(self.cloud.access_key and self.cloud.secret_key)
        elif self.cloud.provider == "gcp":
            return bool(self.cloud.project_id and self.cloud.access_key)
        elif self.cloud.provider == "azure":
            return bool(self.cloud.access_key and self.cloud.secret_key)
        
        return False
    
    def get_effective_config(self) -> Dict[str, Any]:
        """Get all effective configuration as a dictionary."""
        return {
            "cloud": self.cloud.__dict__,
            "security": self.security.__dict__,
            "video": self.video.__dict__,
            "audio": self.audio.__dict__,
            "davinci": self.davinci.__dict__,
            "logging": self.logging.__dict__,
        }
    
    def update_setting(self, section: str, key: str, value: Any) -> None:
        """
        Update a specific setting.
        
        Args:
            section: Configuration section (cloud, security, etc.)
            key: Setting key
            value: New value
        """
        if hasattr(self, section):
            section_obj = getattr(self, section)
            if hasattr(section_obj, key):
                setattr(section_obj, key, value)
                self.logger.info(f"Updated {section}.{key} = {value}")
            else:
                raise ValueError(f"Invalid key '{key}' in section '{section}'")
        else:
            raise ValueError(f"Invalid section '{section}'")
        
        # Re-validate after update
        self._validate_configuration()