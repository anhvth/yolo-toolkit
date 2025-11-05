#!/usr/bin/env python3
"""
Configuration loader for Label Studio + YOLO toolkit
Loads settings from ls_settings.json instead of .env files
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional


class Config:
    """Central configuration manager for all scripts"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Load configuration from ls_settings.json
        
        Args:
            config_path: Path to settings JSON file (default: ls_settings.json in project root)
        """
        if config_path is None:
            # Find project root (go up from scripts/ to project root)
            script_dir = Path(__file__).parent
            project_root = script_dir.parent
            config_path = str(project_root / "ls_settings.json")
        
        self.config_path = Path(config_path)
        
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"❌ Configuration file not found: {self.config_path}\n"
                f"💡 Copy ls_settings.json.example to ls_settings.json and update with your settings"
            )
        
        with open(self.config_path, 'r') as f:
            self._config = json.load(f)
    
    # Label Studio settings
    @property
    def ls_url(self) -> str:
        """Label Studio URL"""
        return self._config["label_studio"]["url"]
    
    @property
    def ls_api_key(self) -> str:
        """Label Studio API key"""
        return self._config["label_studio"]["api_key"]
    
    @property
    def project_id(self) -> int:
        """Label Studio project ID"""
        return self._config["label_studio"]["project_id"]
    
    @property
    def local_files_serving_enabled(self) -> bool:
        """Whether local file serving is enabled"""
        return self._config["label_studio"].get("local_files_serving_enabled", False)
    
    @property
    def local_files_document_root(self) -> str:
        """Document root for local file serving"""
        return self._config["label_studio"].get("local_files_document_root", "")
    
    # Path settings
    @property
    def image_dir(self) -> str:
        """Directory containing images"""
        return self._config["paths"]["image_dir"]
    
    @property
    def export_dir(self) -> str:
        """Directory for YOLO exports"""
        return self._config["paths"]["export_dir"]
    
    @property
    def predictions_dir(self) -> str:
        """Directory for predictions"""
        return self._config["paths"]["predictions_dir"]
    
    @property
    def base_model_path(self) -> str:
        """Path to base YOLO model"""
        return self._config["paths"]["base_model_path"]
    
    @property
    def updated_model_path(self) -> str:
        """Path to updated/fine-tuned YOLO model"""
        return self._config["paths"]["updated_model_path"]
    
    # YOLO settings
    @property
    def epochs(self) -> int:
        """Training epochs"""
        return self._config["yolo"]["epochs"]
    
    @property
    def image_size(self) -> int:
        """Training image size"""
        return self._config["yolo"]["image_size"]
    
    @property
    def model_score_threshold(self) -> float:
        """Model prediction confidence threshold"""
        return self._config["yolo"]["model_score_threshold"]
    
    def update_project_id(self, project_id: int):
        """
        Update PROJECT_ID in the settings file
        
        Args:
            project_id: New project ID to save
        """
        self._config["label_studio"]["project_id"] = project_id
        with open(self.config_path, 'w') as f:
            json.dump(self._config, f, indent=2)
        print(f"✅ Updated PROJECT_ID={project_id} in {self.config_path}")
    
    def get_raw(self) -> Dict[str, Any]:
        """Get raw configuration dictionary"""
        return self._config


# Singleton instance for convenience
_config = None

def get_config(config_path: Optional[str] = None) -> Config:
    """
    Get or create the global configuration instance
    
    Args:
        config_path: Optional path to settings JSON file
    
    Returns:
        Config instance
    """
    global _config
    if _config is None:
        _config = Config(config_path)
    return _config
