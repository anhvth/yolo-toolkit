#!/usr/bin/env python3
"""
Script 1: Start Label Studio Server
This script launches Label Studio locally (for non-Docker setups).
For Docker users, use: docker compose up
"""

import os
import sys
from pathlib import Path
from config import get_config

def start_labelstudio():
    """Start Label Studio server with local file serving enabled"""
    
    # Get configuration from ls_settings.json
    config = get_config()
    project_root = Path(__file__).parent.parent.absolute()
    data_dir = project_root / "data"
    
    # Set required environment variables for local file serving
    os.environ['LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED'] = str(config.local_files_serving_enabled).lower()
    os.environ['LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT'] = config.local_files_document_root or str(data_dir)
    
    print("🚀 Starting Label Studio server...")
    print("Note: If using Docker, run 'docker compose up' instead")
    print(f"\n📂 Local files directory: {data_dir}")
    print(f"   - Images at: {data_dir}/images")
    print(f"   - This allows Label Studio to serve local image files")
    print("\n🔗 Access Label Studio at: http://localhost:8080")
    print("\nPress Ctrl+C to stop the server\n")
    
    # Start label-studio with environment variables set
    os.system("label-studio start")

if __name__ == "__main__":
    try:
        start_labelstudio()
    except KeyboardInterrupt:
        print("\n\n✅ Label Studio server stopped")
        sys.exit(0)
