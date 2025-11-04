#!/usr/bin/env python3
"""
Script 3: Upload Images to Label Studio
Uploads images from the data/images directory to Label Studio project.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from label_studio_sdk import LabelStudio

# Load environment variables
load_dotenv()

LS_URL = os.getenv("LABEL_STUDIO_URL", "http://localhost:8080")
API_KEY = os.getenv("LABEL_STUDIO_API_KEY")
PROJECT_ID = os.getenv("PROJECT_ID", "1")
IMAGE_DIR = os.getenv("IMAGE_DIR", "data/images")

# Supported image extensions
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}

def upload_images(project_id=None, image_dir=None):
    """Upload images from directory to Label Studio project"""
    
    if not API_KEY:
        print("❌ Error: LABEL_STUDIO_API_KEY not found in environment")
        sys.exit(1)
    
    project_id = project_id or PROJECT_ID
    image_dir = image_dir or IMAGE_DIR
    image_path = Path(image_dir)
    
    if not image_path.exists():
        print(f"❌ Error: Image directory not found: {image_dir}")
        sys.exit(1)
    
    # Collect image files
    image_files = [
        f for f in image_path.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    ]
    
    if not image_files:
        print(f"❌ No images found in {image_dir}")
        print(f"   Supported formats: {', '.join(IMAGE_EXTENSIONS)}")
        sys.exit(1)
    
    print(f"🔗 Connecting to Label Studio at {LS_URL}...")
    print(f"📁 Found {len(image_files)} images in {image_dir}")
    
    try:
        client = LabelStudio(base_url=LS_URL, api_key=API_KEY)
        
        # Create tasks for each image
        tasks = []
        for img in image_files:
            # Use absolute path for Docker setup
            task_data = {
                "data": {
                    "image": f"/data/images/{img.name}"
                }
            }
            tasks.append(task_data)
        
        print(f"⬆️  Uploading {len(tasks)} images to project {project_id}...")
        client.tasks.create_many(project=int(project_id), request=tasks)
        
        print(f"✅ Successfully uploaded {len(tasks)} images!")
        print(f"   View them at: {LS_URL}/projects/{project_id}")
        
    except Exception as e:
        print(f"❌ Error uploading images: {e}")
        sys.exit(1)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Upload images to Label Studio")
    parser.add_argument("--project-id", help="Label Studio project ID")
    parser.add_argument("--image-dir", help="Directory containing images")
    args = parser.parse_args()
    
    upload_images(args.project_id, args.image_dir)
