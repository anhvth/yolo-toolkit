#!/usr/bin/env python3
"""
Script 2: Create Label Studio Project
Creates a new Label Studio project with YOLO object detection configuration.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from label_studio_sdk import Client

# Load environment variables
load_dotenv()

LS_URL = os.getenv("LABEL_STUDIO_URL", "http://localhost:8080")
API_KEY = os.getenv("LABEL_STUDIO_API_KEY")

# YOLO object detection label config
LABEL_CONFIG = '''
<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="image" model_score_threshold="0.25">
    <Label value="Person" background="red"/>
    <Label value="Car" background="blue"/>
    <Label value="Bicycle" background="green"/>
    <Label value="Motorcycle" background="yellow"/>
  </RectangleLabels>
</View>
'''

def create_project(project_title="YOLO Detection Project"):
    """Create a new Label Studio project with YOLO configuration"""
    
    if not API_KEY:
        print("❌ Error: LABEL_STUDIO_API_KEY not found in environment")
        print("Please set your API key in .env file")
        print("You can get it from Label Studio UI → Account → Access Token")
        sys.exit(1)
    
    print(f"🔗 Connecting to Label Studio at {LS_URL}...")
    
    try:
        client = Client(url=LS_URL, api_key=API_KEY)
        
        print(f"📝 Creating project: {project_title}")
        project = client.start_project(
            title=project_title,
            label_config=LABEL_CONFIG
        )
        
        print(f"✅ Project created successfully!")
        print(f"   Project ID: {project.id}")
        print(f"   Project URL: {LS_URL}/projects/{project.id}")
        print(f"\n💡 Update PROJECT_ID={project.id} in your .env file")
        
        return project.id
        
    except Exception as e:
        print(f"❌ Error creating project: {e}")
        sys.exit(1)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create Label Studio project for YOLO")
    parser.add_argument("--title", default="YOLO Detection Project", help="Project title")
    args = parser.parse_args()
    
    create_project(args.title)
