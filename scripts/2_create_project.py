#!/usr/bin/env python3
"""
Script 2: Create Label Studio Project
Creates a new Label Studio project with YOLO object detection configuration.
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

def create_project(project_title="YOLO Detection Project", allow_duplicate=False):
    """Create a new Label Studio project with YOLO configuration"""
    
    if not API_KEY:
        print("❌ Error: LABEL_STUDIO_API_KEY not found in environment")
        print("\n📋 Steps to get your API key:")
        print("   1. Open Label Studio: http://localhost:8080")
        print("   2. Sign up or log in")
        print("   3. Go to: Account & Settings → Access Token")
        print("   4. Copy the token and add to .env file:")
        print("      LABEL_STUDIO_API_KEY=your_token_here")
        sys.exit(1)
    
    print(f"🔗 Connecting to Label Studio at {LS_URL}...")
    
    try:
        client = LabelStudio(base_url=LS_URL, api_key=API_KEY)
        
        # Check for existing projects with the same title
        if not allow_duplicate:
            print(f"🔍 Checking for existing projects...")
            existing_projects = list(client.projects.list(title=project_title))
            
            if existing_projects:
                print(f"\n⚠️  Project '{project_title}' already exists!")
                print(f"   Found {len(existing_projects)} project(s) with this name:")
                for proj in existing_projects:
                    print(f"   - ID: {proj.id}, Title: {proj.title}")
                    print(f"     URL: {LS_URL}/projects/{proj.id}")
                
                print(f"\n💡 Options:")
                print(f"   1. Use existing project ID: {existing_projects[0].id}")
                print(f"   2. Choose a different project name")
                print(f"   3. Delete the existing project first:")
                print(f"      client.projects.delete(id={existing_projects[0].id})")
                print(f"   4. Use --allow-duplicate flag to create anyway")
                sys.exit(1)
        
        print(f"📝 Creating project: {project_title}")
        project = client.projects.create(
            title=project_title,
            label_config=LABEL_CONFIG
        )
        
        print(f"✅ Project created successfully!")
        print(f"   Project ID: {project.id}")
        print(f"   Project URL: {LS_URL}/projects/{project.id}")
        print(f"\n💡 Update PROJECT_ID={project.id} in your .env file")
        
        return project.id
        
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Error creating project: {error_msg}")
        
        if "401" in error_msg or "Unauthorized" in error_msg:
            print("\n🔑 Authentication failed. Your API token may be invalid or expired.")
            print("\n📋 To get a new API token:")
            print("   1. Open Label Studio: http://localhost:8080")
            print("   2. Sign up or log in (create account if first time)")
            print("   3. Click on your username (top right) → Account & Settings")
            print("   4. Go to 'Access Token' tab")
            print("   5. Copy the token (or click 'Reset Token' if needed)")
            print("   6. Update your .env file:")
            print("      LABEL_STUDIO_API_KEY=your_new_token_here")
            print("\n💡 Note: Label Studio 1.21.0+ requires modern API tokens")
        
        sys.exit(1)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create Label Studio project for YOLO")
    parser.add_argument("--title", default="YOLO Detection Project", help="Project title")
    parser.add_argument("--allow-duplicate", action="store_true", 
                        help="Allow creating project even if name already exists")
    args = parser.parse_args()
    
    create_project(args.title, args.allow_duplicate)
