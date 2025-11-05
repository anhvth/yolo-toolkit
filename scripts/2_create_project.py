#!/usr/bin/env python3
"""
Script 2: Create Label Studio Project
Creates a new Label Studio project with YOLO object detection configuration.
"""

import sys
from pathlib import Path
from label_studio_sdk import LabelStudio
from config import get_config

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

def create_project(project_title="YOLO Detection Project", allow_duplicate=False, force=False):
    """Create a new Label Studio project with YOLO configuration"""
    
    config = get_config()
    
    if not config.ls_api_key:
        print("❌ Error: LABEL_STUDIO_API_KEY not found in ls_settings.json")
        print("\n📋 Steps to get your API key:")
        print("   1. Open Label Studio: http://localhost:8080")
        print("   2. Sign up or log in")
        print("   3. Go to: Account & Settings → Access Token")
        print("   4. Copy the token and add to ls_settings.json:")
        print('      "api_key": "your_token_here"')
        sys.exit(1)
    
    print(f"🔗 Connecting to Label Studio at {config.ls_url}...")
    
    try:
        client = LabelStudio(base_url=config.ls_url, api_key=config.ls_api_key)
        
        # Check for existing projects with the same title
        if force:
            print(f"🔍 Checking for existing projects to delete...")
            existing_projects = list(client.projects.list(title=project_title))
            
            if existing_projects:
                print(f"🗑️  Found {len(existing_projects)} project(s) with name '{project_title}' - deleting...")
                for proj in existing_projects:
                    print(f"   Deleting project ID: {proj.id}")
                    client.projects.delete(id=proj.id)
                print(f"✅ Deleted all existing projects with name '{project_title}'")
        elif not allow_duplicate:
            print(f"🔍 Checking for existing projects...")
            existing_projects = list(client.projects.list(title=project_title))
            
            if existing_projects:
                print(f"\n⚠️  Project '{project_title}' already exists!")
                print(f"   Found {len(existing_projects)} project(s) with this name:")
                for proj in existing_projects:
                    print(f"   - ID: {proj.id}, Title: {proj.title}")
                    print(f"     URL: {config.ls_url}/projects/{proj.id}")
                
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
        print(f"   Project URL: {config.ls_url}/projects/{project.id}")
        print(f"\n💡 Updating project_id in ls_settings.json...")
        
        # Update the settings file with new project ID
        config.update_project_id(project.id)


        return project.id
        
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Error creating project: {error_msg}")
        
        if "401" in error_msg or "Unauthorized" in error_msg:
            print("\n🔑 Authentication failed. Your API token may be invalid or expired.")
            print("\n📋 To get a new API token:")
            print("   1. Open Label Studio: http://localhost:8080/user/account/personal-access-token")
            print("   2. Sign up or log in (create account if first time)")
            print("   3. Click on your username (top right) → Account & Settings")
            print("   4. Go to 'Access Token' tab")
            print("   5. Copy the token (or click 'Reset Token' if needed)")
            print("   6. Update your ls_settings.json file:")
            print('      "api_key": "your_new_token_here"')
            print("\n💡 Note: Label Studio 1.21.0+ requires modern API tokens")
        
        sys.exit(1)

if __name__ == "__main__":
    import argparse

    
    parser = argparse.ArgumentParser(description="Create Label Studio project for YOLO")
    parser.add_argument("--title", help="Project title", default='YOLO Detection Project')
    parser.add_argument("--allow-duplicate", action="store_true", 
                        help="Allow creating project even if name already exists")
    parser.add_argument("--force", action="store_true",
                        help="Delete any existing projects with the same name before creating")
    args = parser.parse_args()
    
    create_project(args.title, args.allow_duplicate, args.force)
