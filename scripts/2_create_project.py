#!/usr/bin/env python3
"""
Script 2: Create Label Studio Project
Creates a new Label Studio project with YOLO object detection configuration.
"""

import sys
from pathlib import Path
from label_studio_sdk import LabelStudio
from config import get_config
from config_label import build_label_config


def prompt_for_api_key(config):
    """Prompt user for API key and save it to settings"""
    print("❌ API key is missing, invalid, or expired in ls_settings.json")
    print("\n📋 To get your API key:")
    print("   1. Open Label Studio: http://localhost:8080")
    print("   2. Sign up or log in (create account if first time)")
    print("   3. Click on your username (top right) → Account & Settings")
    print("   4. Go to 'Access Token' tab")
    print("   5. Copy the token (or click 'Reset Token' if needed)")
    print("\n🔑 Please paste your API key here (press Ctrl+C to cancel):")
    
    try:
        api_key = input("API Key: ").strip()
        if not api_key:
            print("❌ No API key provided. Exiting...")
            sys.exit(1)
        
        # Basic validation - Label Studio API keys are typically JWT tokens
        if len(api_key) < 10:
            print("⚠️  Warning: API key seems too short. Make sure you copied the complete token.")
        
        # Save the API key to settings file
        config.update_api_key(api_key)
        print("✅ API key saved successfully!")
        
        return api_key
        
    except KeyboardInterrupt:
        print("\n❌ Operation cancelled by user.")
        sys.exit(1)
    except EOFError:
        print("\n❌ No input provided. Exiting...")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error saving API key: {e}")
        sys.exit(1)


def create_project():
    """Create a new Label Studio project with YOLO configuration"""
    
    config = get_config()
    
    # Check if API key is missing or placeholder
    if not config.ls_api_key or config.ls_api_key == "YOUR_API_KEY_HERE":
        api_key = prompt_for_api_key(config)
        # Reload config to get the updated key
        config = get_config()
    
    project_title = config.project_title
    allow_duplicate = config.allow_duplicate
    force = config.force
    print(f"🔗 Connecting to Label Studio at {config.ls_url}...")
    
    try:
        client = LabelStudio(base_url=config.ls_url, api_key=config.ls_api_key)        # Check for existing projects with the same title
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
        
        # prepare labels
        LABEL_CONFIG = build_label_config(config.labels)

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
        
        if "401" in error_msg or "Unauthorized" in error_msg or "Token is invalid" in error_msg:
            print("\n🔑 Authentication failed. Your API token may be invalid or expired.")
            print("\n💡 Let's get a new API token...")
            
            # Prompt for new API key
            api_key = prompt_for_api_key(config)
            
            # Retry with new API key
            print(f"\n🔄 Retrying with new API token...")
            try:
                client = LabelStudio(base_url=config.ls_url, api_key=api_key)
                
                # Repeat the project creation logic with new client
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
                
            except Exception as retry_e:
                print(f"❌ Still failed after updating API key: {retry_e}")
                sys.exit(1)
        else:
            if "Connection" in error_msg or "ConnectionError" in error_msg or "refused" in error_msg:
                print(f"\n🔌 Connection failed - Label Studio is not running at {config.ls_url}")
                print("\n📋 To start Label Studio:")
                print("   Option 1 (Docker): docker compose up -d")
                print("   Option 2 (Local):  python3 scripts/1_start_labelstudio.py")
                print("   Option 3 (Manual): label-studio start")
                print(f"\n💡 Then verify it's accessible at {config.ls_url}")
            else:
                print(f"\n💡 Note: Check that Label Studio is running and accessible at {config.ls_url}")
            
        sys.exit(1)

if __name__ == "__main__":

    create_project()
