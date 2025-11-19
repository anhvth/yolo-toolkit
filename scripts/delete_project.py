#!/usr/bin/env python3
"""
Helper Script: Delete Label Studio Project
Deletes a Label Studio project by ID or title.
"""

import sys
from pathlib import Path
from label_studio_sdk import LabelStudio
from label_studio_sdk_wrapper.config import get_config

def delete_project(project_id=None, project_title=None, confirm=True):
    """Delete a Label Studio project"""
    
    config = get_config()
    
    if not config.ls_api_key:
        print("❌ Error: LABEL_STUDIO_API_KEY not found in ls_settings.yaml")
        sys.exit(1)
    
    if not project_id and not project_title:
        print("❌ Error: Either --id or --title must be provided")
        sys.exit(1)
    
    print(f"🔗 Connecting to Label Studio at {config.ls_url}...")
    
    try:
        client = LabelStudio(base_url=config.ls_url, api_key=config.ls_api_key)
        
        # If title provided, find project by title
        if project_title and not project_id:
            print(f"🔍 Searching for project: '{project_title}'")
            projects = list(client.projects.list(title=project_title))
            
            if not projects:
                print(f"❌ No project found with title: '{project_title}'")
                sys.exit(1)
            
            if len(projects) > 1:
                print(f"⚠️  Found {len(projects)} projects with title '{project_title}':")
                for proj in projects:
                    print(f"   - ID: {proj.id}, Title: {proj.title}")
                print("\n💡 Please specify project ID using --id")
                sys.exit(1)
            
            project_id = projects[0].id
            project_title = projects[0].title
        
        # Get project details
        if not project_title:
            project = client.projects.get(id=int(project_id))
            project_title = project.title
        
        # Confirm deletion
        if confirm:
            print(f"\n⚠️  WARNING: This will permanently delete the project!")
            print(f"   Project ID: {project_id}")
            print(f"   Title: {project_title}")
            print(f"   URL: {config.ls_url}/projects/{project_id}")
            
            response = input("\nType 'yes' to confirm deletion: ")
            if response.lower() != 'yes':
                print("❌ Deletion cancelled")
                sys.exit(0)
        
        # Delete project
        print(f"\n🗑️  Deleting project {project_id}...")
        client.projects.delete(id=int(project_id))
        
        print(f"✅ Project deleted successfully!")
        print(f"   Project ID: {project_id}")
        print(f"   Title: {project_title}")
        
    except Exception as e:
        print(f"❌ Error deleting project: {e}")
        sys.exit(1)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Delete Label Studio project")
    parser.add_argument("--id", type=int, help="Project ID to delete")
    parser.add_argument("--title", help="Project title to search and delete")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation prompt")
    args = parser.parse_args()
    
    delete_project(args.id, args.title, confirm=not args.yes)
