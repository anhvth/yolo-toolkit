#!/usr/bin/env python3
"""
Helper Script: Label Studio Project Management
Provides utilities to list and delete Label Studio projects.

Usage:
    List all projects:
        python scripts/helper.py --list-projects
    
    Delete project by ID:
        python scripts/helper.py --delete-project 123
    
    Delete project by name:
        python scripts/helper.py --delete-project "My Project"
    
    Delete without confirmation:
        python scripts/helper.py --delete-project 123 --yes
"""

import sys
from pathlib import Path
from label_studio_sdk import LabelStudio
from label_studio_sdk_wrapper.config import get_config


def list_projects(client, config):
    """List all Label Studio projects"""
    print(f"🔗 Connecting to Label Studio at {config.ls_url}...\n")
    
    try:
        projects = list(client.projects.list())
        
        if not projects:
            print("📋 No projects found.")
            return
        
        print(f"📋 Found {len(projects)} project(s):\n")
        print(f"{'ID':<8} {'Title':<40} {'Tasks':<8} {'Created'}")
        print("=" * 90)
        
        for proj in projects:
            # Get task count
            tasks = list(client.tasks.list(project=proj.id))
            task_count = len(tasks)
            
            # Format created date if available
            created = getattr(proj, 'created_at', 'N/A')
            if created and created != 'N/A':
                try:
                    # Try to format the date nicely
                    from datetime import datetime
                    if isinstance(created, str):
                        dt = datetime.fromisoformat(created.replace('Z', '+00:00'))
                        created = dt.strftime('%Y-%m-%d')
                except:
                    pass
            
            # Truncate title if too long
            title = proj.title
            if len(title) > 38:
                title = title[:35] + "..."
            
            print(f"{proj.id:<8} {title:<40} {task_count:<8} {created}")
            print(f"         URL: {config.ls_url}/projects/{proj.id}")
        
        print()
        
    except Exception as e:
        print(f"❌ Error listing projects: {e}")
        sys.exit(1)


def delete_project_by_id_or_name(client, config, identifier, confirm=True):
    """Delete a project by ID (int) or name (str)"""
    
    try:
        # Try to parse as integer (ID)
        try:
            project_id = int(identifier)
            is_id = True
        except ValueError:
            # It's a name/title
            project_id = None
            project_title = identifier
            is_id = False
        
        # If identifier is a name, search for it
        if not is_id:
            print(f"🔍 Searching for project: '{project_title}'")
            projects = list(client.projects.list(title=project_title))
            
            if not projects:
                print(f"❌ No project found with title: '{project_title}'")
                sys.exit(1)
            
            if len(projects) > 1:
                print(f"⚠️  Found {len(projects)} projects with title '{project_title}':")
                for proj in projects:
                    print(f"   - ID: {proj.id}, Title: {proj.title}")
                    print(f"     URL: {config.ls_url}/projects/{proj.id}")
                print("\n💡 Please specify project ID instead")
                sys.exit(1)
            
            project_id = projects[0].id
            project_title = projects[0].title
        else:
            # Get project details by ID
            try:
                project = client.projects.get(id=project_id)
                project_title = project.title
            except Exception as e:
                print(f"❌ Project with ID {project_id} not found: {e}")
                sys.exit(1)
        
        # Confirm deletion
        if confirm:
            print("\n⚠️  WARNING: This will permanently delete the project!")
            print(f"   Project ID: {project_id}")
            print(f"   Title: {project_title}")
            print(f"   URL: {config.ls_url}/projects/{project_id}")
            
            response = input("\nType 'yes' to confirm deletion: ")
            if response.lower() != 'yes':
                print("❌ Deletion cancelled")
                sys.exit(0)
        
        # Delete project
        print(f"\n🗑️  Deleting project {project_id}...")
        client.projects.delete(id=project_id)
        
        print("✅ Project deleted successfully!")
        print(f"   Project ID: {project_id}")
        print(f"   Title: {project_title}")
        
    except Exception as e:
        print(f"❌ Error deleting project: {e}")
        sys.exit(1)


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Label Studio Project Management Helper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  List all projects:
    python scripts/helper.py --list-projects
  
  Delete project by ID:
    python scripts/helper.py --delete-project 123
  
  Delete project by name:
    python scripts/helper.py --delete-project "My Project"
  
  Delete without confirmation:
    python scripts/helper.py --delete-project 123 --yes
        """
    )
    
    parser.add_argument("--list-projects", action="store_true",
                       help="List all Label Studio projects")
    parser.add_argument("--delete-project", metavar="ID_OR_NAME",
                       help="Delete project by ID (number) or name (string)")
    parser.add_argument("--yes", action="store_true",
                       help="Skip confirmation prompt for deletion")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.list_projects and not args.delete_project:
        parser.print_help()
        sys.exit(1)
    
    # Get configuration
    config = get_config()
    
    if not config.ls_api_key:
        print("❌ Error: LABEL_STUDIO_API_KEY not found in ls_settings.json")
        print("\n📋 Steps to fix:")
        print("   1. Open Label Studio: http://localhost:8080")
        print("   2. Go to Account & Settings → Access Token")
        print("   3. Copy your API key")
        print("   4. Update ls_api_key in ls_settings.json")
        sys.exit(1)
    
    print(f"🔗 Connecting to Label Studio at {config.ls_url}...")
    
    try:
        client = LabelStudio(base_url=config.ls_url, api_key=config.ls_api_key)
        
        # Execute requested action
        if args.list_projects:
            list_projects(client, config)
        
        if args.delete_project:
            delete_project_by_id_or_name(
                client, 
                config, 
                args.delete_project, 
                confirm=not args.yes
            )
    
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Error: {error_msg}")
        
        if "401" in error_msg or "Unauthorized" in error_msg:
            print("\n🔑 Authentication failed - your API token may be invalid or expired")
            print("\n📋 To get a new token:")
            print("   1. Open: http://localhost:8080/user/account")
            print("   2. Go to 'Access Token' tab")
            print("   3. Click 'Reset Token' if needed")
            print("   4. Update ls_api_key in ls_settings.json")
        
        sys.exit(1)


if __name__ == "__main__":
    main()
