#!/usr/bin/env python3
"""
Initialize and start Label Studio with local file serving enabled.

This script:
1. Sets up environment variables for local file serving
2. Starts Label Studio with admin credentials
3. Uses the project's data directory as the document root
"""

import os
import subprocess
import sys
from pathlib import Path


def get_project_root() -> Path:
    """Get the absolute path to the project root directory."""
    return Path(__file__).parents[1].absolute()


def setup_environment(project_root: Path) -> dict[str, str]:
    """
    Configure environment variables for Label Studio.
    
    Args:
        project_root: Path to the project root directory
        
    Returns:
        Dictionary of environment variables to set
    """
    data_dir = project_root# / "data"
    
    env_vars = {
        "LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED": "true",
        "LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT": str(data_dir),
    }
    
    return env_vars


def start_label_studio(
    project_root: Path,
    username: str = "admin@example.com",
    password: str = "admin",
    data_dir: str = "label_studio_data"
) -> None:
    """
    Start Label Studio server with specified configuration.
    
    Args:
        project_root: Path to the project root directory
        username: Admin username (default: admin@example.com)
        password: Admin password (default: admin)
        data_dir: Directory for Label Studio data (default: label_studio_data)
        
    Raises:
        FileNotFoundError: If label-studio command is not found
        subprocess.CalledProcessError: If label-studio fails to start
    """
    # Setup environment
    env_vars = setup_environment(project_root)
    env = os.environ.copy()
    env['LABEL_STUDIO_WORKER_COUNT'] = '16'
    env.update(env_vars)
    
    # Change to project root
    os.chdir(project_root)
    
    # Prepare command
    cmd = [
        "label-studio",
        "start",
        "--username", username,
        "--password", password,
        "--data-dir", data_dir,
    ]
    
    print("🚀 Starting Label Studio...")
    print(f"📁 Project root: {project_root}")
    print(f"📂 Data directory: {project_root / data_dir}")
    print(f"🗂️  Document root: {env_vars['LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT']}")
    print()
    
    try:
        # Start Label Studio
        subprocess.run(cmd, env=env, check=True)
    except FileNotFoundError:
        print("❌ Error: label-studio command not found")
        print("\n📋 Steps to fix:")
        print("1. Install Label Studio: pip install label-studio")
        print("2. Or activate your virtual environment")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: Label Studio failed to start: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n✅ Label Studio stopped")
        print("👤 Admin user: admin@example.com / admin")
        print("📋 To upload images, use: /data/local-files/?d=images/filename.jpg")
        sys.exit(0)


def main() -> None:
    """Main entry point for the script."""
    project_root = get_project_root()
    
    print("=" * 60)
    print("Label Studio Initialization")
    print("=" * 60)
    print()
    
    start_label_studio(project_root)


if __name__ == "__main__":
    main()
