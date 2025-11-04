#!/usr/bin/env python3
"""
Script 4: Export Annotations in YOLO Format
Exports labeled data from Label Studio in YOLO format for training.
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
EXPORT_DIR = os.getenv("EXPORT_DIR", "data/exports")

def export_yolo(project_id=None, export_dir=None):
    """Export annotations from Label Studio in YOLO format"""
    
    if not API_KEY:
        print("❌ Error: LABEL_STUDIO_API_KEY not found in environment")
        sys.exit(1)
    
    project_id = project_id or PROJECT_ID
    export_dir = export_dir or EXPORT_DIR
    export_path = Path(export_dir)
    export_path.mkdir(parents=True, exist_ok=True)
    
    print(f"🔗 Connecting to Label Studio at {LS_URL}...")
    
    try:
        client = LabelStudio(base_url=LS_URL, api_key=API_KEY)
        
        print(f"📦 Exporting annotations from project {project_id}...")
        export_data = client.projects.exports.create(
            id=int(project_id),
            export_type="YOLO"
        )
        
        # Save export to file
        export_file = export_path / "export.zip"
        with open(export_file, "wb") as f:
            f.write(export_data)
        
        print(f"✅ Export completed successfully!")
        print(f"   Saved to: {export_file}")
        print(f"\n💡 Next steps:")
        print(f"   1. Extract the zip file: unzip {export_file}")
        print(f"   2. Review the data.yaml file for training configuration")
        print(f"   3. Run script 5 to train the YOLO model")
        
    except Exception as e:
        print(f"❌ Error exporting annotations: {e}")
        sys.exit(1)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Export YOLO format annotations")
    parser.add_argument("--project-id", help="Label Studio project ID")
    parser.add_argument("--export-dir", help="Directory to save exports")
    args = parser.parse_args()
    
    export_yolo(args.project_id, args.export_dir)
