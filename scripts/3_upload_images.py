import os
import argparse
import requests
from label_studio_sdk.client import LabelStudio
from config import get_config

def main():
    parser = argparse.ArgumentParser(description="Import local images to Label Studio (local storage)")
    parser.add_argument("--image_dir", help="Local image directory (overrides config)")
    args = parser.parse_args()

    config = get_config()
    
    LABEL_STUDIO_URL = config.ls_url
    LABEL_STUDIO_API_KEY = config.ls_api_key
    PROJECT_ID = config.project_id

    if not LABEL_STUDIO_API_KEY:
        raise RuntimeError("LABEL_STUDIO_API_KEY is missing in ls_settings.json")

    # Use command-line arg or config value
    image_dir = args.image_dir or config.image_dir
    
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Directory not found: {image_dir}")

    path = os.path.abspath(image_dir)

    # Client with the same class you used
    ls = LabelStudio(base_url=LABEL_STUDIO_URL, api_key=LABEL_STUDIO_API_KEY)

    # Check if this path already exists in the project's import storages
    existing_storages = ls.import_storage.local.list(project=PROJECT_ID)
    for existing in existing_storages:
        if existing.path == path:
            print(f"⚠️  Warning: Path '{path}' already exists in project {PROJECT_ID}")
            print(f"   Storage ID: {existing.id}")
            print(f"   Skipping duplicate import. Use a different path or delete the existing storage first.")
            return

    # Create local import storage (same API family you wrote)
    storage = ls.import_storage.local.create(
        project=PROJECT_ID,
        path=path,
        title=f"Images from {path}",
        use_blob_urls=True,
        regex_filter=r".*.(jpe?g|png|gif)$",
    )
    

    print(f"✅ Imported {path} into project {PROJECT_ID}")
    print(f"   Storage ID: {storage.id}")
    
    # After uploading - show project URL
    project_url = f"http://localhost:8080/projects/{PROJECT_ID}/settings/storage"
    print(f"🔗 View project at: {project_url}")

if __name__ == "__main__":
    main()