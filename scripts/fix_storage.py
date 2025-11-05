#!/usr/bin/env python3
"""Fix local storage to use 'Tasks' import method instead of parsing files as JSON."""

import os
import requests
from dotenv import load_dotenv
from label_studio_sdk.client import LabelStudio

load_dotenv()

LABEL_STUDIO_URL = os.getenv("LABEL_STUDIO_URL", "http://localhost:8080")
LABEL_STUDIO_API_KEY = os.getenv("LABEL_STUDIO_API_KEY")
PROJECT_ID = int(os.getenv("PROJECT_ID", "4"))

if not LABEL_STUDIO_API_KEY:
    raise RuntimeError("LABEL_STUDIO_API_KEY is missing")

# Use SDK to get storage ID
ls = LabelStudio(base_url=LABEL_STUDIO_URL, api_key=LABEL_STUDIO_API_KEY)
storages = ls.import_storage.local.list(project=PROJECT_ID)

if not storages:
    print("❌ No local storage found for this project")
    exit(1)

storage = storages[0]  # Get the first one
storage_id = storage.id

print(f"📦 Found storage ID: {storage_id}")
print(f"   Path: {storage.path}")

# Use direct API call to update the storage with the missing parameter
# The SDK doesn't expose this field, but the API supports it
url = f"{LABEL_STUDIO_URL}/api/storages/localfiles/{storage_id}"
headers = {"Authorization": f"Token {LABEL_STUDIO_API_KEY}"}

# Patch the storage to disable "treat every bucket object as source file"
payload = {
    "treat_every_bucket_object_as_a_source_file": False
}

response = requests.patch(url, json=payload, headers=headers)

if response.status_code == 200:
    print("✅ Storage updated successfully")
    print("   Setting: treat_every_bucket_object_as_a_source_file = False")
    
    # Now sync
    print("\n🔄 Syncing storage...")
    ls.import_storage.local.sync(id=storage_id)
    print("✅ Sync complete!")
    
    project_url = f"{LABEL_STUDIO_URL}/projects/{PROJECT_ID}/"
    print(f"\n🔗 View project at: {project_url}")
else:
    print(f"❌ Failed to update storage: {response.status_code}")
    print(f"   Response: {response.text}")
