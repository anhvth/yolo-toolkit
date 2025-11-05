#!/usr/bin/env python3
"""Check the status of local storage."""

import os
from dotenv import load_dotenv
from label_studio_sdk.client import LabelStudio

load_dotenv()

LABEL_STUDIO_URL = os.getenv("LABEL_STUDIO_URL")
LABEL_STUDIO_API_KEY = os.getenv("LABEL_STUDIO_API_KEY")
PROJECT_ID = int(os.getenv("PROJECT_ID", "4"))

ls = LabelStudio(base_url=LABEL_STUDIO_URL, api_key=LABEL_STUDIO_API_KEY)

# Get all storages for the project
storages = list(ls.import_storage.local.list(project=PROJECT_ID))

if not storages:
    print(f"❌ No storages found for project {PROJECT_ID}")
else:
    for storage in storages:
        print(f"\n📦 Storage ID: {storage.id}")
        print(f"   Title: {storage.title}")
        print(f"   Path: {storage.path}")
        print(f"   Status: {storage.status}")
        print(f"   Last sync: {storage.last_sync}")
        print(f"   Synced tasks: {storage.synced}")
        
        # Check for the problematic field via __dict__
        if hasattr(storage, '__dict__'):
            if 'treat_every_bucket_object_as_a_source_file' in storage.__dict__:
                print(f"   Treat as source: {storage.__dict__['treat_every_bucket_object_as_a_source_file']}")
        
        # If status is failed, suggest syncing
        if storage.status == "failed":
            print(f"\n   ⚠️  Status is FAILED - check logs in UI")
            print(f"   You may need to manually change 'Import method' to 'Tasks' in:")
            print(f"   {LABEL_STUDIO_URL}/projects/{PROJECT_ID}/settings/cloud")
