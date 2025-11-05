#!/bin/bash

# 0. setup everything in ls_settings.json
# 1. create project
# 2. upload images to Label Studio
echo "All information will be read from ls_settings.json"

echo "1. Create Project"
python scripts/2_create_project.py

echo "2. Upload Images"
python scripts/3_upload_images.py