#!/bin/bash


# Start Label Studio with the data directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
echo "Project root: $PROJECT_ROOT"
cd "$PROJECT_ROOT"
export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/mnt/disk1/khaipt4/project/yolo-toolkit/data/
label-studio start --username admin@example.com --password admin --data-dir label_studio_data

echo ""
echo "✅ Label Studio started"
echo "👤 Admin user: admin@example.com / admin"
echo "📋 To upload images, use: /data/local-files/?d=images/filename.jpg"
