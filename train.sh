#!/bin/bash

# Read configuration from ls_settings.json
CONFIG_FILE="ls_settings.json"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Error: $CONFIG_FILE not found!"
    exit 1
fi

# Extract YOLO configuration using jq
EPOCHS=$(jq -r '.yolo.epochs' "$CONFIG_FILE")
IMAGE_SIZE=$(jq -r '.yolo.image_size' "$CONFIG_FILE")
BATCH_SIZE=$(jq -r '.yolo.batch_size' "$CONFIG_FILE")
MODEL_THRESHOLD=$(jq -r '.yolo.model_score_threshold' "$CONFIG_FILE")

# Get the nano model path (default model)
MODEL=$(jq -r '.yolo.model_name' "$CONFIG_FILE")

# Get paths
DATA_PATH=$(jq -r '.paths.export_dir' "$CONFIG_FILE")/yolo_dataset/data.yaml
UPDATED_MODEL_PATH=$(jq -r '.paths.updated_model_path' "$CONFIG_FILE")

echo "🔧 Training Configuration:"
echo "   Model: $MODEL"
echo "   Data: $DATA_PATH"
echo "   Epochs: $EPOCHS"
echo "   Image Size: $IMAGE_SIZE"
echo "   Batch Size: $BATCH_SIZE"
echo "   Model Threshold: $MODEL_THRESHOLD"
echo "   Output: $UPDATED_MODEL_PATH"
echo ""

# Check if data.yaml exists
if [ ! -f "$DATA_PATH" ]; then
    echo "❌ Error: $DATA_PATH not found!"
    echo "💡 Run 'python scripts/4_export_yolo.py' first to export annotations"
    exit 1
fi

# Run YOLO training with config parameters
echo "🚀 Starting YOLO training..."
yolo train model="$MODEL" data="$DATA_PATH" epochs="$EPOCHS" imgsz="$IMAGE_SIZE" batch="$BATCH_SIZE" conf="$MODEL_THRESHOLD"

# Check if training was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Training completed successfully!"
    
    # Find the latest training run directory
    LATEST_RUN=$(find runs/detect -name "train*" -type d | sort -V | tail -1)
    
    if [ -d "$LATEST_RUN" ]; then
        echo "📁 Results saved to: $LATEST_RUN"
        
        # Copy best model to updated_model_path if it exists
        if [ -f "$LATEST_RUN/weights/best.pt" ]; then
            mkdir -p $(dirname "$UPDATED_MODEL_PATH")
            cp "$LATEST_RUN/weights/best.pt" "$UPDATED_MODEL_PATH"
            echo "📋 Best model copied to: $UPDATED_MODEL_PATH"
        fi
    fi
else
    echo "❌ Training failed!"
    exit 1
fi 