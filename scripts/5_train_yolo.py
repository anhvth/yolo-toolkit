#!/usr/bin/env python3
"""
Script 5: Train YOLO Model
Fine-tunes a YOLO model using exported annotations from Label Studio.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from ultralytics import YOLO

# Load environment variables
load_dotenv()

BASE_MODEL_PATH = os.getenv("BASE_MODEL_PATH", "models/base_model.pt")
UPDATED_MODEL_PATH = os.getenv("UPDATED_MODEL_PATH", "models/updated_model.pt")
EPOCHS = int(os.getenv("EPOCHS", "30"))
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "640"))

def train_yolo(data_yaml, base_model=None, output_model=None, epochs=None, imgsz=None):
    """Train YOLO model on exported dataset"""
    
    base_model = base_model or BASE_MODEL_PATH
    output_model = output_model or UPDATED_MODEL_PATH
    epochs = epochs or EPOCHS
    imgsz = imgsz or IMAGE_SIZE
    
    data_path = Path(data_yaml)
    if not data_path.exists():
        print(f"❌ Error: Data YAML file not found: {data_yaml}")
        print("\n💡 Make sure you've:")
        print("   1. Exported annotations using script 4")
        print("   2. Extracted the export.zip file")
        print("   3. Located the data.yaml file in the extracted folder")
        sys.exit(1)
    
    base_model_path = Path(base_model)
    
    # Download base model if it doesn't exist
    if not base_model_path.exists():
        print(f"📥 Base model not found. Downloading pretrained model...")
        base_model = "yolo11n.pt"  # Use nano model as default
        print(f"   Using: {base_model}")
    
    print(f"🚀 Starting YOLO training...")
    print(f"   Base model: {base_model}")
    print(f"   Data config: {data_yaml}")
    print(f"   Epochs: {epochs}")
    print(f"   Image size: {imgsz}")
    print()
    
    try:
        # Load model
        model = YOLO(base_model)
        
        # Train model
        results = model.train(
            data=str(data_path),
            epochs=epochs,
            imgsz=imgsz,
            project="runs/train",
            name="yolo_labelstudio",
            exist_ok=True
        )
        
        # Save trained model
        output_path = Path(output_model)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Copy best weights
        best_weights = Path("runs/train/yolo_labelstudio/weights/best.pt")
        if best_weights.exists():
            import shutil
            shutil.copy(best_weights, output_path)
            print(f"\n✅ Training completed successfully!")
            print(f"   Model saved to: {output_path}")
            print(f"   Training results: runs/train/yolo_labelstudio/")
        else:
            print(f"\n⚠️  Training completed but best.pt not found")
            print(f"   Check results in: runs/train/yolo_labelstudio/")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        sys.exit(1)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train YOLO model on labeled data")
    parser.add_argument("--data", required=True, help="Path to data.yaml file")
    parser.add_argument("--base-model", help="Base model path (default: yolo11n.pt)")
    parser.add_argument("--output", help="Output model path")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, help="Training image size")
    args = parser.parse_args()
    
    train_yolo(
        args.data,
        args.base_model,
        args.output,
        args.epochs,
        args.imgsz
    )
