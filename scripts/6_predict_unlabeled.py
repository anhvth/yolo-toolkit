#!/usr/bin/env python3
"""
Script 6: Generate Predictions for Unlabeled Images
Uses trained YOLO model to predict on unlabeled images and generate pre-annotations.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from ultralytics import YOLO

# Load environment variables
load_dotenv()

UPDATED_MODEL_PATH = os.getenv("UPDATED_MODEL_PATH", "models/updated_model.pt")
IMAGE_DIR = os.getenv("IMAGE_DIR", "data/images")
PREDICTIONS_DIR = os.getenv("PREDICTIONS_DIR", "data/predictions")

def predict_unlabeled(model_path=None, image_dir=None, output_dir=None, conf_threshold=0.25):
    """Generate predictions for unlabeled images"""
    
    model_path = model_path or UPDATED_MODEL_PATH
    image_dir = image_dir or IMAGE_DIR
    output_dir = output_dir or PREDICTIONS_DIR
    
    model_file = Path(model_path)
    if not model_file.exists():
        print(f"❌ Error: Model not found: {model_path}")
        print("\n💡 Train a model first using script 5")
        sys.exit(1)
    
    image_path = Path(image_dir)
    if not image_path.exists():
        print(f"❌ Error: Image directory not found: {image_dir}")
        sys.exit(1)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
    image_files = [
        f for f in image_path.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ]
    
    if not image_files:
        print(f"❌ No images found in {image_dir}")
        sys.exit(1)
    
    print(f"🔮 Loading model: {model_path}")
    print(f"📁 Found {len(image_files)} images")
    print(f"💾 Predictions will be saved to: {output_dir}")
    print()
    
    try:
        # Load model
        model = YOLO(model_path)
        
        # Run predictions
        print(f"🚀 Running predictions...")
        for i, img_file in enumerate(image_files, 1):
            print(f"   [{i}/{len(image_files)}] {img_file.name}")
            
            results = model.predict(
                source=str(img_file),
                conf=conf_threshold,
                save=True,
                save_txt=True,
                save_conf=True,
                project=str(output_path),
                name="predict",
                exist_ok=True
            )
        
        print(f"\n✅ Predictions completed!")
        print(f"   Results saved to: {output_path}/predict/")
        print(f"   - Images with bounding boxes: {output_path}/predict/")
        print(f"   - YOLO format labels: {output_path}/predict/labels/")
        print(f"\n💡 Next steps:")
        print(f"   1. Review predictions in {output_path}/predict/")
        print(f"   2. Import predictions to Label Studio for correction")
        print(f"   3. Export corrected annotations and retrain")
        
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        sys.exit(1)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Predict on unlabeled images")
    parser.add_argument("--model", help="Path to trained model")
    parser.add_argument("--image-dir", help="Directory containing images")
    parser.add_argument("--output-dir", help="Directory to save predictions")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    args = parser.parse_args()
    
    predict_unlabeled(
        args.model,
        args.image_dir,
        args.output_dir,
        args.conf
    )
