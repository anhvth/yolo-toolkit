#!/usr/bin/env python3
"""
Script 4: Export & Train - Export annotations from Label Studio and train YOLO model
Combines export and training into a single workflow.
Usage: python scripts/4_train.py --model yolo11n.pt --project 2
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from label_studio_sdk import LabelStudio
import torch
from label_studio_sdk_wrapper.config import get_config


def convert_to_yolo_format(exported_json, output_dir="yolo_dataset", image_base_dir=None, train_split=0.8):
    """
    Convert Label Studio JSON export to YOLO format with proper dataset structure.
    
    Args:
        exported_json: List of tasks from Label Studio export
        output_dir: Directory to save YOLO dataset
        image_base_dir: Base directory where images are stored (for symlinks)
        train_split: Fraction of data for training (default 0.8 = 80% train, 20% val)
    
    Returns:
        dict with labels mapping
    """
    output_path = Path(output_dir)
    
    # Create YOLO dataset structure
    train_images_dir = output_path / "images" / "train"
    val_images_dir = output_path / "images" / "val"
    train_labels_dir = output_path / "labels" / "train"
    val_labels_dir = output_path / "labels" / "val"
    
    for dir_path in [train_images_dir, val_images_dir, train_labels_dir, val_labels_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # Collect unique labels to assign class ids
    labels = {}
    next_class_id = 0
    train_count = 0
    val_count = 0

    # Split data
    import random
    random.seed(42)
    random.shuffle(exported_json)
    split_idx = int(len(exported_json) * train_split)
    
    for idx, task in enumerate(exported_json):
        is_train = idx < split_idx
        images_dir = train_images_dir if is_train else val_images_dir
        labels_dir = train_labels_dir if is_train else val_labels_dir
        
        image_path = task["data"]["image"]
        # extract filename from local path like "/data/local-files/?d=data/images/img.jpg"
        filename = os.path.basename(image_path.split("d=")[-1])
        label_file = os.path.splitext(filename)[0] + ".txt"

        # Find actual image file
        if image_base_dir:
            actual_image = Path(image_base_dir) / filename
        else:
            # Try to extract from path
            if "d=" in image_path:
                rel_path = image_path.split("d=")[-1]
                actual_image = Path.cwd() / rel_path
            else:
                actual_image = Path(image_path)
        
        # Create symlink to image
        symlink_target = images_dir / filename
        if actual_image.exists():
            if symlink_target.exists() or symlink_target.is_symlink():
                symlink_target.unlink()
            symlink_target.symlink_to(actual_image.absolute())
        
        yolo_lines = []
        for ann in task.get("annotations", []):
            for r in ann.get("result", []):
                if r["type"] != "rectanglelabels":
                    continue

                label_name = r["value"]["rectanglelabels"][0]
                if label_name not in labels:
                    labels[label_name] = next_class_id
                    next_class_id += 1

                bbox = r["value"]
                x = bbox["x"] / 100
                y = bbox["y"] / 100
                w = bbox["width"] / 100
                h = bbox["height"] / 100
                x_center = x + w / 2
                y_center = y + h / 2

                cls_id = labels[label_name]
                yolo_lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

        if yolo_lines:
            with open(labels_dir / label_file, "w") as f:
                f.write("\n".join(yolo_lines))
            if is_train:
                train_count += 1
            else:
                val_count += 1

    # Create data.yaml
    yaml_content = f"""# YOLO dataset configuration
path: {output_path.absolute()}
train: images/train
val: images/val

# Classes
nc: {len(labels)}
names: {list(labels.keys())}
"""
    
    with open(output_path / "data.yaml", "w") as f:
        f.write(yaml_content)
    
    # Save label mapping for reference
    with open(output_path / "classes.txt", "w") as f:
        for label, idx in sorted(labels.items(), key=lambda x: x[1]):
            f.write(f"{idx}: {label}\n")

    print(f"✅ Created YOLO dataset in '{output_dir}'")
    print(f"   📊 Train: {train_count} images | Val: {val_count} images")
    print(f"   📋 Classes ({len(labels)}): {', '.join(labels.keys())}")
    
    return str(output_path / "data.yaml")


def export_annotations(project_id, export_dir, image_base_dir):
    """Export annotations from Label Studio"""
    try:
        config = get_config()
    except FileNotFoundError as e:
        print(str(e))
        sys.exit(1)
    except json.JSONDecodeError:
        print("❌ Error: Invalid JSON in ls_settings.json")
        print("💡 Check the JSON syntax in your settings file")
        sys.exit(1)
    
    if not config.ls_api_key:
        print("❌ Error: LABEL_STUDIO_API_KEY not set in ls_settings.json")
        print("💡 Get your API key from Label Studio UI → Account & Settings → Access Token")
        sys.exit(1)
    
    export_path = Path(export_dir)
    export_path.mkdir(parents=True, exist_ok=True)
    
    print(f"🔗 Connecting to Label Studio at {config.ls_url}...")
    
    try:
        client = LabelStudio(base_url=config.ls_url, api_key=config.ls_api_key)
        print(f"📦 Exporting annotations from project {project_id}...")
        
        # Create export job
        export_job = client.projects.exports.create(id=project_id, title="YOLO Export")
        export_id = export_job.id
        
        # Wait for completion
        print("   Waiting for export to complete...", end="", flush=True)
        while client.projects.exports.get(id=project_id, export_pk=export_id).status != 'completed':
            print(".", end="", flush=True)
            time.sleep(1)
        print(" Done!")
        
        # Download JSON
        json_file = export_path / f"project_{project_id}_{export_id}.json"
        with open(json_file, "wb") as f:
            for chunk in client.projects.exports.download(
                id=project_id,
                export_pk=export_id,
                export_type="JSON",
                request_options={"chunk_size": 1024},
            ):
                f.write(chunk)
        
        # Load data
        with open(json_file, "r") as f:
            data = json.load(f)
        
        if not data:
            print("⚠️  No annotations found to export")
            print("💡 Label some images in Label Studio first")
            sys.exit(1)
        
        print(f"✅ Downloaded {len(data)} tasks")
        
        # Convert to YOLO format
        yolo_dir = export_path / "yolo_dataset"
        print(f"\n🔄 Converting to YOLO dataset format...")
        data_yaml = convert_to_yolo_format(data, str(yolo_dir), image_base_dir=image_base_dir)
        
        return data_yaml
        
    except Exception as e:
        print(f"❌ Error exporting annotations: {e}")
        if "401" in str(e).lower():
            print("💡 Check your API key - it may be expired or incorrect")
        elif "404" in str(e).lower():
            print("💡 Check the project ID - the project may not exist")
        elif "connection" in str(e).lower():
            print("💡 Check Label Studio URL and ensure the server is running")
        sys.exit(1)


def train_yolo(model_path, data_yaml, epochs, image_size, output_model_path):
    """Train YOLO model"""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("❌ Error: ultralytics package not installed")
        print("💡 Install it with: pip install ultralytics")
        sys.exit(1)
    
    data_path = Path(data_yaml)
    if not data_path.exists():
        print(f"❌ Error: {data_yaml} not found!")
        sys.exit(1)
    
    print(f"\n🚀 Starting YOLO training...")
    print(f"   Model: {model_path}")
    print(f"   Data: {data_yaml}")
    print(f"   Epochs: {epochs}")
    print(f"   Image Size: {image_size}")
    
    try:
        model = YOLO(model_path)
        results = model.train(
            data=str(data_path),
            epochs=epochs,
            imgsz=image_size,
            project="runs/detect",
            name="train",
            exist_ok=True,
            device='mps' if torch.backends.mps.is_available() else 'cpu'
        )
        
        print("\n✅ Training completed successfully!")
        
        # Find the latest training run
        runs_dir = Path("runs/detect")
        train_dirs = sorted(runs_dir.glob("train*"), key=lambda x: x.stat().st_mtime)
        if train_dirs:
            latest_run = train_dirs[-1]
            best_model = latest_run / "weights" / "best.pt"
            
            if best_model.exists():
                # Copy to output path
                output_path = Path(output_model_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                import shutil
                shutil.copy(best_model, output_path)
                print(f"📋 Best model saved to: {output_path}")
                print(f"📁 Training results: {latest_run}")
            
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Export Label Studio annotations and train YOLO model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/4_train.py --model yolo11n.pt --project 2
  python scripts/4_train.py --model yolo11n.pt --project 2 --epochs 50
        """
    )
    parser.add_argument("--model", required=True, help="YOLO model path (e.g., yolo11n.pt)")
    parser.add_argument("--project", type=int, required=True, help="Label Studio project ID")
    parser.add_argument("--epochs", type=int, help="Training epochs (default: from config)")
    parser.add_argument("--imgsz", type=int, help="Image size (default: from config)")
    parser.add_argument("--output", help="Output model path (default: from config)")
    
    args = parser.parse_args()
    
    # Load config
    try:
        config = get_config()
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        sys.exit(1)
    
    # Use config defaults if not specified
    epochs = args.epochs or config.epochs
    image_size = args.imgsz or config.image_size
    output_model = args.output or config.updated_model_path
    
    print("=" * 60)
    print("🎯 Export & Train Pipeline")
    print("=" * 60)
    
    # Step 1: Export annotations
    data_yaml = export_annotations(
        project_id=args.project,
        export_dir=config.export_dir,
        image_base_dir=config.image_dir
    )
    
    # Step 2: Train model
    train_yolo(
        model_path=args.model,
        data_yaml=data_yaml,
        epochs=epochs,
        image_size=image_size,
        output_model_path=output_model
    )
    
    print("\n" + "=" * 60)
    print("✅ Pipeline completed successfully!")
    print("=" * 60)
    print(f"💡 Next steps:")
    print(f"   - Review results in runs/detect/train*/")
    print(f"   - Use model: {output_model}")
    print(f"   - Run predictions: python scripts/6_predict_unlabeled.py")


if __name__ == "__main__":
    main()
