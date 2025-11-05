#!/usr/bin/env python3
"""
Script 4: Export Annotations in YOLO Format
Exports labeled data from Label Studio in YOLO format for training.
"""

import json
import os
import sys
import time
from pathlib import Path
from label_studio_sdk import LabelStudio
from config import get_config
from dotenv import load_dotenv
load_dotenv()

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
    converted_count = 0
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
            converted_count += 1
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
    print(f"   📊 Train: {train_count} images")
    print(f"   📊 Val: {val_count} images")
    print(f"   📋 Classes ({len(labels)}): {', '.join(labels.keys())}")
    print(f"   📝 Config: {output_path / 'data.yaml'}")
    
    return labels


def export_yolo(project_id=None, export_dir=None):
    """Export annotations from Label Studio in YOLO format"""
    
    config = get_config()
    
    if not config.ls_api_key:
        print("❌ Error: LABEL_STUDIO_API_KEY not found in ls_settings.json")
        sys.exit(1)
    
    project_id = project_id or config.project_id
    export_dir = export_dir or config.export_dir
    export_path = Path(export_dir)
    export_path.mkdir(parents=True, exist_ok=True)
    
    print(f"🔗 Connecting to Label Studio at {config.ls_url}...")
    
    try:
        client = LabelStudio(base_url=config.ls_url, api_key=config.ls_api_key)
        
        print(f"📦 Exporting annotations from project {project_id}...")
        
        # Snapshot-based export (your working code)
        export_job = client.projects.exports.create(id=project_id, title="YOLO Export")
        export_id = export_job.id
        
        # Poll until export_job.status becomes 'completed'
        print("   Waiting for export to complete...", end="", flush=True)
        while client.projects.exports.get(id=project_id, export_pk=export_id).status != 'completed':
            print(".", end="", flush=True)
            time.sleep(1)
        print(" Done!")
        
        # Download the snapshot as JSON
        json_file = export_path / f"project_{project_id}_{export_id}.json"
        print(f"   Downloading to {json_file}...")
        with open(json_file, "wb") as f:
            for chunk in client.projects.exports.download(
                id=project_id,
                export_pk=export_id,
                export_type="JSON",
                request_options={"chunk_size": 1024},
            ):
                f.write(chunk)
        
        # Load the JSON data
        print(f"   Loading JSON data...")
        with open(json_file, "r") as f:
            data = json.load(f)
        
        if not data:
            print("⚠️  No annotations found to export")
            sys.exit(1)
        
        print(f"✅ Downloaded {len(data)} tasks")
        
        # Convert to YOLO format with dataset structure
        yolo_dir = export_path / "yolo_dataset"
        print(f"\n🔄 Converting to YOLO dataset format...")
        
        # Get image base directory from config
        image_base_dir = config.image_dir
        labels = convert_to_yolo_format(data, str(yolo_dir), image_base_dir=image_base_dir)
        
        # Save JSON for reference
        json_export_file = export_path / "export.json"
        with open(json_export_file, "w") as f:
            json.dump(data, f, indent=2)
        print(f"💾 Saved JSON export to: {json_export_file}")
        
        print(f"\n✅ Export completed successfully!")
        print(f"   📁 Dataset: {yolo_dir}/")
        print(f"   📁 Structure:")
        print(f"      ├── images/")
        print(f"      │   ├── train/  (symlinks)")
        print(f"      │   └── val/    (symlinks)")
        print(f"      ├── labels/")
        print(f"      │   ├── train/")
        print(f"      │   └── val/")
        print(f"      ├── data.yaml")
        print(f"      └── classes.txt")
        
        print(f"\n💡 Next steps:")
        print(f"   1. Review the dataset: {yolo_dir}")
        print(f"   2. Train with: python scripts/5_train_yolo.py --data {yolo_dir}/data.yaml")
        
    except Exception as e:
        print(f"❌ Error exporting annotations: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Export YOLO format annotations")
    parser.add_argument("--project-id", type=int, help="Label Studio project ID")
    parser.add_argument("--export-dir", help="Directory to save exports")
    parser.add_argument("--train-split", type=float, default=0.8, help="Train/val split ratio (default: 0.8)")
    args = parser.parse_args()
    
    export_yolo(args.project_id, args.export_dir)
