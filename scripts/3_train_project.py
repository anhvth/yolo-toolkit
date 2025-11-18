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


def convert_to_yolo_format(exported_json, output_dir="yolo_dataset", image_base_dir=None, train_split=0.8, filter_cls_ids=None):
    """
    Convert Label Studio JSON export to YOLO format with proper dataset structure.
    Only processes tasks with submitted annotations.
    
    Args:
        exported_json: List of tasks from Label Studio export (should be pre-filtered for submitted tasks)
        output_dir: Directory to save YOLO dataset
        image_base_dir: Base directory where images are stored (for symlinks)
        train_split: Fraction of data for training (default 0.8 = 80% train, 20% val)
        filter_cls_ids: List of class IDs to filter by (only include tasks with these classes)
    
    Returns:
        Path to data.yaml file
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

    # Filter tasks by class IDs if specified
    filtered_json = exported_json
    if filter_cls_ids is not None:
        # First pass: collect all labels to build label mapping
        for task in exported_json:
            for ann in task.get("annotations", []):
                for r in ann.get("result", []):
                    if r["type"] == "rectanglelabels":
                        label_name = r["value"]["rectanglelabels"][0]
                        if label_name not in labels:
                            labels[label_name] = next_class_id
                            next_class_id += 1
        
        # Second pass: filter tasks by class IDs
        filtered_json = []
        for task in exported_json:
            task_cls_ids = set()
            for ann in task.get("annotations", []):
                for r in ann.get("result", []):
                    if r["type"] == "rectanglelabels":
                        label_name = r["value"]["rectanglelabels"][0]
                        if label_name in labels:
                            task_cls_ids.add(labels[label_name])
            
            # Check if task contains any of the requested class IDs
            if task_cls_ids & set(filter_cls_ids):
                filtered_json.append(task)
        
        print(f"   🔍 Filtered by class IDs {filter_cls_ids}: {len(filtered_json)}/{len(exported_json)} tasks")
        if not filtered_json:
            print("⚠️  No tasks match the specified class IDs")
            sys.exit(1)
    
    # Split data
    import random
    random.seed(42)
    random.shuffle(filtered_json)
    split_idx = int(len(filtered_json) * train_split)
    
    for idx, task in enumerate(filtered_json):
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


def export_annotations(project_id, export_dir, image_base_dir, filter_cls_ids=None):
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
    
    # Remove old export directory if it exists
    if export_path.exists():
        import shutil
        print(f"🗑️  Removing old export directory: {export_path}")
        shutil.rmtree(export_path)
        print("   ✅ Old export directory removed")
    
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
        
        # Filter for tasks with submitted annotations only
        tasks_with_labels = []
        for task in data:
            # Check if task has annotations and they are completed
            if task.get("annotations"):
                for ann in task["annotations"]:
                    # Check if annotation is completed and has actual results
                    if not ann.get("was_cancelled", False) and ann.get("result"):
                        # Check if there are actual bounding boxes
                        has_boxes = any(
                            r.get("type") == "rectanglelabels" 
                            for r in ann.get("result", [])
                        )
                        if has_boxes:
                            tasks_with_labels.append(task)
                            break  # One valid annotation is enough
        
        if not tasks_with_labels:
            print("⚠️  No submitted annotations found")
            print(f"   Total tasks: {len(data)}, Tasks with labels: 0")
            print("💡 Submit some annotations in Label Studio first")
            sys.exit(1)
        
        print(f"   📝 Tasks with submitted labels: {len(tasks_with_labels)}/{len(data)}")
        if len(tasks_with_labels) < len(data):
            print(f"   ⏭️  Skipping {len(data) - len(tasks_with_labels)} unlabeled tasks")
        
        # Convert to YOLO format
        yolo_dir = export_path / "yolo_dataset"
        print(f"\n🔄 Converting to YOLO dataset format...")
        data_yaml = convert_to_yolo_format(tasks_with_labels, str(yolo_dir), image_base_dir=image_base_dir, filter_cls_ids=filter_cls_ids)
        
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


def train_yolo(model_path, data_yaml, epochs, image_size, output_model_path, 
               lr0=1e-4, mosaic=0.5, batch=4, half=True, device=None, freeze_ratio=0.95, close_mosaic=10):
    """Train YOLO model with configurable parameters"""
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
    
    # Auto-detect device if not specified
    if device is None:
        device = 'mps' if torch.backends.mps.is_available() else 'cuda'
    
    print(f"\n🚀 Starting YOLO training...")
    print(f"   Model: {model_path}")
    print(f"   Data: {data_yaml}")
    print(f"   Epochs: {epochs}")
    print(f"   Image Size: {image_size}")
    print(f"   Learning Rate: {lr0}")
    print(f"   Mosaic: {mosaic}")
    print(f"   Close Mosaic: {close_mosaic}")
    print(f"   Batch Size: {batch}")
    print(f"   Half Precision: {half}")
    print(f"   Device: {device}")
    print(f"   Freeze Ratio: {freeze_ratio}")
    
    try:
        model = YOLO(model_path)
        def frozen_model(model, ratio: float):
            """
            Freeze ratio (0~1) of parameters.
            Example: ratio=0.8 -> freeze first 80% of params.
            """
            params = list(model.model.named_parameters())
            total = len(params)
            k = int(total * ratio)

            for i, (name, p) in enumerate(params):
                p.requires_grad = (i >= k)

            print(f"Frozen {k}/{total} params ({ratio*100:.1f}%)")
            return model
        
        # Apply model freezing if freeze_ratio > 0
        if freeze_ratio > 0:
            model = frozen_model(model, freeze_ratio)
        
        results = model.train(
            data=str(data_path),
            epochs=epochs,
            imgsz=image_size,
            project="runs/detect",
            name="train",
            exist_ok=True,
            lr0=lr0,
            mosaic=mosaic,
            close_mosaic=close_mosaic,
            batch=batch,
            half=half,
            device=device
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
  python scripts/4_train.py --project 2 --export-project-only
  python scripts/4_train.py --model yolo11n.pt --project 2 --cls-ids 0 1 2
        """
    )
    parser.add_argument("--model", help="YOLO model path (e.g., yolo11n.pt)")
    parser.add_argument("--project", type=int, required=True, help="Label Studio project ID")
    parser.add_argument("--epochs", type=int, help="Training epochs (default: from config)")
    parser.add_argument("--imgsz", type=int, help="Image size (default: from config)")
    parser.add_argument("--output", help="Output model path (default: from config)")
    parser.add_argument("--export-project-only", action="store_true", help="Only export annotations, skip training")
    parser.add_argument("--cls-ids", type=int, nargs="+", default=None, help="Filter tasks by class IDs (only include tasks with these classes)")
    
    # Training hyperparameters
    parser.add_argument("--lr0", type=float, default=1e-4, help="Initial learning rate (default: 1e-4)")
    parser.add_argument("--mosaic", type=float, default=0.5, help="Mosaic augmentation probability (default: 0.5)")
    parser.add_argument("--close-mosaic", type=int, default=2, help="Epochs to disable mosaic augmentation before end (default: 10)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--half", action="store_true", default=True, help="Use half precision (FP16) training (default: True)")
    parser.add_argument("--no-half", dest="half", action="store_false", help="Disable half precision training")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda/mps/cpu, default: auto-detect)")
    parser.add_argument("--freeze-ratio", type=float, default=0.5, help="Ratio of parameters to freeze (0-1, default: 0.5)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.export_project_only and not args.model:
        parser.error("--model is required unless --export-project-only is specified")
    
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
    if args.export_project_only:
        print("📦 Export Annotations Only")
    else:
        print("🎯 Export & Train Pipeline")
    print("=" * 60)
    
    # Step 1: Export annotations
    data_yaml = export_annotations(
        project_id=args.project,
        export_dir=config.export_dir,
        image_base_dir=config.image_dir,
        filter_cls_ids=args.cls_ids
    )
    
    if args.export_project_only:
        print("\n" + "=" * 60)
        print("✅ Export completed successfully!")
        print("=" * 60)
        print("💡 Next steps:")
        print(f"   - Dataset ready at: {Path(data_yaml).parent}")
        print(f"   - Train with: python scripts/3_train_project.py --model yolo11n.pt --project {args.project}")
        return
    
    # Step 2: Train model
    train_yolo(
        model_path=args.model,
        data_yaml=data_yaml,
        epochs=epochs,
        image_size=image_size,
        output_model_path=output_model,
        lr0=args.lr0,
        mosaic=args.mosaic,
        close_mosaic=args.close_mosaic,
        batch=args.batch,
        half=args.half,
        device=args.device,
        freeze_ratio=args.freeze_ratio
    )
    
    print("\n" + "=" * 60)
    print("✅ Pipeline completed successfully!")
    print("=" * 60)
    print("💡 Next steps:")
    print("   - Review results in runs/detect/train*/")
    print(f"   - Use model: {output_model}")
    # print(f"   - Run predictions: python scripts/4_predict_unlabeled.py --model runs/detect/train/weights/best.pt --project-id 7")
    print(f"   - Retrain with new data: python scripts/4_train.py --model {output_model} --project {args.project}")


if __name__ == "__main__":
    main()
