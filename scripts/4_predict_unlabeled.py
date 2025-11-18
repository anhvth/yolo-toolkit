#!/usr/bin/env python3
"""
Script 4: Generate Predictions for Unlabeled Images and Upload to Label Studio
Uses trained YOLO model to predict on unlabeled tasks and upload pre-annotations.
"""

import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any
from ultralytics import YOLO
from label_studio_sdk import LabelStudio
from label_studio_sdk_wrapper.config import get_config
from tqdm import tqdm

def is_submitted(task):
    """Return True if the task has manual annotation (submitted by user)."""
    return bool(task.is_labeled or getattr(task, "annotations", []))

def get_unlabeled_tasks(client: LabelStudio, project_id: int) -> List[Dict[str, Any]]:
    """
    Get all tasks from the project that don't have submitted annotations.
    
    Args:
        client: Label Studio client
        project_id: Project ID
    
    Returns:
        List of unlabeled tasks
    """
    print(f"📋 Fetching tasks from project {project_id}...")
    
    # Get all tasks - convert paginator to list
    tasks_paginator = client.tasks.list(project=project_id)
    tasks = list(tasks_paginator)
    
    unlabeled_tasks = []
    for task in tasks:
        # Check if task has any submitted annotations
        if is_submitted(task):
            continue
        has_annotation = False
        if hasattr(task, 'annotations') and task.annotations:
            for ann in task.annotations:
                # Check if annotation is completed and has actual results
                if (hasattr(ann, 'was_cancelled') and not ann.was_cancelled and 
                    hasattr(ann, 'result') and ann.result):
                    # Check if there are actual bounding boxes
                    has_boxes = any(
                        r.get("type") == "rectanglelabels" 
                        for r in ann.result
                    )
                    if has_boxes:
                        has_annotation = True
                        break
        
        if not has_annotation:
            unlabeled_tasks.append(task)
    
    print(f"   Total tasks: {len(tasks)}")
    print(f"   Unlabeled tasks: {len(unlabeled_tasks)}")
    return unlabeled_tasks


def yolo_to_ls_format(yolo_boxes: List[tuple], img_width: int, img_height: int, 
                       class_names: List[str]) -> List[Dict[str, Any]]:
    """
    Convert YOLO predictions to Label Studio format.
    
    Args:
        yolo_boxes: List of (class_id, confidence, x_center, y_center, width, height) in normalized coords
        img_width: Original image width
        img_height: Original image height
        class_names: List of class names
    
    Returns:
        List of Label Studio prediction results
    """
    results = []
    
    for box in yolo_boxes:
        class_id, conf, x_center, y_center, width, height = box
        
        # Convert from YOLO format (center coords, normalized) to LS format (top-left, percentage)
        x = (x_center - width / 2) * 100  # Convert to percentage
        y = (y_center - height / 2) * 100
        w = width * 100
        h = height * 100
        
        # Ensure bounds
        x = max(0, min(100, x))
        y = max(0, min(100, y))
        w = max(0, min(100 - x, w))
        h = max(0, min(100 - y, h))
        
        result = {
            "type": "rectanglelabels",
            "value": {
                "x": x,
                "y": y,
                "width": w,
                "height": h,
                "rotation": 0,
                "rectanglelabels": [class_names[int(class_id)]]
            },
            "to_name": "image",
            "from_name": "label",
            "image_rotation": 0,
            "original_width": img_width,
            "original_height": img_height
        }
        results.append(result)
    
    return results


def predict_unlabeled(model_path=None, project_id=None, conf_threshold=None, upload=True, max_boxes=None):
    """
    Generate predictions for unlabeled tasks and optionally upload to Label Studio.
    
    Args:
        model_path: Path to trained YOLO model
        project_id: Label Studio project ID
        conf_threshold: Confidence threshold for predictions
        upload: Whether to upload predictions to Label Studio
        max_boxes: Maximum number of boxes to keep per image (highest confidence), None = unlimited
    """
    
    config = get_config()
    
    model_path = model_path or config.updated_model_path
    project_id = project_id or config.project_id
    conf_threshold = conf_threshold or config.model_score_threshold
    
    # Validate inputs
    if not project_id:
        print("❌ Error: PROJECT_ID not set")
        print("\n📋 Steps to fix:")
        print("   1. Run script 2 to create a project")
        print("   2. The project ID will be saved to ls_settings.json")
        sys.exit(1)
    
    model_file = Path(model_path)
    if not model_file.exists():
        print(f"❌ Error: Model not found: {model_path}")
        print("\n💡 Train a model first using script 3")
        sys.exit(1)
    
    print(f"🔗 Connecting to Label Studio at {config.ls_url}...")
    
    try:
        client = LabelStudio(base_url=config.ls_url, api_key=config.ls_api_key)
        
        # Get unlabeled tasks
        unlabeled_tasks = get_unlabeled_tasks(client, project_id)
        
        if not unlabeled_tasks:
            print("\n✅ No unlabeled tasks found - all tasks have annotations!")
            print("💡 Upload more images or create a new project to continue labeling")
            return
        
        print(f"\n📊 Found {len(unlabeled_tasks)} unsubmitted tasks to predict")
        
        print(f"\n🔮 Loading model: {model_path}")
        model = YOLO(model_path)
        
        # Get class names from model
        class_names = model.names  # Dict: {0: 'class1', 1: 'class2', ...}
        class_list = [class_names[i] for i in sorted(class_names.keys())]
        print(f"📝 Classes: {', '.join(class_list)}")
        print(f"🎯 Confidence threshold: {conf_threshold}")
        if max_boxes:
            print(f"📦 Max boxes per image: {max_boxes}")
        
        # Process each unlabeled task
        print(f"\n🚀 Running predictions on {len(unlabeled_tasks)} unlabeled tasks...")
        successful_uploads = 0
        failed_uploads = 0
        
        for task in tqdm(unlabeled_tasks, desc="Predicting", unit="task"):
            try:
                task_id = task.id if hasattr(task, 'id') else task.get('id')
                # Get image path from task data
                task_data = task.data if hasattr(task, 'data') else {}
                image_url = task_data.get('image', '')
                
                # Extract filename from URL
                # Format: /data/local-files/?d=data/images/filename.jpg or /data/images/filename.jpg
                if '/data/local-files/?d=' in image_url:
                    image_rel_path = image_url.split('/data/local-files/?d=')[-1]
                elif '/data/images/' in image_url:
                    image_rel_path = image_url.split('/data/images/')[-1]
                    image_rel_path = f"data/images/{image_rel_path}"
                else:
                    tqdm.write(f"   ⚠️  Skipping - unknown image format: {image_url}")
                    failed_uploads += 1
                    continue
                
                # Construct full path (relative to project root or absolute)
                image_path = Path(image_rel_path)
                if not image_path.exists():
                    # Try from project root
                    project_root = Path(__file__).parent.parent
                    image_path = project_root / image_rel_path
                
                if not image_path.exists():
                    tqdm.write(f"   ⚠️  Image not found: {image_path}")
                    failed_uploads += 1
                    continue
                
                # Run prediction
                results = model.predict(
                    source=str(image_path),
                    conf=conf_threshold,
                    verbose=False
                )
                
                if not results or len(results) == 0:
                    tqdm.write(f"#{task_id} (no detections)")
                    continue
                
                result = results[0]
                
                # Get image dimensions
                img_height, img_width = result.orig_shape
                
                # Extract boxes in YOLO format
                yolo_boxes = []
                if result.boxes is not None and len(result.boxes) > 0:
                    for box in result.boxes:
                        # box.xywhn: normalized [x_center, y_center, width, height]
                        if hasattr(box, 'xywhn') and hasattr(box, 'cls') and hasattr(box, 'conf'):
                            x_center, y_center, width, height = box.xywhn[0].tolist()
                            class_id = int(box.cls[0])
                            confidence = float(box.conf[0])
                            yolo_boxes.append((class_id, confidence, x_center, y_center, width, height))
                
                # Sort by confidence and limit to max_boxes if specified
                if max_boxes and len(yolo_boxes) > max_boxes:
                    yolo_boxes = sorted(yolo_boxes, key=lambda x: x[1], reverse=True)[:max_boxes]
                
                if not yolo_boxes:
                    tqdm.write("(no detections)")
                    continue
                
                # Convert to Label Studio format
                ls_results = yolo_to_ls_format(yolo_boxes, img_width, img_height, class_list)
                
                detail_msg = f"#{task_id} {image_path.name} ({len(ls_results)} boxes)"
                
                if upload:
                    # Create prediction
                    prediction_data = {
                        "result": ls_results,
                        "score": sum(box[1] for box in yolo_boxes) / len(yolo_boxes),  # Average confidence
                        "model_version": str(model_path)
                    }
                    
                    # Upload prediction to Label Studio
                    task_id = task.id if hasattr(task, 'id') else task.get('id')
                    client.predictions.create(
                        task=task_id,
                        **prediction_data
                    )
                    tqdm.write(detail_msg + " ✓")
                    successful_uploads += 1
                else:
                    tqdm.write(detail_msg)
                
            except Exception as e:
                tqdm.write(f" ✗ Error: {e}")
                failed_uploads += 1
        
        # Summary
        print(f"\n{'='*60}")
        print("📊 PREDICTION SUMMARY")
        print(f"{'='*60}")
        print(f"✅ Successfully predicted and uploaded: {successful_uploads}")
        if failed_uploads > 0:
            print(f"❌ Failed: {failed_uploads}")
        print(f"\n🔗 View predictions at: {config.ls_url}/projects/{project_id}")
        print("\n💡 Next steps:")
        print("   1. Review predictions in Label Studio")
        print("   2. Correct any errors in the pre-annotations")
        print("   3. Submit the corrected annotations")
        print("   4. Export and retrain (script 3) for improved accuracy")
        
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Error: {error_msg}")
        
        if "401" in error_msg or "Unauthorized" in error_msg:
            print("\n🔑 Authentication failed - API key is invalid or expired")
            print("\n📋 To fix:")
            print("   1. Go to Label Studio → Account & Settings → Access Token")
            print("   2. Generate a new API token")
            print("   3. Update LABEL_STUDIO_API_KEY in ls_settings.json")
        elif "Connection" in error_msg or "refused" in error_msg:
            print(f"\n🔌 Connection failed - Label Studio is not running at {config.ls_url}")
            print("\n📋 To start Label Studio:")
            print("   Option 1 (Docker): docker compose up -d")
            print("   Option 2 (Local):  python scripts/1_start_labelstudio.py")
        
        sys.exit(1)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Predict on unlabeled tasks and upload to Label Studio"
    )
    parser.add_argument("--model", help="Path to trained YOLO model")
    parser.add_argument("--project-id", type=int, help="Label Studio project ID")
    parser.add_argument("--conf", type=float, help="Confidence threshold")
    parser.add_argument("--max-boxes", type=int, help="Maximum number of boxes to keep per image (highest confidence)")
    parser.add_argument("--no-upload", action="store_true", 
                        help="Don't upload predictions to Label Studio (prediction only)")
    args = parser.parse_args()
    
    predict_unlabeled(
        model_path=args.model,
        project_id=args.project_id,
        conf_threshold=args.conf,
        upload=not args.no_upload,
        max_boxes=args.max_boxes
    )

