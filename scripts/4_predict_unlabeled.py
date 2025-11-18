#!/usr/bin/env python3
"""
Script 4: Generate Predictions for Unlabeled Images and Upload to Label Studio
Uses trained YOLO model to predict on unlabeled tasks and upload pre-annotations.
"""

import sys
import json
import time
import asyncio
import cv2
from pathlib import Path
from typing import List, Dict, Any
from urllib.parse import unquote
from ultralytics import YOLO
from label_studio_sdk import LabelStudio
from label_studio_sdk_wrapper.config import get_config
from label_studio_sdk_wrapper.inference_roi_moto_plate import REGISTERED_PIPELINES
from tqdm.asyncio import tqdm as async_tqdm

DEBUG_TASKS = [233]

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
    if DEBUG_TASKS:
        tasks = [t for t in tasks if t.id in DEBUG_TASKS]
        print(f"⚠️  DEBUG MODE: only processing tasks: {DEBUG_TASKS}")
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


async def upload_prediction_async(
    client: LabelStudio,
    task_id: int,
    prediction_data: Dict[str, Any]
) -> bool:
    """Upload a single prediction asynchronously."""
    try:
        await asyncio.to_thread(
            client.predictions.create,
            task=task_id,
            **prediction_data
        )
        return True
    except Exception as e:
        print(f"Failed to upload prediction for task {task_id}: {e}")
        return False


def predict_unlabeled(
    model_path=None,
    project_id=None,
    conf_threshold=None,
    upload=True,
    max_boxes=None,
    batch_size=10,
    pipeline_name=None
):
    """
    Generate predictions for unlabeled tasks and optionally upload to Label Studio.
    
    Args:
        model_path: Path to trained YOLO model
        project_id: Label Studio project ID
        conf_threshold: Confidence threshold for predictions
        upload: Whether to upload predictions to Label Studio
        max_boxes: Maximum number of boxes per image (highest confidence), None = unlimited
        batch_size: Number of concurrent uploads (default: 10)
        pipeline_name: Name of custom pipeline (e.g., 'yolo_plate'), None for standard YOLO
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
        
        # Load model - either custom pipeline or standard YOLO
        if pipeline_name:
            if pipeline_name not in REGISTERED_PIPELINES:
                print(f"❌ Unknown pipeline: {pipeline_name}")
                print(f"Available pipelines: {list(REGISTERED_PIPELINES.keys())}")
                sys.exit(1)
            
            pipeline_class = REGISTERED_PIPELINES[pipeline_name]
            model = pipeline_class(
                model_path=model_path,
                roi_conf_thres=conf_threshold,
                veh_conf_thres=conf_threshold
            )
            print(f"✅ Using custom pipeline: {pipeline_name}")
            # Get class names from pipeline
            class_names = model.class_id_to_name
            class_list = [class_names[i] for i in sorted(class_names.keys())]
            use_custom_pipeline = True
        else:
            model = YOLO(model_path)
            # Get class names from model
            class_names = model.names  # Dict: {0: 'class1', 1: 'class2', ...}
            class_list = [class_names[i] for i in sorted(class_names.keys())]
            use_custom_pipeline = False
        
        print(f"📝 Classes: {', '.join(class_list)}")
        print(f"🎯 Confidence threshold: {conf_threshold}")
        if max_boxes:
            print(f"📦 Max boxes per image: {max_boxes}")
        
        # Process predictions and prepare upload data
        print(f"\n🚀 Running predictions on {len(unlabeled_tasks)} unlabeled tasks...")
        print(f"📦 Batch size: {batch_size} concurrent uploads")
        
        predictions_to_upload = []
        failed_predictions = 0
        
        from tqdm import tqdm
        

        predictions_to_upload = []
        failed_predictions = 0
        
        from tqdm import tqdm as tqdm_sync
        
        for task in tqdm_sync(unlabeled_tasks, desc="Predicting"):
            try:
                task_id = task.id if hasattr(task, "id") else task["id"]

                # Extract image path
                img_url = task.data.get("image", "")
                if "/data/local-files/?d=" in img_url:
                    img_path = img_url.replace("/data/local-files/?d=", "")
                    # URL decode to handle special characters like Đ (%C4%90)
                    img_path = unquote(img_path)
                else:
                    tqdm_sync.write(f"⚠️ unknown image format: {img_url}")
                    failed_predictions += 1
                    continue

                img_path = Path(img_path)
                if not img_path.exists():
                    tqdm_sync.write(f"⚠️ not found: {img_path}")
                    failed_predictions += 1
                    continue

                # Load image
                img = cv2.imread(str(img_path))
                if img is None:
                    tqdm_sync.write(f"⚠️ failed to read: {img_path}")
                    failed_predictions += 1
                    continue

                H, W = img.shape[:2]

                # Run model
                preds = model.run_on_image(img[..., ::-1], path=img_path)

                if not preds:
                    continue

                # Convert to Label Studio format
                ls_results = []
                for i, (cls, cx, cy, bw, bh, score) in enumerate(preds):
                    x = (cx - bw/2) * 100
                    y = (cy - bh/2) * 100
                    w = bw * 100
                    h = bh * 100

                    # Clamp bounds
                    x = max(0, min(100, x))
                    y = max(0, min(100, y))
                    w = max(0, min(100 - x, w))
                    h = max(0, min(100 - y, h))

                    cls_name = class_names.get(int(cls), str(cls))
                    ls_results.append({
                        "type": "rectanglelabels",
                        "value": {
                            "x": x,
                            "y": y,
                            "width": w,
                            "height": h,
                            "rotation": 0,
                            "rectanglelabels": [cls_name],
                        },
                        "to_name": "image",
                        "from_name": "label",
                        "image_rotation": 0,
                        "original_width": W,
                        "original_height": H,
                        "id": f"pred-{i}"
                    })

                if not ls_results:
                    continue

                # Calculate average score
                avg_score = sum(p[5] for p in preds) / len(preds)

                # Store prediction data for upload
                predictions_to_upload.append({
                    "task_id": task_id,
                    "result": ls_results,
                    "score": float(avg_score),
                    "model_version": "custom-yolo",
                    "img_name": img_path.name
                })

            except Exception as e:
                tqdm_sync.write(f"✗ Prediction error: {e}")
                failed_predictions += 1

        print(f"\n✅ Generated {len(predictions_to_upload)} predictions")
        if failed_predictions > 0:
            print(f"⚠️  Failed: {failed_predictions}")

        # Phase 2: Upload predictions asynchronously
        successful_uploads = 0
        failed_uploads = 0

        if predictions_to_upload:
            print(f"\n📤 Uploading {len(predictions_to_upload)} predictions...")

            async def upload_batch():
                nonlocal successful_uploads, failed_uploads
                semaphore = asyncio.Semaphore(batch_size)

                async def upload_with_semaphore(pred_info):
                    async with semaphore:
                        try:
                            await asyncio.to_thread(
                                client.predictions.create,
                                task=pred_info["task_id"],
                                result=pred_info["result"],
                                score=pred_info["score"],
                                model_version=pred_info["model_version"]
                            )
                            return True, pred_info["task_id"], pred_info["img_name"]
                        except Exception as e:
                            return False, pred_info["task_id"], str(e)

                tasks = [
                    upload_with_semaphore(pred_info)
                    for pred_info in predictions_to_upload
                ]

                for completed in async_tqdm(
                    asyncio.as_completed(tasks),
                    total=len(tasks),
                    desc="Uploading",
                    unit="pred"
                ):
                    success, task_id, info = await completed
                    if success:
                        successful_uploads += 1
                    else:
                        failed_uploads += 1
                        print(f"\n✗ Upload error (task {task_id}): {info}")

            asyncio.run(upload_batch())
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
    parser.add_argument(
        "--max-boxes", 
        type=int, 
        help="Maximum number of boxes to keep per image (highest confidence)"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=10,
        help="Number of concurrent uploads (default: 10)"
    )
    parser.add_argument(
        "--pipeline",
        type=str,
        help="Custom pipeline name (e.g., 'yolo_plate')"
    )
    parser.add_argument(
        "--no-upload", 
        action="store_true", 
        help="Don't upload predictions to Label Studio (prediction only)"
    )
    args = parser.parse_args()
    
    predict_unlabeled(
        model_path=args.model,
        project_id=args.project_id,
        conf_threshold=args.conf,
        upload=not args.no_upload,
        max_boxes=args.max_boxes,
        batch_size=args.batch_size,
        pipeline_name=args.pipeline
    )

