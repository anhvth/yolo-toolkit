#!/usr/bin/env python3
"""
CLI Tool for Advanced YOLO Predictions with Label Studio Integration
Supports batch prediction jobs with async processing to avoid timeouts.
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Optional
import argparse
from dotenv import load_dotenv
from label_studio_sdk import Client
from ultralytics import YOLO

# Configure logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class YOLOLabelStudioCLI:
    """CLI tool for YOLO + Label Studio integration"""
    
    def __init__(self, ls_url: str, api_key: str, model_path: str):
        self.ls_url = ls_url
        self.api_key = api_key
        self.model_path = model_path
        self.client = Client(url=ls_url, api_key=api_key)
        self.model = None
        
    def load_model(self):
        """Load YOLO model"""
        logger.info(f"Loading model: {self.model_path}")
        self.model = YOLO(self.model_path)
        logger.info("Model loaded successfully")
        
    def get_unlabeled_tasks(self, project_id: int) -> List[dict]:
        """Get all unlabeled tasks from project"""
        logger.info(f"Fetching unlabeled tasks from project {project_id}")
        
        project = self.client.get_project(project_id)
        tasks = project.get_tasks()
        
        unlabeled = [t for t in tasks if not t.get('annotations')]
        logger.info(f"Found {len(unlabeled)} unlabeled tasks")
        return unlabeled
    
    def predict_task(self, task: dict, conf_threshold: float = 0.25) -> dict:
        """Run prediction on a single task"""
        if not self.model:
            self.load_model()
            
        task_id = task['id']
        image_url = task['data'].get('image', '')
        
        logger.debug(f"Processing task {task_id}: {image_url}")
        
        try:
            # Run prediction
            results = self.model.predict(
                source=image_url,
                conf=conf_threshold,
                verbose=False
            )
            
            # Convert to Label Studio format
            predictions = self._convert_to_ls_format(results[0], task)
            
            return {
                'task_id': task_id,
                'success': True,
                'predictions': predictions
            }
            
        except Exception as e:
            logger.error(f"Error processing task {task_id}: {e}")
            return {
                'task_id': task_id,
                'success': False,
                'error': str(e)
            }
    
    def _convert_to_ls_format(self, result, task: dict) -> dict:
        """Convert YOLO results to Label Studio format"""
        # Get image dimensions
        img_width = result.orig_shape[1]
        img_height = result.orig_shape[0]
        
        predictions = []
        
        if result.boxes is not None:
            for box in result.boxes:
                # Get box coordinates (xyxy format)
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                # Convert to Label Studio percentage format
                x_percent = (x1 / img_width) * 100
                y_percent = (y1 / img_height) * 100
                width_percent = ((x2 - x1) / img_width) * 100
                height_percent = ((y2 - y1) / img_height) * 100
                
                # Get class and confidence
                class_id = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = result.names[class_id]
                
                prediction = {
                    "value": {
                        "x": x_percent,
                        "y": y_percent,
                        "width": width_percent,
                        "height": height_percent,
                        "rectanglelabels": [class_name]
                    },
                    "score": conf,
                    "from_name": "label",
                    "to_name": "image",
                    "type": "rectanglelabels"
                }
                predictions.append(prediction)
        
        return {
            "result": predictions,
            "score": float(result.boxes.conf.mean()) if result.boxes is not None else 0.0
        }
    
    def batch_predict(
        self, 
        project_id: int, 
        task_ids: Optional[List[int]] = None,
        conf_threshold: float = 0.25
    ):
        """Run predictions on multiple tasks"""
        
        # Get tasks
        if task_ids:
            logger.info(f"Processing {len(task_ids)} specified tasks")
            tasks = [self.client.get_task(tid) for tid in task_ids]
        else:
            tasks = self.get_unlabeled_tasks(project_id)
        
        if not tasks:
            logger.warning("No tasks to process")
            return
        
        # Load model
        self.load_model()
        
        # Process tasks
        results = []
        for i, task in enumerate(tasks, 1):
            logger.info(f"Processing task {i}/{len(tasks)}")
            result = self.predict_task(task, conf_threshold)
            results.append(result)
            
            # Upload prediction to Label Studio
            if result['success']:
                try:
                    self.client.create_prediction(
                        task_id=result['task_id'],
                        **result['predictions']
                    )
                    logger.info(f"✓ Uploaded prediction for task {result['task_id']}")
                except Exception as e:
                    logger.error(f"Failed to upload prediction: {e}")
        
        # Summary
        successful = sum(1 for r in results if r['success'])
        failed = len(results) - successful
        
        logger.info(f"\n{'='*50}")
        logger.info(f"Batch prediction completed")
        logger.info(f"Total tasks: {len(results)}")
        logger.info(f"Successful: {successful}")
        logger.info(f"Failed: {failed}")
        logger.info(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser(
        description="YOLO + Label Studio CLI Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predict all unlabeled tasks in project
  python cli.py --ls-url http://localhost:8080 --ls-api-key XXX --project 1 --model models/updated_model.pt
  
  # Predict specific tasks
  python cli.py --ls-url http://localhost:8080 --ls-api-key XXX --project 1 --tasks 1,2,3 --model models/updated_model.pt
  
  # Debug mode
  LOG_LEVEL=DEBUG python cli.py --ls-url http://localhost:8080 --ls-api-key XXX --project 1 --model models/updated_model.pt
        """
    )
    
    parser.add_argument(
        "--ls-url",
        default=os.getenv("LABEL_STUDIO_URL", "http://localhost:8080"),
        help="Label Studio URL"
    )
    parser.add_argument(
        "--ls-api-key",
        default=os.getenv("LABEL_STUDIO_API_KEY"),
        help="Label Studio API key"
    )
    parser.add_argument(
        "--project",
        type=int,
        required=True,
        help="Project ID"
    )
    parser.add_argument(
        "--model",
        default=os.getenv("UPDATED_MODEL_PATH", "models/updated_model.pt"),
        help="Path to YOLO model"
    )
    parser.add_argument(
        "--tasks",
        help="Comma-separated task IDs (optional, predicts all unlabeled if not specified)"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold"
    )
    
    args = parser.parse_args()
    
    if not args.ls_api_key:
        logger.error("Label Studio API key not provided")
        sys.exit(1)
    
    # Parse task IDs
    task_ids = None
    if args.tasks:
        task_ids = [int(tid.strip()) for tid in args.tasks.split(',')]
    
    # Run CLI
    cli = YOLOLabelStudioCLI(
        ls_url=args.ls_url,
        api_key=args.ls_api_key,
        model_path=args.model
    )
    
    cli.batch_predict(
        project_id=args.project,
        task_ids=task_ids,
        conf_threshold=args.conf
    )


if __name__ == "__main__":
    main()
