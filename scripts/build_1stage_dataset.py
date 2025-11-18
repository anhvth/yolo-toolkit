#!/usr/bin/env python3
"""
Generate 1-stage YOLO dataset from stacked images using a 2-stage pipeline.

- Input image: top (front) + bottom (rear) stacked vertically.
- Step 1: ROI model -> vehicle region on each half (top/bottom).
- Step 2: Person/motorbike model -> objects inside that ROI.

Output:
    OUT_ROOT/
        images/
            <stem>_left.jpg   (top half)
            <stem>_right.jpg  (bottom half)
        labels/
            <stem>_left.txt
            <stem>_right.txt

YOLO label format:
    <class_id> <cx> <cy> <w> <h>   (normalized)

Classes for the NEW 1-stage model (you will configure this in data.yaml):
    0 = roi
    1 = person
    2 = motorbike
"""

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

# =====================================================================================
# Config
# =====================================================================================
PROJECT_ROOT = Path("/mnt/disk1/anhvth8/projects/yolo-toolkit").resolve()
RAW_ROOT = PROJECT_ROOT / "data/raw"

# New YOLO dataset output root
OUT_ROOT = PROJECT_ROOT / "data/yolo_1stage_dataset"
OUT_IMAGES = OUT_ROOT / "images"
OUT_LABELS = OUT_ROOT / "labels"

# 2-stage models
MODEL_ROI_PATH = PROJECT_ROOT / "runs/detect/train/weights/best.pt"  # ROI model
MODEL_PM_PATH = PROJECT_ROOT / "yolo12x.pt"                           # person/motorbike model

# COCO indices for the pm model
COCO_MAP = {
    "person": 0,
    "motorbike": 3,
}

# New 1-stage dataset class ids
CLASS_IDS = {
    "roi": 0,
    "person": 1,
    "motorbike": 2,
}

# globals (loaded in main)
model_roi: Optional[YOLO] = None
model_pm: Optional[YOLO] = None


# =====================================================================================
# Helpers
# =====================================================================================
def collect_raw_paths(raw_root: Path) -> List[str]:
    raw_root = raw_root.expanduser().resolve()
    date_folders = [d for d in raw_root.iterdir() if d.is_dir()]
    paths: List[str] = []
    for d in sorted(date_folders):
        for p in sorted(d.glob("*.jpg")):
            if p.is_file():
                paths.append(str(p.resolve()))
    return paths


def best_person_motorbike_boxes(pred) -> Dict[str, Optional[Tuple[int, int, int, int]]]:
    """Return best bbox for person and motorbike from a YOLO prediction (ROI coords)."""
    best = {"person": None, "motorbike": None}
    confs = {"person": 0.0, "motorbike": 0.0}

    for box in pred.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        for name, coco_idx in COCO_MAP.items():
            if cls == coco_idx and conf > confs[name]:
                confs[name] = conf
                best[name] = (x1, y1, x2, y2)

    return best


def normalize_yolo_box(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    w: int,
    h: int,
) -> Tuple[float, float, float, float]:
    """Convert xyxy (pixels) -> (cx, cy, bw, bh) normalized."""
    bw = x2 - x1
    bh = y2 - y1
    cx = x1 + bw / 2.0
    cy = y1 + bh / 2.0
    return cx / w, cy / h, bw / w, bh / h


def ensure_dirs():
    OUT_IMAGES.mkdir(parents=True, exist_ok=True)
    OUT_LABELS.mkdir(parents=True, exist_ok=True)


# =====================================================================================
# Core: batch processing of stacked images
# =====================================================================================
def process_batch(
    paths: List[str],
    roi_conf: float = 0.1,
    pm_conf: float = 0.05,
    pm_iou: float = 0.6,
    overwrite: bool = False,
) -> None:
    """
    Process a batch of stacked images for better performance.
    
    - Split all images into top/bottom halves
    - Run ROI model on all halves in batch
    - Run PM model on all ROIs in batch
    - Save all results
    """
    global model_roi, model_pm
    assert model_roi is not None and model_pm is not None, "Models are not loaded"

    # Prepare batch data
    batch_data = []
    for path in paths:
        img = cv2.imread(path)
        if img is None:
            continue
        H, W = img.shape[:2]
        mid = H // 2
        img_top = img[:mid, :, :].copy()
        img_bot = img[mid:, :, :].copy()
        stem = Path(path).stem
        
        # Check if already processed
        img_left_path = OUT_IMAGES / f"{stem}_left.jpg"
        img_right_path = OUT_IMAGES / f"{stem}_right.jpg"
        label_left_path = OUT_LABELS / f"{stem}_left.txt"
        label_right_path = OUT_LABELS / f"{stem}_right.txt"
        
        if (not overwrite) and img_left_path.exists() and img_right_path.exists() \
                and label_left_path.exists() and label_right_path.exists():
            continue
            
        batch_data.append({
            'stem': stem,
            'img_top': img_top,
            'img_bot': img_bot,
            'img_left_path': img_left_path,
            'img_right_path': img_right_path,
            'label_left_path': label_left_path,
            'label_right_path': label_right_path,
        })
    
    if not batch_data:
        return
    
    # Step 1: Run ROI model on all halves in batch
    all_half_images = []
    for data in batch_data:
        all_half_images.append(data['img_top'])
        all_half_images.append(data['img_bot'])
    
    preds_roi = model_roi(all_half_images, verbose=False, conf=roi_conf)
    
    # Step 2: Process ROI predictions and prepare PM model inputs
    pm_inputs = []
    pm_metadata = []  # Track which image/side each PM input belongs to
    
    for img_idx, data in enumerate(batch_data):
        data['half_boxes'] = [[], []]  # [top_boxes, bot_boxes]
        
        for side in [0, 1]:  # 0=top, 1=bottom
            pred_idx = img_idx * 2 + side
            pred = preds_roi[pred_idx]
            sub_img = data['img_top'] if side == 0 else data['img_bot']
            
            boxes = pred.boxes
            if boxes is None or len(boxes) == 0:
                continue
            
            # Select ROI with largest area
            best_area = -1
            best_box = None
            best_conf = 0.0
            
            for b in boxes:
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                conf = float(b.conf)
                area = (x2 - x1) * (y2 - y1)
                if area > best_area:
                    best_area = area
                    best_box = (int(x1), int(y1), int(x2), int(y2))
                    best_conf = conf
            
            if best_box is None or best_conf < roi_conf:
                continue
            
            x1, y1, x2, y2 = best_box
            data['half_boxes'][side].append((CLASS_IDS["roi"], x1, y1, x2, y2))
            
            # Prepare ROI crop for PM model
            roi_crop = sub_img[y1:y2, x1:x2]
            if roi_crop.size == 0:
                continue
            
            pm_inputs.append(roi_crop)
            pm_metadata.append({
                'img_idx': img_idx,
                'side': side,
                'roi_offset': (x1, y1),
            })
    
    # Step 3: Run PM model on all ROIs in batch
    if pm_inputs:
        pm_preds = model_pm(pm_inputs, verbose=False, conf=pm_conf, iou=pm_iou)
        
        # Process PM predictions
        for pm_idx, (pm_pred, meta) in enumerate(zip(pm_preds, pm_metadata)):
            img_idx = meta['img_idx']
            side = meta['side']
            ox, oy = meta['roi_offset']
            data = batch_data[img_idx]
            
            best = best_person_motorbike_boxes(pm_pred)
            
            # Convert PM boxes from ROI coords to half-image coords
            for obj_name in ("person", "motorbike"):
                bbox_roi = best[obj_name]
                if bbox_roi is None:
                    continue
                bx1, by1, bx2, by2 = bbox_roi
                bx1 += ox
                by1 += oy
                bx2 += ox
                by2 += oy
                class_id = CLASS_IDS[obj_name]
                data['half_boxes'][side].append((class_id, bx1, by1, bx2, by2))
    
    # Step 4: Save all images and labels
    for data in batch_data:
        # Save images
        cv2.imwrite(str(data['img_left_path']), data['img_top'])
        cv2.imwrite(str(data['img_right_path']), data['img_bot'])
        
        
        # Save labels
        for side, (sub_img, label_path, boxes) in enumerate([
            (data['img_top'], data['label_left_path'], data['half_boxes'][0]),
            (data['img_bot'], data['label_right_path'], data['half_boxes'][1]),
        ]):
            h, w = sub_img.shape[:2]
            lines: List[str] = []
            for class_id, x1, y1, x2, y2 in boxes:
                cx, cy, bw, bh = normalize_yolo_box(x1, y1, x2, y2, w, h)
                lines.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            
            with open(label_path, "w", encoding="utf-8") as f:
                if lines:
                    f.write("\n".join(lines))
            print(f"Saved: {data['img_left_path']} + {data['img_right_path']}")

# =====================================================================================
# Main
# =====================================================================================
def main():
    global model_roi, model_pm, OUT_ROOT, OUT_IMAGES, OUT_LABELS

    parser = argparse.ArgumentParser(
        description="Generate 1-stage YOLO dataset from stacked images (keep ROI class)."
    )
    parser.add_argument(
        "--raw-root",
        type=str,
        default=str(RAW_ROOT),
        help=f"Root directory containing raw date folders (default: {RAW_ROOT})",
    )
    parser.add_argument(
        "--out-root",
        type=str,
        default=str(OUT_ROOT),
        help="Output root for YOLO dataset (images/ + labels/).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of images to process (prefix).",
    )
    parser.add_argument("--roi-conf", type=float, default=0.1)
    parser.add_argument("--pm-conf", type=float, default=0.05)
    parser.add_argument("--pm-iou", type=float, default=0.6)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing images/labels instead of skipping.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for model inference (default: 32).",
    )
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    if not raw_root.exists():
        raise SystemExit(f"Raw root does not exist: {raw_root}")

    OUT_ROOT = Path(args.out_root).resolve()
    OUT_IMAGES = OUT_ROOT / "images"
    OUT_LABELS = OUT_ROOT / "labels"
    ensure_dirs()

    print(f"[Config] PROJECT_ROOT = {PROJECT_ROOT}")
    print(f"[Config] RAW_ROOT     = {raw_root.resolve()}")
    print(f"[Config] OUT_ROOT     = {OUT_ROOT}")
    print(f"[Config] MODEL_ROI    = {MODEL_ROI_PATH}")
    print(f"[Config] MODEL_PM     = {MODEL_PM_PATH}")
    print(f"[Config] CLASSES      = {CLASS_IDS}")

    all_raw_paths = collect_raw_paths(raw_root)
    print(f"Found {len(all_raw_paths)} raw stacked images")

    if args.limit is not None:
        all_raw_paths = all_raw_paths[: args.limit]
        print(f"Using prefix limit -> {len(all_raw_paths)} images")

    # Load models once
    print("Loading ROI model...")
    model_roi = YOLO(str(MODEL_ROI_PATH))
    print("Loading person/motorbike model...")
    model_pm = YOLO(str(MODEL_PM_PATH))

    # Process images in batches
    batch_size = args.batch_size
    print(f"Processing with batch size: {batch_size}")
    
    for i in tqdm(range(0, len(all_raw_paths), batch_size), desc="build_yolo_dataset"):
        batch_paths = all_raw_paths[i:i + batch_size]
        process_batch(
            batch_paths,
            roi_conf=args.roi_conf,
            pm_conf=args.pm_conf,
            pm_iou=args.pm_iou,
            overwrite=args.overwrite,
        )

    print("Done.")
    print(f"Images saved to: {OUT_IMAGES}")
    print(f"Labels saved to: {OUT_LABELS}")
    print("Remember to create data.yaml with class names [roi, person, motorbike].")


if __name__ == "__main__":
    main()