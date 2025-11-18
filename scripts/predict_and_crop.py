"""
Predict up to 2 boxes, save crops grouped by vehicle (plate_id or KB_*).
"""

import os
import re
import json
import random
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO
from PIL import Image

# ------------------ Setup ------------------
model = YOLO("runs/detect/train/weights/best.pt")

# Configuration
ROOT_DATA_PATH = "data/raw"  # Root path containing date folders
OUTPUT_ROOT = Path("data/roi_concat")
conf_threshold = 0.1

import cv2
import argparse

def extract_output_dir_from_path(path: str) -> str:
    # path: data/raw/images-2025-11-11_2/Plate_timestamp_*.jpg
    p = Path(path)
    parts = p.parts
    if len(parts) < 2:
        return "unknown"
    return parts[-2]

def read_image(path):
    # return two image as list
    # an image is top/bot concatenated vertically so to get 
    img = cv2.imread(str(path))
    h, w, _ = img.shape
    mid = h // 2
    img_top = img[:mid, :, :]
    img_bot = img[mid:, :, :]
    return [img_top, img_bot]


def decompose_path(path: str):
    p = Path(path)
    name = p.stem
    ext = p.suffix.lstrip(".")
    parts = name.split("_")
    out = {
        "raw": path,
        "filename": p.name,
        "ext": ext,
        "type": None,
        "plate": None,
        "timestamp_ms": None,
        "timestamp_iso": None,
        "view": None,
        "errors": [],
    }

    # Detect view (in_front/out_back)
    if len(parts) >= 2 and parts[-2] in ("in", "out"):
        out["view"] = f"{parts[-2]}_{parts[-1]}"
        core = parts[:-2]
    else:
        out["view"] = parts[-1]
        core = parts[:-1]

    # Timestamp
    ts_idx = next((len(core) - 1 - i for i, token in enumerate(core[::-1])
                   if re.fullmatch(r"\d{10,16}", token)), None)
    if ts_idx is not None:
        ts_token = core[ts_idx]
        ts = int(ts_token)
        ts_ms = ts if ts >= 1e12 else ts * 1000
        out["timestamp_ms"] = ts_ms
        out["timestamp_iso"] = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).isoformat()
        core = core[:ts_idx]
    else:
        out["errors"].append("timestamp missing")

    # Determine plate / type
    if not core:
        out["errors"].append("missing plate info")
    elif core[0] == "KB":
        out["type"] = "unknown"
        out["plate"] = "_".join(core)
    else:
        out["type"] = "plate"
        out["plate"] = "_".join(core)

    return out


def predict_image(model, image_path: Path, output_base_dir: Path, skip_existing=True):
    """
    Predict and crop image, saving to output_base_dir/plate/filename
    
    Args:
        model: YOLO model
        image_path: Path to input image
        output_base_dir: Base output directory (e.g., data/roi_concat/images-2025-11-11)
        skip_existing: If True, skip processing if output file already exists
    """
    # Extract plate info from filename to determine output path
    info = decompose_path(str(image_path))
    plate_id = info["plate"]
    
    # Create output directory: output_base_dir/plate/
    output_dir = output_base_dir / plate_id
    output_path = output_dir / image_path.name
    
    # Check if output already exists
    if skip_existing and output_path.exists():
        print(f"⏭️  Skipping (already exists): {output_path}")
        return str(output_path)
    
    top, bot = read_image(str(image_path))
    predictions = model([top, bot])
    
    # take 1 bbox cropped from top and bot each
    croped = []
    for i, pred in enumerate(predictions):
        img = top if i == 0 else bot
        boxes = pred.boxes
        scores = boxes.conf.cpu().numpy()
        selected_boxes = [
            box for box, score in zip(boxes, scores) if score >= conf_threshold
        ]
        
        # Skip if no boxes detected
        if not selected_boxes:
            print(f"No boxes detected for {'top' if i == 0 else 'bot'} image")
            continue
            
        most_confident_box = sorted(selected_boxes, key=lambda b: b.conf, reverse=True)[0]
        
        # Get bounding box coordinates in xyxy format
        x1, y1, x2, y2 = map(int, most_confident_box.xyxy[0].cpu().numpy())
        print('BBOX:', x1, y1, x2, y2)
        # Crop the image using the bounding box coordinates
        crop_img = img[y1:y2, x1:x2]
        croped.append(crop_img)
    
    # Skip if we don't have any crops
    if not croped:
        print("No crops detected, skipping image")
        return None
    
    # Resize crops to same height (use max height) before concatenating horizontally
    if len(croped) == 2:
        max_height = max(croped[0].shape[0], croped[1].shape[0])
        resized_crops = []
        for crop in croped:
            if crop.shape[0] != max_height:
                # Calculate new width to maintain aspect ratio
                aspect_ratio = crop.shape[1] / crop.shape[0]
                new_width = int(max_height * aspect_ratio)
                resized = cv2.resize(crop, (new_width, max_height))
                resized_crops.append(resized)
            else:
                resized_crops.append(crop)
        final_crop = cv2.hconcat(resized_crops)
    else:
        final_crop = croped[0]
    
    if final_crop is None:
        return None
    
    # Create output directory if needed
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save with original filename
    cv2.imwrite(str(output_path), final_crop)
    print(f"✅ Saved: {output_path}")
    return str(output_path)
def process_date_folder(date_folder: Path, root_data_path: Path, output_root: Path, skip_existing=True):
    """
    Process all images in a date folder.
    
    Args:
        date_folder: Path to date folder (e.g., data/raw/images-2025-11-11)
        root_data_path: Root data path (e.g., data/raw)
        output_root: Output root (e.g., data/roi_concat)
        skip_existing: If True, skip images that have already been processed
    """
    # Get date folder name (e.g., "images-2025-11-11")
    date_folder_name = date_folder.name
    
    # Create output directory for this date folder
    output_base_dir = output_root / date_folder_name
    output_base_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all .jpg images in this date folder
    image_paths = list(date_folder.glob("*.jpg"))
    
    if not image_paths:
        print(f"⚠️  No images found in {date_folder}")
        return []
    
    print(f"\n📁 Processing folder: {date_folder_name}")
    print(f"   Found {len(image_paths)} images")
    print(f"   Output: {output_base_dir}")
    
    processed_paths = []
    skipped_count = 0
    
    for image_path in tqdm(image_paths, desc=f"Processing {date_folder_name}"):
        result_path = predict_image(model, image_path, output_base_dir, skip_existing=skip_existing)
        if result_path:
            processed_paths.append(result_path)
            # Check if file was skipped (existed before)
            info = decompose_path(str(image_path))
            output_path = output_base_dir / info["plate"] / image_path.name
            if skip_existing and output_path.exists() and output_path.stat().st_mtime < image_path.stat().st_mtime:
                pass  # File was newly created
            elif skip_existing and output_path.exists():
                skipped_count += 1
    
    new_count = len(processed_paths) - skipped_count
    print(f"✅ Processed {new_count} new, {skipped_count} skipped, {len(image_paths) - len(processed_paths)} failed from {date_folder_name}\n")
    return processed_paths


def main():
    """Main function to process all date folders in ROOT_DATA_PATH"""
    parser = argparse.ArgumentParser(description="YOLO predict and crop with automatic folder processing")
    parser.add_argument("--root", type=str, default=ROOT_DATA_PATH, 
                       help=f"Root data path (default: {ROOT_DATA_PATH})")
    parser.add_argument("--output", type=str, default=str(OUTPUT_ROOT),
                       help=f"Output root path (default: {OUTPUT_ROOT})")
    parser.add_argument("--force", action="store_true",
                       help="Force reprocess all images even if output exists")
    args = parser.parse_args()
    
    root_path = Path(args.root)
    output_root = Path(args.output)
    skip_existing = not args.force
    
    if not root_path.exists():
        print(f"❌ Error: Root path does not exist: {root_path}")
        return
    
    # Find all subdirectories (date folders) in root path
    date_folders = [f for f in root_path.iterdir() if f.is_dir()]
    
    if not date_folders:
        print(f"❌ No folders found in {root_path}")
        return
    
    print(f"🚀 Starting batch processing")
    print(f"   Root: {root_path}")
    print(f"   Output: {output_root}")
    print(f"   Found {len(date_folders)} date folders")
    print(f"   Mode: {'Force reprocess' if args.force else 'Skip existing'}")
    print("="*60)
    
    all_processed = []
    for date_folder in date_folders:
        processed = process_date_folder(date_folder, root_path, output_root, skip_existing=skip_existing)
        all_processed.extend(processed)
    
    print("="*60)
    print(f"🎉 Batch processing complete!")
    print(f"   Total images processed: {len(all_processed)}")
    
    # Generate labels.csv for all processed images
    generate_labels_csv(output_root)


def generate_labels_csv(output_root: Path):
    """Generate labels.csv from all processed images"""
    print(f"\n📊 Generating labels.csv...")
    
    paths = list(output_root.glob('*/*/*.jpg'))
    
    if not paths:
        print("⚠️  No processed images found for CSV generation")
        return
    
    data = []
    for path in paths:
        # label_str is the plate folder name
        label_str = path.parent.name
        data.append((str(path.absolute()), label_str))
    
    mapping_label_str_to_id = {label: idx for idx, label in enumerate(sorted(set(x[1] for x in data)))}
    data = [(p, ls, mapping_label_str_to_id[ls]) for p, ls in data]
    
    df = pd.DataFrame(data, columns=['path', 'label_str', 'label'])
    csv_path = output_root / 'labels.csv'
    df.to_csv(csv_path, index=False)
    
    print(f"✅ Saved labels.csv with {len(df)} entries to {csv_path}")
    print(f"   Unique plates: {len(mapping_label_str_to_id)}")


if __name__ == "__main__":
    main()