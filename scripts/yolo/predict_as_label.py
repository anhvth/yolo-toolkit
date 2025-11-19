#!/usr/bin/env python3
"""
Predict on images using YOLO or custom pipelines and save as YOLO-format labels.
Creates a complete YOLO dataset structure with symlinked images and generated labels.
"""

import sys
import os
import cv2
import yaml
from pathlib import Path
from typing import Optional
from tqdm import tqdm
from ultralytics import YOLO

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.label_studio_sdk_wrapper.inference_roi_moto_plate import (
    REGISTERED_PIPELINES,
)


def predict_and_save_labels(
    model_path: str,
    image_dir: str,
    output_dir: str,
    pipeline: Optional[str] = None,
    conf_threshold: float = 0.25,
    batch_size: int = 32,
    train_ratio: float = 0.85,
    img_extensions: tuple = ('.jpg', '.jpeg', '.png', '.bmp'),
):
    """
    Run predictions on images and save as YOLO-format labels.
    Creates complete dataset structure with symlinked images and train/val split.
    
    Args:
        model_path: Path to YOLO model (.pt file)
        image_dir: Directory containing images to predict on
        output_dir: Root output directory for YOLO dataset
        pipeline: Name of custom pipeline (e.g., 'yolo_plate'), None for standard YOLO
        conf_threshold: Confidence threshold for predictions
        batch_size: Number of images to process in each batch
        train_ratio: Ratio of images to use for training (default: 0.85)
        img_extensions: Tuple of valid image extensions
    """
    
    # Validate inputs
    model_file = Path(model_path)
    if not model_file.exists():
        print(f"❌ Error: Model not found: {model_path}")
        sys.exit(1)
    
    img_dir = Path(image_dir).resolve()
    if not img_dir.exists():
        print(f"❌ Error: Image directory not found: {image_dir}")
        sys.exit(1)
    
    # Create output directory structure
    output_root = Path(output_dir).resolve()
    
    print(f"🔮 Loading model: {model_path}")
    
    # Load model - either custom pipeline or standard YOLO
    if pipeline:
        if pipeline not in REGISTERED_PIPELINES:
            print(f"❌ Unknown pipeline: {pipeline}")
            print(f"Available pipelines: {list(REGISTERED_PIPELINES.keys())}")
            sys.exit(1)
        
        pipeline_class = REGISTERED_PIPELINES[pipeline]
        model = pipeline_class(
            model_path=model_path,
            roi_conf_thres=conf_threshold,
            veh_conf_thres=conf_threshold,
        )
        print(f"✅ Using custom pipeline: {pipeline}")
        class_names = model.class_id_to_name
        use_custom_pipeline = True
    else:
        model = YOLO(model_path)
        class_names = model.names
        use_custom_pipeline = False
    
    # Get sorted class list
    class_list = [class_names[i] for i in sorted(class_names.keys())]
    print(f"📝 Classes ({len(class_list)}): {', '.join(class_list)}")
    print(f"🎯 Confidence threshold: {conf_threshold}")
    
    # Find all images
    image_files = []
    for ext in img_extensions:
        image_files.extend(img_dir.rglob(f'*{ext}'))
        image_files.extend(img_dir.rglob(f'*{ext.upper()}'))
    
    image_files = sorted(set(image_files))
    
    if not image_files:
        print(f"❌ No images found in {image_dir}")
        sys.exit(1)
    
    # Split images into train/val (85/15)
    import random
    random.seed(42)  # For reproducible splits
    random.shuffle(image_files)
    
    train_count = int(len(image_files) * train_ratio)
    train_images = image_files[:train_count]
    val_images = image_files[train_count:]
    
    print(f"📂 Found {len(image_files)} images")
    print(f"📊 Split: {len(train_images)} train, {len(val_images)} val ({train_ratio:.0%}/{1-train_ratio:.0%})")
    print(f"📦 Batch size: {batch_size}")
    
    # Process both splits
    splits = [('train', train_images), ('val', val_images)]
    all_class_names = None
    
    for split_name, split_images in splits:
        if not split_images:
            print(f"⚠️  No images for {split_name} split, skipping...")
            continue
            
        print(f"\n🔄 Processing {split_name} split ({len(split_images)} images)...")
        
        # Create directories for this split
        images_dir = output_root / 'images' / split_name
        label_dir = output_root / 'labels' / split_name
        images_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🔗 Creating symlinks in: {images_dir}")
        print(f"💾 Saving labels to: {label_dir}")
        
        # Process images in batches
        processed = 0
        skipped = 0
        total_batches = (len(split_images) + batch_size - 1) // batch_size
        
        for batch_idx in range(0, len(split_images), batch_size):
            batch_paths = split_images[batch_idx:batch_idx + batch_size]
            batch_num = batch_idx // batch_size + 1
            
            # Run batch prediction
            if use_custom_pipeline:
                # Custom pipelines process one at a time (path-dependent logic)
                batch_results = []
                for img_path in batch_paths:
                    try:
                        img = cv2.imread(str(img_path))
                        if img is None:
                            batch_results.append(None)
                            continue
                        result = model(img[..., ::-1], path=img_path)
                        if isinstance(result, tuple):
                            result, _ = result
                        batch_results.append(result)
                    except Exception:
                        batch_results.append(None)
            else:
                # Standard YOLO - use batch prediction (MUCH FASTER!)
                try:
                    batch_results = model(
                        [str(p) for p in batch_paths],
                        conf=conf_threshold,
                        verbose=False
                    )
                except Exception as e:
                    tqdm.write(f"✗ Batch prediction error: {e}")
                    batch_results = [None] * len(batch_paths)
            
            # Process results for each image in batch
            pbar = tqdm(
                zip(batch_paths, batch_results),
                desc=f"{split_name.capitalize()} Batch {batch_num}/{total_batches}",
                total=len(batch_paths),
                leave=False
            )
            
            for img_path, results in pbar:
                try:
                    if results is None:
                        tqdm.write(f"⚠️  Failed to process: {img_path.name}")
                        skipped += 1
                        continue
                    
                    # Get image dimensions
                    H, W = results.orig_shape
                    
                    # Extract boxes
                    boxes = results.boxes
                    
                    # Convert to YOLO format (class_id cx cy w h)
                    yolo_lines = []
                    for j in range(len(boxes)):
                        cls_id = int(boxes.cls[j].item())
                        conf = float(boxes.conf[j].item())
                        
                        # Skip low confidence if using standard YOLO
                        if not use_custom_pipeline and conf < conf_threshold:
                            continue
                        
                        xyxy = boxes.xyxy[j].cpu().numpy()
                        x1, y1, x2, y2 = xyxy
                        
                        # Convert to normalized center coordinates
                        cx = (x1 + x2) / 2 / W
                        cy = (y1 + y2) / 2 / H
                        bw = (x2 - x1) / W
                        bh = (y2 - y1) / H
                        
                        # YOLO format: class_id cx cy w h
                        yolo_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                    
                    # Save label file (even if empty - YOLO convention)
                    label_path = label_dir / f"{img_path.stem}.txt"
                    with open(label_path, 'w') as f:
                        f.write('\n'.join(yolo_lines))
                        if yolo_lines:
                            f.write('\n')
                    
                    # Create symlink for image
                    symlink_path = images_dir / img_path.name
                    if symlink_path.exists() or symlink_path.is_symlink():
                        symlink_path.unlink()
                    
                    # Use relative path if possible for portability
                    try:
                        rel_path = os.path.relpath(img_path, images_dir)
                        symlink_path.symlink_to(rel_path)
                    except (ValueError, OSError):
                        # Fall back to absolute path if relative doesn't work
                        symlink_path.symlink_to(img_path)
                    
                    processed += 1
                    
                except Exception as e:
                    tqdm.write(f"✗ Error processing {img_path.name}: {e}")
                    skipped += 1
        
        print(f"✅ {split_name.capitalize()}: {processed} processed")
        if skipped > 0:
            print(f"⚠️  {split_name.capitalize()}: {skipped} skipped")
    
    print(f"\n{'='*60}")
    print("📊 PREDICTION SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Total processed: {len(train_images) + len(val_images)}")
    print(f"📊 Train: {len(train_images)} images")
    print(f"📊 Val: {len(val_images)} images")
    print(f"🔗 Images symlinked to: {output_root / 'images'}")
    print(f"💾 Labels saved to: {output_root / 'labels'}")
    
    return class_list, len(train_images) + len(val_images), output_root


def generate_data_yaml(
    dataset_root: str,
    class_names: list,
    train_rel_path: str = 'images/train',
    val_rel_path: str = 'images/val',
):
    """
    Generate data.yaml configuration file for YOLO dataset.
    
    Args:
        dataset_root: Root directory of YOLO dataset
        class_names: List of class names
        train_rel_path: Relative path to training images from dataset root
        val_rel_path: Relative path to validation images from dataset root
    """
    
    dataset_path = Path(dataset_root).resolve()
    dataset_path.mkdir(parents=True, exist_ok=True)
    
    # Create data.yaml content
    data_config = {
        'path': str(dataset_path),
        'train': train_rel_path,
        'val': val_rel_path,
        'nc': len(class_names),
        'names': class_names,
    }
    
    # Save data.yaml at dataset root
    yaml_path = dataset_path / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False, sort_keys=False)
    
    print(f"\n📄 Generated data.yaml at: {yaml_path}")
    print("\nContents:")
    print(f"{'='*60}")
    with open(yaml_path) as f:
        print(f.read())
    print(f"{'='*60}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Predict on images and save as YOLO-format labels',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard YOLO model
  python scripts/yolo/predict_as_label.py \\
      --model models/best.pt \\
      --image-dir data/images \\
      --output-dir data/yolo_dataset

  # Custom pipeline (e.g., yolo_plate)
  python scripts/yolo/predict_as_label.py \\
      --model models/yolo12x-4classes.pt \\
      --image-dir data/images \\
      --output-dir data/yolo_dataset \\
      --pipeline yolo_plate \\
      --conf 0.3 \\
      --split train

Available pipelines: {pipelines}
        """.format(pipelines=', '.join(REGISTERED_PIPELINES.keys()) or 'None')
    )
    
    parser.add_argument(
        '--model',
        required=True,
        help='Path to trained YOLO model (.pt file)'
    )
    parser.add_argument(
        '--image-dir',
        required=True,
        help='Directory containing images to predict on'
    )
    parser.add_argument(
        '--output-dir',
        required=True,
        help='Root output directory for YOLO dataset (will create images/ and labels/ subdirs with train/val splits)'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.85,
        help='Ratio of images to use for training (default: 0.85, so 85%% train / 15%% val)'
    )
    parser.add_argument(
        '--pipeline',
        type=str,
        help=f"Custom pipeline name (available: {', '.join(REGISTERED_PIPELINES.keys())})"
    )
    parser.add_argument(
        '--conf',
        type=float,
        default=0.25,
        help='Confidence threshold for predictions (default: 0.25)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Number of images to process in each batch (default: 32)'
    )
    parser.add_argument(
        '--no-yaml',
        action='store_true',
        help='Skip generating data.yaml file'
    )
    
    args = parser.parse_args()
    
    # Run predictions
    class_names, processed, dataset_root = predict_and_save_labels(
        model_path=args.model,
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        pipeline=args.pipeline,
        conf_threshold=args.conf,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
    )
    
    # Generate data.yaml at dataset root
    if not args.no_yaml and processed > 0:
        generate_data_yaml(
            dataset_root=dataset_root,
            class_names=class_names,
        )
    
    print("\n✅ Done!")
    print("\n💡 Use this dataset for training:")
    print(f"   yolo train data={dataset_root / 'data.yaml'} ...")
