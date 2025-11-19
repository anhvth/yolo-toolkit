#!/usr/bin/env python3
"""Predict on images using YOLO or custom pipelines and save YOLO-style labels."""

import argparse
import os
import random
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import yaml
from tqdm import tqdm
from ultralytics import YOLO

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.label_studio_sdk_wrapper import PIPELINE_REGISTRY


def _ensure_path_exists(path: Path, label: str) -> Path:
    if not path.exists():
        print(f'❌ Error: {label} not found: {path}')
        sys.exit(1)
    return path


def _load_model(
    model_path: str, pipeline: Optional[str], conf_threshold: float
) -> Tuple[object, dict, bool]:
    print(f'🔮 Loading model: {model_path}')
    if pipeline:
        if pipeline not in PIPELINE_REGISTRY:
            print(f'❌ Unknown pipeline: {pipeline}')
            print(f'Available pipelines: {list(PIPELINE_REGISTRY.keys())}')
            sys.exit(1)

        pipeline_class = PIPELINE_REGISTRY[pipeline]
        model = pipeline_class(
            model_path=model_path,
            roi_conf_thres=conf_threshold,
            veh_conf_thres=conf_threshold,
        )
        print(f'✅ Using custom pipeline: {pipeline}')
        class_map = getattr(model, 'class_id_to_name', {})
        use_custom = True
    else:
        model = YOLO(model_path)
        raw_names = getattr(model, 'names', {})
        class_map = (
            raw_names
            if isinstance(raw_names, dict)
            else {idx: name for idx, name in enumerate(raw_names)}
        )
        use_custom = False

    return model, class_map, use_custom


def _collect_images(img_dir: Path, extensions: Sequence[str]) -> List[Path]:
    seen: set[Path] = set()
    for ext in extensions:
        seen.update(img_dir.rglob(f'*{ext}'))
        seen.update(img_dir.rglob(f'*{ext.upper()}'))
    return sorted(seen)


def _split_images(image_paths: Sequence[Path], train_ratio: float) -> Tuple[List[Path], List[Path]]:
    image_list = list(image_paths)
    random.seed(42)
    random.shuffle(image_list)
    train_count = int(len(image_list) * train_ratio)
    return image_list[:train_count], image_list[train_count:]


def _prepare_split_dirs(output_root: Path, split_name: str) -> Tuple[Path, Path]:
    images_dir = output_root / 'images' / split_name
    labels_dir = output_root / 'labels' / split_name
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    return images_dir, labels_dir


def _predict_batch(
    model: object,
    batch_paths: Sequence[Path],
    conf_threshold: float,
    use_custom_pipeline: bool,
) -> List[Optional[object]]:
    if use_custom_pipeline:
        expect_plates = ["_left" in path.name for path in batch_paths]
        try:
            return model(batch_paths, expect_plates)
        except Exception as exc:  # pylint: disable=broad-except
            tqdm.write(f'✗ Custom pipeline batch prediction error: {exc}')
            return [None] * len(batch_paths)

    try:
        return model(
            [str(path) for path in batch_paths],
            conf=conf_threshold,
            verbose=False,
        )
    except Exception as exc:  # pylint: disable=broad-except
        tqdm.write(f'✗ Batch prediction error: {exc}')
        return [None] * len(batch_paths)


def _format_prediction_lines(
    results: object, conf_threshold: float, use_custom_pipeline: bool
) -> List[str]:
    if results is None:
        return []

    height, width = results.orig_shape
    boxes = results.boxes
    lines: List[str] = []

    for cls_tensor, conf_tensor, xyxy in zip(boxes.cls, boxes.conf, boxes.xyxy):
        confidence = float(conf_tensor.item())
        if not use_custom_pipeline and confidence < conf_threshold:
            continue

        x1, y1, x2, y2 = xyxy.cpu().numpy()
        cx = (x1 + x2) / 2 / width
        cy = (y1 + y2) / 2 / height
        bw = (x2 - x1) / width
        bh = (y2 - y1) / height
        cls_id = int(cls_tensor.item())
        lines.append(f'{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}')

    return lines


def _write_label_file(label_path: Path, lines: Sequence[str]) -> None:
    with open(label_path, 'w') as handler:
        handler.write('\n'.join(lines))
        if lines:
            handler.write('\n')


def _create_symlink(source: Path, dest_dir: Path) -> None:
    symlink_path = dest_dir / source.name
    if symlink_path.exists() or symlink_path.is_symlink():
        symlink_path.unlink()

    try:
        rel_path = os.path.relpath(source, dest_dir)
        symlink_path.symlink_to(rel_path)
    except (ValueError, OSError):
        symlink_path.symlink_to(source)


def predict_and_save_labels(
    model_path: str,
    image_dir: str,
    output_dir: str,
    pipeline: Optional[str] = None,
    conf_threshold: float = 0.25,
    batch: int = 32,
    train_ratio: float = 0.85,
    img_extensions: Sequence[str] = ('.jpg', '.jpeg', '.png', '.bmp'),
) -> Tuple[List[str], int, Path]:
    model_file = _ensure_path_exists(Path(model_path), 'Model')
    img_dir = _ensure_path_exists(Path(image_dir).resolve(), 'Image directory')
    output_root = Path(output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    model, class_map, use_custom_pipeline = _load_model(
        str(model_file), pipeline, conf_threshold
    )

    class_list = [class_map[idx] for idx in sorted(class_map.keys())]
    print(f"📝 Classes ({len(class_list)}): {', '.join(class_list)}")
    print(f'🎯 Confidence threshold: {conf_threshold}')

    image_files = _collect_images(img_dir, img_extensions)
    if not image_files:
        print(f'❌ No images found in {img_dir}')
        sys.exit(1)

    train_images, val_images = _split_images(image_files, train_ratio)
    print(f'📂 Found {len(image_files)} images')
    print(
        f'📊 Split: {len(train_images)} train, {len(val_images)} val '
        f'({train_ratio:.0%}/{1 - train_ratio:.0%})'
    )
    print(f'📦 Batch size: {batch}')

    total_processed = 0
    splits = [('train', train_images), ('val', val_images)]

    for split_name, split_images in splits:
        if not split_images:
            print(f'⚠️  No images for {split_name} split, skipping...')
            continue

        print(f'\n🔄 Processing {split_name} split ({len(split_images)} images)...')
        images_dir, labels_dir = _prepare_split_dirs(output_root, split_name)
        print(f'🔗 Creating symlinks in: {images_dir}')
        print(f'💾 Saving labels to: {labels_dir}')

        processed = 0
        skipped = 0
        total_batches = (len(split_images) + batch - 1) // batch

        for batch_idx in range(0, len(split_images), batch):
            batch_paths = split_images[batch_idx : batch_idx + batch]
            batch_num = batch_idx // batch + 1
            batch_results = _predict_batch(
                model, batch_paths, conf_threshold, use_custom_pipeline
            )
            pbar = tqdm(
                zip(batch_paths, batch_results),
                desc=f'{split_name.capitalize()} Batch {batch_num}/{total_batches}',
                total=len(batch_paths),
                leave=False,
            )

            for img_path, results in pbar:
                if results is None:
                    tqdm.write(f'⚠️  Failed to process: {img_path.name}')
                    skipped += 1
                    continue

                try:
                    lines = _format_prediction_lines(
                        results, conf_threshold, use_custom_pipeline
                    )
                    _write_label_file(labels_dir / f'{img_path.stem}.txt', lines)
                    _create_symlink(img_path, images_dir)
                    processed += 1
                except Exception as exc:  # pylint: disable=broad-except
                    tqdm.write(f'✗ Error processing {img_path.name}: {exc}')
                    skipped += 1

        print(f'✅ {split_name.capitalize()}: {processed} processed')
        if skipped > 0:
            print(f'⚠️  {split_name.capitalize()}: {skipped} skipped')
        total_processed += processed

    print('\n' + '=' * 60)
    print('📊 PREDICTION SUMMARY')
    print('=' * 60)
    print(f'✅ Total processed: {total_processed}')
    print(f'📊 Train: {len(train_images)} images')
    print(f'📊 Val: {len(val_images)} images')
    print(f"🔗 Images symlinked to: {output_root / 'images'}")
    print(f"💾 Labels saved to: {output_root / 'labels'}")

    return class_list, total_processed, output_root


def generate_data_yaml(
    dataset_root: str,
    class_names: Sequence[str],
    train_rel_path: str = 'images/train',
    val_rel_path: str = 'images/val',
) -> None:
    dataset_path = Path(dataset_root).resolve()
    dataset_path.mkdir(parents=True, exist_ok=True)

    data_config = {
        'path': str(dataset_path),
        'train': train_rel_path,
        'val': val_rel_path,
        'nc': len(class_names),
        'names': list(class_names),
    }

    yaml_path = dataset_path / 'data.yaml'
    with open(yaml_path, 'w') as handler:
        yaml.dump(data_config, handler, default_flow_style=False, sort_keys=False)

    print(f'\n📄 Generated data.yaml at: {yaml_path}')
    print('\nContents:')
    print('=' * 60)
    with open(yaml_path) as handler:
        print(handler.read())
    print('=' * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Predict on images and save as YOLO-format labels',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard YOLO model with 85/15 train/val split
  python scripts/yolo/predict_as_label.py \
      --model models/best.pt \
      --image-dir data/images \
      --output-dir data/yolo_dataset

  # Custom pipeline with custom split ratio
  python scripts/yolo/predict_as_label.py \
      --model models/yolo12x-4classes.pt \
      --image-dir data/images \
      --output-dir data/yolo_dataset \
      --pipeline yolo_plate \
      --train-ratio 0.8 \
      --conf 0.3

Available pipelines: {pipelines}
        """.format(
            pipelines=', '.join(PIPELINE_REGISTRY.keys()) or 'None'
        ),
    )

    parser.add_argument('--model', required=True, help='Path to trained YOLO model (.pt file)')
    parser.add_argument('--image-dir', required=True, help='Directory containing images to predict on')
    parser.add_argument(
        '--output-dir',
        required=True,
        help='Root output directory for YOLO dataset (creates images/ and labels/ splits)',
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.85,
        help='Ratio of images to use for training (default: 0.85, so 85%% train / 15%% val)',
    )
    parser.add_argument(
        '--pipeline',
        type=str,
        help=f"Custom pipeline name (available: {', '.join(PIPELINE_REGISTRY.keys())})",
    )
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold for predictions')
    parser.add_argument('--batch', type=int, default=32, help='Number of images per batch (default: 32)')
    parser.add_argument('--no-yaml', action='store_true', help='Skip generating data.yaml file')

    args = parser.parse_args()

    class_names, processed, dataset_root = predict_and_save_labels(
        model_path=args.model,
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        pipeline=args.pipeline,
        conf_threshold=args.conf,
        batch=args.batch,
        train_ratio=args.train_ratio,
    )

    if not args.no_yaml and processed > 0:
        generate_data_yaml(dataset_root=str(dataset_root), class_names=class_names)

    print('\n✅ Done!')
    print('\n💡 Use this dataset for training:')
    print(f"   yolo train data={dataset_root / 'data.yaml'} ...")


if __name__ == '__main__':
    main()
