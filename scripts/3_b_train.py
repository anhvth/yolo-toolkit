#!/usr/bin/env python3
"""
Script 3b: Train - Train YOLO model with exported dataset
Usage: python scripts/3_b_train.py --model yolo11n.pt --data data/exports/yolo_dataset/data.yaml
       python scripts/3_b_train.py --model yolo11n.pt --data data/exports/yolo_dataset/data.yaml --epochs 100
"""

import argparse
import sys
from pathlib import Path
import torch
from label_studio_sdk_wrapper.config import get_config


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
        print("💡 Make sure you've run 3_a_export.py first to create the dataset")
        sys.exit(1)
    
    # Auto-detect device if not specified
    if device is None:
        device = 'mps' if torch.backends.mps.is_available() else 'cuda'
    
    print("\n🚀 Starting YOLO training...")
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
        description="Train YOLO model with exported dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/3_b_train.py --model yolo11n.pt --data data/exports/yolo_dataset/data.yaml
  python scripts/3_b_train.py --model yolo11n.pt --data data/exports/yolo_dataset/data.yaml --epochs 100
  python scripts/3_b_train.py --model yolo11n.pt --data data/exports/yolo_dataset/data.yaml --freeze-ratio 0.8
        """
    )
    parser.add_argument("--model", required=True, help="YOLO model path (e.g., yolo11n.pt)")
    parser.add_argument("--data", required=True, help="Path to data.yaml file")
    parser.add_argument("--epochs", type=int, help="Training epochs (default: from config)")
    parser.add_argument("--imgsz", type=int, help="Image size (default: from config)")
    parser.add_argument("--output", help="Output model path (default: from config)")
    
    # Training hyperparameters
    parser.add_argument("--lr0", type=float, default=1e-4, help="Initial learning rate (default: 1e-4)")
    parser.add_argument("--mosaic", type=float, default=0.5, help="Mosaic augmentation probability (default: 0.5)")
    parser.add_argument("--close-mosaic", type=int, default=10, help="Epochs to disable mosaic augmentation before end (default: 10)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--half", action="store_true", default=True, help="Use half precision (FP16) training (default: True)")
    parser.add_argument("--no-half", dest="half", action="store_false", help="Disable half precision training")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda/mps/cpu, default: auto-detect)")
    parser.add_argument("--freeze-ratio", type=float, default=0.5, help="Ratio of parameters to freeze (0-1, default: 0.5)")
    
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
    print("🎯 Train YOLO Model")
    print("=" * 60)
    
    train_yolo(
        model_path=args.model,
        data_yaml=args.data,
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
    print("✅ Training completed successfully!")
    print("=" * 60)
    print("💡 Next steps:")
    print("   - Review results in runs/detect/train*/")
    print(f"   - Use model: {output_model}")
    print(f"   - Retrain with new data: python scripts/3_b_train.py --model {output_model} --data {args.data}")


if __name__ == "__main__":
    main()
