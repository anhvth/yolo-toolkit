from pathlib import Path

def clean_yolo(root):
    root = Path(root)
    img_root = root / "images"
    lbl_root = root / "labels"

    # collect all images recursively
    imgs = []
    for ext in ["*.jpg", "*.png", "*.jpeg"]:
        imgs.extend(img_root.rglob(ext))

    removed = 0

    for img in imgs:
        # map image path → label path by replacing 'images' → 'labels'
        rel = img.relative_to(img_root)
        lbl = lbl_root / rel.with_suffix(".txt")

        # case 1: label missing
        if not lbl.exists():
            img.unlink()
            removed += 1
            continue

        # case 2: label empty
        if lbl.stat().st_size == 0:
            img.unlink()
            lbl.unlink()
            removed += 1
            continue

    print("removed:", removed)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("usage: python clean_yolo_recursive.py <dataset_root>")
        exit(1)

    clean_yolo(sys.argv[1])
