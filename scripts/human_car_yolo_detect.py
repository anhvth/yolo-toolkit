from ultralytics import YOLO
import cv2
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# --------- CONFIG ---------
model_path = "/mnt/disk1/anhvth8/projects/yolo-toolkit/yolo12s.pt"

input_root  = Path("/mnt/disk1/anhvth8/projects/yolo-toolkit/data/cropped_2")
output_root = Path("/mnt/disk1/anhvth8/projects/yolo-toolkit/data/predicted")
num_workers = 4   # chạy song song 4 task
# --------------------------

# Load model 1 lần, share cho các thread
model = YOLO(model_path)

def process_image(img_path: Path):
    """Detect + split + save left/right cho 1 ảnh."""
    plate = img_path.parent.name

    # Tạo folder output cho biển số này
    left_dir  = output_root / plate / "left"
    right_dir = output_root / plate / "right"
    left_dir.mkdir(parents=True, exist_ok=True)
    right_dir.mkdir(parents=True, exist_ok=True)

    # Inference
    results = model(str(img_path))
    r = results[0]

    # 1) Ảnh full có bbox
    annotated = r.plot()

    # 2) Split ở giữa
    h, w = annotated.shape[:2]
    center = w // 2
    left_img  = annotated[:, :center]
    right_img = annotated[:, center:]

    # Lưu cùng tên file, tách sang 2 folder left/right
    out_left_path  = left_dir / img_path.name
    out_right_path = right_dir / img_path.name

    cv2.imwrite(str(out_left_path), left_img)
    cv2.imwrite(str(out_right_path), right_img)

    return img_path, out_left_path, out_right_path


def main():
    # Lấy toàn bộ ảnh *.jpg trong các folder biển số
    all_images = sorted(input_root.rglob("*.jpg"))
    print(f"Found {len(all_images)} images")

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_image, img_path) for img_path in all_images]

        for fut in as_completed(futures):
            img_path, out_left, out_right = fut.result()
            print(f"[OK] {img_path} ->")
            print(f"     left : {out_left}")
            print(f"     right: {out_right}")

    print("Done!")


if __name__ == "__main__":
    main()
