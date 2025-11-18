# pipeline.py
from pathlib import Path
from unittest import result

import cv2
import numpy as np
from ultralytics import YOLO
import ultralytics

# ============================================================
# Configuration Parameters
# ============================================================
DEFAULT_IMG_SIZE = 640
DEFAULT_ROI_CONF_THRES = 0.25
DEFAULT_VEH_CONF_THRES = 0.25
DEFAULT_PLATE_CONF_THRES = 0.1

DEFAULT_ROI_CLASS_NAME = 'ROI'
DEFAULT_VEHICLE_CLASS_NAMES = ('motobike', 'xedap')
DEFAULT_PLATE_CLASS_NAME = 'bienso'

# Fixed palette; we index with class_id % len(colors)
colors = [
    (255, 0, 0),  # red
    (0, 255, 0),  # green
    (0, 0, 255),  # blue
    (255, 255, 0),  # yellow
    (255, 0, 255),  # magenta
    (0, 255, 255),  # cyan
    (255, 128, 0),  # orange
    (128, 0, 255),  # purple
    (0, 128, 255),  # sky blue
    (128, 255, 0),  # lime
]
REGISTERED_PIPELINES = {}


class SingleYoloPipeline:
    """
    Class names are expected to include:
        - "ROI"
        - "motobike"
        - "xedap"
        - "bienso"
    Mapping is inferred from self.model.names.
    """

    def __init__(
        self,
        model_path,
        img_size: int = DEFAULT_IMG_SIZE,
        roi_class_name: str = DEFAULT_ROI_CLASS_NAME,
        vehicle_class_names=DEFAULT_VEHICLE_CLASS_NAMES,
        plate_class_name: str = DEFAULT_PLATE_CLASS_NAME,
        roi_conf_thres: float = DEFAULT_ROI_CONF_THRES,
        veh_conf_thres: float = DEFAULT_VEH_CONF_THRES,
        plate_conf_thres: float = DEFAULT_PLATE_CONF_THRES,
    ):
        self.model = YOLO(model_path)
        self.img_size = img_size

        # ---------------------------------------------------------
        # Build id <-> name mapping from self.model.names
        # ---------------------------------------------------------
        self.class_id_to_name, self.class_name_to_id = self._build_class_mappings()

        # Resolve class IDs from class names
        self.roi_cls_id = self._get_class_id(roi_class_name)
        self.vehicle_class_ids = [self._get_class_id(n) for n in vehicle_class_names]
        self.plate_cls_id = self._get_class_id(plate_class_name)

        # Thresholds
        self.roi_conf_thres = roi_conf_thres
        self.veh_conf_thres = veh_conf_thres
        self.plate_conf_thres = plate_conf_thres

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_class_mappings(self):
        names = self.model.names
        # Ultralytics sometimes gives list, sometimes dict
        if isinstance(names, dict):
            id_to_name = names
        else:
            # assume list-like
            id_to_name = {i: n for i, n in enumerate(names)}
        name_to_id = {v: k for k, v in id_to_name.items()}
        return id_to_name, name_to_id

    def _get_class_id(self, class_name: str) -> int:
        if class_name not in self.class_name_to_id:
            raise ValueError(
                f"Class name '{class_name}' not found in model.names: "
                f"{list(self.class_name_to_id.keys())}"
            )
        return self.class_name_to_id[class_name]

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def predict(self, img):
        """
        Run YOLO and return the full Results object.
        """
        pred = self.model.predict(img, imgsz=self.img_size, verbose=False, conf=0.001)[0]
        return pred

    def postproc(self, results: ultralytics.engine.results.Results, expect_plate: bool):
        """
        Convert model outputs → filtered YOLO-format predictions.

        Args:
            results: ultralytics Results object
            expect_plate: whether to look for plate inside vehicle

        Output: list of [class_id, cx, cy, w, h, score]

        Rules:
          - 1 ROI (best by confidence over roi_conf_thres)
          - 1 vehicle (best motobike/xedap over veh_conf_thres)
          - If expect_plate and a plate is inside vehicle -> 1 plate
        """
        # Extract data from Results object
        xyxy = results.boxes.xyxy.cpu().numpy()
        cls = results.boxes.cls.cpu().numpy().astype(int)
        conf = results.boxes.conf.cpu().numpy()
        H, W = results.orig_shape
        
        out = []
        is_weird = False

        # =========================================================
        # 1) Pick BEST ROI
        # =========================================================
        roi_indices = [
            i
            for i, c in enumerate(cls)
            if c == self.roi_cls_id and conf[i] >= self.roi_conf_thres
        ]

        if not roi_indices:
            return []  # must have ROI

        best_roi_i = max(roi_indices, key=lambda i: conf[i])
        x1, y1, x2, y2 = xyxy[best_roi_i]
        roi_conf = float(conf[best_roi_i])

        # normalize
        roi_cx = (x1 + x2) / 2 / W
        roi_cy = (y1 + y2) / 2 / H
        roi_w = (x2 - x1) / W
        roi_h = (y2 - y1) / H

        out.append([self.roi_cls_id, roi_cx, roi_cy, roi_w, roi_h, roi_conf])

        # =========================================================
        # 2) Pick BEST vehicle (motobike/xedap)
        # =========================================================
        veh_candidates = [
            i
            for i, c in enumerate(cls)
            if c in self.vehicle_class_ids and conf[i] >= self.veh_conf_thres
        ]

        if not veh_candidates:
            # ROI only
            return out

        best_veh_i = max(veh_candidates, key=lambda i: conf[i])
        vx1, vy1, vx2, vy2 = xyxy[best_veh_i]
        veh_cls = int(cls[best_veh_i])
        veh_conf = float(conf[best_veh_i])

        v_cx = (vx1 + vx2) / 2 / W
        v_cy = (vy1 + vy2) / 2 / H
        v_w = (vx2 - vx1) / W
        v_h = (vy2 - vy1) / H

        out.append([veh_cls, v_cx, v_cy, v_w, v_h, veh_conf])

        # =========================================================
        # 3) Plate inside vehicle
        # =========================================================
        if expect_plate:
            bx1, by1, bx2, by2 = vx1, vy1, vx2, vy2

            plate_indices = []
            for i, c in enumerate(cls):
                if c != self.plate_cls_id:
                    continue
                if conf[i] < self.plate_conf_thres:
                    continue

                px1, py1, px2, py2 = xyxy[i]

                # plate inside vehicle bbox
                if px1 >= bx1 and py1 >= by1 and px2 <= bx2 and py2 <= by2:
                    plate_indices.append(i)

            if plate_indices:
                best_plate_i = max(plate_indices, key=lambda i: conf[i])
                px1, py1, px2, py2 = xyxy[best_plate_i]
                plate_conf = float(conf[best_plate_i])

                p_cx = (px1 + px2) / 2 / W
                p_cy = (py1 + py2) / 2 / H
                p_w = (px2 - px1) / W
                p_h = (py2 - py1) / H

                out.append([self.plate_cls_id, p_cx, p_cy, p_w, p_h, plate_conf])
            else:
                # this is weird case: expect plate but none found inside vehicle
                is_weird = True
        return out, is_weird

    def __call__(self, img, path=None) -> ultralytics.engine.results.Results:
        """
        Main inference method compatible with standard YOLO API.
        Returns ultralytics Results object with filtered predictions.
        
        path is used only to decide whether we expect a plate.
        Example: expect plate if '_left' in path.
        """
        results = self.predict(img)

        expect_plate = False
        if path is not None:
            expect_plate = "_left" in str(path)

        # Get filtered predictions in normalized format
        filtered_preds, is_weird = self.postproc(results, expect_plate=expect_plate)
        if is_weird:
            print(f"⚠️  Weird case detected for image {path}: expect_plate={expect_plate} but no plate found inside vehicle.")
            
        
        # Store filtered predictions as attribute for easy access
        results.filtered_preds = filtered_preds
        
        return results


    def visualize(self, img, preds, thickness: int = 3):
        """
        preds: list of [class_id, cx, cy, w, h, score] in normalized YOLO format.
        """
        H, W = img.shape[:2]
        out = img.copy()

        for class_id, cx, cy, bw, bh, score in preds:
            class_id = int(class_id)

            x1 = int((cx - bw / 2) * W)
            y1 = int((cy - bh / 2) * H)
            x2 = int((cx + bw / 2) * W)
            y2 = int((cy + bh / 2) * H)

            label = self.class_id_to_name.get(class_id, str(class_id))
            color = colors[class_id % len(colors)]

            # Box
            cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)

            # Label background
            text = f"{label}:{score:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)

            # Text with outline
            cv2.putText(
                out,
                text,
                (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                3,
            )
            cv2.putText(
                out,
                text,
                (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                1,
            )

        return out


REGISTERED_PIPELINES["yolo_plate"] = SingleYoloPipeline
