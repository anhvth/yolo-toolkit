"""Moto plate detection pipeline implementation."""

from __future__ import annotations

from typing import Sequence

import cv2
import torch
import ultralytics
from ultralytics import YOLO

from .base import (
    BasePipeline,
    COLORS,
    DEFAULT_IMG_SIZE,
    DEFAULT_PLATE_CLASS_NAME,
    DEFAULT_PLATE_CONF_THRES,
    DEFAULT_ROI_CLASS_NAME,
    DEFAULT_ROI_CONF_THRES,
    DEFAULT_VEHICLE_CLASS_NAMES,
    DEFAULT_VEH_CONF_THRES,
    register_pipeline,
)


@register_pipeline('yolo_plate')
class SingleYoloPipeline(BasePipeline):
    """Pipeline that keeps a single ROI, vehicle, and optional plate detection."""

    def __init__(
        self,
        model_path: str,
        img_size: int = DEFAULT_IMG_SIZE,
        roi_class_name: str = DEFAULT_ROI_CLASS_NAME,
        vehicle_class_names: Sequence[str] = DEFAULT_VEHICLE_CLASS_NAMES,
        plate_class_name: str = DEFAULT_PLATE_CLASS_NAME,
        roi_conf_thres: float = DEFAULT_ROI_CONF_THRES,
        veh_conf_thres: float = DEFAULT_VEH_CONF_THRES,
        plate_conf_thres: float = DEFAULT_PLATE_CONF_THRES,
    ) -> None:
        super().__init__(model_path, img_size)
        self.model = YOLO(model_path)
        self.class_id_to_name, self.class_name_to_id = self._build_class_mappings()
        self.roi_cls_id = self._get_class_id(roi_class_name)
        self.vehicle_class_ids = [self._get_class_id(n) for n in vehicle_class_names]
        self.plate_cls_id = self._get_class_id(plate_class_name)
        self.roi_conf_thres = roi_conf_thres
        self.veh_conf_thres = veh_conf_thres
        self.plate_conf_thres = plate_conf_thres

    def _build_class_mappings(self) -> tuple[dict[int, str], dict[str, int]]:
        names = self.model.names
        if isinstance(names, dict):
            id_to_name = names
        else:
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

    def postproc(
        self, results: ultralytics.engine.results.Results, expect_plate: bool
    ) -> tuple[ultralytics.engine.results.Results, bool]:
        """Filter model detections to one ROI, one vehicle, and optional plate."""

        xyxy = results.boxes.xyxy.cpu().numpy()
        cls = results.boxes.cls.cpu().numpy().astype(int)
        conf = results.boxes.conf.cpu().numpy()
        selected_indices: list[int] = []
        is_weird = False

        roi_indices = [
            i
            for i, c in enumerate(cls)
            if c == self.roi_cls_id and conf[i] >= self.roi_conf_thres
        ]

        if not roi_indices:
            empty_boxes = torch.empty(0, 6).to(results.boxes.data.device)
            results.boxes.data = empty_boxes
            return results, is_weird

        best_roi_i = max(roi_indices, key=lambda i: conf[i])
        selected_indices.append(best_roi_i)

        veh_candidates = [
            i
            for i, c in enumerate(cls)
            if c in self.vehicle_class_ids and conf[i] >= self.veh_conf_thres
        ]

        if not veh_candidates:
            results.boxes.data = results.boxes.data[selected_indices]
            return results, is_weird

        best_veh_i = max(veh_candidates, key=lambda i: conf[i])
        selected_indices.append(best_veh_i)
        vx1, vy1, vx2, vy2 = xyxy[best_veh_i]

        if expect_plate:
            bx1, by1, bx2, by2 = vx1, vy1, vx2, vy2
            plate_indices: list[int] = []
            for i, c in enumerate(cls):
                if c != self.plate_cls_id or conf[i] < self.plate_conf_thres:
                    continue
                px1, py1, px2, py2 = xyxy[i]
                if px1 >= bx1 and py1 >= by1 and px2 <= bx2 and py2 <= by2:
                    plate_indices.append(i)

            if plate_indices:
                best_plate_i = max(plate_indices, key=lambda i: conf[i])
                selected_indices.append(best_plate_i)
            else:
                vehicle_id = self.class_name_to_id['motobike']
                is_motobike_exists = any(cls[i] == vehicle_id for i in range(len(cls)))
                is_weird = True if is_motobike_exists else False

        results.boxes.data = results.boxes.data[selected_indices]
        return results, is_weird

    def __call__(
        self,
        imgs: Sequence[ultralytics.engine.results.Results],
        expect_plates: Sequence[bool],
    ) -> list[ultralytics.engine.results.Results]:
        """Run inference and post-process each frame."""

        results = self.model.predict(imgs, imgsz=self.img_size, verbose=False, conf=0.001)
        filtered_results: list[ultralytics.engine.results.Results] = []
        for r, e in zip(results, expect_plates):
            filtered_result, _ = self.postproc(r, e)
            filtered_results.append(filtered_result)
        return filtered_results

    def visualize(self, img, preds, thickness: int = 3):
        """Render detections on the source image."""

        H, W = img.shape[:2]
        out = img.copy()

        for class_id, cx, cy, bw, bh, score in preds:
            class_id = int(class_id)
            x1 = int((cx - bw / 2) * W)
            y1 = int((cy - bh / 2) * H)
            x2 = int((cx + bw / 2) * W)
            y2 = int((cy + bh / 2) * H)
            label = self.class_id_to_name.get(class_id, str(class_id))
            color = COLORS[class_id % len(COLORS)]
            cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
            text = f"{label}:{score:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(out, text, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
            cv2.putText(out, text, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

        return out
