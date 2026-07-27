import torch
import numpy as np
from ultralytics import YOLO
import logging
from config import config

logger = logging.getLogger(__name__)

class PersonDetector:
    def __init__(self, model_path: str = None, device: str = None):
        self.model_path = model_path or config.yolo_model
        self.device = device or config.device
        if self.device == "0" and not torch.cuda.is_available():
            self.device = "cpu"
            logger.warning("GPU não disponível, usando CPU")
        self.model = YOLO(self.model_path)
        self.model.to(self.device)
        self.confidence = config.confidence
        self.iou = config.iou
        logger.info(f"YOLO carregado em {self.device}")

    def detect(self, frame: np.ndarray) -> list:
        results = self.model(frame, conf=self.confidence, iou=self.iou, classes=[0], verbose=False)
        detections = []
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = box.conf[0].item()
                    cls_id = int(box.cls[0].item())
                    if cls_id == 0:
                        detections.append((x1, y1, x2, y2, conf, cls_id))
        return detections

    def track(self, frame: np.ndarray, persist: bool = True) -> tuple:
        results = self.model.track(
            frame,
            persist=persist,
            conf=self.confidence,
            iou=self.iou,
            classes=[0],
            tracker="bytetrack.yaml",
            verbose=False
        )
        annotated_frame = results[0].plot()
        tracks = []
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            confs = results[0].boxes.conf.cpu().numpy()
            for box, track_id, conf in zip(boxes, ids, confs):
                x1, y1, x2, y2 = box
                tracks.append((x1, y1, x2, y2, track_id, conf))
        return annotated_frame, tracks
