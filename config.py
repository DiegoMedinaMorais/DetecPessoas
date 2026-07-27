import json
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

@dataclass
class Config:
    # YOLO
    yolo_model: str = "yolov8n.pt"
    confidence: float = 0.5
    iou: float = 0.45
    device: str = "0"  # "0" para GPU, "cpu" para CPU

    # Câmera
    camera_sources: List[str] = field(default_factory=lambda: ["0"])
    resolution: Tuple[int, int] = (1280, 720)
    fps_limit: int = 30

    # Rastreamento
    tracker_type: str = "bytetrack"  # "bytetrack" ou "botsort"
    track_high_thresh: float = 0.5
    track_low_thresh: float = 0.1
    new_track_thresh: float = 0.6
    track_buffer: int = 30
    match_thresh: float = 0.8

    # Linha de contagem (opcional)
    line_position: Optional[float] = None   # 0.0 a 1.0 (proporção)
    line_orientation: str = "horizontal"    # "horizontal" ou "vertical"

    # ROI (opcional) - coordenadas em pixels (x1, y1, x2, y2)
    roi: Optional[Tuple[int, int, int, int]] = None

    # Face detection
    enable_face_detection: bool = True
    face_confidence: float = 0.5

    # Persistência
    counter_file: str = "data/counter.json"
    log_dir: str = "logs"
    snapshot_dir: str = "snapshots"
    save_snapshots: bool = False

    # Interface
    show_fps: bool = True
    show_gpu_info: bool = True
    show_time: bool = True

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    enable_api: bool = True
    enable_websocket: bool = True

    @classmethod
    def from_json(cls, path: str = "config.json") -> "Config":
        if os.path.exists(path):
            with open(path, "r") as f:
                data = json.load(f)
            return cls(**data)
        return cls()

    def to_json(self, path: str = "config.json"):
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=2)

config = Config.from_json()
