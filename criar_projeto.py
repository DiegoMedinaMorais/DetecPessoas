import os
import shutil

# Conteúdo de cada arquivo
files = {
    "requirements.txt": """ultralytics>=8.0.0
opencv-python>=4.8.0
numpy>=1.24.0
mediapipe>=0.10.0
fastapi>=0.100.0
uvicorn>=0.23.0
websockets>=11.0
pydantic>=2.0.0
python-multipart>=0.0.6
PySide6>=6.5.0
python-dotenv>=1.0.0
""",

    "config.py": """import json
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
""",

    "camera_manager.py": """import cv2
import threading
import time
import numpy as np
import logging

logger = logging.getLogger(__name__)

class CameraManager:
    def __init__(self, source: str, resolution: tuple = (1280, 720), fps_limit: int = 30):
        self.source = source
        self.resolution = resolution
        self.fps_limit = fps_limit
        self.cap = None
        self.lock = threading.Lock()
        self.running = False
        self.frame = None
        self.thread = None

    def start(self):
        if self.running:
            return
        self.cap = cv2.VideoCapture(self._parse_source())
        if not self.cap.isOpened():
            raise RuntimeError(f"Não foi possível abrir a câmera: {self.source}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        logger.info(f"Câmera {self.source} iniciada")

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
        logger.info(f"Câmera {self.source} parada")

    def _parse_source(self):
        if self.source.isdigit():
            return int(self.source)
        return self.source

    def _capture_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                logger.warning(f"Falha ao ler frame da câmera {self.source}, reconectando...")
                time.sleep(0.5)
                self._reconnect()
                continue
            with self.lock:
                self.frame = frame
            time.sleep(1.0 / self.fps_limit)

    def _reconnect(self):
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(self._parse_source())
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])

    def get_frame(self) -> np.ndarray:
        with self.lock:
            if self.frame is not None:
                return self.frame.copy()
            return None

    def is_running(self) -> bool:
        return self.running
""",

    "detector.py": """import torch
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
""",

    "face_detector.py": """import mediapipe as mp
import numpy as np
import cv2
from config import config
import logging

logger = logging.getLogger(__name__)

class FaceDetector:
    def __init__(self, min_detection_confidence: float = None):
        self.min_conf = min_detection_confidence or config.face_confidence
        self.mp_face = mp.solutions.face_detection
        self.face_detection = self.mp_face.FaceDetection(
            model_selection=1, min_detection_confidence=self.min_conf
        )
        logger.info("FaceDetector inicializado com MediaPipe")

    def detect(self, frame: np.ndarray) -> list:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb)
        faces = []
        if results.detections:
            h, w, _ = frame.shape
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x1 = int(bbox.xmin * w)
                y1 = int(bbox.ymin * h)
                x2 = int((bbox.xmin + bbox.width) * w)
                y2 = int((bbox.ymin + bbox.height) * h)
                conf = detection.score[0]
                faces.append((x1, y1, x2, y2, conf))
        return faces
""",

    "counter.py": """import json
import os
import time
from collections import defaultdict
from typing import Dict, Set, List
import logging
import cv2
import numpy as np
from config import config

logger = logging.getLogger(__name__)

class PeopleCounter:
    def __init__(self):
        self.counter_file = config.counter_file
        self.log_dir = config.log_dir
        self.snapshot_dir = config.snapshot_dir
        os.makedirs(os.path.dirname(self.counter_file), exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.snapshot_dir, exist_ok=True)

        self.total_count = 0
        self.known_ids: Set[int] = set()
        self._load_counter()

        self.active_ids: Dict[int, dict] = {}
        self.crossed_ids: Set[int] = set()

        self.line_position = config.line_position
        self.line_orientation = config.line_orientation
        self.roi = config.roi

    def _load_counter(self):
        if os.path.exists(self.counter_file):
            try:
                with open(self.counter_file, 'r') as f:
                    data = json.load(f)
                    self.total_count = data.get('total_people', 0)
                    self.known_ids = set(data.get('known_ids', []))
                    logger.info(f"Contador carregado: total={self.total_count}, ids_conhecidos={len(self.known_ids)}")
            except Exception as e:
                logger.error(f"Erro ao carregar contador: {e}")

    def _save_counter(self):
        try:
            data = {
                'total_people': self.total_count,
                'known_ids': list(self.known_ids)
            }
            with open(self.counter_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Erro ao salvar contador: {e}")

    def update(self, tracks: List[tuple], frame: np.ndarray = None) -> int:
        h, w = frame.shape[:2] if frame is not None else (720, 1280)

        if self.roi is not None:
            x1r, y1r, x2r, y2r = self.roi
            tracks = [t for t in tracks if self._inside_roi(t, x1r, y1r, x2r, y2r)]

        active_ids_now = set()
        for (x1, y1, x2, y2, track_id, conf) in tracks:
            active_ids_now.add(track_id)
            if track_id not in self.active_ids:
                self.active_ids[track_id] = {
                    'first_seen': time.time(),
                    'last_seen': time.time(),
                    'prev_center': None,
                    'confs': [conf]
                }
            else:
                self.active_ids[track_id]['last_seen'] = time.time()
                self.active_ids[track_id]['confs'].append(conf)

            if self.line_position is not None and track_id not in self.crossed_ids:
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                prev_center = self.active_ids[track_id]['prev_center']
                if prev_center is not None:
                    if self._crossed_line(prev_center, (cx, cy), w, h):
                        self.crossed_ids.add(track_id)
                        self._increment(track_id, frame)
                self.active_ids[track_id]['prev_center'] = (cx, cy)

        current_time = time.time()
        for tid in list(self.active_ids.keys()):
            if tid not in active_ids_now and current_time - self.active_ids[tid]['last_seen'] > 5.0:
                del self.active_ids[tid]

        return len(self.active_ids)

    def _inside_roi(self, track, x1r, y1r, x2r, y2r) -> bool:
        x1, y1, x2, y2, _, _ = track
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        return x1r <= cx <= x2r and y1r <= cy <= y2r

    def _crossed_line(self, prev_center, curr_center, frame_w, frame_h) -> bool:
        if self.line_position is None:
            return False
        px, py = prev_center
        cx, cy = curr_center
        if self.line_orientation == "horizontal":
            line_y = self.line_position * frame_h
            return (py < line_y and cy >= line_y) or (py >= line_y and cy < line_y)
        else:
            line_x = self.line_position * frame_w
            return (px < line_x and cx >= line_x) or (px >= line_x and cx < line_x)

    def _increment(self, track_id: int, frame: np.ndarray = None):
        if track_id not in self.known_ids:
            self.known_ids.add(track_id)
            self.total_count += 1
            self._save_counter()
            self._log_detection(track_id)
            if config.save_snapshots and frame is not None:
                self._save_snapshot(track_id, frame)

    def _log_detection(self, track_id: int):
        import csv
        log_file = os.path.join(self.log_dir, f"{time.strftime('%Y-%m-%d')}.csv")
        file_exists = os.path.isfile(log_file)
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['timestamp', 'track_id', 'total'])
            writer.writerow([time.time(), track_id, self.total_count])

    def _save_snapshot(self, track_id: int, frame: np.ndarray):
        date_dir = os.path.join(self.snapshot_dir, time.strftime('%Y-%m-%d'))
        os.makedirs(date_dir, exist_ok=True)
        filename = os.path.join(date_dir, f"person_{track_id:06d}.jpg")
        cv2.imwrite(filename, frame)

    def get_total(self) -> int:
        return self.total_count

    def get_active_count(self) -> int:
        return len(self.active_ids)

    def reset(self):
        self.total_count = 0
        self.known_ids.clear()
        self.active_ids.clear()
        self.crossed_ids.clear()
        self._save_counter()
""",

    "draw.py": """import cv2
import numpy as np
import time
from datetime import datetime
from config import config

def draw_overlay(frame, tracks, faces, fps, active_count, total_count, gpu_info=""):
    h, w = frame.shape[:2]

    for (x1, y1, x2, y2, track_id, conf) in tracks:
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        label = f"ID:{track_id} {conf:.2f}"
        cv2.putText(frame, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    for (x1, y1, x2, y2, conf) in faces:
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        cv2.putText(frame, f"Face {conf:.2f}", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

    info_lines = []
    if config.show_fps:
        info_lines.append(f"FPS: {fps:.1f}")
    if config.show_gpu_info and gpu_info:
        info_lines.append(f"GPU: {gpu_info}")
    info_lines.append(f"Pessoas na cena: {active_count}")
    info_lines.append(f"Total: {total_count}")

    if config.show_time:
        now = datetime.now()
        info_lines.append(f"Data: {now.strftime('%d/%m/%Y')}")
        info_lines.append(f"Hora: {now.strftime('%H:%M:%S')}")

    y_offset = 30
    for line in info_lines:
        cv2.putText(frame, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        y_offset += 30

    if config.line_position is not None:
        if config.line_orientation == "horizontal":
            line_y = int(config.line_position * h)
            cv2.line(frame, (0, line_y), (w, line_y), (0, 0, 255), 2)
        else:
            line_x = int(config.line_position * w)
            cv2.line(frame, (line_x, 0), (line_x, h), (0, 0, 255), 2)

    if config.roi is not None:
        x1, y1, x2, y2 = config.roi
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)

    return frame
""",

    "utils.py": """import torch
import time
import logging

logger = logging.getLogger(__name__)

class FPSCounter:
    def __init__(self, alpha=0.9):
        self.alpha = alpha
        self.fps = 0.0
        self.last_time = time.time()

    def update(self):
        current = time.time()
        delta = current - self.last_time
        self.last_time = current
        if delta > 0:
            self.fps = self.alpha * self.fps + (1 - self.alpha) * (1.0 / delta)
        return self.fps

def get_gpu_info():
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "CPU"
""",

    "api.py": """from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
import asyncio
import json
from datetime import datetime
from typing import List, Dict, Any

app = FastAPI()

state = {
    "people_now": 0,
    "total_people": 0,
    "fps": 0,
    "camera": False,
    "active_ids": []
}

class StatusResponse(BaseModel):
    people_now: int
    total_people: int
    fps: float
    camera: bool

@app.get("/status", response_model=StatusResponse)
async def get_status():
    return StatusResponse(
        people_now=state["people_now"],
        total_people=state["total_people"],
        fps=state["fps"],
        camera=state["camera"]
    )

@app.get("/people")
async def get_people():
    return {"people_now": state["people_now"]}

@app.get("/fps")
async def get_fps():
    return {"fps": state["fps"]}

@app.get("/total")
async def get_total():
    return {"total_people": state["total_people"]}

@app.post("/reset")
async def reset_counter():
    return {"status": "reset requested"}

@app.post("/start")
async def start_camera():
    return {"status": "start requested"}

@app.post("/stop")
async def stop_camera():
    return {"status": "stop requested"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = {
                "people_now": state["people_now"],
                "total_people": state["total_people"],
                "fps": state["fps"],
                "timestamp": datetime.now().isoformat(),
                "active_ids": state["active_ids"]
            }
            await websocket.send_json(data)
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        pass
""",

    "app.py": """import sys
import os
import threading
import time
import logging
import cv2
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QLineEdit, QWidget
)
from PySide6.QtCore import QTimer, Qt, QThread, Signal
from PySide6.QtGui import QImage, QPixmap

from config import config
from camera_manager import CameraManager
from detector import PersonDetector
from face_detector import FaceDetector
from counter import PeopleCounter
from draw import draw_overlay
from utils import FPSCounter, get_gpu_info
import api
import uvicorn

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

counter = PeopleCounter()
detector = PersonDetector()
face_detector = FaceDetector() if config.enable_face_detection else None
fps_counter = FPSCounter()
running = False
camera_manager = None

api_state = api.state

class VideoThread(QThread):
    frame_signal = Signal(np.ndarray)

    def __init__(self, source):
        super().__init__()
        self.source = source
        self.running = True

    def run(self):
        global running, camera_manager
        camera_manager = CameraManager(self.source, config.resolution, config.fps_limit)
        camera_manager.start()
        running = True
        while running:
            frame = camera_manager.get_frame()
            if frame is None:
                continue

            annotated, tracks = detector.track(frame)
            active_count = counter.update(tracks, frame)

            faces = []
            if face_detector:
                faces = face_detector.detect(frame)

            fps = fps_counter.update()
            gpu_info = get_gpu_info() if config.show_gpu_info else ""
            output_frame = draw_overlay(annotated, tracks, faces, fps, active_count, counter.get_total(), gpu_info)

            api_state["people_now"] = active_count
            api_state["total_people"] = counter.get_total()
            api_state["fps"] = fps
            api_state["camera"] = True
            api_state["active_ids"] = list(counter.active_ids.keys())

            self.frame_signal.emit(output_frame)

        camera_manager.stop()
        running = False

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sistema de Monitoramento YOLO")
        self.setGeometry(100, 100, 900, 700)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: #1a1a1a;")
        self.video_label.setMinimumSize(640, 480)
        layout.addWidget(self.video_label)

        controls = QHBoxLayout()
        self.source_combo = QComboBox()
        self.source_combo.addItems(["Webcam (0)", "Arquivo de vídeo", "RTSP"])
        self.source_combo.currentIndexChanged.connect(self.on_source_changed)
        controls.addWidget(QLabel("Fonte:"))
        controls.addWidget(self.source_combo)

        self.source_input = QLineEdit()
        self.source_input.setPlaceholderText("0, caminho, ou rtsp://...")
        controls.addWidget(self.source_input)

        self.start_btn = QPushButton("Iniciar")
        self.start_btn.clicked.connect(self.start_capture)
        controls.addWidget(self.start_btn)

        self.stop_btn = QPushButton("Parar")
        self.stop_btn.clicked.connect(self.stop_capture)
        controls.addWidget(self.stop_btn)

        self.reset_btn = QPushButton("Resetar Contagem")
        self.reset_btn.clicked.connect(self.reset_counter)
        controls.addWidget(self.reset_btn)

        layout.addLayout(controls)

        self.status_label = QLabel("Status: Parado")
        layout.addWidget(self.status_label)

        self.thread = None
        self.current_frame = None

        if config.enable_api:
            threading.Thread(target=self.run_api, daemon=True).start()

    def on_source_changed(self, index):
        if index == 0:
            self.source_input.setText("0")
        elif index == 1:
            self.source_input.setText("caminho/do/video.mp4")
        elif index == 2:
            self.source_input.setText("rtsp://usuario:senha@ip:554/stream")

    def start_capture(self):
        if self.thread and self.thread.isRunning():
            return
        source = self.source_input.text().strip()
        if not source:
            source = "0"
        self.thread = VideoThread(source)
        self.thread.frame_signal.connect(self.update_frame)
        self.thread.start()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_label.setText(f"Status: Rodando - Fonte: {source}")

    def stop_capture(self):
        global running
        running = False
        if self.thread:
            self.thread.running = False
            self.thread.wait()
            self.thread = None
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("Status: Parado")
        api_state["camera"] = False

    def reset_counter(self):
        counter.reset()
        self.status_label.setText("Status: Contagem resetada")

    def update_frame(self, frame):
        self.current_frame = frame
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qt_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(qt_img)
        self.video_label.setPixmap(pixmap.scaled(self.video_label.width(), self.video_label.height(), Qt.KeepAspectRatio))

    def run_api(self):
        uvicorn.run(api.app, host=config.api_host, port=config.api_port, log_level="warning")

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("snapshots", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
""",

    "config.json": """{
  "yolo_model": "yolov8n.pt",
  "confidence": 0.5,
  "iou": 0.45,
  "device": "0",
  "camera_sources": ["0"],
  "resolution": [1280, 720],
  "fps_limit": 30,
  "tracker_type": "bytetrack",
  "track_high_thresh": 0.5,
  "track_low_thresh": 0.1,
  "new_track_thresh": 0.6,
  "track_buffer": 30,
  "match_thresh": 0.8,
  "line_position": 0.6,
  "line_orientation": "horizontal",
  "roi": null,
  "enable_face_detection": true,
  "face_confidence": 0.5,
  "counter_file": "data/counter.json",
  "log_dir": "logs",
  "snapshot_dir": "snapshots",
  "save_snapshots": true,
  "show_fps": true,
  "show_gpu_info": true,
  "show_time": true,
  "api_host": "0.0.0.0",
  "api_port": 8000,
  "enable_api": true,
  "enable_websocket": true
}
"""
}

# Criar diretórios e arquivos
for filename, content in files.items():
    # Se o nome contiver "/", cria subdiretórios
    if "/" in filename:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Criado: {filename}")

print("\nProjeto criado com sucesso!")
print("Execute 'pip install -r requirements.txt' para instalar as dependências.")
print("Depois execute 'python app.py' para iniciar o sistema.")