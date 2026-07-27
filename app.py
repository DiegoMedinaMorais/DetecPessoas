import sys
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
