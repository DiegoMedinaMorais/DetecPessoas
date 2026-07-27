import cv2
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
