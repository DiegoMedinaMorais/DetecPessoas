import json
import os
import time
import csv
import cv2
import numpy as np
from config import config
import logging

logger = logging.getLogger(__name__)

class PeopleCounter:
    def __init__(self):
        self.counter_file = config.counter_file
        self.log_dir = config.log_dir
        self.snapshot_dir = config.snapshot_dir
        os.makedirs(os.path.dirname(self.counter_file), exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.snapshot_dir, exist_ok=True)

        # Dados persistentes
        self.total_count = 0
        self.known_ids = set()
        self._load_counter()

        # Dados em execução
        self.active_ids = {}  # track_id -> dict
        self.crossed_ids = set()

        # Configurações
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
                'total_people': int(self.total_count),
                'known_ids': [int(i) for i in self.known_ids]
            }
            with open(self.counter_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Erro ao salvar contador: {e}")

    def update(self, tracks, frame=None):
        if frame is None:
            h, w = 720, 1280
        else:
            h, w = frame.shape[:2]

        # Aplica ROI se configurado
        if self.roi is not None:
            x1r, y1r, x2r, y2r = self.roi
            tracks = [t for t in tracks if self._inside_roi(t, x1r, y1r, x2r, y2r)]

        active_ids_now = set()
        for (x1, y1, x2, y2, track_id, conf) in tracks:
            # Garante que track_id seja int nativo
            track_id = int(track_id)
            active_ids_now.add(track_id)

            if track_id not in self.active_ids:
                self.active_ids[track_id] = {
                    'first_seen': time.time(),
                    'last_seen': time.time(),
                    'prev_center': None,
                    'confs': [float(conf)]
                }
            else:
                self.active_ids[track_id]['last_seen'] = time.time()
                self.active_ids[track_id]['confs'].append(float(conf))

            # Verifica cruzamento da linha
            if self.line_position is not None and track_id not in self.crossed_ids:
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                prev_center = self.active_ids[track_id]['prev_center']

                if prev_center is not None:
                    if self._crossed_line(prev_center, (cx, cy), w, h):
                        self.crossed_ids.add(track_id)
                        self._increment(track_id, frame)

                self.active_ids[track_id]['prev_center'] = (cx, cy)

        # Remove IDs inativos (desaparecidos > 5 segundos)
        current_time = time.time()
        for tid in list(self.active_ids.keys()):
            if tid not in active_ids_now and current_time - self.active_ids[tid]['last_seen'] > 5.0:
                del self.active_ids[tid]

        return len(self.active_ids)

    def _inside_roi(self, track, x1r, y1r, x2r, y2r):
        x1, y1, x2, y2, _, _ = track
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        return (x1r <= cx <= x2r) and (y1r <= cy <= y2r)

    def _crossed_line(self, prev_center, curr_center, frame_w, frame_h):
        if self.line_position is None:
            return False

        px, py = prev_center
        cx, cy = curr_center

        if self.line_orientation == "horizontal":
            line_y = self.line_position * frame_h
            return (py < line_y and cy >= line_y) or (py >= line_y and cy < line_y)
        else:  # vertical
            line_x = self.line_position * frame_w
            return (px < line_x and cx >= line_x) or (px >= line_x and cx < line_x)

    def _increment(self, track_id, frame=None):
        if track_id not in self.known_ids:
            self.known_ids.add(track_id)
            self.total_count += 1
            self._save_counter()
            self._log_detection(track_id)
            if config.save_snapshots and frame is not None:
                self._save_snapshot(track_id, frame)

    def _log_detection(self, track_id):
        log_file = os.path.join(self.log_dir, f"{time.strftime('%Y-%m-%d')}.csv")
        file_exists = os.path.isfile(log_file)
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['timestamp', 'track_id', 'total'])
            writer.writerow([time.time(), int(track_id), int(self.total_count)])

    def _save_snapshot(self, track_id, frame):
        date_dir = os.path.join(self.snapshot_dir, time.strftime('%Y-%m-%d'))
        os.makedirs(date_dir, exist_ok=True)
        filename = os.path.join(date_dir, f"person_{int(track_id):06d}.jpg")
        cv2.imwrite(filename, frame)

    def get_total(self):
        return int(self.total_count)

    def get_active_count(self):
        return len(self.active_ids)

    def reset(self):
        self.total_count = 0
        self.known_ids.clear()
        self.active_ids.clear()
        self.crossed_ids.clear()
        self._save_counter()