import cv2
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
