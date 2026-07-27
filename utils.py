import torch
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
