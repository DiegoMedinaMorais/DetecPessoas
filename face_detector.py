import cv2
import numpy as np
from config import config
import logging
import os

logger = logging.getLogger(__name__)

class FaceDetector:
    def __init__(self, conf_threshold: float = None):
        self.conf_threshold = conf_threshold or config.face_confidence
        model_path = "models/res10_300x300_ssd_iter_140000_fp16.caffemodel"
        config_path = "models/deploy.prototxt"
        
        # Verifica se os arquivos existem
        if not os.path.exists(model_path) or not os.path.exists(config_path):
            logger.error("Arquivos do modelo de face DNN não encontrados. "
                         "Baixe-os de https://github.com/opencv/opencv_extra e coloque em 'models/'")
            self.net = None
        else:
            self.net = cv2.dnn.readNetFromCaffe(config_path, model_path)
            logger.info("FaceDetector DNN inicializado com sucesso")

    def detect(self, frame: np.ndarray) -> list:
        """
        Detecta rostos usando OpenCV DNN.
        Retorna lista de (x1, y1, x2, y2, conf)
        """
        if self.net is None:
            return []
        
        h, w = frame.shape[:2]
        # Pré-processamento para o modelo
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        detections = self.net.forward()
        
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.conf_threshold:
                # Coordenadas relativas (0-1) e converte para pixels
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")
                # Garantir que não ultrapasse os limites do frame
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)
                faces.append((x1, y1, x2, y2, float(confidence)))
        return faces