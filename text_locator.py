"""
Cliente leve do serviço de OCR — fica no seu projeto principal.
Dependências: requests, numpy, opencv-python (você já usa para os frames).
Nenhuma dependência do PaddleOCR aqui.
"""
import atexit
import logging
import subprocess
import time
from difflib import SequenceMatcher

import cv2
import numpy as np
import requests

from utils.unit_converter import calcular_ponto_central


def start_ocr_service(python_exe, server_dir, port=8765):
    """
    Sobe o serviço como subprocesso usando o Python do OUTRO venv.
    Ex: start_ocr_service("C:/ocr_service/.venv/Scripts/python.exe", "C:/ocr_service")
    """
    proc = subprocess.Popen(
        [python_exe, "-m", "uvicorn", "server:app",
         "--host", "127.0.0.1", "--port", str(port), "--workers", "1"],
        cwd=server_dir,
    )
    atexit.register(proc.terminate)
    return proc


class TextLocator:

    def __init__(self, logger=None, similarity_threshold=0.8,
                 url="http://127.0.0.1:8765", startup_timeout=180):
        self.logger = logger or logging.getLogger(__name__)
        self.similarity_threshold = similarity_threshold
        self.url = url.rstrip("/")
        self.session = requests.Session()
        self.last_detections = []
        self._wait_ready(startup_timeout)

    def _wait_ready(self, timeout):
        """Bloqueia até o serviço terminar de carregar o modelo."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                r = self.session.get(f"{self.url}/health", timeout=2)
                if r.ok and r.json().get("ready"):
                    self.logger.info("Serviço OCR pronto.")
                    return
            except requests.RequestException:
                pass
            time.sleep(1)
        raise TimeoutError(f"Serviço OCR não respondeu em {self.url}")

    def detect(self, frame):
        """Envia o frame ao serviço e salva as detecções."""
        ok, buf = cv2.imencode(".png", frame)  # sem perda; use ".jpg" se quiser mais velocidade
        if not ok:
            raise ValueError("Falha ao codificar frame")

        r = self.session.post(
            f"{self.url}/detect",
            files={"image": ("frame.png", buf.tobytes(), "image/png")},
            timeout=30,
        )
        r.raise_for_status()

        self.last_detections = [
            {"text": d["text"], "box": np.array(d["box"])}
            for d in r.json()["detections"]
        ]
        self.logger.warning([d["text"] for d in self.last_detections])
        return self.last_detections

    def find_text(self, target_text, frame):
        """Retorna o melhor match de texto."""
        self.detect(frame)
        candidates = []

        for item in self.last_detections:
            score = self._similarity(item["text"], target_text)
            if score >= self.similarity_threshold:
                candidates.append({"text": item["text"], "score": score, "box": item["box"]})

        return max(candidates, key=lambda x: x["score"]) if candidates else None

    def find_text_partial(self, target_text, frame):
        """
        Encontra texto mesmo que esteja dentro de uma frase.
        Ex: "Login" dentro de "Login with Google"
        """
        self.detect(frame)
        target = target_text.lower()
        candidates = []

        for item in self.last_detections:
            detected = item["text"].lower()

            if target in detected:
                candidates.append({"text": item["text"], "score": 1.0, "box": item["box"]})
                continue

            for w in detected.split():
                score = self._similarity(w, target)
                if score >= self.similarity_threshold:
                    candidates.append({"text": item["text"], "score": score, "box": item["box"]})

        return max(candidates, key=lambda x: x["score"]) if candidates else None

    def get_text_center(self, target_text, frame, partial=False):
        """Retorna centro do texto encontrado."""
        match = (self.find_text_partial(target_text, frame) if partial
                 else self.find_text(target_text, frame))
        if not match:
            return None
        return calcular_ponto_central(match["box"])

    def _similarity(self, a, b):
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()