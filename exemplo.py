"""
Exemplo mínimo: envia uma foto ao serviço de OCR e imprime a resposta.

Uso:
    python exemplo_ocr.py caminho/da/foto.png
"""
import sys

import requests

URL = "http://127.0.0.1:8765"

caminho = sys.argv[1] if len(sys.argv) > 1 else "img.png"

# 1. Confere se o serviço está no ar e com o modelo carregado
status = requests.get(f"{URL}/health", timeout=5).json()
print("Serviço pronto?", status["ready"])

# 2. Envia a foto
with open(caminho, "rb") as f:
    resposta = requests.post(
        f"{URL}/detect",
        files={"image": (caminho, f, "image/png")},
        timeout=60,
    )

resposta.raise_for_status()

# 3. Mostra o resultado
for d in resposta.json()["detections"]:
    print(f"{d['text']!r:30}  caixa={d['box']}")