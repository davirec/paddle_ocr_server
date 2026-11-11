# """
# Serviço de OCR isolado (roda em um venv/container próprio, só com PaddleOCR).
#
# Iniciar:
#     uvicorn server:app --host 127.0.0.1 --port 8765 --workers 1
#
# O modelo é carregado UMA vez na inicialização e fica em memória
# aguardando requisições.
# """
# import os
# os.environ.setdefault("OMP_NUM_THREADS", "1")
#
# import logging
# import threading
# from contextlib import asynccontextmanager
#
# import cv2
# import numpy as np
# from fastapi import FastAPI, File, HTTPException, UploadFile
# from paddleocr import PaddleOCR
#
# logger = logging.getLogger("ocr_service")
# logging.basicConfig(level=logging.INFO)
#
# DEVICE = os.getenv("OCR_DEVICE", "gpu")
# LANG = os.getenv("OCR_LANG", "en")
#
# state = {"ocr": None}
# # PaddleOCR não é thread-safe: serializa as chamadas ao predict
# ocr_lock = threading.Lock()
#
#
# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     logger.info("Carregando PaddleOCR (device=%s, lang=%s)...", DEVICE, LANG)
#     state["ocr"] = PaddleOCR(
#         device=DEVICE,
#         lang=LANG,
#         use_doc_orientation_classify=False,
#         use_doc_unwarping=False,
#         use_textline_orientation=False,
#     )
#     # Aquecimento: a primeira inferência costuma ser lenta
#     with ocr_lock:
#         state["ocr"].predict(np.zeros((64, 64, 3), dtype=np.uint8))
#     logger.info("OCR pronto.")
#     yield
#     state["ocr"] = None
#
#
# app = FastAPI(title="OCR Service", lifespan=lifespan)
#
#
# @app.get("/health")
# def health():
#     return {"ready": state["ocr"] is not None}
#
#
# @app.post("/detect")
# def detect(image: UploadFile = File(...)):
#     """Recebe uma imagem (PNG/JPEG) e devolve textos + polígonos."""
#     if state["ocr"] is None:
#         raise HTTPException(503, "OCR ainda carregando")
#
#     data = np.frombuffer(image.file.read(), dtype=np.uint8)
#     frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
#     if frame is None:
#         raise HTTPException(400, "Imagem inválida")
#
#     with ocr_lock:
#         result = state["ocr"].predict(frame)
#
#     if not result or "rec_texts" not in result[0]:
#         return {"detections": []}
#
#     texts = result[0]["rec_texts"]
#     boxes = result[0]["dt_polys"]
#
#     detections = [
#         {"text": t, "box": np.asarray(b).tolist()}
#         for t, b in zip(texts, boxes)
#     ]
#     logger.info("Textos: %s", texts)
#     return {"detections": detections}

"""
Serviço de OCR isolado (roda em um venv/container próprio, só com PaddleOCR).

Iniciar (local):
    python server.py

Iniciar com URL pública temporária (Cloudflare, sem conta):
    python server.py --tunnel cloudflare

Iniciar com ngrok (precisa de NGROK_AUTHTOKEN no ambiente):
    python server.py --tunnel ngrok

Variáveis de ambiente opcionais:
    OCR_DEVICE   = gpu | cpu          (padrão: gpu)
    OCR_LANG     = en                 (padrão: en)
    OCR_API_KEY  = uma-chave-secreta  (se definida, o cliente precisa mandar X-API-Key)
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import logging
import secrets
import threading
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from paddleocr import PaddleOCR

logger = logging.getLogger("ocr_service")
logging.basicConfig(level=logging.INFO)

DEVICE = os.getenv("OCR_DEVICE", "gpu")
LANG = os.getenv("OCR_LANG", "en")
API_KEY = os.getenv("OCR_API_KEY")
PORT = int(os.getenv("OCR_PORT", "8765"))
TUNNEL = os.getenv("OCR_TUNNEL", "")  # preenchido pelo --tunnel

state = {"ocr": None}
# PaddleOCR não é thread-safe: serializa as chamadas ao predict
ocr_lock = threading.Lock()


# ---------------------------------------------------------------- túnel
def abrir_tunel(tipo, porta):
    """Abre um túnel público e devolve (url, função_para_fechar)."""
    if tipo == "cloudflare":
        # pip install pycloudflared  (baixa o cloudflared sozinho, sem conta)
        from pycloudflared import try_cloudflare
        urls = try_cloudflare(port=porta, verbose=False)
        return urls.tunnel, lambda: try_cloudflare.terminate(porta)

    if tipo == "ngrok":
        # pip install pyngrok  + variável NGROK_AUTHTOKEN
        from pyngrok import ngrok
        token = os.getenv("NGROK_AUTHTOKEN")
        if token:
            ngrok.set_auth_token(token)
        tunel = ngrok.connect(porta, "http")
        return tunel.public_url, lambda: ngrok.disconnect(tunel.public_url)

    raise ValueError(f"Túnel desconhecido: {tipo}")


# ---------------------------------------------------------------- app
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Carregando PaddleOCR (device=%s, lang=%s)...", DEVICE, LANG)
    state["ocr"] = PaddleOCR(
        device=DEVICE,
        lang=LANG,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )
    # Aquecimento: a primeira inferência costuma ser lenta
    with ocr_lock:
        state["ocr"].predict(np.zeros((64, 64, 3), dtype=np.uint8))
    logger.info("OCR pronto.")

    fechar_tunel = None
    if TUNNEL:
        url, fechar_tunel = abrir_tunel(TUNNEL, PORT)
        print("\n" + "=" * 60)
        print(f"  URL pública: {url}")
        if API_KEY:
            print(f"  X-API-Key:   {API_KEY}")
        print("=" * 60 + "\n", flush=True)

    yield

    if fechar_tunel:
        fechar_tunel()
    state["ocr"] = None


app = FastAPI(title="OCR Service", lifespan=lifespan)


def checar_chave(x_api_key: str = Header(None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(401, "Chave inválida")


@app.get("/health")
def health():
    return {"ready": state["ocr"] is not None}


@app.post("/detect", dependencies=[Depends(checar_chave)])
def detect(image: UploadFile = File(...)):
    """Recebe uma imagem (PNG/JPEG) e devolve textos + polígonos."""
    if state["ocr"] is None:
        raise HTTPException(503, "OCR ainda carregando")

    data = np.frombuffer(image.file.read(), dtype=np.uint8)
    frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(400, "Imagem inválida")

    with ocr_lock:
        result = state["ocr"].predict(frame)

    if not result or "rec_texts" not in result[0]:
        return {"detections": []}

    texts = result[0]["rec_texts"]
    boxes = result[0]["dt_polys"]

    detections = [
        {"text": t, "box": np.asarray(b).tolist()}
        for t, b in zip(texts, boxes)
    ]
    logger.info("Textos: %s", texts)
    return {"detections": detections}


# ---------------------------------------------------------------- main
if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--tunnel", choices=["cloudflare", "ngrok"], default=None)
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args()

    PORT = args.port
    if args.tunnel:
        TUNNEL = args.tunnel
        # URL pública sem chave é perigoso: gera uma se não houver
        if not API_KEY:
            API_KEY = secrets.token_urlsafe(16)

    uvicorn.run(app, host="127.0.0.1", port=PORT, workers=1)