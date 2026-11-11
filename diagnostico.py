"""
Testa o PaddleOCR direto, sem o serviço, para saber onde está o problema.
Rode com o Python do venv do serviço:

    .venv/bin/python diagnostico.py caminho/da/foto.png
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys

import cv2
import paddle
from paddleocr import PaddleOCR

caminho = sys.argv[1]
frame = cv2.imread(caminho, cv2.IMREAD_COLOR)
print("Imagem:", frame.shape, frame.dtype)
print("Paddle:", paddle.__version__, "| compilado com CUDA:", paddle.is_compiled_with_cuda())
paddle.utils.run_check()


def testar(device, **extra):
    print(f"\n===== device={device} {extra} =====")
    ocr = PaddleOCR(
        device=device,
        lang="en",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        **extra,
    )
    res = ocr.predict(frame)[0]
    for t, s in zip(res["rec_texts"], res["rec_scores"]):
        print(f"{s:.2f}  {t}")
    print("Total de caixas:", len(res["rec_texts"]))


testar("cpu")
testar("cpu", enable_mkldnn=False)
if paddle.is_compiled_with_cuda():
    testar("gpu")