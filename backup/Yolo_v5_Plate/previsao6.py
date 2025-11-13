import torch
from pathlib import Path
import cv2
import numpy as np
import pytesseract

# Se necessário, configure o caminho do executável do Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def detectar_e_recortar_placa(imagem_path, modelo, min_width=50, min_height=20):
    results = modelo(str(imagem_path))
    boxes = results.xyxy[0]
    coords_filtradas = []

    for box in boxes:
        x1, y1, x2, y2, conf, cls = box.tolist()
        w = x2 - x1
        h = y2 - y1

        if w >= min_width and h >= min_height:
            coords_filtradas.append([int(x1), int(y1), int(x2), int(y2)])

    return coords_filtradas

# --- código principal ---
base_dir = Path(__file__).resolve().parent
model = torch.hub.load('ultralytics/yolov5', 'custom', path=base_dir/'best.pt')

# Configurações
model.conf = 0.20
model.iou = 0.1
model.augment = False
model.force_reload = True

# Caminho da imagem
imagem = base_dir / 'imagens/teste19.jpg'

# Detecta e recorta
coordenadas = detectar_e_recortar_placa(imagem, model, min_width=50, min_height=20)
