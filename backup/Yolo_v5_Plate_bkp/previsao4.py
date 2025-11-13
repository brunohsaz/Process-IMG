import torch
from pathlib import Path
import cv2
import numpy as np

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

def aplicar_cinza_com_tons(imagem_path, coordenadas, niveis=64):
    imagem = cv2.imread(str(imagem_path))

    for x1, y1, x2, y2 in coordenadas:
        regiao = imagem[y1:y2, x1:x2]
        regiao_cinza = cv2.cvtColor(regiao, cv2.COLOR_BGR2GRAY)

        fator = 255 // (niveis - 1)
        regiao_cinza = (regiao_cinza // fator) * fator

        # imagem[y1:y2, x1:x2] = cv2.cvtColor(regiao_cinza, cv2.COLOR_GRAY2BGR)

    cv2.imshow("Imagem com tons de cinza", regiao_cinza)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# --- código principal ---
base_dir = Path(__file__).resolve().parent
model = torch.hub.load('ultralytics/yolov5', 'custom', path=base_dir/'best.pt')

# Configurações
model.conf = 0.20
model.iou = 0.1
model.augment = False
model.force_reload = True

# Caminho da imagem
imagem = base_dir / 'imagens/teste14.jpg'

# Detecta e recorta
coordenadas = detectar_e_recortar_placa(imagem, model, min_width=50, min_height=20)

# Aplicar tons de cinza
aplicar_cinza_com_tons(imagem, coordenadas)
