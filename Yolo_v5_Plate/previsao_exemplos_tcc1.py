import torch
from pathlib import Path
import cv2
import warnings
import numpy as np
import sys
import requests
sys.path.append(str(Path(__file__).resolve().parent.parent))
from lib.plate_reader import PlateReader

warnings.simplefilter("ignore")

base_dir = Path(__file__).resolve().parent

# ---------------------- YOLOv5 ----------------------
model = torch.hub.load('ultralytics/yolov5', 'custom', path=base_dir/'best.pt')
model.conf = 0.2
model.iou = 0.1
model.augment = False

# ---------------------- PlateReader ----------------------
templates_path = Path(__file__).resolve().parent.parent / 'lib' / 'templates'
reader = PlateReader(template_root=str(templates_path))

# ---------------------- PIPELINE ----------------------
def detectar_e_recortar_placa(frame, modelo, min_width=50, min_height=20):
    results = modelo(frame)
    boxes = results.xyxy[0]
    coords_filtradas = []
    for box in boxes:
        x1, y1, x2, y2, conf, cls = box.tolist()
        w = x2 - x1
        h = y2 - y1
        if w >= min_width and h >= min_height:
            coords_filtradas.append([int(x1), int(y1), int(x2), int(y2)])
    return coords_filtradas

def aplicar_pre_processamento(frame, coordenadas, crop_ratio_x=0.07, crop_ratio_y=0.15, fator_topo=1.4):
    placas_processadas = []
    for x1, y1, x2, y2 in coordenadas:
        largura, altura = x2 - x1, y2 - y1
        margem_x = int(largura * crop_ratio_x)
        margem_y = int(altura * crop_ratio_y)
        margem_topo = int(margem_y * fator_topo)

        x1 = max(0, x1 + margem_x)
        y1 = max(0, y1 + margem_topo)
        x2 = min(frame.shape[1], x2 - margem_x)
        y2 = min(frame.shape[0], y2 - margem_y)

        img = frame[y1:y2, x1:x2]
        img = cv2.resize(img, (800, int(800 * img.shape[0] / img.shape[1])))
        img_cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_suavizada = cv2.bilateralFilter(img_cinza, 9, 75, 75)
        _, img_thresh = cv2.threshold(img_suavizada, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        img_thresh = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, kernel)
        placas_processadas.append(img_thresh)
    return placas_processadas

# ---------------------- EXECUÇÃO HÍBRIDA (MELHOR DOS DOIS MUNDOS) ----------------------
frame = cv2.imread(str(base_dir / 'imagens/teste36.jpg'))
if frame is None:
    print("Erro: Não foi possível carregar a imagem")
    exit()

# 1. DETECÇÃO
coordenadas = detectar_e_recortar_placa(frame, model)

# Mostra imagem original com as caixas detectadas
frame_detectado = frame.copy()
for (x1, y1, x2, y2) in coordenadas:
    cv2.rectangle(frame_detectado, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imshow("Resultado da Detecção (Função 1)", frame_detectado)
cv2.imwrite("resultado_detectar_placa_1.jpg", frame_detectado)  # salva para o TCC

# 2. PRÉ-PROCESSAMENTO
placas_processadas = aplicar_pre_processamento(frame, coordenadas)

# Exibe e salva cada imagem pré-processada
for i, placa in enumerate(placas_processadas):
    titulo = f"Placa Processada {i+2} (Função 2)"
    cv2.imshow(titulo, placa)
    cv2.imwrite(f"resultado_preprocessada_{i+1}.jpg", placa)

cv2.waitKey(0)
cv2.destroyAllWindows()