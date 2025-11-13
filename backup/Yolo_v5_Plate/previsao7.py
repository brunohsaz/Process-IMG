import torch
from pathlib import Path
import cv2
import numpy as np
import pytesseract

# Configurar o caminho do executável do Tesseract se necessário
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

# Configurações do modelo
model.conf = 0.20
model.iou = 0.1
model.augment = False
model.force_reload = True

while True:
    nome_arquivo = input("Digite o nome da imagem (ou 'sair' para encerrar): ").strip()
    if nome_arquivo.lower() == 'sair':
        break

    imagem_path = base_dir / 'imagens' / nome_arquivo

    if not imagem_path.exists():
        print(f"Arquivo '{nome_arquivo}' não encontrado.\n")
        continue

    img = cv2.imread(str(imagem_path))
    coordenadas = detectar_e_recortar_placa(imagem_path, model)

    for (x1, y1, x2, y2) in coordenadas:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    saida_path = base_dir / 'imagens' / f'detectado_{nome_arquivo}'
    cv2.imwrite(str(saida_path), img)

    print(f"Imagem salva como: detectado_{nome_arquivo}\n")
