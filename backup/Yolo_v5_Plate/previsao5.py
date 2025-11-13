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

def aplicar_cinza_com_ocr(imagem_path, coordenadas, niveis=64):
    imagem = cv2.imread(str(imagem_path))

    for i, (x1, y1, x2, y2) in enumerate(coordenadas):
        regiao = imagem[y1:y2, x1:x2]
        
        # Converte pra escala de cinza
        regiao_cinza = cv2.cvtColor(regiao, cv2.COLOR_BGR2GRAY)

        # Aplica filtro bilateral pra suavizar sem perder bordas
        regiao_suavizada = cv2.bilateralFilter(regiao_cinza, d=9, sigmaColor=75, sigmaSpace=75)

        # Faz binarização com threshold adaptativo
        regiao_thresh = cv2.adaptiveThreshold(
            regiao_suavizada,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=11,
            C=2
        )

        # Opcional: aplicar morfologia pra remover ruído (erosão seguida de dilatação)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        regiao_morph = cv2.morphologyEx(regiao_thresh, cv2.MORPH_OPEN, kernel)

        # OCR com config mais focado pra leitura de placas
        config = '--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        texto = pytesseract.image_to_string(regiao_morph, config=config)
        print(f"Região {i+1}: {texto.strip()}")
        
        cv2.imshow("Imagem",regiao_morph)
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
imagem = base_dir / 'imagens/teste16.jpg'

# Detecta e recorta
coordenadas = detectar_e_recortar_placa(imagem, model, min_width=50, min_height=20)

# Aplica tons de cinza + OCR
aplicar_cinza_com_ocr(imagem, coordenadas)
