import torch
from pathlib import Path
import cv2
import warnings
import numpy as np
import sys
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
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        img_thresh = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, kernel)
        placas_processadas.append(img_thresh)
    return placas_processadas

def encontrar_e_reconstruir_caracteres(img_bin, min_area=100, max_chars=7, debug=False, pad_x=30, pad_y=20):
    img_inv = cv2.bitwise_not(img_bin)
    contornos, _ = cv2.findContours(img_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    altura_img = img_bin.shape[0]

    selecionados = []
    for cnt in contornos:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area < min_area or h < altura_img * 0.4 or w > altura_img * 1.2:
            continue
        selecionados.append((x, y, w, h))

    selecionados.sort(key=lambda t: t[0])
    selecionados = selecionados[:max_chars]

    chars = []
    for (x, y, w, h) in selecionados:
        roi = img_bin[y:y+h, x:x+w]
        chars.append(cv2.resize(roi, (50, 80)))

    if not chars:
        return img_bin

    espaco = 10
    largura_total = sum(c.shape[1] for c in chars) + espaco * (len(chars) - 1) + 2 * pad_x
    altura = max(c.shape[0] for c in chars) + 2 * pad_y
    reconstruida = np.full((altura, largura_total), 255, dtype=np.uint8)

    x_offset = pad_x
    for ch in chars:
        reconstruida[pad_y:pad_y+ch.shape[0], x_offset:x_offset+ch.shape[1]] = ch
        x_offset += ch.shape[1] + espaco

    if debug:
        cv2.imshow("Placa Reconstruida", reconstruida)

    return reconstruida



def desenhar_resultados(frame, coordenadas, textos):
    for i, coord in enumerate(coordenadas):
        x1, y1, x2, y2 = coord
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        if i < len(textos) and textos[i]:
            cv2.putText(frame, textos[i], (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            print(f"Placa: {textos[i]}")

# ---------------------- EXECUÇÃO ----------------------
frame = cv2.imread(str(base_dir / 'imagens/teste26.jpg'))
coordenadas = detectar_e_recortar_placa(frame, model)
placas = aplicar_pre_processamento(frame, coordenadas)

textos = []
for placa_proc in placas:
    reconstruida = encontrar_e_reconstruir_caracteres(placa_proc, debug=True)
    reconstruida_inv = cv2.bitwise_not(reconstruida)  # inverte branco/preto
    reconstruida_bgr = cv2.cvtColor(reconstruida_inv, cv2.COLOR_GRAY2BGR)    
    texto, confs = reader.read(reconstruida_bgr, plate_hint='auto')
    textos.append(texto)
    cv2.imshow("Placa Reconstruida2", reconstruida_bgr)

desenhar_resultados(frame, coordenadas, textos)

cv2.imshow("Resultado Final", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
