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

def extrair_rois_dos_caracteres(placa_binaria, min_area=100, max_chars=7):
    """
    Usa a imagem binária da placa para encontrar ROIs (caracteres).
    Agora com filtros adicionais para reduzir falsos positivos.
    Retorna caracteres com letra branca e fundo preto.
    """
    img_inv = cv2.bitwise_not(placa_binaria)  
    contornos, _ = cv2.findContours(img_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    altura_img, largura_img = placa_binaria.shape[:2]

    candidatos = []
    for cnt in contornos:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h

        # --- FILTROS ---
        if area < min_area:
            continue
        if h < altura_img * 0.4 or h > altura_img * 0.95:  
            continue
        if w > altura_img * 1.2 or w < altura_img * 0.1:  
            continue
        if w/h > 1.2:  
            continue

        candidatos.append((x, y, w, h))

    # --- Ordenar pela posição X ---
    candidatos.sort(key=lambda t: t[0])

    # --- Calcular largura média e remover outliers ---
    if candidatos:
        larguras = [w for (_, _, w, _) in candidatos]
        largura_media = np.mean(larguras)
        candidatos = [
            (x, y, w, h) for (x, y, w, h) in candidatos
            if 0.5*largura_media <= w <= 1.8*largura_media
        ]

    # --- Limitar ao número máximo de caracteres ---
    candidatos = candidatos[:max_chars]

    # --- Recortar ROIs ---
    rois = []
    for (x, y, w, h) in candidatos:
        roi = placa_binaria[y:y+h, x:x+w]
        roi = cv2.bitwise_not(roi)  
        rois.append(roi)

    return rois

def desenhar_resultados(frame, coordenadas, textos):
    for i, coord in enumerate(coordenadas):
        x1, y1, x2, y2 = coord
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        if i < len(textos) and textos[i]:
            cv2.putText(frame, textos[i], (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            print(f"Placa: {textos[i]}")

# ---------------------- EXECUÇÃO HÍBRIDA (MELHOR DOS DOIS MUNDOS) ----------------------
frame = cv2.imread(str(base_dir / 'imagens/teste20.jpg'))
if frame is None:
    print(f"Erro: Não foi possível carregar a imagem")
    exit()

coordenadas = detectar_e_recortar_placa(frame, model)
placas_processadas = aplicar_pre_processamento(frame, coordenadas) # Usamos seu pré-processamento

textos = []
for i, placa_proc in enumerate(placas_processadas):
    # 1. Usamos sua lógica para extrair os ROIs dos caracteres
    rois_caracteres = extrair_rois_dos_caracteres(placa_proc)

    # Para debug, vamos visualizar os ROIs extraídos
    for j, roi in enumerate(rois_caracteres):
        cv2.imshow(f"Placa {i+1} - ROI {j+1}", roi)
    
    if rois_caracteres:
        # 2. Passamos a lista de ROIs para o novo método do PlateReader
        # texto, confs = reader.read_from_rois(rois_caracteres, plate_hint='auto')
        texto, confs = reader.read_from_rois_hybrid(rois_caracteres, plate_hint='auto')
        textos.append(texto)
    else:
        textos.append("") # Se nenhum ROI foi encontrado

desenhar_resultados(frame, coordenadas, textos)

cv2.imshow("Resultado Final", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()