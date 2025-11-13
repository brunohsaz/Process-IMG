import torch
from pathlib import Path
import cv2
import warnings
import numpy as np
import sys
import pytesseract
from PIL import Image # Importar a biblioteca PIL

warnings.simplefilter("ignore")

base_dir = Path(__file__).resolve().parent

# --- Configuração do Tesseract ---
# (Se necessário) Adicione o caminho para o executável do Tesseract, ex:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# ---------------------- YOLOv5 ----------------------
model = torch.hub.load('ultralytics/yolov5', 'custom', path=base_dir/'best.pt')
model.conf = 0.2
model.iou = 0.1
model.augment = False

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
    """
    Esta função é mantida, pois o pré-processamento (binarização)
    ajuda o Tesseract.
    """
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
        if img.size == 0:
            continue
            
        img = cv2.resize(img, (800, int(800 * img.shape[0] / img.shape[1])))
        img_cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_suavizada = cv2.bilateralFilter(img_cinza, 9, 75, 75)
        _, img_thresh = cv2.threshold(img_suavizada, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        img_thresh = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, kernel)
        placas_processadas.append(img_thresh)
    return placas_processadas

# A função extrair_rois_dos_caracteres() foi REMOVIDA, pois não é mais necessária.

def desenhar_resultados(frame, coordenadas, textos):
    for i, coord in enumerate(coordenadas):
        x1, y1, x2, y2 = coord
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        if i < len(textos) and textos[i]:
            # Limpa o texto para exibir
            texto_placa = "".join(filter(str.isalnum, textos[i])).upper()
            if texto_placa:
                cv2.putText(frame, texto_placa, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                print(f"Placa (Tesseract): {texto_placa}")

# ---------------------- EXECUÇÃO COM TESSERACT ----------------------
frame = cv2.imread(str(base_dir / 'imagens/teste33.jpg'))
if frame is None:
    print(f"Erro: Não foi possível carregar a imagem")
    exit()

coordenadas = detectar_e_recortar_placa(frame, model)
placas_processadas = aplicar_pre_processamento(frame, coordenadas) # Usamos seu pré-processamento

textos = []
for placa_proc in placas_processadas:
    # O Tesseract funciona melhor com imagens PIL
    pil_img = Image.fromarray(placa_proc)
    
    # Configuração do Tesseract:
    # --psm 7: Trata a imagem como uma única linha de texto.
    # -c tessedit_char_whitelist: Restringe os caracteres (ótimo para placas)
    config_tesseract = '--psm 9 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    
    try:
        texto = pytesseract.image_to_string(pil_img, config=config_tesseract)
        textos.append(texto)
    except Exception as e:
        print(f"Erro no Tesseract: {e}")
        textos.append("")

desenhar_resultados(frame, coordenadas, textos)

cv2.imshow("Resultado Final (Tesseract)", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()