import torch
from pathlib import Path
import cv2
import pytesseract
import warnings
import re
from collections import Counter

warnings.simplefilter("ignore")

base_dir = Path(__file__).resolve().parent
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

OCR_CONFIG = r'-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 --psm 8'

CORRECOES = {
    '0': ['O', 'Q'], '1': ['I', 'L'], '2': ['Z'], '4': ['A'], '5': ['S'], '6': ['G'], '8': ['B'],
    'O': ['0', 'Q'], 'I': ['1', 'L'], 'Z': ['2'], 'S': ['5'], 'G': ['6']
}

MAX_ERROS = 1  # permite 1 caractere errado no consenso

# --- Funções ---
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

def aplicar_pre_processamento(frame, coordenadas, crop_ratio_x=0.08, crop_ratio_y=0.15, fator_topo=1.4):
    placas_processadas = []
    for x1, y1, x2, y2 in coordenadas:
        largura, altura = x2 - x1, y2 - y1
        margem_x = int(largura * crop_ratio_x)
        margem_y = int(altura * crop_ratio_y)

        # aumentar apenas o corte de cima
        margem_topo = int(margem_y * fator_topo)

        x1 = max(0, x1 + margem_x)
        y1 = max(0, y1 + margem_topo)  # corta mais do topo
        x2 = min(frame.shape[1], x2 - margem_x)
        y2 = min(frame.shape[0], y2 - margem_y)

        img = frame[y1:y2, x1:x2]
        img = cv2.resize(img, (800, int(800 * img.shape[0] / img.shape[1])))
        img_cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_suavizada = cv2.bilateralFilter(img_cinza, 9, 75, 75)
        _, img_thresh = cv2.threshold(img_suavizada, 75, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        img_thresh = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, kernel)
        placas_processadas.append(img_thresh)
    return placas_processadas


def ocr_imagem(img):
    texto_normal = pytesseract.image_to_string(img, lang='eng', config=OCR_CONFIG)
    texto_invertido = pytesseract.image_to_string(cv2.bitwise_not(img), lang='eng', config=OCR_CONFIG)
    return texto_invertido if len(texto_invertido.strip()) > len(texto_normal.strip()) else texto_normal

def limpar_texto(texto):
    return re.sub(r'[^A-Z0-9]', '', texto.strip().upper())

def identificar_formato(texto):
    if len(texto)!=7: return None
    return "mercosul" if texto[4].isalpha() else "antiga"

def corrigir_char(ch, deve_ser_letra):
    for destino, origens in CORRECOES.items():
        if ch in origens:
            if deve_ser_letra and destino.isalpha(): return destino
            elif not deve_ser_letra and destino.isdigit(): return destino
    return ch

def aplicar_correcoes(texto, formato):
    texto_corrigido = ""
    for i, ch in enumerate(texto):
        if formato=="mercosul": texto_corrigido += corrigir_char(ch, i in [0,1,2,4])
        elif formato=="antiga": texto_corrigido += corrigir_char(ch, i<3)
        else: texto_corrigido += corrigir_char(ch, ch.isalpha())
    return texto_corrigido

def aplicar_ocr(placa_binaria):
    texto = ocr_imagem(placa_binaria)
    texto = limpar_texto(texto)
    formato = identificar_formato(texto)
    return aplicar_correcoes(texto, formato)

def placas_sao_similares(a,b):
    if len(a)!=len(b): return False
    erros = sum(1 for x,y in zip(a,b) if x!=y)
    return erros <= MAX_ERROS

def consenso_textos(textos):
    # Remove duplicados parecidos
    textos_validos = []
    for t in textos:
        if t and all(not placas_sao_similares(t,v) for v in textos_validos):
            textos_validos.append(t)
    return textos_validos

def desenhar_resultados(frame, coordenadas, textos):
    for i, coord in enumerate(coordenadas):
        x1,y1,x2,y2 = coord
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
        if i < len(textos) and textos[i]:
            cv2.putText(frame,textos[i],(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            print(f"Placa: {textos[i]}")  # imprime apenas se houver texto

# --- carregamento do modelo ---
model = torch.hub.load('ultralytics/yolov5', 'custom', path=base_dir/'best.pt')
model.conf = 0.2
model.iou = 0.1
model.augment = False

# --- teste com imagem ---
frame = cv2.imread(str(base_dir / 'imagens/teste16.jpg'))
coordenadas = detectar_e_recortar_placa(frame, model)
placas = aplicar_pre_processamento(frame, coordenadas)

# Mostrar cada placa processada
for i, placa_proc in enumerate(placas):
    cv2.imshow(f"Placa Processada {i+1}", placa_proc)

textos = [aplicar_ocr(p) for p in placas]
textos = consenso_textos(textos)
desenhar_resultados(frame, coordenadas, textos)

cv2.imshow("Resultado Final", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
