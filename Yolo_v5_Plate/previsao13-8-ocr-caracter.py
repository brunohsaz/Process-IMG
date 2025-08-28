import torch
from pathlib import Path
import cv2
import easyocr
import warnings
import re
import numpy as np
from collections import Counter

warnings.simplefilter("ignore")

base_dir = Path(__file__).resolve().parent

# --- Configuração OCR (EasyOCR) ---
reader = easyocr.Reader(['en'], gpu=False)

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
        _, img_thresh = cv2.threshold(img_suavizada, 75, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        img_thresh = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, kernel)
        placas_processadas.append(img_thresh)
    return placas_processadas


def segmentar_caracteres(img_bin, min_area=100, max_chars=7, debug=False):
    img_inv = cv2.bitwise_not(img_bin)
    contornos, _ = cv2.findContours(img_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidatos = []
    altura_img = img_bin.shape[0]

    for cnt in contornos:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area < min_area:
            continue
        if h < altura_img * 0.4:
            continue
        if w > altura_img * 1.2:
            continue
        candidatos.append((x, y, w, h))

    candidatos = sorted(candidatos, key=lambda c: c[0])
    candidatos = candidatos[:max_chars]

    if debug:
        debug_img = cv2.cvtColor(img_bin, cv2.COLOR_GRAY2BGR)
        for i, (x, y, w, h) in enumerate(candidatos):
            cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(debug_img, str(i+1), (x, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Caracteres Contornados", debug_img)

    return [(x, y, w, h) for (x, y, w, h) in candidatos]


# --- NOVO: reconstruir a placa com caracteres recortados ---
def reconstruir_placa_a_partir_caracteres(img_bin, min_area=100, max_chars=7):
    candidatos = segmentar_caracteres(img_bin, min_area=min_area, max_chars=max_chars, debug=False)

    if not candidatos:
        return None

    altura_padrao = 100
    largura_padrao = 60
    caracteres_norm = []

    for (x, y, w, h) in candidatos:
        char_img = img_bin[y:y+h, x:x+w]
        ch_resized = cv2.resize(char_img, (largura_padrao, altura_padrao))
        caracteres_norm.append(ch_resized)

    largura_total = largura_padrao * len(caracteres_norm)
    nova_img = 255 * np.ones((altura_padrao, largura_total), dtype=np.uint8)

    for i, ch in enumerate(caracteres_norm):
        x_offset = i * largura_padrao
        nova_img[:, x_offset:x_offset+largura_padrao] = ch

    return nova_img


def aplicar_ocr_placa_reconstruida(placa_binaria, mostrar=True):
    nova_img = reconstruir_placa_a_partir_caracteres(placa_binaria)
    if nova_img is None:
        return ""

    if mostrar:
        cv2.imshow("Placa Reconstruída", nova_img)

    results = reader.readtext(nova_img, detail=0, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
    if not results:
        return ""

    texto = "".join(results)
    texto = limpar_texto(texto)
    formato = identificar_formato(texto)
    return aplicar_correcoes(texto, formato)


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

def placas_sao_similares(a,b):
    if len(a)!=len(b): return False
    erros = sum(1 for x,y in zip(a,b) if x!=y)
    return erros <= MAX_ERROS

def consenso_textos(textos):
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
            print(f"Placa: {textos[i]}")

# --- carregamento do modelo ---
model = torch.hub.load('ultralytics/yolov5', 'custom', path=base_dir/'best.pt')
model.conf = 0.2
model.iou = 0.1
model.augment = False

# --- teste com imagem ---
frame = cv2.imread(str(base_dir / 'imagens/teste16.jpg'))
coordenadas = detectar_e_recortar_placa(frame, model)
placas = aplicar_pre_processamento(frame, coordenadas)

for i, placa_proc in enumerate(placas):
    cv2.imshow(f"Placa Processada {i+1}", placa_proc)

# --- OCR reconstruído ---
textos = [aplicar_ocr_placa_reconstruida(p, mostrar=True) for p in placas]
textos = consenso_textos(textos)
desenhar_resultados(frame, coordenadas, textos)

cv2.imshow("Resultado Final", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
