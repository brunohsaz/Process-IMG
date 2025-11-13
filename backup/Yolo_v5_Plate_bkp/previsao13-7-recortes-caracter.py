import torch
from pathlib import Path
import cv2
import easyocr
import warnings
import re
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


def segmentar_caracteres(img_bin, min_area=100, max_chars=7, debug=True):
    # Inverter para letras = branco
    img_inv = cv2.bitwise_not(img_bin)

    # Achar contornos
    contornos, _ = cv2.findContours(img_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidatos = []
    altura_img = img_bin.shape[0]

    for cnt in contornos:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h

        # --- Filtros para descartar o contorno da placa e sujeiras ---
        if area < min_area:  # muito pequeno
            continue
        if h < altura_img * 0.4:  # muito baixo comparado à altura da placa
            continue
        if w > altura_img * 1.2:  # largura exagerada (pega a placa inteira)
            continue

        candidatos.append((x, y, w, h))

    # Ordenar da esquerda para a direita
    candidatos = sorted(candidatos, key=lambda c: c[0])

    # Limitar ao número de caracteres esperado
    candidatos = candidatos[:max_chars]

    caracteres = []
    if debug:
        debug_img = cv2.cvtColor(img_bin, cv2.COLOR_GRAY2BGR)
        for i, (x, y, w, h) in enumerate(candidatos):
            cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(debug_img, str(i+1), (x, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            char_img = img_bin[y:y+h, x:x+w]
            cv2.imshow(f"Caractere {i+1}", char_img)
            caracteres.append(char_img)

        cv2.imshow("Caracteres Contornados", debug_img)
    else:
        caracteres = [img_bin[y:y+h, x:x+w] for (x,y,w,h) in candidatos]

    return caracteres

# --- Alterado: OCR de caractere por caractere ---
def ocr_caracter(img):
    results = reader.readtext(img, detail=0, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
    if not results:
        return ""
    return max(results, key=len)

def aplicar_ocr_por_caractere(placa_binaria):
    caracteres = segmentar_caracteres(placa_binaria)
    texto = ""
    for ch_img in caracteres:
        ch = ocr_caracter(ch_img)
        ch = limpar_texto(ch)
        texto += ch
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
frame = cv2.imread(str(base_dir / 'imagens/teste20.jpg'))
coordenadas = detectar_e_recortar_placa(frame, model)
placas = aplicar_pre_processamento(frame, coordenadas)

for i, placa_proc in enumerate(placas):
    cv2.imshow(f"Placa Processada {i+1}", placa_proc)

# --- AQUI troquei a chamada do OCR ---
textos = [aplicar_ocr_por_caractere(p) for p in placas]
textos = consenso_textos(textos)
desenhar_resultados(frame, coordenadas, textos)

cv2.imshow("Resultado Final", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
