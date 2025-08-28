import torch
from pathlib import Path
import cv2
import easyocr
import warnings
import re

warnings.simplefilter("ignore")

base_dir = Path(__file__).resolve().parent

# --- Inicializa EasyOCR (somente inglês, mas funciona bem em placas) ---
reader = easyocr.Reader(['en'], gpu=False)

# --- Correções e formatos ---
CORRECOES = {
    '0': ['O', 'Q', 'D'],
    '1': ['I', 'L'],
    '2': ['Z'],
    '4': ['A', 'L'],   # <- adicionado L aqui
    '5': ['S'],
    '6': ['G'],
    '8': ['B'],
    'O': ['0', 'Q'],
    'I': ['1', 'L'],
    'Z': ['2'],
    'S': ['5'],
    'G': ['6'],
    'L': ['1', '4']   # <- também mapeado para corrigir casos de L ser na verdade 4
}

PADRAO_ANTIGA = re.compile(r'^[A-Z]{3}[0-9]{4}$')
PADRAO_MERCOSUL = re.compile(r'^[A-Z]{3}[0-9][A-Z][0-9]{2}$')
MAX_ERROS = 1

# --- Funções ---
def detectar_e_recortar_placa(frame, modelo, min_width=50, min_height=20):
    results = modelo(frame)
    boxes = results.xyxy[0]
    coords_filtradas = []
    for box in boxes:
        x1, y1, x2, y2, conf, cls = box.tolist()
        w, h = x2 - x1, y2 - y1
        if w >= min_width and h >= min_height:
            coords_filtradas.append([int(x1), int(y1), int(x2), int(y2)])
    return coords_filtradas

def aplicar_pre_processamento(frame, coordenadas, crop_ratio_x=0.08, crop_ratio_y=0.15, fator_topo=1.4):
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

        # CLAHE melhora contraste em iluminação ruim
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(5,5))
        img_eq = clahe.apply(img_cinza)

        # Limiar adaptativo
        img_bin = cv2.adaptiveThreshold(img_eq, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 67, 11)

        # Pequeno fechamento morfológico para unir falhas nas letras
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
        img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_CLOSE, kernel)

        placas_processadas.append(img_bin)
    return placas_processadas

def limpar_texto(texto):
    return re.sub(r'[^A-Z0-9]', '', texto.strip().upper())

def identificar_formato(texto):
    if PADRAO_ANTIGA.match(texto): return "antiga"
    if PADRAO_MERCOSUL.match(texto): return "mercosul"
    return None

def corrigir_char(ch, deve_ser_letra):
    for destino, origens in CORRECOES.items():
        if ch in origens:
            if deve_ser_letra and destino.isalpha(): return destino
            elif not deve_ser_letra and destino.isdigit(): return destino
    return ch

def aplicar_correcoes(texto, formato):
    texto_corrigido = ""
    for i, ch in enumerate(texto):
        if formato == "mercosul":
            texto_corrigido += corrigir_char(ch, i in [0,1,2,4])
        elif formato == "antiga":
            texto_corrigido += corrigir_char(ch, i < 3)
        else:
            texto_corrigido += corrigir_char(ch, ch.isalpha())
    return texto_corrigido

def aplicar_ocr(img):
    resultados = reader.readtext(img, detail=0, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
    if not resultados: return ""
    texto = limpar_texto(resultados[0])
    formato = identificar_formato(texto)
    # return aplicar_correcoes(texto, formato)
    return texto

def placas_sao_similares(a, b):
    if len(a) != len(b): return False
    erros = sum(1 for x, y in zip(a, b) if x != y)
    return erros <= MAX_ERROS

def consenso_textos(textos):
    textos_validos = []
    for t in textos:
        if t and all(not placas_sao_similares(t, v) for v in textos_validos):
            textos_validos.append(t)
    return textos_validos

def desenhar_resultados(frame, coordenadas, textos):
    for i, coord in enumerate(coordenadas):
        x1, y1, x2, y2 = coord
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if i < len(textos) and textos[i]:
            cv2.putText(frame, textos[i], (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            print(f"Placa: {textos[i]}")

# --- carregamento do modelo YOLO ---
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
