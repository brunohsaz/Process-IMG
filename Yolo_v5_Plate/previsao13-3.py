import torch
from pathlib import Path
import cv2
import easyocr
import warnings
import re
from collections import deque, Counter

warnings.simplefilter("ignore")  # ignora warnings temporariamente

base_dir = Path(__file__).resolve().parent

# --- CORREÇÕES DE CARACTERES ---
CORRECOES = {
    '0': ['O', 'Q'],
    '1': ['I', 'L'],
    '2': ['Z'],
    '4': ['A'],
    '5': ['S'],
    '6': ['G'],
    '8': ['B'],
    'O': ['0', 'Q'],
    'I': ['1', 'L'],
    'Z': ['2'],
    'S': ['5'],
    'G': ['6']
}

# --- parâmetros de histórico e consenso ---
HISTORICO_TAMANHO = 5
CONSENSO_FRAMES = 3
MAX_ERROS = 1  # permite 1 caractere errado no consenso

historico_textos = deque(maxlen=HISTORICO_TAMANHO)

# --- Inicializa EasyOCR ---
reader = easyocr.Reader(['en'])  # ou ['pt'] se preferir

# --- funções OCR e pré-processamento ---
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

def aplicar_pre_processamento(frame, coordenadas, crop_ratio_x=0.08, crop_ratio_y=0.15):
    placas_processadas = []
    for x1, y1, x2, y2 in coordenadas:
        largura, altura = x2 - x1, y2 - y1
        margem_x, margem_y = int(largura*crop_ratio_x), int(altura*crop_ratio_y)
        x1, y1, x2, y2 = max(0, x1+margem_x), max(0, y1+margem_y), min(frame.shape[1], x2-margem_x), min(frame.shape[0], y2-margem_y)
        img = frame[y1:y2, x1:x2]
        img = cv2.resize(img, (800, int(800*img.shape[0]/img.shape[1])))
        img_cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_suavizada = cv2.bilateralFilter(img_cinza, 9, 75, 75)
        _, img_thresh = cv2.threshold(img_suavizada, 50, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
        img_thresh = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, kernel)
        placas_processadas.append(img_thresh)
    return placas_processadas

def limpar_texto(texto):
    return re.sub(r'[^A-Z0-9]', '', texto.strip().upper())

def identificar_formato(texto):
    if len(texto) != 7:
        return None
    return "mercosul" if texto[4].isalpha() else "antiga"

def corrigir_char(ch, deve_ser_letra):
    for destino, origens in CORRECOES.items():
        if ch in origens:
            if deve_ser_letra and destino.isalpha():
                return destino
            elif not deve_ser_letra and destino.isdigit():
                return destino
    return ch

def aplicar_correcoes(texto, formato):
    texto_corrigido = ""
    for i, ch in enumerate(texto):
        if formato == "mercosul":
            texto_corrigido += corrigir_char(ch, i in [0, 1, 2, 4])
        elif formato == "antiga":
            texto_corrigido += corrigir_char(ch, i < 3)
        else:
            texto_corrigido += corrigir_char(ch, ch.isalpha())
    return texto_corrigido

def aplicar_ocr(placa_binaria):
    resultado = reader.readtext(placa_binaria, detail=0)
    if not resultado:
        return ""
    texto = resultado[0].replace(" ", "")
    texto = limpar_texto(texto)
    formato = identificar_formato(texto)
    return aplicar_correcoes(texto, formato)

# --- Funções auxiliares ---
def capturar_frame(cap):
    ret, frame = cap.read()
    return frame if ret else None

def redimensionar_frame(frame, largura=640, altura=480):
    return cv2.resize(frame, (largura, altura))

def ajustar_coordenadas(coordenadas, frame_ori, frame_red):
    h_ori, w_ori = frame_ori.shape[:2]
    h_red, w_red = frame_red.shape[:2]
    scale_x = w_ori / w_red
    scale_y = h_ori / h_red
    return [[int(x1*scale_x), int(y1*scale_y), int(x2*scale_x), int(y2*scale_y)] for x1, y1, x2, y2 in coordenadas]

def processar_frame(frame, model):
    coordenadas = detectar_e_recortar_placa(frame, model)
    placas = aplicar_pre_processamento(frame, coordenadas)
    textos = [aplicar_ocr(placa) for placa in placas]
    return coordenadas, textos

def desenhar_resultados(frame, coordenadas, textos):
    for i, coord in enumerate(coordenadas):
        x1, y1, x2, y2 = coord
        if i < len(textos) and textos[i]:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, textos[i], (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            print(f"Placa identificada: {textos[i]}")

# --- Consenso ---
def placas_sao_similares(a, b):
    if len(a) != len(b):
        return False
    erros = sum(1 for x, y in zip(a, b) if x != y)
    return erros <= MAX_ERROS

def consenso_com_tolerancia(historico):
    textos_validos = []
    if not historico: return textos_validos
    num_placas = max(len(h) for h in historico)
    for i in range(num_placas):
        candidatos = [h[i] for h in historico if i < len(h) and h[i]]
        if len(candidatos) >= CONSENSO_FRAMES:
            contagem = Counter()
            for c in candidatos:
                adicionado = False
                for k in contagem:
                    if placas_sao_similares(c, k):
                        contagem[k] += 1
                        adicionado = True
                        break
                if not adicionado:
                    contagem[c] += 1
            placa_mais_comum, freq = contagem.most_common(1)[0]
            if freq >= CONSENSO_FRAMES:
                textos_validos.append(placa_mais_comum)
            else:
                textos_validos.append("")
        else:
            textos_validos.append("")
    return textos_validos

# --- Loop principal ---
def main_loop(cap, model, processar_a_cada=2):
    frame_count = 0
    ultima_coordenadas = []
    ultimo_texto_placas = []

    while True:
        frame = capturar_frame(cap)
        if frame is None:
            break

        frame_count += 1
        frame_red = redimensionar_frame(frame)

        if frame_count % processar_a_cada == 0:
            coordenadas_red, textos = processar_frame(frame_red, model)
            coordenadas = ajustar_coordenadas(coordenadas_red, frame, frame_red)
            ultima_coordenadas = coordenadas
            historico_textos.append(textos)
            ultimo_texto_placas = consenso_com_tolerancia(historico_textos)

        desenhar_resultados(frame, ultima_coordenadas, ultimo_texto_placas)
        cv2.imshow("Video com Reconhecimento", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# --- Configurações do modelo YOLO ---
model = torch.hub.load('ultralytics/yolov5', 'custom', path=base_dir/'best.pt')
model.conf = 0.20
model.iou = 0.1
model.augment = False

# --- Captura de vídeo ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Erro ao acessar a câmera")
    exit()

# --- Executar loop principal ---
main_loop(cap, model)

cap.release()
cv2.destroyAllWindows()
