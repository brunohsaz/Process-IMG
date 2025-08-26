import torch
from pathlib import Path
import cv2
import pytesseract
import warnings
import re

warnings.simplefilter("ignore")  # ignora warnings temporariamente

base_dir = Path(__file__).resolve().parent
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# -------- VARIÁVEIS GLOBAIS APLICADAS NAS FUNÇÕES --------

# Configura o modelo do tesseractOCR
OCR_CONFIG = r'-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 --psm 8'

# Array de correções para os caracteres que podem parecer outro
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

def detectar_e_recortar_placa(frame, modelo, min_width=50, min_height=20):
    # Utilizar o modelo YOLO que já é pré-treinado para localizar a placa
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

def aplicar_pre_processamento(frame, coordenadas, crop_ratio_x=0.08, crop_ratio_y=0.15):
    # Aplica pre-processamento para facilitar o reconhecimento dos caracteres
    
    placas_processadas = []
    for x1, y1, x2, y2 in coordenadas:
        largura = x2 - x1
        altura = y2 - y1
        margem_x = int(largura * crop_ratio_x)
        margem_y = int(altura * crop_ratio_y)

        # Corte interno
        x1 += margem_x
        y1 += margem_y
        x2 -= margem_x
        y2 -= margem_y

        # Garantir limites
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)

        img = frame[y1:y2, x1:x2]

        # Redimensionar para mais pixels (melhor OCR)
        img = cv2.resize(img, (800, int(800 * img.shape[0] / img.shape[1])))

        # Escala de cinza
        img_cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Suavização sem perder bordas
        img_suavizada = cv2.bilateralFilter(img_cinza, d=9, sigmaColor=75, sigmaSpace=75)

        # Binarização com Otsu
        _, img_thresh = cv2.threshold(
            img_suavizada, 50, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # Fechamento para reforçar caracteres
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        img_thresh = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, kernel)

        placas_processadas.append(img_thresh)

    return placas_processadas

def ocr_imagem(img):
    # Aplica OCR normal e invertido e retorna o resultado mais confiável
    texto_normal = pytesseract.image_to_string(img, lang='eng', config=OCR_CONFIG)
    img_invertida = cv2.bitwise_not(img)
    texto_invertido = pytesseract.image_to_string(img_invertida, lang='eng', config=OCR_CONFIG)
    return texto_invertido if len(texto_invertido.strip()) > len(texto_normal.strip()) else texto_normal

def limpar_texto(texto):
    #Remove caracteres indesejados e padroniza para maiúsculas
    return re.sub(r'[^A-Z0-9]', '', texto.strip().upper())

def identificar_formato(texto):
    # Identifica se a placa é 'mercosul' ou 'antiga'
    if len(texto) != 7:
        return None
    return "mercosul" if texto[4].isalpha() else "antiga"

def corrigir_char(ch, deve_ser_letra):
    # Corrige caracteres comuns confusos entre letras e números
    for destino, origens in CORRECOES.items():
        if ch in origens:
            if deve_ser_letra and destino.isalpha():
                return destino
            elif not deve_ser_letra and destino.isdigit():
                return destino
    return ch

def aplicar_correcoes(texto, formato):
    # Aplica correções de caracteres baseado no formato da placa
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
    texto = ocr_imagem(placa_binaria)
    texto = limpar_texto(texto)
    formato = identificar_formato(texto)
    return aplicar_correcoes(texto, formato)

def capturar_frame(cap):
    # Captura um frame da câmera
    ret, frame = cap.read()
    if not ret:
        return None
    return frame

def redimensionar_frame(frame, largura=640, altura=480):
    return cv2.resize(frame, (largura, altura))

def ajustar_coordenadas(coordenadas, frame_ori, frame_red):
    # Ajusta coordenadas redimensionadas para o frame original
    h_ori, w_ori = frame_ori.shape[:2]
    h_red, w_red = frame_red.shape[:2]
    scale_x = w_ori / w_red
    scale_y = h_ori / h_red
    return [[
        int(x1 * scale_x), int(y1 * scale_y),
        int(x2 * scale_x), int(y2 * scale_y)
    ] for (x1, y1, x2, y2) in coordenadas]

def processar_frame(frame, model):
    # Detecta placas, aplica preprocessamento e OCR
    coordenadas = detectar_e_recortar_placa(frame, model)
    placas = aplicar_pre_processamento(frame, coordenadas)
    textos = [aplicar_ocr(placa) for placa in placas]
    return coordenadas, textos

def desenhar_resultados(frame, coordenadas, textos):
    # Desenha retângulos e textos das placas no frame
    for i, coord in enumerate(coordenadas):
        x1, y1, x2, y2 = coord
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if i < len(textos):
            cv2.putText(frame, textos[i], (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            print(f"Placa {i+1}: {textos[i]}")
            
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
            ultimo_texto_placas = textos

        desenhar_resultados(frame, ultima_coordenadas, ultimo_texto_placas)
        cv2.imshow("Video com Reconhecimento", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break            


# Configurar modelo para detecção de placas
model = torch.hub.load('ultralytics/yolov5', 'custom', path=base_dir/'best.pt')

model.conf = 0.20
model.iou = 0.1
model.augment = False
# model.force_reload = True

# Código para a captura de vídeo
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Erro ao acessar a camera")
    exit()

processar_a_cada = 2  # Processa 1 frame a cada 2 para aliviar carga

ultima_coordenadas = []
ultimo_texto_placas = []

# Chamada da função principal
main_loop(cap, model)

cap.release()
cv2.destroyAllWindows()
