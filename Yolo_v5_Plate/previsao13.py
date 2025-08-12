import torch
from pathlib import Path
import cv2
import pytesseract
import warnings
import re

warnings.simplefilter("ignore")  # ignora warnings temporariamente

base_dir = Path(__file__).resolve().parent
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

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

def aplicar_pre_processamento(frame, coordenadas, crop_ratio_x=0.08, crop_ratio_y=0.15):
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

def aplicar_ocr(placa_binaria):
    config = r'-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 --psm 8'

    # OCR normal
    texto_normal = pytesseract.image_to_string(placa_binaria, lang='eng', config=config)
    # OCR invertido
    placa_invertida = cv2.bitwise_not(placa_binaria)
    texto_invertido = pytesseract.image_to_string(placa_invertida, lang='eng', config=config)

    # Escolhe o resultado mais longo
    texto = texto_invertido if len(texto_invertido.strip()) > len(texto_normal.strip()) else texto_normal
    texto = texto.strip().upper()

    # Remove qualquer caractere que não seja letra ou número
    texto = re.sub(r'[^A-Z0-9]', '', texto)

    # Correções comuns (mapa de caracteres confusos)
    correcoes = {
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

    def corrigir_char(ch, deve_ser_letra):
        for destino, origem_lista in correcoes.items():
            if ch in origem_lista:
                if deve_ser_letra and destino.isalpha():
                    return destino
                elif not deve_ser_letra and destino.isdigit():
                    return destino
        return ch

    # Escolher formato baseado no tamanho e composição
    formato = None
    if len(texto) == 7:
        # Conta letras e números nas posições finais
        if texto[4].isalpha():
            formato = "mercosul"  # LLLNLNN
        else:
            formato = "antiga"    # LLLNNNN

    # Corrige de acordo com o formato
    texto_corrigido = ""
    for i, ch in enumerate(texto):
        if formato == "mercosul":
            if i in [0, 1, 2, 4]:  # Letras
                texto_corrigido += corrigir_char(ch, True)
            else:  # Números
                texto_corrigido += corrigir_char(ch, False)
        elif formato == "antiga":
            if i < 3:  # Letras
                texto_corrigido += corrigir_char(ch, True)
            else:  # Números
                texto_corrigido += corrigir_char(ch, False)
        else:
            # Caso sem formato definido, tenta corrigir genericamente
            if ch.isalpha():
                texto_corrigido += corrigir_char(ch, True)
            else:
                texto_corrigido += corrigir_char(ch, False)

    return texto_corrigido

# --- código principal ---
model = torch.hub.load('ultralytics/yolov5', 'custom', path=base_dir/'best.pt')

model.conf = 0.20
model.iou = 0.1
model.augment = False
model.force_reload = True

cap = cv2.VideoCapture(0)

frame_count = 0
processar_a_cada = 2  # Processa 1 frame a cada 2 para aliviar carga

ultima_coordenadas = []
ultimo_texto_placas = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    frame_red = cv2.resize(frame, (640, 480))

    if frame_count % processar_a_cada == 0:
        coordenadas = detectar_e_recortar_placa(frame_red, model)

        h_ori, w_ori = frame.shape[:2]
        h_red, w_red = frame_red.shape[:2]
        scale_x = w_ori / w_red
        scale_y = h_ori / h_red
        coordenadas = [[
            int(x1 * scale_x), int(y1 * scale_y),
            int(x2 * scale_x), int(y2 * scale_y)
        ] for (x1, y1, x2, y2) in coordenadas]

        placas = aplicar_pre_processamento(frame, coordenadas)

        texto_placas = []
        for placa in placas:
            texto = aplicar_ocr(placa)
            texto_placas.append(texto)

        ultima_coordenadas = coordenadas
        ultimo_texto_placas = texto_placas

    # Desenha sempre a última detecção e texto
    for i, coord in enumerate(ultima_coordenadas):
        x1, y1, x2, y2 = coord
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if i < len(ultimo_texto_placas):
            cv2.putText(frame, ultimo_texto_placas[i], (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            print(f"Placa {i+1}: {ultimo_texto_placas[i]}")

    cv2.imshow("Vídeo com Detecção", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
