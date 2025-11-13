import torch
from pathlib import Path
import cv2
import pytesseract

# Configurar caminho do Tesseract
base_dir = Path(__file__).resolve().parent
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def detectar_e_recortar_placa(frame, modelo, min_width=50, min_height=20):
    results = modelo(frame)  # Passa o frame direto para o YOLO
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
        # Calcula margens internas
        largura = x2 - x1
        altura = y2 - y1
        margem_x = int(largura * crop_ratio_x)
        margem_y = int(altura * crop_ratio_y)

        # Aplica corte interno para remover bordas e para-choques
        x1 += margem_x
        y1 += margem_y
        x2 -= margem_x
        y2 -= margem_y

        # Garante que os valores ainda são válidos
        if x1 < 0: x1 = 0
        if y1 < 0: y1 = 0
        if x2 > frame.shape[1]: x2 = frame.shape[1]
        if y2 > frame.shape[0]: y2 = frame.shape[0]

        img = frame[y1:y2, x1:x2]
        img = cv2.resize(img, (560, 360))

        # Aplica filtro de média para reduzir ruídos
        img = cv2.blur(img, (3, 3))
        # Aplica filtro de mediana
        img = cv2.medianBlur(img, 3)
        # Converte para escala de cinza
        img_cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Aplica filtro bilateral para suavizar sem perder bordas
        img_suavizada = cv2.bilateralFilter(img_cinza, d=9, sigmaColor=75, sigmaSpace=75)
        
        # Binarização simples
        _, img_thresh = cv2.threshold(
            img_suavizada,
            80,
            255,
            cv2.THRESH_BINARY
        )
        
        placas_processadas.append(img_thresh)

    return placas_processadas


# --- código principal ---
model = torch.hub.load('ultralytics/yolov5', 'custom', path=base_dir/'best.pt')

# Configurações do modelo
model.conf = 0.20
model.iou = 0.1
model.augment = False
model.force_reload = True

frame = cv2.imread(str(base_dir / 'imagens/teste18.jpg'))

coordenadas = detectar_e_recortar_placa(frame, model)
placas = aplicar_pre_processamento(frame, coordenadas)

# Mostra todas as placas processadas
for i, placa in enumerate(placas):
    cv2.imshow(f"Placa {i+1}", placa)

cv2.waitKey(0)
cv2.destroyAllWindows()
