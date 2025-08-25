import torch
from pathlib import Path
import cv2
import easyocr

# Configurar caminhos
base_dir = Path(__file__).resolve().parent

# --- Funções ---
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

        # Remover manchas pretas internas (limpeza)
        kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        img_thresh = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, kernel_clean)

        # Encontrar maior contorno (provavelmente toda área da placa)
        contours, _ = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            img_thresh = img_thresh[y:y+h, x:x+w]

        placas_processadas.append(img_thresh)

    return placas_processadas


# --- Inicializa YOLO ---
model = torch.hub.load('ultralytics/yolov5', 'custom', path=base_dir/'best.pt')
model.conf = 0.20
model.iou = 0.1
model.augment = False
model.force_reload = True

# --- Inicializa EasyOCR com o modelo fine-tuned ---
reader = easyocr.Reader(
    ['en'], 
    model_storage_directory=str(base_dir / 'runs/final_model'),  # pasta do fine-tuning
    recog_network='custom',  # usa o modelo ajustado
)

# --- Leitura da imagem ---
frame = cv2.imread(str(base_dir / 'imagens/teste19.jpg'))

coordenadas = detectar_e_recortar_placa(frame, model)
placas = aplicar_pre_processamento(frame, coordenadas)

# --- OCR e exibição ---
for i, placa in enumerate(placas):
    resultado = reader.readtext(placa, detail=0)
    print(f"Placa {i+1}: {resultado}")
    cv2.imshow(f"Placa {i+1}", placa)

cv2.waitKey(0)
cv2.destroyAllWindows()

