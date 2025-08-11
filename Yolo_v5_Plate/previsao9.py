import torch
from pathlib import Path
import cv2
import pytesseract

# Configurar caminho do Tesseract
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

# --- código principal ---
base_dir = Path(__file__).resolve().parent
model = torch.hub.load('ultralytics/yolov5', 'custom', path=base_dir/'best.pt')

# Configurações do modelo
model.conf = 0.20
model.iou = 0.1
model.augment = False
model.force_reload = True

# Captura da câmera (0 = webcam padrão, ou substitua por IP/USB)
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    coordenadas = detectar_e_recortar_placa(frame, model)

    for (x1, y1, x2, y2) in coordenadas:
        # Desenha retângulo verde ao redor da placa
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Mostra vídeo ao vivo com retângulo
    cv2.imshow("Detecção de Placas", frame)

    # Pressione 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()