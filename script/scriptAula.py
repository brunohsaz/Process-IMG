import torch
from pathlib import Path
import cv2
import warnings

warnings.simplefilter("ignore")

base_dir = Path(__file__).resolve().parent

# ---------------------- YOLOv5 ----------------------
model = torch.hub.load('ultralytics/yolov5', 'custom', path=base_dir/'best.pt')
model.conf = 0.2
model.iou = 0.1
model.augment = False

# ---------------------- DETECTAR E DESENHAR ----------------------
def detectar_e_desenhar_placa(frame, modelo, min_width=50, min_height=20):
    results = modelo(frame)
    boxes = results.xyxy[0]

    for box in boxes:
        x1, y1, x2, y2, conf, cls = box.tolist()
        w = x2 - x1
        h = y2 - y1
        if w >= min_width and h >= min_height:
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

    return frame

# ---------------------- EXECUÇÃO ----------------------
frame = cv2.imread(str(base_dir / 'imagens/teste32.jpg'))
if frame is None:
    print("Erro: Não foi possível carregar a imagem")
    exit()

resultado = detectar_e_desenhar_placa(frame, model)

cv2.imshow("Placa Detectada", resultado)
cv2.waitKey(0)
cv2.destroyAllWindows()
