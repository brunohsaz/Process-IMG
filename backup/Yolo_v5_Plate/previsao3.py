import torch
from pathlib import Path
from PIL import Image

def detectar_e_recortar_placa(imagem_path, modelo, min_width=50, min_height=20, salvar=True):
    results = modelo(imagem_path)
    boxes = results.xyxy[0]
    imagem = Image.open(imagem_path)
    coords_filtradas = []

    for i, box in enumerate(boxes):
        x1, y1, x2, y2, conf, cls = box.tolist()
        w = x2 - x1
        h = y2 - y1

        if w >= min_width and h >= min_height:
            coords_filtradas.append([x1, y1, x2, y2])

    return coords_filtradas

def aplicar_cinza_com_tons(imagem_path, coordenadas, niveis=64):
    imagem = Image.open(imagem_path).convert('RGB')
    for x1, y1, x2, y2 in coordenadas:
        regiao = imagem.crop((x1, y1, x2, y2)).convert('L')

        # Reduz o número de tons de cinza onde // arredonda o valor de saída
        fator = 255 // (niveis - 1)
        regiao = regiao.point(lambda x: (x // fator) * fator)

    regiao.show()

# --- código principal ---
base_dir = Path(__file__).resolve().parent
model = torch.hub.load('ultralytics/yolov5', 'custom', path=base_dir/'best.pt')

# Configurações
model.conf = 0.20
model.iou = 0.1
model.augment = False
model.force_reload = True

# Caminho da imagem
imagem = base_dir / 'imagens/teste14.jpg'

# Detecta e recorta
coordenadas = detectar_e_recortar_placa(imagem, model, min_width=50, min_height=20)

# Transformar a região do recorte em tons de cinza
aplicar_cinza_com_tons(imagem, coordenadas)