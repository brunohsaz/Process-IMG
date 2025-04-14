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
            recorte = imagem.crop((x1, y1, x2, y2))

            if salvar:
                recorte.save(imagem_path.parent / f"placa_{i}.jpg")

            recorte.show()  # Exibe o recorte da placa

    return coords_filtradas


# --- código principal ---
base_dir = Path(__file__).resolve().parent
model = torch.hub.load('ultralytics/yolov5', 'custom', path=base_dir/'best.pt')

# Configurações
model.conf = 0.75
model.iou = 0.1
model.augment = False

# Caminho da imagem
imagem = base_dir / 'carroteste2.jpg'

# Detecta e recorta
coordenadas = detectar_e_recortar_placa(imagem, model, min_width=60, min_height=25)

print('Coordenadas filtradas:', coordenadas)

# Visualiza detecção original
model(imagem).show()
