import torch
from pathlib import Path

base_dir = Path(__file__).resolve().parent
model = torch.hub.load('ultralytics/yolov5', 'custom', path=base_dir/'best.pt')

#configurações para o modelo
model.conf = 0.75  # nível mínimo de confiança
model.iou = 0.1  # nível mínimo de IOU (considera sobreposições)
model.augment = False  # não usar aumentação de dados para aumentar velocidade


results = model(base_dir/'carroteste2.jpg')

#coordenadas
coords = results.xyxy[0][:, :4].tolist()
print(coords)  # Lista de [x1, y1, x2, y2] para cada objeto


results.show()
