from ultralytics import YOLO
from pathlib import Path

base_dir = Path(__file__).resolve().parent
modelo_path = base_dir / 'runs/detect/placas_yolo2/weights/best.pt'
imagem_path = base_dir / 'pessoasteste3.jpg'

model = YOLO(str(modelo_path))

# Roda a predição e guarda os resultados
resultados = model.predict(
    source=str(imagem_path),
    show=True,
    save=True,
    conf=0.3
)

# Verifica se houve detecção
for resultado in resultados:
    if resultado.boxes is not None and len(resultado.boxes) > 0:
        print('Placa(s) detectada(s) com sucesso!')
    else:
        print('Nenhuma placa detectada.')
