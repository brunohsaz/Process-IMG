from ultralytics import YOLO
from pathlib import Path

base_dir = Path(__file__).resolve().parent
yaml_path = base_dir / 'placas.yaml'

model = YOLO('yolov8n.yaml')  # pode usar yolov8s.yaml se quiser um pouco mais robusto
model.train(data=str(yaml_path), epochs=100, imgsz=640, batch=16, name='placas_yolo')
