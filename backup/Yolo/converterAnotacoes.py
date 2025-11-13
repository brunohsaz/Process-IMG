from pathlib import Path
import pandas as pd
import shutil
from tqdm import tqdm

base_dir = Path(__file__).resolve().parent
csv_path = base_dir / 'train-annotations-bbox.csv'
imagens_path = base_dir / 'placas_dataset/imagens_originais'  # ajuste se necessário
saida_imgs = base_dir / 'placas_dataset/images/train'
saida_labels = base_dir / 'placas_dataset/labels/train'

saida_imgs.mkdir(parents=True, exist_ok=True)
saida_labels.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(csv_path)
df = df[df['LabelName'] == '/m/01g317']  # somente placas

agrupado = df.groupby('ImageID')

for image_id, group in tqdm(agrupado):
    label_file = saida_labels / f"{image_id}.txt"
    with label_file.open('w') as f:
        for _, row in group.iterrows():
            x_center = (row['XMin'] + row['XMax']) / 2
            y_center = (row['YMin'] + row['YMax']) / 2
            width = row['XMax'] - row['XMin']
            height = row['YMax'] - row['YMin']
            f.write(f'0 {x_center} {y_center} {width} {height}\n')
    
    img_file = imagens_path / f"{image_id}.jpg"
    if img_file.exists():
        shutil.copyfile(img_file, saida_imgs / f"{image_id}.jpg")
