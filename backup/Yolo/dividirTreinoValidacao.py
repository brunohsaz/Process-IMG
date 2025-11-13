from pathlib import Path
import shutil
import random

base_dir = Path(__file__).resolve().parent
train_img_dir = base_dir / 'placas_dataset/images/train'
val_img_dir = base_dir / 'placas_dataset/images/val'
train_label_dir = base_dir / 'placas_dataset/labels/train'
val_label_dir = base_dir / 'placas_dataset/labels/val'

val_img_dir.mkdir(parents=True, exist_ok=True)
val_label_dir.mkdir(parents=True, exist_ok=True)

imagens = list(train_img_dir.glob('*.jpg'))
random.shuffle(imagens)

val_count = int(0.2 * len(imagens))
val_imgs = imagens[:val_count]

for img_path in val_imgs:
    nome = img_path.name
    label_path = train_label_dir / (img_path.stem + '.txt')

    shutil.move(str(img_path), str(val_img_dir / nome))
    if label_path.exists():
        shutil.move(str(label_path), str(val_label_dir / label_path.name))
