import os
import random
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
import cv2

# Caminhos
saida = "dataset_mercosul"
os.makedirs(saida, exist_ok=True)

# Fonte da placa (baixe a "Mandatory.ttf" ou use outra parecida)
font = ImageFont.truetype("C:/Users/bruno/OneDrive/Documentos/Process-IMG/Treino_EasyOCR/fe-font.ttf", 120)

# Função para gerar texto de placa Mercosul
def gerar_placa():
    letras = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    numeros = "0123456789"
    return (
        "".join(random.choices(letras, k=3))
        + random.choice(numeros)
        + random.choice(letras)
        + "".join(random.choices(numeros, k=2))
    )

# Função para aplicar augmentations com OpenCV
def augmentar(img):
    arr = np.array(img)

    # ruído
    if random.random() > 0.5:
        ruido = np.random.normal(0, 25, arr.shape).astype(np.uint8)
        arr = cv2.add(arr, ruido)

    # blur
    if random.random() > 0.5:
        arr = cv2.GaussianBlur(arr, (5, 5), 0)

    # brilho/contraste
    if random.random() > 0.5:
        alpha = random.uniform(0.7, 1.3)  # contraste
        beta = random.randint(-40, 40)    # brilho
        arr = cv2.convertScaleAbs(arr, alpha=alpha, beta=beta)

    # rotação leve
    if random.random() > 0.5:
        h, w = arr.shape[:2]
        M = cv2.getRotationMatrix2D((w//2, h//2), random.uniform(-5, 5), 1)
        arr = cv2.warpAffine(arr, M, (w, h), borderValue=(255, 255, 255))

    return Image.fromarray(arr)

# Gerar dataset
num_imgs = 2000  # quantidade de placas
labels_path = os.path.join(saida, "labels.txt")

with open(labels_path, "w") as f:
    for i in range(num_imgs):
        placa = gerar_placa()

        # cria imagem branca
        img = Image.new("RGB", (520, 110), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.text((30, -10), placa, font=font, fill=(0, 0, 0))

        # aplica augmentação
        img = augmentar(img)

        # salva imagem
        nome = f"{placa}_{i}.jpg"
        caminho = os.path.join(saida, nome)
        img.save(caminho)

        # grava no arquivo de labels
        f.write(f"{nome} {placa}\n")

print("Dataset gerado em:", saida)
