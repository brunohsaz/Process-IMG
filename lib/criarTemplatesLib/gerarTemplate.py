from pathlib import Path
from PIL import Image, ImageFont, ImageDraw
import cv2
import numpy as np
import string

# Caminhos das fontes
fe_font_path = Path(__file__).parent / "fe-font.ttf"           # sua fonte FE-Schrift
old_font_path = Path(__file__).parent / "Mandatory.otf"          # baixe a DIN 1451 (ou instale no sistema)

# Pasta de saída
base_dir = Path("templates")
fe_dir = base_dir / "fe_schrift"
old_dir = base_dir / "old"
fe_dir.mkdir(parents=True, exist_ok=True)
old_dir.mkdir(parents=True, exist_ok=True)

def save_glyph(ch, font, outdir, size=(96,56)):
    img = Image.new("L", (120, 160), color=0)  # fundo preto
    draw = ImageDraw.Draw(img)
    bbox = draw.textbbox((0, 0), ch, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]

    draw.text(((120-w)//2, (160-h)//2), ch, font=font, fill=255)
    arr = np.array(img)
    # binarizar
    _, arr = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # recortar bbox
    coords = cv2.findNonZero(arr)
    x,y,w,h = cv2.boundingRect(coords)
    arr = arr[y:y+h, x:x+w]
    # redimensionar
    target_h, target_w = size
    scale = min(target_w/arr.shape[1], target_h/arr.shape[0])
    nw, nh = int(arr.shape[1]*scale), int(arr.shape[0]*scale)
    resized = cv2.resize(arr, (nw,nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_h,target_w), dtype=np.uint8)
    cx, cy = (target_w-nw)//2, (target_h-nh)//2
    canvas[cy:cy+nh, cx:cx+nw] = resized
    cv2.imwrite(str(outdir/f"{ch}.png"), canvas)

# Carregar fontes
fe_font = ImageFont.truetype(str(fe_font_path), 120)
old_font = ImageFont.truetype(str(old_font_path), 120)

# Gerar para A-Z e 0-9
for ch in string.ascii_uppercase + string.digits:
    save_glyph(ch, fe_font, fe_dir)
    save_glyph(ch, old_font, old_dir)

print("Templates criados em:", base_dir)
