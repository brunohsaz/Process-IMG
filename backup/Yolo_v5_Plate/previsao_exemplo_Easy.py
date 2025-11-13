import torch
from pathlib import Path
import cv2
import warnings
import numpy as np
import sys
import easyocr # Importar o EasyOCR

warnings.simplefilter("ignore")

base_dir = Path(__file__).resolve().parent

# ---------------------- YOLOv5 ----------------------
model = torch.hub.load('ultralytics/yolov5', 'custom', path=base_dir/'best.pt')
model.conf = 0.2
model.iou = 0.1
model.augment = False

# ---------------------- EasyOCR ----------------------
# Carrega o modelo do EasyOCR na memória (faça isso apenas uma vez)
# Usamos 'pt' (português) e 'en' (inglês)
# gpu=True tentará usar a GPU se disponível (requer CUDA)
print("Carregando modelo EasyOCR...")
reader_easyocr = easyocr.Reader(['pt', 'en'], gpu=torch.cuda.is_available())
print("Modelo EasyOCR carregado.")

# ---------------------- PIPELINE ----------------------
def detectar_e_recortar_placa(frame, modelo, min_width=50, min_height=20):
    results = modelo(frame)
    boxes = results.xyxy[0]
    coords_filtradas = []
    for box in boxes:
        x1, y1, x2, y2, conf, cls = box.tolist()
        w = x2 - x1
        h = y2 - y1
        if w >= min_width and h >= min_height:
            coords_filtradas.append([int(x1), int(y1), int(x2), int(y2)])
    return coords_filtradas

# A função aplicar_pre_processamento() foi REMOVIDA.
# A função extrair_rois_dos_caracteres() foi REMOVIDA.

def desenhar_resultados(frame, coordenadas, textos):
    for i, coord in enumerate(coordenadas):
        x1, y1, x2, y2 = coord
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        if i < len(textos) and textos[i]:
            # O EasyOCR já retorna o texto limpo, só precisamos juntar
            texto_placa = "".join(textos[i].split()).upper()
            if texto_placa:
                cv2.putText(frame, texto_placa, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                print(f"Placa (EasyOCR): {texto_placa}")

# ---------------------- EXECUÇÃO COM EASYOCR ----------------------
frame = cv2.imread(str(base_dir / 'imagens/teste33.jpg'))
if frame is None:
    print(f"Erro: Não foi possível carregar a imagem")
    exit()

coordenadas = detectar_e_recortar_placa(frame, model)

textos = []
# Lista de caracteres permitidos para o EasyOCR
char_whitelist = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

for (x1, y1, x2, y2) in coordenadas:
    # 1. Recortar a placa do frame original (colorido ou cinza)
    # Adicionamos uma pequena margem para garantir que não corte caracteres
    margem = 5
    y1_m = max(0, y1 - margem)
    y2_m = min(frame.shape[0], y2 + margem)
    x1_m = max(0, x1 - margem)
    x2_m = min(frame.shape[1], x2 + margem)
    
    placa_recortada = frame[y1_m:y2_m, x1_m:x2_m]

    # 2. (Opcional, mas recomendado) Converter para escala de cinza
    placa_cinza = cv2.cvtColor(placa_recortada, cv2.COLOR_BGR2GRAY)

    # 3. Passar a imagem em escala de cinza (numpy array) para o EasyOCR
    # 'detail=0': Retorna apenas a lista de textos encontrados
    # 'paragraph=True': Tenta juntar textos que estão na mesma linha
    try:
        resultado_ocr = reader_easyocr.readtext(
            placa_cinza,
            detail=0,
            paragraph=True,
            allowlist=char_whitelist
        )
        
        if resultado_ocr:
            # Juntar os resultados (caso ele separe 'ABC' e '1234')
            texto_completo = "".join(resultado_ocr)
            textos.append(texto_completo)
        else:
            textos.append("")
            
    except Exception as e:
        print(f"Erro no EasyOCR: {e}")
        textos.append("")
        

desenhar_resultados(frame, coordenadas, textos)

cv2.imshow("Resultado Final (EasyOCR)", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()