import torch
from pathlib import Path
import cv2
import warnings
import numpy as np
import sys
import requests
import time

sys.path.append(str(Path(__file__).resolve().parent.parent))
from lib.plate_reader import PlateReader

warnings.simplefilter("ignore")

base_dir = Path(__file__).resolve().parent

# ---------------------- YOLOv5 ----------------------
model = torch.hub.load('ultralytics/yolov5', 'custom', path=base_dir/'best.pt')
model.conf = 0.2
model.iou = 0.1
model.augment = False

# ---------------------- PlateReader ----------------------
templates_path = Path(__file__).resolve().parent.parent / 'lib' / 'templates'
reader = PlateReader(template_root=str(templates_path))

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

def aplicar_pre_processamento(frame, coordenadas, crop_ratio_x=0.07, crop_ratio_y=0.15, fator_topo=1.4):
    placas_processadas = []
    for x1, y1, x2, y2 in coordenadas:
        largura, altura = x2 - x1, y2 - y1
        margem_x = int(largura * crop_ratio_x)
        margem_y = int(altura * crop_ratio_y)
        margem_topo = int(margem_y * fator_topo)

        x1 = max(0, x1 + margem_x)
        y1 = max(0, y1 + margem_topo)
        x2 = min(frame.shape[1], x2 - margem_x)
        y2 = min(frame.shape[0], y2 - margem_y)

        img = frame[y1:y2, x1:x2]
        img = cv2.resize(img, (800, int(800 * img.shape[0] / img.shape[1])))
        img_cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_suavizada = cv2.bilateralFilter(img_cinza, 9, 75, 75)
        _, img_thresh = cv2.threshold(img_suavizada, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        img_thresh = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, kernel)
        placas_processadas.append(img_thresh)
    return placas_processadas

def extrair_rois_dos_caracteres(placa_binaria, min_area=100, max_chars=7):
    """
    Usa a imagem binária da placa para encontrar e retornar uma lista de ROIs
    (imagens individuais) para cada caractere detectado.
    Os ROIs são retornados com CARACTERE BRANCO e FUNDO PRETO.
    """
    img_inv = cv2.bitwise_not(placa_binaria)  # Inverte para encontrar contornos de objetos brancos
    contornos, _ = cv2.findContours(img_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    altura_img = placa_binaria.shape[0]

    selecionados = []
    for cnt in contornos:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area < min_area or h < altura_img * 0.4 or w > altura_img * 1.2:
            continue
        selecionados.append((x, y, w, h))

    selecionados.sort(key=lambda t: t[0])
    selecionados = selecionados[:max_chars]

    rois = []
    for (x, y, w, h) in selecionados:
        # 1. Recorta o ROI (que tem letra preta, fundo branco)
        roi = placa_binaria[y:y+h, x:x+w]
        
        # 2. INVERTE AS CORES para o formato correto (letra branca, fundo preto)
        roi = cv2.bitwise_not(roi)
        
        rois.append(roi)
        
    return rois

def desenhar_resultados(frame, coordenadas, textos):
    for i, coord in enumerate(coordenadas):
        x1, y1, x2, y2 = coord
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        # if i < len(textos) and textos[i]:
        #     cv2.putText(frame, textos[i], (x1, y1-10),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            # print(f"Placa: {textos[i]}")

def caracteres_iguais(a, b):
    if not a or not b:
        return 0
    a, b = a.upper(), b.upper()
    return sum(1 for x, y in zip(a, b) if x == y)

# ---------------------- EXECUÇÃO COM VÍDEO ----------------------
video_path = str(base_dir / 'videos/video-5.mp4')
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Erro: Não foi possível abrir o vídeo em {video_path}")
    exit()
    
ultima_placa_reconhecida = None
texto_anterior = ""
ultimo_request = 0
intervalo_request = 5

target_fps = 30
target_interval = 1.0 / target_fps

while True:
    start = time.time()

    ret, frame = cap.read()
    if not ret:
        print("Fim do vídeo ou erro na leitura do frame.")
        break

    agora = time.time()

    if (agora - ultimo_request) < intervalo_request:
        frame_redimensionado = cv2.resize(frame, (1024, 600))
        cv2.imshow("Detecção de Placas", frame_redimensionado)        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        loop_time = time.time() - start
        sleep_time = max(0, target_interval - loop_time)
        time.sleep(sleep_time)
        continue

    coordenadas = detectar_e_recortar_placa(frame, model)
    placas_processadas = aplicar_pre_processamento(frame, coordenadas)
    textos = []

    for i, placa_proc in enumerate(placas_processadas):
        rois_caracteres = extrair_rois_dos_caracteres(placa_proc)
        if not rois_caracteres:
            textos.append("")
            continue

        texto, confs = reader.read_from_rois(rois_caracteres, plate_hint='auto')
        textos.append(texto)

        deve_enviar = len(texto) == 7 and texto != texto_anterior

        if ultima_placa_reconhecida:
            iguais = caracteres_iguais(texto, ultima_placa_reconhecida)
            if iguais >= 4:
                deve_enviar = False

        if deve_enviar:
            texto_anterior = texto
            ultimo_request = agora

            print(texto)
            url = "https://localhost:4040/api/veiculos/placa/" + texto
            resposta = requests.get(url, verify=False)

            if resposta.status_code == 200:
                dados = resposta.json()
                ultima_placa_reconhecida = dados.get("placa")
                if dados.get("acesso"):
                    intervalo_request = 24
                else:
                    intervalo_request = 3
            else:
                ultima_placa_reconhecida = None
                intervalo_request = 3

    desenhar_resultados(frame, coordenadas, textos)
    frame_redimensionado = cv2.resize(frame, (1024, 600))
    cv2.imshow("Detecção de Placas", frame_redimensionado)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    loop_time = time.time() - start
    sleep_time = max(0, target_interval - loop_time)
    time.sleep(sleep_time)

cap.release()
cv2.destroyAllWindows()
