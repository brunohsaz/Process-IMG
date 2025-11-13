import torch
from pathlib import Path
import cv2
import warnings
import numpy as np
import sys
import requests
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

# ---------------------- FUNÇÕES HELPER ----------------------
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
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        img_thresh = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, kernel)
        placas_processadas.append(img_thresh)
    return placas_processadas

def extrair_rois_dos_caracteres(placa_binaria, min_area=100, max_chars=7):
    img_inv = cv2.bitwise_not(placa_binaria)
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
        roi = placa_binaria[y:y+h, x:x+w]
        # NOTA: Aqui, retornamos o ROI no formato original (char branco, fundo preto)
        # O PlateReader deve esperar este formato.
        # A inversão para visualização (char preto, fundo branco) é feita na função de visualização.
        rois.append(roi) 
        
    return rois

def desenhar_resultados(frame, coordenadas, textos):
    for i, coord in enumerate(coordenadas):
        x1, y1, x2, y2 = coord
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        if i < len(textos) and textos[i]:
            cv2.putText(frame, textos[i], (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            print(f"Placa: {textos[i]}")

# ---------------------- NOVA FUNÇÃO DE VISUALIZAÇÃO DOS ROIs SEPARADOS ----------------------
def criar_visualizacao_rois_com_borda(lista_rois, padding_externo=20, padding_interno=10, cor_borda=(0, 0, 0), espessura_borda=2):
    """
    Junta uma lista de ROIs (imagens de caracteres) em uma única imagem horizontal
    com fundo branco, adicionando bordas a cada ROI e espaçamento.
    Espera ROIs com (char branco, fundo preto).
    """
    if not lista_rois:
        return None

    rois_com_borda = []
    max_altura_com_borda = 0
    largura_total_com_borda = 0

    for roi in lista_rois:
        # Invertemos para visualização (char preto, fundo branco)
        roi_vis = cv2.bitwise_not(roi)
        
        # Converte para 3 canais para que a borda colorida funcione
        roi_vis_bgr = cv2.cvtColor(roi_vis, cv2.COLOR_GRAY2BGR)
        
        # Cria uma borda para cada ROI
        roi_com_borda = cv2.copyMakeBorder(
            roi_vis_bgr,
            padding_interno + espessura_borda, # Top
            padding_interno + espessura_borda, # Bottom
            padding_interno + espessura_borda, # Left
            padding_interno + espessura_borda, # Right
            cv2.BORDER_CONSTANT,
            value=[255, 255, 255] # Cor do padding interno (branco)
        )
        
        # Desenha o retângulo da borda principal
        cv2.rectangle(
            roi_com_borda,
            (espessura_borda, espessura_borda),
            (roi_com_borda.shape[1] - espessura_borda, roi_com_borda.shape[0] - espessura_borda),
            cor_borda, # Cor preta para a borda (BGR)
            espessura_borda
        )
        
        rois_com_borda.append(roi_com_borda)
        
        h_cb, w_cb, _ = roi_com_borda.shape # Pega as dimensões da imagem BGR
        if h_cb > max_altura_com_borda:
            max_altura_com_borda = h_cb
        largura_total_com_borda += w_cb

    # Adiciona o padding externo entre os ROIs e nas bordas da imagem final
    largura_final = largura_total_com_borda + padding_externo * (len(rois_com_borda) + 1)
    altura_final = max_altura_com_borda + padding_externo * 2

    # Cria o canvas branco final (agora em 3 canais para consistência)
    canvas_final = np.full((altura_final, largura_final, 3), 255, dtype=np.uint8)

    # Cola cada ROI com borda no canvas final
    x_atual = padding_externo
    for roi_cb in rois_com_borda:
        h_cb, w_cb, _ = roi_cb.shape
        # Centraliza verticalmente
        y_offset = padding_externo + (max_altura_com_borda - h_cb) // 2
        
        canvas_final[y_offset : y_offset + h_cb, x_atual : x_atual + w_cb] = roi_cb
        
        x_atual += w_cb + padding_externo

    return canvas_final

# ---------------------- EXECUÇÃO HÍBRIDA (MELHOR DOS DOIS MUNDOS) ----------------------
frame = cv2.imread(str(base_dir / 'imagens/teste26.jpg'))
if frame is None:
    print(f"Erro: Não foi possível carregar a imagem")
    exit()

coordenadas = detectar_e_recortar_placa(frame, model)
placas_processadas = aplicar_pre_processamento(frame, coordenadas)

textos = []
for i, placa_proc in enumerate(placas_processadas):
    rois_caracteres = extrair_rois_dos_caracteres(placa_proc)
    
    if rois_caracteres:
        # Usa a nova função para criar a visualização dos ROIs com borda
        imagem_rois_separados = criar_visualizacao_rois_com_borda(
            rois_caracteres,
            padding_externo=20,    # Espaçamento entre os caracteres e borda da imagem final
            padding_interno=0,     # Espaçamento entre o caractere e a borda interna
            cor_borda=(0, 0, 0),   # Cor da borda (preto)
            espessura_borda=0      # Espessura da borda
        )
        
        if imagem_rois_separados is not None:
            nome_arquivo = f"placa_{i+1}_rois_separados.png"
            cv2.imwrite(nome_arquivo, imagem_rois_separados)
            print(f"Imagem de ROIs separados de exemplo salva em: {nome_arquivo}")

            cv2.imshow(f"ROIs Separados - Placa {i+1}", imagem_rois_separados)

        texto, confs = reader.read_from_rois(rois_caracteres, plate_hint='auto')
        textos.append(texto)
    else:
        textos.append("")

desenhar_resultados(frame, coordenadas, textos)

cv2.imshow("Resultado Final", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()