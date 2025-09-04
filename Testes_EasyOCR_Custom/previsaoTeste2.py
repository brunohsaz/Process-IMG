import torch
from pathlib import Path
import cv2
import warnings
import re
import numpy as np
from PIL import Image
from torchvision import transforms

# --- Importações dos arquivos do nosso modelo de OCR ---
from ocr_model.model import Model
from ocr_model.utils import AttnLabelConverter

warnings.simplefilter("ignore")
base_dir = Path(__file__).resolve().parent

# =================================================================================
# PARTE 1: CARREGAMENTO E FUNÇÕES DO MODELO OCR CUSTOMIZADO
# =================================================================================

class OCRConfig:
    """ Objeto para simular as opções (opt) com as quais o modelo foi treinado. """
    def __init__(self, character):
        self.Transformation = 'TPS'
        self.FeatureExtraction = 'ResNet'
        self.SequenceModeling = 'BiLSTM'
        self.Prediction = 'Attn'
        self.num_fiducial = 20
        self.input_channel = 1
        self.output_channel = 256
        self.hidden_size = 256
        self.character = character
        self.batch_max_length = 25
        self.imgH = 32
        self.imgW = 100
        self.rgb = False
        self.PAD = False
        self.num_class = len(self.character) + 2

def carregar_modelo_ocr(caminho_modelo, character_set):
    """Carrega o modelo de OCR treinado e o conversor de texto."""
    print(f"Carregando modelo OCR de: {caminho_modelo}")
    opt = OCRConfig(character_set)
    converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)
    
    model = Model(opt)
    model = torch.nn.DataParallel(model)
    
    try:
        model.load_state_dict(torch.load(caminho_modelo, map_location='cpu'))
    except FileNotFoundError:
        print(f"ERRO: Arquivo do modelo OCR não encontrado em: {caminho_modelo}")
        return None, None, None
        
    model.eval()
    return model, converter, opt

def pre_processar_para_ocr_custom(recorte_placa, img_height=32, img_width=100):
    """Pre-processa a imagem da placa para o formato que o modelo OCR customizado espera."""
    img = Image.fromarray(recorte_placa).convert('L')

    w, h = img.size
    ratio = w / float(h)
    new_w = int(ratio * img_height)

    resized_pil = img.resize((new_w, img_height), Image.BICUBIC)
    
    final_img = Image.new('L', (img_width, img_height), 255)
    if new_w < img_width:
        final_img.paste(resized_pil, (0, 0))
    else:
        final_img = resized_pil.resize((img_width, img_height), Image.BICUBIC)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    
    tensor = transform(final_img)
    return tensor.unsqueeze(0)

def reconhecer_texto_custom(modelo_ocr, converter, tensor_placa, opt):
    """Usa o modelo OCR customizado para prever o texto na imagem."""
    with torch.no_grad():
        batch_size = tensor_placa.size(0)
        length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size)
        text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0)

        preds = modelo_ocr(tensor_placa, text_for_pred, is_train=False)

        _, preds_index = preds.max(2)
        preds_str = converter.decode(preds_index, length_for_pred)
        
        pred_clean = preds_str[0].strip().split('[s]')[0]
    return pred_clean

# =================================================================================
# PARTE 2: PIPELINE DE DETECÇÃO E PRÉ-PROCESSAMENTO (SEU CÓDIGO)
# =================================================================================

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

def aplicar_pre_processamento_binario(frame, coordenadas, crop_ratio_x=0.07, crop_ratio_y=0.15, fator_topo=1.4):
    placas_processadas = []
    for x1, y1, x2, y2 in coordenadas:
        largura, altura = x2 - x1, y2 - y1
        margem_x = int(largura * crop_ratio_x)
        margem_y = int(altura * crop_ratio_y)
        margem_topo = int(margem_y * fator_topo)
        
        x1, y1 = max(0, x1 + margem_x), max(0, y1 + margem_topo)
        x2, y2 = min(frame.shape[1], x2 - margem_x), min(frame.shape[0], y2 - margem_y)

        img = frame[y1:y2, x1:x2]
        img = cv2.resize(img, (800, int(800 * img.shape[0] / img.shape[1])))
        img_cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_suavizada = cv2.bilateralFilter(img_cinza, 9, 75, 75)
        _, img_thresh = cv2.threshold(img_suavizada, 75, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        img_thresh = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, kernel)
        placas_processadas.append(img_thresh)
    return placas_processadas

# =================================================================================
# PARTE 3: LÓGICA DE FILTRAGEM POR CONTORNOS E OCR (MESCLADO)
# =================================================================================

def encontrar_contornos_caracteres(img_bin, min_area=100, max_chars=7):
    img_inv = cv2.bitwise_not(img_bin)
    contornos, _ = cv2.findContours(img_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    altura_img = img_bin.shape[0]
    selecionados = []
    for cnt in contornos:
        x, y, w, h = cv2.boundingRect(cnt)
        if w*h < min_area or h < altura_img * 0.4 or w > altura_img * 1.2:
            continue
        selecionados.append((x, cnt))
    
    selecionados.sort(key=lambda t: t[0])
    return [t[1] for t in selecionados[:max_chars]]

def filtrar_externo_por_contornos(img_bin, contornos, dilatacao=1):
    mask = np.zeros_like(img_bin)
    cv2.drawContours(mask, contornos, -1, 255, thickness=cv2.FILLED)

    if dilatacao and dilatacao > 0:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (2*dilatacao+1, 2*dilatacao+1))
        mask = cv2.dilate(mask, k, iterations=1)

    out = np.full_like(img_bin, 255)
    out[mask == 255] = img_bin[mask == 255]
    return out

def aplicar_ocr_filtrado_custom(placa_binaria, modelo_ocr, ocr_converter, ocr_opt, mostrar_debug=True):
    contornos = encontrar_contornos_caracteres(placa_binaria)

    if not contornos:
        img_para_ocr = placa_binaria # Usa a imagem original se não achar contornos
    else:
        img_para_ocr = filtrar_externo_por_contornos(placa_binaria, contornos)
    
    if mostrar_debug:
        cv2.imshow("Placa Limpa (Filtrada)", img_para_ocr)

    # Converte a imagem binária (1 canal) para BGR (3 canais) para a proxima função
    img_para_ocr_bgr = cv2.cvtColor(img_para_ocr, cv2.COLOR_GRAY2BGR)

    # Pipeline do nosso modelo customizado
    tensor_placa = pre_processar_para_ocr_custom(img_para_ocr_bgr, ocr_opt.imgH, ocr_opt.imgW)
    texto = reconhecer_texto_custom(modelo_ocr, ocr_converter, tensor_placa, ocr_opt)
    
    # Aplica as correções de pós-processamento
    texto_limpo = limpar_texto(texto)
    formato = identificar_formato(texto_limpo)
    return aplicar_correcoes(texto_limpo, formato)

# =================================================================================
# PARTE 4: FUNÇÕES UTILITÁRIAS (PÓS-PROCESSAMENTO)
# =================================================================================

CORRECOES = {'0': ['O', 'Q'], '1': ['I', 'L'], '2': ['Z'], '4': ['A'], '5': ['S'], '6': ['G'], '8': ['B'], 'O': ['0', 'Q'], 'I': ['1', 'L'], 'Z': ['2'], 'S': ['5'], 'G': ['6']}
MAX_ERROS = 1

def limpar_texto(texto): return re.sub(r'[^A-Z0-9]', '', texto.strip().upper())
def identificar_formato(texto):
    if len(texto) != 7: return None
    return "mercosul" if texto[4].isalpha() else "antiga"

def corrigir_char(ch, deve_ser_letra):
    for d, o in CORRECOES.items():
        if ch in o and ((deve_ser_letra and d.isalpha()) or (not deve_ser_letra and d.isdigit())):
            return d
    return ch

def aplicar_correcoes(texto, formato):
    corrigido = ""
    for i, c in enumerate(texto):
        deve_ser_letra = (formato == "mercosul" and i in [0,1,2,4]) or (formato == "antiga" and i < 3)
        corrigido += corrigir_char(c, deve_ser_letra)
    return corrigido

def placas_sao_similares(a, b):
    return sum(1 for x, y in zip(a, b) if x != y) <= MAX_ERROS if len(a) == len(b) else False

def consenso_textos(textos):
    validos = []
    for t in textos:
        if t and all(not placas_sao_similares(t, v) for v in validos):
            validos.append(t)
    return validos

def desenhar_resultados(frame, coordenadas, textos):
    for i, coord in enumerate(coordenadas):
        x1, y1, x2, y2 = coord
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if i < len(textos) and textos[i]:
            cv2.putText(frame, textos[i], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            print(f"Placa: {textos[i]}")

# =================================================================================
# PARTE 5: EXECUÇÃO PRINCIPAL
# =================================================================================

if __name__ == '__main__':
    # --- CONFIGURE OS CAMINHOS AQUI ---
    CAMINHO_DETECTOR_YOLO = base_dir / 'best.pt'
    CAMINHO_OCR_TREINADO = base_dir / 'best_norm_ED.pth'
    CAMINHO_IMAGEM_TESTE = base_dir / '../Yolo_v5_Plate/imagens/teste20.jpg'
    CARACTERES_DO_MODELO = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

    # --- Carregamento dos modelos ---
    print("Carregando modelo de detecção YOLOv5...")
    detector_yolo = torch.hub.load('ultralytics/yolov5', 'custom', path=CAMINHO_DETECTOR_YOLO)
    detector_yolo.conf = 0.2
    detector_yolo.iou = 0.1

    modelo_ocr, ocr_converter, ocr_opt = carregar_modelo_ocr(CAMINHO_OCR_TREINADO, CARACTERES_DO_MODELO)
    
    if not modelo_ocr: exit()

    # --- Processamento da imagem ---
    frame = cv2.imread(str(CAMINHO_IMAGEM_TESTE))
    if frame is None:
        print(f"Erro ao ler a imagem: {CAMINHO_IMAGEM_TESTE}")
        exit()

    coordenadas = detectar_e_recortar_placa(frame, detector_yolo)
    placas_binarias = aplicar_pre_processamento_binario(frame, coordenadas)
    
    # Para cada placa binarizada, aplicar o pipeline de OCR filtrado com nosso modelo
    textos = [aplicar_ocr_filtrado_custom(p, modelo_ocr, ocr_converter, ocr_opt) for p in placas_binarias]
    textos = consenso_textos(textos)
    desenhar_resultados(frame, coordenadas, textos)

    cv2.imshow("Resultado Final", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()