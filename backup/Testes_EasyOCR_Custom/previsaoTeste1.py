import torch
from pathlib import Path
import cv2
import warnings
from PIL import Image
from torchvision import transforms

# --- Importações dos arquivos do nosso modelo de OCR ---
# (Isso funciona por causa da estrutura de pastas acima)
from ocr_model.model import Model
from ocr_model.utils import AttnLabelConverter

warnings.simplefilter("ignore")
base_dir = Path(__file__).resolve().parent

# =================================================================================
# PARTE 1: FUNÇÕES PARA O NOSSO MODELO DE OCR TREINADO
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
        self.output_channel = 256 # Ajustado para o modelo pré-treinado
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
        return None, None
        
    model.eval()
    return model, converter

def pre_processar_para_ocr(recorte_placa, img_height=32, img_width=100):
    """Pre-processa a imagem da placa para o formato que o modelo OCR espera."""
    img = Image.fromarray(recorte_placa).convert('L') # Converte numpy (cv2) para PIL e Cinza

    # Redimensionar mantendo a proporção e adicionar padding (preenchimento)
    w, h = img.size
    ratio = w / float(h)
    new_w = int(ratio * img_height)

    resized_pil = img.resize((new_w, img_height), Image.BICUBIC)

    final_img = Image.new('L', (img_width, img_height), 255) # Fundo branco
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

def reconhecer_texto(modelo_ocr, converter, tensor_placa, opt):
    """Usa o modelo OCR para prever o texto na imagem."""
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
# PARTE 2: FUNÇÕES DO DETECTOR E VISUALIZAÇÃO
# =================================================================================

def detectar_placas(frame, modelo_detector):
    """Usa o YOLOv5 para encontrar as placas e retorna os recortes e coordenadas."""
    results = modelo_detector(frame)
    boxes = results.xyxy[0]
    recortes = []
    coordenadas = []
    for box in boxes:
        x1, y1, x2, y2, conf, cls = box.tolist()
        recorte = frame[int(y1):int(y2), int(x1):int(x2)]
        recortes.append(recorte)
        coordenadas.append([int(x1), int(y1), int(x2), int(y2)])
    return recortes, coordenadas

def desenhar_resultados(frame, coordenadas, textos):
    """Desenha os retângulos e o texto reconhecido na imagem."""
    for i, coord in enumerate(coordenadas):
        x1, y1, x2, y2 = coord
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if i < len(textos) and textos[i]:
            texto_placa = textos[i].upper()
            cv2.putText(frame, texto_placa, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            print(f"Placa Reconhecida: {texto_placa}")

# =================================================================================
# PARTE 3: EXECUÇÃO PRINCIPAL
# =================================================================================

if __name__ == '__main__':
    # --- CONFIGURE OS CAMINHOS AQUI ---
    CAMINHO_DETECTOR_YOLO = base_dir / 'best.pt'
    CAMINHO_OCR_TREINADO = base_dir / 'best_norm_ED.pth'
    CAMINHO_IMAGEM_TESTE = base_dir / '../Yolo_v5_Plate/imagens/teste20.jpg'
    CARACTERES_DO_MODELO = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    
    # --- Carregamento dos modelos ---
    print("Carregando modelo de detecção YOLOv5...")
    try:
        detector_yolo = torch.hub.load('ultralytics/yolov5', 'custom', path=CAMINHO_DETECTOR_YOLO)
        detector_yolo.conf = 0.4 # Ajuste a confiança conforme necessário
    except Exception as e:
        print(f"Erro ao carregar o modelo YOLOv5: {e}")
        exit()

    modelo_ocr, ocr_converter = carregar_modelo_ocr(CAMINHO_OCR_TREINADO, CARACTERES_DO_MODELO)
    
    if not modelo_ocr:
        exit()

    # --- Processamento da imagem ---
    try:
        frame = cv2.imread(str(CAMINHO_IMAGEM_TESTE))
        if frame is None: raise FileNotFoundError
    except FileNotFoundError:
        print(f"Erro: não foi possível ler a imagem de teste em: {CAMINHO_IMAGEM_TESTE}")
        exit()

    # 1. Detectar e recortar as placas com YOLO
    recortes_das_placas, coordenadas = detectar_placas(frame, detector_yolo)
    
    textos_reconhecidos = []
    
    # 2. Para cada placa recortada, usar nosso OCR
    for recorte in recortes_das_placas:
        opt_config = OCRConfig(CARACTERES_DO_MODELO)
        tensor_placa = pre_processar_para_ocr(recorte, opt_config.imgH, opt_config.imgW)
        texto = reconhecer_texto(modelo_ocr, ocr_converter, tensor_placa, opt_config)
        textos_reconhecidos.append(texto)

    # 3. Desenhar os resultados na imagem original
    desenhar_resultados(frame, coordenadas, textos_reconhecidos)

    cv2.imshow("Resultado Final", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()