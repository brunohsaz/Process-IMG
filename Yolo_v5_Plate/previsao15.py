# -*- coding: utf-8 -*-
"""
Detector + OCR para placas BR (Mercosul/antiga) com YOLOv5, retificação,
múltiplos pré-processamentos, ensemble de Tesseract e pós-processamento
regrado por formato com votação por caractere.

Requer: torch, opencv-python, pytesseract, numpy
Opcional: ajustar o caminho do executável do Tesseract no Windows.
"""

import cv2
import numpy as np
import torch
import pytesseract
import re
from pathlib import Path
from collections import Counter, defaultdict

# ===================== CONFIG =====================

BASE_DIR = Path(__file__).resolve().parent  # ajuste se quiser rodar de outra pasta
CAMINHO_PESO_YOLO = BASE_DIR / "best.pt"    # seu peso custom
SAIDA_DIR = BASE_DIR / "saidas_placa"
SAIDA_DIR.mkdir(exist_ok=True, parents=True)

# Ajuste para Windows, se necessário:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

IMAGENS_DE_TESTE = [
    
    BASE_DIR / "imagens" / "teste16.jpg",
    BASE_DIR / "imagens" / "teste21.jpg",
    BASE_DIR / "imagens" / "teste20.jpg",
    BASE_DIR / "imagens" / "teste5.jpg",
]

# Thresholds mínimos para considerar a detecção da placa
MIN_W, MIN_H = 50, 20

# Correções de confusão visual (mapa bidirecional)
CORRECOES = {
    '0': ['O', 'Q'], '1': ['I', 'L'], '2': ['Z'], '4': ['A'], '5': ['S'], '6': ['G'], '8': ['B'],
    'O': ['0', 'Q'], 'I': ['1', 'L'], 'Z': ['2'], 'S': ['5'], 'G': ['6']
}

# ===================== UTIL =====================

def clamp(v, a, b):
    return max(a, min(b, v))

def iou(a, b):
    # (x1,y1,x2,y2) IoU para eventual *NMS* extra (não usado por padrão)
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    interx1, intery1 = max(ax1, bx1), max(ay1, by1)
    interx2, intery2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, interx2-interx1), max(0, intery2-intery1)
    inter = iw*ih
    area_a = (ax2-ax1)*(ay2-ay1)
    area_b = (bx2-bx1)*(by2-by1)
    union = area_a + area_b - inter + 1e-9
    return inter/union

# ===================== DETECÇÃO (YOLO) =====================

def carregar_modelo_yolo(caminho_peso):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=str(caminho_peso))
    model.conf = 0.25
    model.iou = 0.45
    model.augment = False
    return model

def detectar_placas(frame_bgr, modelo, min_w=MIN_W, min_h=MIN_H):
    """
    Retorna lista de boxes [x1,y1,x2,y2] inteiros.
    """
    res = modelo(frame_bgr)
    boxes = res.xyxy[0]
    saida = []
    for b in boxes:
        x1, y1, x2, y2, conf, cls = b.tolist()
        w = x2 - x1
        h = y2 - y1
        if w >= min_w and h >= min_h:
            saida.append([int(x1), int(y1), int(x2), int(y2)])
    return saida

# ===================== RETIFICAÇÃO =====================

def encontrar_quadrilatero_placa(crop):
    """
    Dentro do recorte do YOLO, tenta achar o maior quadrilátero plausível
    para warpPerspective (placa retangular).
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 50, 50)
    v = max(5, int(np.median(gray)*0.66))
    edges = cv2.Canny(gray, v, v*2)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=1)

    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H, W = gray.shape[:2]
    area_img = H*W
    melhor = None
    melhor_area = 0

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.03*peri, True)
        if len(approx) == 4:
            a = cv2.contourArea(approx)
            if a > melhor_area and a > 0.15*area_img:
                melhor_area = a
                melhor = approx

    if melhor is None:
        return None

    pts = melhor.reshape(-1,2).astype(np.float32)

    # ordenar pontos (tl, tr, br, bl)
    def ordenar(pts):
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1).ravel()
        tl = pts[np.argmin(s)]
        br = pts[np.argmax(s)]
        tr = pts[np.argmin(diff)]
        bl = pts[np.argmax(diff)]
        return np.array([tl, tr, br, bl], dtype=np.float32)

    return ordenar(pts)

def warp_para_retangular(crop, quad):
    """
    Faz homografia do quadrilátero para retângulo proporcional (aspect ~ 4:1–5:1).
    """
    if quad is None:
        return crop

    (tl, tr, br, bl) = quad
    wA = np.linalg.norm(br - bl)
    wB = np.linalg.norm(tr - tl)
    hA = np.linalg.norm(tr - br)
    hB = np.linalg.norm(tl - bl)
    w = int(max(wA, wB))
    h = int(max(hA, hB))
    # força largura mínima e limita proporção
    w = max(w, 200)
    h = clamp(h, 40, 300)
    dst = np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(quad, dst)
    ret = cv2.warpPerspective(crop, M, (w, h), flags=cv2.INTER_LINEAR)
    return ret

# ===================== PRÉ-PROCESSAMENTOS =====================

def gerar_variantes_binarias(img_bgr):
    """
    Cria várias versões binarizadas para robustez (ensemble).
    """
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    variantes = []

    # normalização
    img_eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(img)

    bases = [img, img_eq]

    for base in bases:
        # Suavizações leves
        sm1 = cv2.bilateralFilter(base, 9, 75, 75)
        sm2 = cv2.GaussianBlur(base, (3,3), 0)

        for src in [base, sm1, sm2]:
            # Otsu
            _, th1 = cv2.threshold(src, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            _, th2 = cv2.threshold(src, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # Adaptativas
            th3 = cv2.adaptiveThreshold(src, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                        cv2.THRESH_BINARY, 31, 10)
            th4 = cv2.adaptiveThreshold(src, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 31, 8)

            for t in [th1, th2, th3, th4]:
                # limpeza morfológica leve
                k = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
                t2 = cv2.morphologyEx(t, cv2.MORPH_CLOSE, k, iterations=1)
                variantes.append(t2)

    # remove duplicadas (pela soma dos pixels)
    uniq = []
    hashes = set()
    for v in variantes:
        h = int(v.sum())
        if h not in hashes:
            hashes.add(h)
            uniq.append(v)
    return uniq[:12]  # limite para não explodir o tempo

# ===================== OCR (Tesseract) =====================

PSMS = [7, 8, 6]  # linha única, palavra, semi-automático
OCR_BASE = r'-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

def ocr_um_pass(img_bin, psm=7):
    cfg = f"{OCR_BASE} --psm {psm}"
    # Testa tanto normal quanto invertido e pega o mais informativo
    txt1 = pytesseract.image_to_string(img_bin, lang='eng', config=cfg)
    txt2 = pytesseract.image_to_string(cv2.bitwise_not(img_bin), lang='eng', config=cfg)
    return max([txt1, txt2], key=lambda s: len(s.strip()))

def limpar(txt):
    return re.sub(r'[^A-Z0-9]', '', txt.strip().upper())

def identificar_formato(texto):
    if len(texto) != 7:
        return None
    # Mercosul: ABC1D23  (4ª pos é dígito, 5ª é letra)
    if texto[0:3].isalpha() and texto[3].isdigit() and texto[4].isalpha() and texto[5:7].isdigit():
        return "mercosul"
    # Antigo: ABC1234
    if texto[0:3].isalpha() and texto[3:7].isdigit():
        return "antiga"
    return None

def corrigir_char(ch, deve_ser_letra):
    # Se já atende, retorna
    if deve_ser_letra and ch.isalpha():
        return ch
    if (not deve_ser_letra) and ch.isdigit():
        return ch
    # Tenta coerções
    for destino, origens in CORRECOES.items():
        if ch in origens:
            if deve_ser_letra and destino.isalpha():
                return destino
            if (not deve_ser_letra) and destino.isdigit():
                return destino
    return ch

def aplicar_regras(texto):
    """
    Aplica coerções por posição para tentar forçar um dos formatos válidos.
    Retorna (texto_corrigido, formato_atingido_ou_None)
    """
    if len(texto) != 7:
        return texto, None

    candidatos = []

    # Tentar mercosul
    t = list(texto)
    pos_letra = [0,1,2,4]
    for i in range(7):
        t[i] = corrigir_char(t[i], i in pos_letra)
    t_merc = ''.join(t)
    if identificar_formato(t_merc) == 'mercosul':
        candidatos.append(('mercosul', t_merc))

    # Tentar antiga
    t = list(texto)
    pos_letra = [0,1,2]
    for i in range(7):
        t[i] = corrigir_char(t[i], i in pos_letra)
    t_ant = ''.join(t)
    if identificar_formato(t_ant) == 'antiga':
        candidatos.append(('antiga', t_ant))

    if candidatos:
        # prefere mercosul se ambos válidos (mais atual)
        candidatos.sort(key=lambda x: 0 if x[0]=='mercosul' else 1)
        return candidatos[0][1], candidatos[0][0]
    else:
        return texto, None

# ===================== VOTAÇÃO / CONSENSO =====================

def votar_por_caractere(lista):
    """
    Dada uma lista de strings do MESMO tamanho,
    devolve string com voto majoritário por posição e uma pontuação.
    """
    if not lista:
        return "", 0.0
    L = len(lista[0])
    filtradas = [s for s in lista if len(s)==L]
    if not filtradas:
        return "", 0.0
    out = []
    for i in range(L):
        col = [s[i] for s in filtradas]
        c = Counter(col).most_common(1)[0]
        out.append(c[0])
    # score de acordo com concordância média
    acertos = 0
    for s in filtradas:
        acertos += sum(1 for i,ch in enumerate(s) if ch == out[i])
    score = acertos/(len(filtradas)*L + 1e-9)
    return ''.join(out), score

def normalizar_candidatos(cands):
    # limpa, filtra vazios e limita a 7 chars
    norm = []
    for c in cands:
        t = limpar(c)
        if t:
            if len(t) > 7:
                # às vezes o OCR cola um char extra; preferir manter 7 mais prováveis
                # aqui simplificamos: mantemos primeiros 7
                t = t[:7]
            norm.append(t)
    return norm

# ===================== PIPELINE COMPLETO PARA UM RECORTE =====================

def ocr_de_recorte(img_bgr):
    """
    Retorna (texto_final, melhor_variantes_debug)
    """
    # padroniza tamanho (ajuda Tesseract)
    h, w = img_bgr.shape[:2]
    escala = 1000.0 / max(w, 400)  # força ~1000px largura máx
    img_bgr = cv2.resize(img_bgr, (int(w*escala), int(h*escala)), interpolation=cv2.INTER_CUBIC)

    variantes = gerar_variantes_binarias(img_bgr)

    candidatos = []
    debug_imgs = []

    for vb in variantes:
        for psm in PSMS:
            txt = ocr_um_pass(vb, psm=psm)
            candidatos.append(txt)
        debug_imgs.append(vb)

    candidatos = normalizar_candidatos(candidatos)

    # Votação em duas camadas: (1) por comprimento 7, (2) aplicar regras por formato
    cand7 = [c for c in candidatos if len(c) == 7]
    if not cand7 and candidatos:
        # se ninguém veio com 7, tenta ajustar cortando/pad até 7
        for c in candidatos:
            if len(c) >= 6:
                cand7.append((c + "XXXXXXX")[:7])

    # Voto bruto
    voto_bruto, score_bruto = votar_por_caractere(cand7)

    # Aplica regras por formato nos candidatos e vota de novo
    pos_regra = []
    for c in cand7:
        cr, _ = aplicar_regras(c)
        pos_regra.append(cr)
    voto_regra, score_regra = votar_por_caractere(pos_regra)

    # Escolha final: prioriza quem bate formato válido
    fmt = identificar_formato(voto_regra)
    if fmt is not None:
        final = voto_regra
        score = score_regra + 0.05  # pequeno bônus
    else:
        final = voto_bruto
        score = score_bruto

    # Passo extra: aplicar_regras no final para tentar coerção
    final2, fmt2 = aplicar_regras(final)
    if identificar_formato(final2) and not identificar_formato(final):
        final, fmt = final2, fmt2

    return final, fmt, score, debug_imgs

# ===================== EXECUÇÃO EM UMA IMAGEM =====================

def processar_imagem(caminho, modelo):
    img = cv2.imread(str(caminho))
    if img is None:
        print(f"[ERRO] Não consegui abrir {caminho}")
        return

    boxes = detectar_placas(img, modelo)
    resultados = []

    for (x1,y1,x2,y2) in boxes:
        # margem interna para cortar a faixa azul/topo e bordas
        w, h = x2-x1, y2-y1
        mx, my, mtop = int(w*0.08), int(h*0.12), int(h*0.25)
        cx1 = clamp(x1+mx, 0, img.shape[1]-1)
        cy1 = clamp(y1+mtop, 0, img.shape[0]-1)  # corta mais o topo
        cx2 = clamp(x2-mx, 0, img.shape[1]-1)
        cy2 = clamp(y2-my, 0, img.shape[0]-1)

        crop = img[cy1:cy2, cx1:cx2].copy()
        # retificação por quadrilátero interno
        quad = encontrar_quadrilatero_placa(crop)
        crop_rect = warp_para_retangular(crop, quad)

        texto, formato, score, _dbg = ocr_de_recorte(crop_rect)

        resultados.append(((x1,y1,x2,y2), texto, formato, score, crop_rect))

    # Desenho/overlay e salvamento
    vis = img.copy()
    for (x1,y1,x2,y2), texto, formato, score, crop_rect in resultados:
        cv2.rectangle(vis, (x1,y1), (x2,y2), (0,255,0), 2)
        label = f"{texto} ({formato or '??'}) {score:.2f}"
        cv2.putText(vis, label, (x1, max(0, y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    out_path = SAIDA_DIR / f"{Path(caminho).stem}_resultado.jpg"
    cv2.imwrite(str(out_path), vis)

    for i, (_, texto, formato, score, crop_rect) in enumerate(resultados):
        cv2.imwrite(str(SAIDA_DIR / f"{Path(caminho).stem}_crop{i+1}.png"), crop_rect)
        print(f"[{Path(caminho).name}] Placa {i+1}: {texto}  formato={formato}  score={score:.2f}")

# ===================== MAIN =====================

def main():
    modelo = carregar_modelo_yolo(CAMINHO_PESO_YOLO)
    if not IMAGENS_DE_TESTE:
        print("⚠️  Adicione caminhos em IMAGENS_DE_TESTE.")
        return
    for img_path in IMAGENS_DE_TESTE:
        processar_imagem(img_path, modelo)
    print(f"Imagens de saída salvas em: {SAIDA_DIR}")

if __name__ == "__main__":
    main()
