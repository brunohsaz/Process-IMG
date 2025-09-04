"""
PlateReader — OCR de placas BR (Mercosul e padrão antigo) usando apenas OpenCV.

Requisitos: opencv-python, numpy

A detecção da placa (bounding box ou quadrilátero) é externa (ex.: YOLOv5).
Esta biblioteca faz apenas a leitura e transcrição dos caracteres.

Como preparar templates (obrigatório):
- Crie pastas com gabaritos (PNG/JPG) binários (fundo preto = 0, caractere branco = 255),
  um arquivo por caractere (A.png, B.png, ..., Z.png, 0.png, ..., 9.png),
  todos com o mesmo tamanho (ex.: 56x96 px) e margem mínima (~4 px).
- Dois conjuntos são esperados:
    templates/fe_schrift/   # fonte Mercosul (FE-Schrift)
    templates/old/          # fonte cinza antiga

Uso:
    from plate_reader import PlateReader
    pr = PlateReader(template_root='templates')
    text, confidences = pr.read(image_bgr, quad=quad_points, plate_hint='auto')

Onde quad_points é np.array([(x1,y1),(x2,y2),(x3,y3),(x4,y4)], dtype=np.float32)
no sentido horário, ou use bbox=(x, y, w, h) caso não tenha a perspectiva.

Saída:
- text: string com a transcrição
- confidences: lista de (char, score) por caractere

Observação importante:
- A validação de padrão ajusta ambiguidades (O/0, I/1, S/5, B/8, Z/2) com base
  no formato esperado da placa (Mercosul: LLLNLNN; Antiga: LLLNNNN).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import os
import re
import cv2
import numpy as np

# ----------------------------- Utilidades ------------------------------------

def _ensure_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def _order_quad(quad: np.ndarray) -> np.ndarray:
    q = quad.astype(np.float32)
    if q.shape != (4, 2):
        raise ValueError("quad precisa ser (4,2)")
    # Ordena em sentido horário: top-left, top-right, bottom-right, bottom-left
    s = q.sum(axis=1)
    diff = np.diff(q, axis=1)[:, 0]
    tl = q[np.argmin(s)]
    br = q[np.argmax(s)]
    tr = q[np.argmin(diff)]
    bl = q[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def _four_point_warp(img: np.ndarray, quad: np.ndarray, target_h: int = 160) -> np.ndarray:
    q = _order_quad(quad)
    # Largura estimada pelos lados superior/inferior
    w1 = np.linalg.norm(q[1] - q[0])
    w2 = np.linalg.norm(q[2] - q[3])
    width = int(max(w1, w2))
    height = int(target_h)
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(q, dst)
    warped = cv2.warpPerspective(img, M, (width, height))
    return warped


def _deskew(img: np.ndarray) -> np.ndarray:
    gray = _ensure_gray(img)
    # Usa gradiente horizontal para estimar orientação
    g = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    _, bw = cv2.threshold(cv2.normalize(np.abs(g), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(bw > 0))
    if len(coords) < 50:
        return img
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    if angle < -45:
        angle = 90 + angle
    # Rotaciona
    (h, w) = gray.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return rotated


def _binarize(img: np.ndarray) -> np.ndarray:
    gray = _ensure_gray(img)
    # Normaliza iluminação e aplica limiar adaptativo
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    adap = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 41, 15)
    # Remove ruído fino e fecha buracos
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mor = cv2.morphologyEx(adap, cv2.MORPH_OPEN, kernel, iterations=1)
    mor = cv2.morphologyEx(mor, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mor


def _largest_cc_mask(bw: np.ndarray) -> np.ndarray:
    # Mantém somente o maior componente (placa) quando necessário
    num, labels = cv2.connectedComponents(bw)
    if num <= 2:
        return bw
    areas = [(labels == i).sum() for i in range(1, num)]
    best = 1 + int(np.argmax(areas))
    mask = np.zeros_like(bw)
    mask[labels == best] = 255
    return mask


def _extract_char_regions(bw: np.ndarray) -> List[Tuple[int, int, int, int]]:
    # Encontra contornos e filtra por heurísticas de caractere
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    H, W = bw.shape[:2]
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        ar = h / (w + 1e-6)
        area = w * h
        if area < 80:
            continue
        if h < 0.3 * H or h > 0.98 * H:
            continue
        if ar < 0.8 or ar > 5.5:
            continue
        boxes.append((x, y, w, h))
    # Ordena da esquerda para direita
    boxes = sorted(boxes, key=lambda b: b[0])
    # Agrupa por linhas (caso 2 linhas)
    if len(boxes) > 8:  # pode haver ruído; tenta reduzir
        # k-means 2 clusters em y
        ys = np.array([y + h / 2 for (_, y, _, h) in boxes], dtype=np.float32).reshape(-1, 1)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.1)
        _, labels, _ = cv2.kmeans(ys, 2, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
        line1 = [b for b, l in zip(boxes, labels.flatten()) if l == 0]
        line2 = [b for b, l in zip(boxes, labels.flatten()) if l == 1]
        boxes = sorted(line1, key=lambda b: b[0]) + sorted(line2, key=lambda b: b[0])
    return boxes


def _resize_keep_aspect(img: np.ndarray, target: Tuple[int, int]) -> np.ndarray:
    th, tw = target
    h, w = img.shape[:2]
    scale = min(tw / w, th / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((th, tw), dtype=np.uint8)
    x = (tw - nw) // 2
    y = (th - nh) // 2
    canvas[y:y + nh, x:x + nw] = resized
    return canvas


# ------------------------ Gabaritos/Template Matching -------------------------

@dataclass
class TemplateSet:
    glyphs: Dict[str, List[np.ndarray]]  # cada chave é um caractere, com lista de amostras
    size: Tuple[int, int]

    @staticmethod
    def from_folder(folder: str, target_size: Tuple[int, int] = (96, 56)) -> 'TemplateSet':
        glyphs: Dict[str, List[np.ndarray]] = {}
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Pasta de templates não encontrada: {folder}")
        for fn in os.listdir(folder):
            path = os.path.join(folder, fn)
            if not os.path.isfile(path):
                continue
            name, ext = os.path.splitext(fn)
            if ext.lower() not in ('.png', '.jpg', '.jpeg', '.bmp', '.tif'):
                continue
            ch = name.upper()
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            # binariza e ajusta para alvo
            _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            img = _resize_keep_aspect(img, target_size)
            glyphs.setdefault(ch, []).append(img)
        if not glyphs:
            raise RuntimeError(f"Nenhum template válido encontrado em {folder}")
        return TemplateSet(glyphs=glyphs, size=target_size)

    def match(self, roi: np.ndarray) -> Tuple[str, float]:
        # roi binário no mesmo tamanho de self.size
        roi = _resize_keep_aspect(roi, self.size)
        best_char, best_score = '?', -1.0
        for ch, samples in self.glyphs.items():
            for t in samples:
                # NCC (correlação normalizada)
                res = cv2.matchTemplate(roi, t, cv2.TM_CCOEFF_NORMED)
                score = float(res[0][0])
                if score > best_score:
                    best_char, best_score = ch, score
        return best_char, best_score


# --------------------------- Validação de Padrões -----------------------------

MERCOSUL_PATTERN = "LLLNLNN"  # Brasil: AAA1A23
OLD_PATTERN = "LLLNNNN"       # AAA-1234

CONFUSIONS_NUM = {"O": "0", "I": "1", "Q": "0", "Z": "2", "S": "5", "B": "8"}
CONFUSIONS_LET = {"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B"}


def _enforce_pattern(chars: List[Tuple[str, float]], pattern: str) -> List[Tuple[str, float]]:
    fixed = []
    for i, (ch, sc) in enumerate(chars):
        want = pattern[i] if i < len(pattern) else 'X'
        if want == 'L':
            if not ('A' <= ch <= 'Z'):
                ch = CONFUSIONS_LET.get(ch, ch)
        elif want == 'N':
            if not ('0' <= ch <= '9'):
                ch = CONFUSIONS_NUM.get(ch, ch)
        fixed.append((ch, sc))
    return fixed


def _pattern_score(chars: List[Tuple[str, float]], pattern: str) -> float:
    ok = 0
    for i, (ch, _) in enumerate(chars):
        want = pattern[i] if i < len(pattern) else 'X'
        if want == 'L' and ('A' <= ch <= 'Z'):
            ok += 1
        elif want == 'N' and ('0' <= ch <= '9'):
            ok += 1
    return ok / max(1, len(pattern))


# ------------------------------- Leitor ---------------------------------------

@dataclass
class PlateReader:
    template_root: str = 'templates'
    char_size: Tuple[int, int] = (96, 56)
    expect_len: Tuple[int, int] = (7, 7)  # mínimo, máximo

    def __post_init__(self):
        fe_path = os.path.join(self.template_root, 'fe_schrift')
        old_path = os.path.join(self.template_root, 'old')
        self.templates = {
            'fe': TemplateSet.from_folder(fe_path, self.char_size),
            'old': TemplateSet.from_folder(old_path, self.char_size),
        }

    # --------------------------- API principal --------------------------------
    def read(
        self,
        image_bgr: np.ndarray,
        bbox: Optional[Tuple[int, int, int, int]] = None,
        quad: Optional[np.ndarray] = None,
        plate_hint: str = 'auto',  # 'auto' | 'mercosul' | 'old'
        debug: bool = False,
    ) -> Tuple[str, List[Tuple[str, float]]]:
        """Retorna (texto, lista[(char, score)]).
        - Se quad for fornecido, aplica warp de perspectiva.
        - Caso contrário, usa bbox para recortar. Se nenhum, processa imagem inteira.
        """
        if quad is not None:
            plate = _four_point_warp(image_bgr, quad)
        elif bbox is not None:
            x, y, w, h = bbox
            plate = image_bgr[max(0, y):y + h, max(0, x):x + w]
        else:
            plate = image_bgr.copy()

        plate = _deskew(plate)
        bw = _binarize(plate)

        # Isola região dominante (opcional)
        mask = _largest_cc_mask(bw)
        bw = cv2.bitwise_and(bw, mask)

        boxes = _extract_char_regions(bw)
        if not boxes:
            return "", []

        rois = []
        for (x, y, w, h) in boxes:
            roi = bw[y:y + h, x:x + w]
            roi = _resize_keep_aspect(roi, self.char_size)
            rois.append(roi)

        # Seleção de conjunto de templates
        if plate_hint == 'mercosul':
            sets = [('fe', self.templates['fe'])]
            patterns = [MERCOSUL_PATTERN]
        elif plate_hint == 'old':
            sets = [('old', self.templates['old'])]
            patterns = [OLD_PATTERN]
        else:
            sets = [('fe', self.templates['fe']), ('old', self.templates['old'])]
            patterns = [MERCOSUL_PATTERN, OLD_PATTERN]

        # Matching de caracteres
        candidates: List[Tuple[str, List[Tuple[str, float]], float]] = []
        for name, ts in sets:
            chars: List[Tuple[str, float]] = []
            for roi in rois:
                ch, sc = ts.match(roi)
                chars.append((ch, sc))
            # Ajusta por padrão se comprimentos compatíveis
            if self.expect_len[0] <= len(chars) <= self.expect_len[1]:
                for pat in patterns:
                    adj = _enforce_pattern(chars, pat)
                    pat_sc = _pattern_score(adj, pat)
                    # score global = média das confs + bônus de padrão
                    conf_mean = float(np.mean([s for (_, s) in adj])) if adj else 0.0
                    global_score = 0.8 * conf_mean + 0.2 * pat_sc
                    candidates.append((''.join([c for (c, _) in adj]), adj, global_score))
            else:
                text = ''.join([c for (c, _) in chars])
                conf_mean = float(np.mean([s for (_, s) in chars])) if chars else 0.0
                candidates.append((text, chars, conf_mean))

        if not candidates:
            return "", []

        # Escolhe melhor candidato
        best = max(candidates, key=lambda t: t[2])
        text, adj_chars, _ = best

        # Pós-processamento: remove possíveis duplicatas/ruídos extremos
        text = re.sub(r"[^A-Z0-9]", "", text.upper())
        # Tenta truncar/ajustar para 7 chars, comum nos padrões BR
        if len(text) > 7:
            text = text[:7]
            adj_chars = adj_chars[:7]
        return text, adj_chars

    # --------------------------- Utilitários ----------------------------------
    def draw_debug(self, image_bgr: np.ndarray, bbox: Optional[Tuple[int,int,int,int]] = None,
                   quad: Optional[np.ndarray] = None) -> np.ndarray:
        """Desenha retângulos das regiões de caractere para inspeção visual."""
        if quad is not None:
            plate = _four_point_warp(image_bgr, quad)
            base = plate.copy()
        elif bbox is not None:
            x, y, w, h = bbox
            base = image_bgr[max(0, y):y + h, max(0, x):x + w].copy()
        else:
            base = image_bgr.copy()
        bw = _binarize(base)
        mask = _largest_cc_mask(bw)
        bw = cv2.bitwise_and(bw, mask)
        boxes = _extract_char_regions(bw)
        vis = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
        for (x, y, w, h) in boxes:
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return vis
    
    # DENTRO DA CLASSE PlateReader, em plate_reader.py

    def read_from_rois(
        self,
        rois: List[np.ndarray],
        plate_hint: str = 'auto'
    ) -> Tuple[str, List[Tuple[str, float]]]:
        """
        Executa o OCR a partir de uma lista de imagens de caracteres (ROIs) já segmentadas.
        Pula a detecção interna de caracteres.
        - rois: Lista de imagens binárias, cada uma contendo um caractere.
        """
        if not rois:
            return "", []

        # Seleção de conjunto de templates (lógica copiada do método read)
        if plate_hint == 'mercosul':
            sets = [('fe', self.templates['fe'])]
            patterns = [MERCOSUL_PATTERN]
        elif plate_hint == 'old':
            sets = [('old', self.templates['old'])]
            patterns = [OLD_PATTERN]
        else:
            sets = [('fe', self.templates['fe']), ('old', self.templates['old'])]
            patterns = [MERCOSUL_PATTERN, OLD_PATTERN]

        # Matching de caracteres
        candidates: List[Tuple[str, List[Tuple[str, float]], float]] = []
        for name, ts in sets:
            chars: List[Tuple[str, float]] = []
            for roi in rois:
                # Redimensiona o ROI para o tamanho esperado pelo template antes de comparar
                resized_roi = _resize_keep_aspect(roi, self.char_size)
                ch, sc = ts.match(resized_roi)
                chars.append((ch, sc))

            if self.expect_len[0] <= len(chars) <= self.expect_len[1]:
                for pat in patterns:
                    adj = _enforce_pattern(chars, pat)
                    pat_sc = _pattern_score(adj, pat)
                    conf_mean = float(np.mean([s for (_, s) in adj])) if adj else 0.0
                    global_score = 0.8 * conf_mean + 0.2 * pat_sc
                    candidates.append((''.join([c for (c, _) in adj]), adj, global_score))
            else:
                text = ''.join([c for (c, _) in chars])
                conf_mean = float(np.mean([s for (_, s) in chars])) if chars else 0.0
                candidates.append((text, chars, conf_mean))

        if not candidates:
            return "", []

        best = max(candidates, key=lambda t: t[2])
        text, adj_chars, _ = best

        text = re.sub(r"[^A-Z0-9]", "", text.upper())
        if len(text) > 7:
            text = text[:7]
            adj_chars = adj_chars[:7]
            
        return text, adj_chars


# ------------------------------ CLI opcional ---------------------------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', required=True, help='caminho da imagem BGR da placa ou cena')
    ap.add_argument('--bbox', nargs=4, type=int, help='x y w h')
    ap.add_argument('--quad', nargs=8, type=float, help='x1 y1 x2 y2 x3 y3 x4 y4 (sentido horário)')
    ap.add_argument('--templates', default='templates', help='raiz dos templates (fe_schrift/ e old/)')
    ap.add_argument('--hint', default='auto', choices=['auto','mercosul','old'])
    args = ap.parse_args()

    img = cv2.imread(args.image)
    pr = PlateReader(template_root=args.templates)
    quad = None
    bbox = None
    if args.quad is not None:
        pts = np.array(args.quad, dtype=np.float32).reshape(4,2)
        quad = pts
    elif args.bbox is not None:
        bbox = tuple(args.bbox)

    text, confs = pr.read(img, bbox=bbox, quad=quad, plate_hint=args.hint)
    print(text)
    for ch, sc in confs:
        print(ch, f"{sc:.3f}")
