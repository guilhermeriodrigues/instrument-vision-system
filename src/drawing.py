"""
src/utils/drawing.py
--------------------
Funções de renderização do overlay visual no frame OpenCV.
Centraliza todos os cv2.putText / cv2.rectangle para facilitar customização.
"""

import cv2
import numpy as np
from typing import Tuple, Optional

from src.utils.config_loader import get_section


# Carrega paleta de cores e parâmetros de exibição uma única vez
_disp = get_section("display")

# Cores BGR lidas do config
COLOR_ANALOG   = tuple(_disp["color_analog"])    # Ponteiro analógico
COLOR_DIGITAL  = tuple(_disp["color_digital"])   # Display digital
COLOR_WARNING  = tuple(_disp["color_warning"])   # Alertas / baixa confiança
COLOR_INFO     = tuple(_disp["color_info"])      # Texto secundário (FPS, labels)

FONT           = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE     = _disp["font_scale"]
THICKNESS      = _disp["thickness"]
LINE_TYPE      = cv2.LINE_AA


def draw_roi_border(frame: np.ndarray, roi_rect: Tuple[int, int, int, int],
                    color: Tuple[int, int, int] = COLOR_INFO,
                    thickness: int = 2) -> None:
    """
    Desenha a borda do ROI central no frame.

    Args:
        frame:    Frame OpenCV (BGR, uint8).
        roi_rect: Tupla (x, y, w, h) do retângulo ROI.
        color:    Cor BGR da borda.
        thickness: Espessura da linha.
    """
    x, y, w, h = roi_rect
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness, LINE_TYPE)


def draw_overlay(frame: np.ndarray,
                 class_name: str,
                 confidence: float,
                 reading: Optional[str],
                 fps: float,
                 low_confidence: bool = False) -> None:
    """
    Renderiza o painel de informações principal no frame.

    Layout:
      ┌─────────────────────────────────────┐
      │ [CLASS]  conf: XX.X%                │  ← linha 1
      │ Leitura: <valor>                    │  ← linha 2
      │ FPS: XX.X                           │  ← linha 3 (canto inf. direito)
      └─────────────────────────────────────┘

    Args:
        frame:          Frame OpenCV (modificado in-place).
        class_name:     "Analog_Instruments" | "Digital_Instruments" | "..."
        confidence:     Confiança do classificador [0.0, 1.0].
        reading:        Valor extraído como string (ex: "73.2 %", "220 V").
                        None se leitura não disponível.
        fps:            FPS calculado pelo FPSCounter.
        low_confidence: Se True, usa cor de alerta para sinalizar incerteza.
    """
    # Determina cor principal com base na classe e confiança
    if low_confidence:
        primary_color = COLOR_WARNING
    elif "Analog" in class_name:
        primary_color = COLOR_ANALOG
    else:
        primary_color = COLOR_DIGITAL

    # --- Fundo semitransparente para legibilidade ---
    _draw_transparent_bg(frame, x=8, y=8, w=340, h=90, alpha=0.45)

    # --- Linha 1: Classe + Confiança ---
    label_class = f"{class_name}  |  conf: {confidence * 100:.1f}%"
    cv2.putText(frame, label_class, (14, 32),
                FONT, FONT_SCALE, primary_color, THICKNESS, LINE_TYPE)

    # --- Linha 2: Leitura ---
    label_reading = f"Leitura: {reading}" if reading is not None else "Leitura: ---"
    cv2.putText(frame, label_reading, (14, 64),
                FONT, FONT_SCALE + 0.1, primary_color, THICKNESS + 1, LINE_TYPE)

    # --- Linha 3: FPS (canto inferior direito) ---
    h_frame, w_frame = frame.shape[:2]
    fps_text = f"FPS: {fps:.1f}"
    (tw, th), _ = cv2.getTextSize(fps_text, FONT, FONT_SCALE * 0.9, 1)
    cv2.putText(frame, fps_text,
                (w_frame - tw - 10, h_frame - 12),
                FONT, FONT_SCALE * 0.9, COLOR_INFO, 1, LINE_TYPE)


def draw_hough_lines(frame: np.ndarray,
                     lines: np.ndarray,
                     color: Tuple[int, int, int] = (0, 255, 0),
                     thickness: int = 2) -> None:
    """
    Desenha as linhas retornadas por HoughLinesP (modo debug).

    Args:
        frame:     Frame OpenCV onde desenhar.
        lines:     Array shape (N, 1, 4) de [x1,y1,x2,y2].
        color:     Cor BGR das linhas.
        thickness: Espessura em pixels.
    """
    if lines is None:
        return
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(frame, (x1, y1), (x2, y2), color, thickness, LINE_TYPE)


def draw_pointer_arrow(frame: np.ndarray,
                       center: Tuple[int, int],
                       angle_deg: float,
                       length: int = 60,
                       color: Tuple[int, int, int] = COLOR_ANALOG) -> None:
    """
    Desenha uma seta indicando o ângulo do ponteiro detectado.
    Útil para debug visual do módulo analógico.

    Args:
        frame:     Frame OpenCV (modificado in-place).
        center:    Ponto central (cx, cy) em pixels.
        angle_deg: Ângulo em graus (0° = direita, sentido anti-horário).
        length:    Comprimento da seta em pixels.
        color:     Cor BGR.
    """
    import math
    cx, cy = center
    rad = math.radians(angle_deg)
    ex = int(cx + length * math.cos(rad))
    ey = int(cy - length * math.sin(rad))  # y invertido em coordenadas de imagem
    cv2.arrowedLine(frame, (cx, cy), (ex, ey), color, 2, LINE_TYPE, tipLength=0.3)


def draw_no_detection(frame: np.ndarray, message: str = "Sem detecção") -> None:
    """
    Exibe mensagem centralizada quando nenhum instrumento é detectado.

    Args:
        frame:   Frame OpenCV.
        message: Texto a exibir.
    """
    h, w = frame.shape[:2]
    (tw, th), _ = cv2.getTextSize(message, FONT, FONT_SCALE, THICKNESS)
    cx = (w - tw) // 2
    cy = (h + th) // 2
    cv2.putText(frame, message, (cx, cy),
                FONT, FONT_SCALE, COLOR_WARNING, THICKNESS, LINE_TYPE)


# ---------------------------------------------------------------------------
# Helper interno
# ---------------------------------------------------------------------------

def _draw_transparent_bg(frame: np.ndarray,
                          x: int, y: int, w: int, h: int,
                          alpha: float = 0.4) -> None:
    """
    Sobrepõe um retângulo preto semitransparente para fundo de texto.

    Args:
        frame: Frame de destino (modificado in-place).
        x, y:  Canto superior esquerdo.
        w, h:  Largura e altura do retângulo.
        alpha: Opacidade (0=transparente, 1=opaco).
    """
    sub = frame[y:y + h, x:x + w]
    black_rect = np.zeros_like(sub)
    blended = cv2.addWeighted(sub, 1.0 - alpha, black_rect, alpha, 0)
    frame[y:y + h, x:x + w] = blended