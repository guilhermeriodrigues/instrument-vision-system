"""
src/camera/capture.py
---------------------
Módulo de captura de vídeo em tempo real via OpenCV.

Responsabilidades:
  - Abrir e configurar a câmera (resolução, FPS, buffer).
  - Capturar frames com tratamento de erros robusto.
  - Extrair o ROI (Region of Interest) central para análise.
  - Liberar recursos corretamente ao encerrar.

Notas para Raspberry Pi 5:
  - Prefira índice 0 com backend V4L2: cv2.VideoCapture(0, cv2.CAP_V4L2)
  - Reduza resolução para 320×240 se o FPS cair abaixo de 10.
  - cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) minimiza latência de buffer.
"""

import cv2
import numpy as np
from typing import Tuple, Optional

from src.utils.logger import get_logger
from src.utils.config_loader import get_section

logger = get_logger(__name__)


class CameraCapture:
    """
    Encapsula a câmera OpenCV com configuração automática via config.yaml.

    Usage:
        cam = CameraCapture()
        cam.open()
        frame, roi, roi_rect = cam.read()
        cam.release()

    Ou como context manager:
        with CameraCapture() as cam:
            frame, roi, roi_rect = cam.read()
    """

    def __init__(self):
        cfg = get_section("camera")
        self._index     = cfg["index"]
        self._width     = cfg["width"]
        self._height    = cfg["height"]
        self._fps       = cfg["fps"]
        self._roi_ratio = cfg["roi_ratio"]
        self._cap: Optional[cv2.VideoCapture] = None

        logger.info(
            f"CameraCapture inicializado: "
            f"index={self._index}, resolução={self._width}×{self._height}, "
            f"fps={self._fps}, roi_ratio={self._roi_ratio}"
        )

    # ------------------------------------------------------------------
    # Ciclo de vida
    # ------------------------------------------------------------------

    def open(self) -> None:
        """
        Abre a câmera e aplica configurações de resolução/FPS/buffer.

        Raises:
            RuntimeError: Se a câmera não puder ser aberta.
        """
        # CAP_V4L2 é mais estável no Linux/RPi5; fallback automático se não disponível
        self._cap = cv2.VideoCapture(self._index, cv2.CAP_V4L2)

        if not self._cap.isOpened():
            # Tenta sem forçar backend (útil no macOS / Windows)
            logger.warning("CAP_V4L2 falhou. Tentando backend padrão...")
            self._cap = cv2.VideoCapture(self._index)

        if not self._cap.isOpened():
            raise RuntimeError(
                f"Não foi possível abrir a câmera no índice {self._index}. "
                "Verifique a conexão do dispositivo."
            )

        # Aplica propriedades desejadas
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._cap.set(cv2.CAP_PROP_FPS,          self._fps)
        # Minimiza latência de buffer (crítico para resposta em tempo real)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Lê os valores reais aceitos pelo driver
        real_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        real_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        real_fps = self._cap.get(cv2.CAP_PROP_FPS)
        logger.info(f"Câmera aberta: {real_w}×{real_h} @ {real_fps:.1f} fps")

    def release(self) -> None:
        """Libera o recurso da câmera de forma segura."""
        if self._cap and self._cap.isOpened():
            self._cap.release()
            logger.info("Câmera liberada.")

    # ------------------------------------------------------------------
    # Captura de frames
    # ------------------------------------------------------------------

    def read(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray],
                            Optional[Tuple[int, int, int, int]]]:
        """
        Captura um frame e extrai o ROI central.

        Returns:
            Tupla (frame_completo, roi_crop, roi_rect) onde:
              - frame_completo: Frame BGR original (H×W×3).
              - roi_crop:       Região central recortada (para classificação/OCR/ponteiro).
              - roi_rect:       Tupla (x, y, w, h) do ROI em coordenadas do frame completo.

            Retorna (None, None, None) se a captura falhar.
        """
        if self._cap is None or not self._cap.isOpened():
            logger.error("Tentativa de leitura com câmera fechada.")
            return None, None, None

        ret, frame = self._cap.read()
        if not ret or frame is None:
            logger.warning("Frame inválido recebido da câmera.")
            return None, None, None

        roi_crop, roi_rect = self._extract_roi(frame)
        return frame, roi_crop, roi_rect

    # ------------------------------------------------------------------
    # ROI central
    # ------------------------------------------------------------------

    def _extract_roi(self, frame: np.ndarray) -> Tuple[np.ndarray,
                                                        Tuple[int, int, int, int]]:
        """
        Recorta a região central do frame com base em roi_ratio.

        Estratégia: o ROI é um quadrado centrado no frame.
        Isso favorece instrumentos fotografados de frente e garante
        que o classificador veja apenas o instrumento, não o fundo.

        Args:
            frame: Frame BGR completo.

        Returns:
            (roi_crop, (x, y, w, h))
        """
        h, w = frame.shape[:2]

        # Lado do quadrado central
        side = int(min(h, w) * self._roi_ratio)

        # Coordenadas do canto superior esquerdo do ROI
        x = (w - side) // 2
        y = (h - side) // 2

        roi_crop = frame[y:y + side, x:x + side].copy()
        return roi_crop, (x, y, side, side)

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "CameraCapture":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.release()
        return False  # Não suprime exceções

    # ------------------------------------------------------------------
    # Propriedades
    # ------------------------------------------------------------------

    @property
    def is_opened(self) -> bool:
        """True se a câmera está aberta e operacional."""
        return self._cap is not None and self._cap.isOpened()

    @property
    def frame_size(self) -> Tuple[int, int]:
        """Retorna (width, height) reais do frame da câmera."""
        if not self.is_opened:
            return (self._width, self._height)
        return (
            int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )