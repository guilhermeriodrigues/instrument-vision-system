"""
src/utils/fps_counter.py
------------------------
Contador de FPS com média móvel para exibição estável no overlay.
Evita oscilações bruscas causadas por variações de latência frame a frame.
"""

import time
from collections import deque


class FPSCounter:
    """
    Calcula FPS em tempo real usando uma janela deslizante de timestamps.

    Attributes:
        window_size (int): Número de frames na janela de média.
    """

    def __init__(self, window_size: int = 30):
        """
        Args:
            window_size: Quantos frames recentes usar para calcular a média.
                         Valores maiores = mais suave; menores = mais responsivo.
        """
        self._timestamps: deque = deque(maxlen=window_size)
        self._window_size = window_size

    def tick(self) -> None:
        """
        Registra o timestamp do frame atual.
        Deve ser chamado uma vez por iteração do loop principal.
        """
        self._timestamps.append(time.perf_counter())

    def get_fps(self) -> float:
        """
        Calcula o FPS atual com base nos timestamps armazenados.

        Returns:
            FPS como float. Retorna 0.0 se não houver frames suficientes.
        """
        if len(self._timestamps) < 2:
            return 0.0

        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed <= 0.0:
            return 0.0

        # Número de intervalos = número de frames - 1
        return (len(self._timestamps) - 1) / elapsed

    def reset(self) -> None:
        """Limpa o histórico de timestamps."""
        self._timestamps.clear()