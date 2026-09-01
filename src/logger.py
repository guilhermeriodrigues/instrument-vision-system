"""
src/utils/logger.py
-------------------
Configuração centralizada de logging para o Instrument Vision System.
Garante formato consistente em todos os módulos e suporte a arquivo de log.
"""

import logging
import os
import sys
from pathlib import Path


def setup_logger(name: str, level: str = "INFO", log_to_file: bool = False,
                 log_file: str = "logs/ivs.log") -> logging.Logger:
    """
    Cria e configura um logger nomeado.

    Args:
        name:        Nome do logger (geralmente __name__ do módulo chamador).
        level:       Nível de log: DEBUG | INFO | WARNING | ERROR | CRITICAL.
        log_to_file: Se True, também grava em arquivo.
        log_file:    Caminho do arquivo de log (relativo à raiz do projeto).

    Returns:
        logging.Logger configurado e pronto para uso.
    """
    # Converte string para constante do módulo logging
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    logger = logging.getLogger(name)
    logger.setLevel(numeric_level)

    # Evita handlers duplicados se o logger já foi inicializado
    if logger.handlers:
        return logger

    # --- Formato padrão ---
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # --- Handler: console (stdout) ---
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    # --- Handler: arquivo (opcional) ---
    if log_to_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Retorna um logger existente pelo nome (sem reconfigurar).
    Use nos módulos filhos após setup_logger() já ter sido chamado no main.

    Args:
        name: Nome do logger desejado.

    Returns:
        logging.Logger correspondente.
    """
    return logging.getLogger(name)