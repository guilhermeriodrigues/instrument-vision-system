"""
src/utils/config_loader.py
--------------------------
Carrega e valida o arquivo config.yaml, expondo um objeto de configuração
acessível por todos os módulos sem dependência circular.
"""

import yaml
from pathlib import Path
from typing import Any, Dict

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Caminho padrão do arquivo de configuração (relativo à raiz do projeto)
_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"

# Cache da configuração carregada (singleton simples)
_config_cache: Dict[str, Any] = {}


def load_config(config_path: Path = _DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """
    Lê o arquivo YAML e armazena em cache.

    Args:
        config_path: Caminho para o config.yaml. Usa o padrão se omitido.

    Returns:
        Dicionário com todas as configurações.

    Raises:
        FileNotFoundError: Se config.yaml não for encontrado.
        yaml.YAMLError:    Se o arquivo contiver YAML inválido.
    """
    global _config_cache

    if _config_cache:
        return _config_cache

    if not config_path.exists():
        raise FileNotFoundError(
            f"Arquivo de configuração não encontrado: {config_path}\n"
            "Certifique-se de executar o sistema a partir da raiz do projeto."
        )

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        raise ValueError("config.yaml está vazio ou mal formatado.")

    _config_cache = cfg
    logger.info(f"Configuração carregada de: {config_path}")
    return _config_cache


def get_config() -> Dict[str, Any]:
    """
    Retorna a configuração do cache. Chama load_config() se ainda não foi carregada.

    Returns:
        Dicionário de configuração completo.
    """
    if not _config_cache:
        return load_config()
    return _config_cache


def get_section(section: str) -> Dict[str, Any]:
    """
    Retorna uma seção específica da configuração (ex: 'camera', 'analog').

    Args:
        section: Nome da chave de nível superior no config.yaml.

    Returns:
        Sub-dicionário da seção solicitada.

    Raises:
        KeyError: Se a seção não existir.
    """
    cfg = get_config()
    if section not in cfg:
        raise KeyError(
            f"Seção '{section}' não encontrada no config.yaml. "
            f"Seções disponíveis: {list(cfg.keys())}"
        )
    return cfg[section]