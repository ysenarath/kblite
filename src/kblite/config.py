from dataclasses import field
from pathlib import Path

from nightjar import BaseConfig

__all__ = [
    "config",
]

CACHE_DIR = Path.home() / ".cache" / "kblite"


class LoggingConfig(BaseConfig):
    stream: bool = False
    filename: str = "default.log"
    level: int = 20
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_dir: Path = CACHE_DIR / "logs"
    truncate: int = 100


class Config(BaseConfig):
    cache_dir: Path = CACHE_DIR
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @property
    def resources_dir(self) -> Path:
        return self.cache_dir / "resources"


config = Config()
