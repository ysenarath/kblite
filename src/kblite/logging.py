import logging

from kblite.config import config

__all__ = [
    "get_logger",
]

# levels
DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING = logging.WARNING
ERROR = logging.ERROR
CRITICAL = logging.CRITICAL


def setup_logger(logger: logging.Logger) -> None:
    logger.setLevel(config.logging.level)
    formatter = logging.Formatter(config.logging.format)
    if config.logging.stream:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    if config.logging.filename:
        path = config.logging.log_dir / config.logging.filename
        # create log directory if it does not exist
        path.parent.mkdir(parents=True, exist_ok=True)
        # truncate log file if it exists
        num_lines = config.logging.truncate
        if path.exists() and num_lines >= 0:
            with path.open("r") as f:
                lines = f.readlines()
            with path.open("w") as f:
                f.writelines(lines[-num_lines:])
        file_handler = logging.FileHandler(path, mode="a")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    logger.debug("Logger is set up")


def get_logger(name: str, setup: bool = True) -> logging.Logger:
    logger = logging.getLogger(name)
    if setup:
        setup_logger(logger)
    return logger
