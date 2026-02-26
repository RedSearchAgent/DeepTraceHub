# utils/logger.py
import logging
import sys
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

def get_logger(
    name: str = "app",
    level: int = logging.INFO,
    max_bytes: int = 20 * 1024 * 1024,
    backup_count: int = 10,
) -> logging.Logger:

    log_file = os.environ.get("LOG_FILE", "./output/logs.log")
    if not os.path.exists(os.path.dirname(log_file)):
        os.makedirs(os.path.dirname(log_file))
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger

    # 统一格式
    fmt = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    # 控制台
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    # 文件（轮转）
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        filename=log_file, maxBytes=max_bytes,
        backupCount=backup_count, encoding="utf-8"
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.propagate = False  # 避免重复传到 root 造成重复打印
    return logger
