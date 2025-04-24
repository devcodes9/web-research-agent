import logging

def setup_logger(name: str, level: str = "INFO") -> logging.Logger:
    fmt = "%(asctime)s %(levelname)s %(name)s %(message)s"
    logging.basicConfig(level=level, format=fmt)
    return logging.getLogger(name)