import logging


def get_logger(name, level=logging.INFO):
    """ Get logger with given name and level

    Args:
        name (str): logger name
        level (int): logging level
            default: logging.INFO

    Returns:
        logger (logging.Logger): logger
    """

    logger = logging.getLogger(name)
    logger.setLevel(level)

    formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)

    return logger
