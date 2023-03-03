import logging
import os


class CustomFormatter(logging.Formatter):

    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    custom_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: grey + custom_format + reset,
        logging.INFO: grey + custom_format + reset,
        logging.WARNING: yellow + custom_format + reset,
        logging.ERROR: red + custom_format + reset,
        logging.CRITICAL: bold_red + custom_format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(CustomFormatter())
    file_handler = logging.FileHandler(os.getenv('EXPERIMENT_NAME', 'run') + ".log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(CustomFormatter())
    logger.addHandler(ch)
    logger.addHandler(file_handler)
    return logger
