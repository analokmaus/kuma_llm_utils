import logging


class DefaultLogger:

    def __init__(
            self,
            stream: bool = True,
            file: str | None = None,
            default_level: str = 'INFO'):
        self.stream = stream
        self.file = file
        self.default_level = default_level

    def get_logger(self, name: str = 'default'):
        logger = logging.getLogger(name)
        logger.setLevel(self.default_level)
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            handler.close()
            
        if self.file:
            fh = logging.FileHandler(self.file)
            fh.setFormatter(formatter)
            logger.addHandler(fh)

        if self.stream:
            sh = logging.StreamHandler()
            sh.setFormatter(formatter)
            logger.addHandler(sh)

        formatter = logging.Formatter("%(asctime)s - %(lineno)-4d - %(levelname)-8s - %(message)s")
        return logger
