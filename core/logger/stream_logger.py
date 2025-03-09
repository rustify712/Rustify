import logging
import sys
import uuid

import colorlog

from core.logger.base_logger import BaseLogger


class StreamLogger(BaseLogger):
    """StreamLogger

    Args:
        name (str): The name of the logger.
        level (str): The logging level.
        fmt (str): The format of the log message, default is DEFAULT_FMT.
        date_fmt (str): The format of the date, default is DEFAULT_DATE_FMT.
    """

    DEFAULT_FMT = "%(log_color)s[%(asctime)s] - [%(name)s] - [%(levelname)s] - %(message)s"
    DEFAULT_DATE_FMT = "%Y-%m-%d %H:%M:%S"

    def __init__(
            self,
            name: str,
            level: str,
            fmt: str = None,
            date_fmt: str = None
    ):

        self.enable = True
        self.session_id = str(uuid.uuid4())
        self.logger = logging.getLogger(name)
        self.fmt = fmt or self.DEFAULT_FMT
        self.date_fmt = date_fmt or self.DEFAULT_DATE_FMT

        match level.upper():
            case "DEBUG":
                self.logger.setLevel(logging.DEBUG)
            case "INFO":
                self.logger.setLevel(logging.INFO)
            case "WARNING":
                self.logger.setLevel(logging.WARNING)
            case "ERROR":
                self.logger.setLevel(logging.ERROR)
            case _:
                raise ValueError(f"invalid log level: {level}")
        console_handler = logging.StreamHandler(sys.stdout)
        formatter = colorlog.ColoredFormatter(
            fmt=self.fmt,
            datefmt=self.date_fmt,
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
            }
        )
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

    def info(self, msg: str | dict, *args, **kwargs) -> None:
        if not self.enable:
            return
        self.logger.info(msg)

    def debug(self, msg: str, *args, **kwargs) -> None:
        if not self.enable:
            return
        self.logger.debug(msg)

    def warning(self, msg: str, *args, **kwargs) -> None:
        if not self.enable:
            return
        self.logger.warning(msg)

    def error(self, msg: str, *args, **kwargs) -> None:
        if not self.enable:
            return
        self.logger.error(msg)

    def disable(self):
        self.logger.enable = False
