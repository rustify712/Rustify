import json
import logging
import os
from datetime import datetime
from typing import Optional

from core.logger.base_logger import BaseLogger

LOG_DATE_DIR_FMT = "%Y-%m-%d"


class FileLogger(BaseLogger):
    """FileLogger

    Args:
        name (str): The name of the logger.
        level (str): The logging level.
        logdir (str): The directory to store log files.
        fmt (str): The format of the log message, default is DEFAULT_FMT.
        date_fmt (str): The format of the date, default is DEFAULT_DATE_FMT.
    """
    DEFAULT_FMT = "[%(asctime)s] - [%(name)s] - [%(levelname)s] - %(message)s"
    DEFAULT_DATE_FMT = "%Y-%m-%d %H:%M:%S"

    def __init__(
            self,
            name: str,
            level: str,
            logdir: str,
            filename: str,
            fmt: Optional[str] = None,
            timestamp_folder: Optional[str] = None,
            date_fmt: Optional[str] = DEFAULT_DATE_FMT,
    ):
        self.logdir = logdir
        self.timestamp_folder = timestamp_folder
        self.filename = filename
        self.fmt = fmt or self.DEFAULT_FMT
        self.date_fmt = date_fmt or self.DEFAULT_DATE_FMT

        # create logdir if not exists
        self.logfile = self.get_logfile()
        self.app_logfile = self.get_application_logfile()

        self.logger = logging.getLogger(name)

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
                raise ValueError(f"invalid logging level: {level}")

        self.file_handler = None
        self.app_file_handler = None
        self.set_file_handler()

    def get_logfile(self):
        if self.timestamp_folder:
            return os.path.join(self.logdir, datetime.now().strftime(LOG_DATE_DIR_FMT), self.timestamp_folder,
                                self.filename)
        return os.path.join(self.logdir, datetime.now().strftime(LOG_DATE_DIR_FMT), self.filename)

    def get_application_logfile(self):
        if self.timestamp_folder:
            return os.path.join(self.logdir, datetime.now().strftime(LOG_DATE_DIR_FMT), self.timestamp_folder,
                                "application.log")
        return os.path.join(self.logdir, datetime.now().strftime(LOG_DATE_DIR_FMT), "application.log")

    def set_file_handler(self):
        """Set the file handler for the logger."""
        if self.file_handler:
            self.file_handler.close()
            self.logger.removeHandler(self.file_handler)
        os.makedirs(os.path.dirname(self.logfile), exist_ok=True)
        self.file_handler = logging.FileHandler(self.logfile)
        formatter = logging.Formatter(
            self.fmt,
            self.date_fmt
        )
        self.file_handler.setFormatter(formatter)
        self.logger.addHandler(self.file_handler)
        # Application log file
        if self.app_file_handler:
            self.app_file_handler.close()
            self.logger.removeHandler(self.app_file_handler)
        os.makedirs(os.path.dirname(self.app_logfile), exist_ok=True)
        self.app_file_handler = logging.FileHandler(self.app_logfile)
        formatter = logging.Formatter(
            self.fmt,
            self.date_fmt
        )
        self.app_file_handler.setFormatter(formatter)
        self.logger.addHandler(self.app_file_handler)

    def check_date(self):
        """Check if the date has changed and update the log file."""
        if datetime.now().strftime(LOG_DATE_DIR_FMT) != self.logfile:
            self.logfile = self.get_logfile()
            self.app_logfile = self.get_application_logfile()
            self.set_file_handler()

    def info(self, msg: str | dict, *args, **kwargs) -> None:
        if isinstance(msg, dict):
            msg = json.dumps(msg, indent=4)
        self.check_date()
        self.logger.info(msg)

    def debug(self, msg: str, *args, **kwargs) -> None:
        if isinstance(msg, dict):
            msg = json.dumps(msg, indent=4)
        self.check_date()
        self.logger.debug(msg)

    def warning(self, msg: str, *args, **kwargs) -> None:
        if isinstance(msg, dict):
            msg = json.dumps(msg, indent=4)
        self.check_date()
        self.logger.warning(msg)

    def error(self, msg: str, *args, **kwargs) -> None:
        if isinstance(msg, dict):
            msg = json.dumps(msg, indent=4)
        self.check_date()
        self.logger.error(msg)

    def disable(self):
        self.logger.enable = False
