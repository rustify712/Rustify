import warnings
from typing import Literal, Optional

from pydantic import BaseModel, Field

from core.logger.base_logger import BaseLogger

try:
    from core.config import Config

    LOG_LEVEL = Config.LOG_LEVEL
    LOG_TYPE = Config.LOG_TYPE
    LOG_DIR = Config.LOG_DIR
except ImportError:
    LOG_LEVEL = "DEBUG"
    LOG_TYPE = "console"
    LOG_DIR = "logs"


class LoggerConfig(BaseModel):
    level: str = Field(description="Log level", default="INFO")
    type: Literal["console", "file"] = Field(description="Log output type", default="console")
    fmt: Optional[str] = Field(description="Log format", default=None)
    logdir: Optional[str] = Field(description="Log file path", default=None)


# default logger config
default_logger_config = LoggerConfig(
    level=LOG_LEVEL,
    type=LOG_TYPE,
    logdir=LOG_DIR,
)
TIMESTAMP_FOLDER_FORMAT = "%Y%m%d%H%M%S"
TIMESTAMP_FOLDER_NAME = None


class LoggerManager:
    logger_dict = {}
    logger_config: LoggerConfig = default_logger_config

    @classmethod
    def set_logger_config(cls, logger_config: LoggerConfig):
        cls.logger_config = logger_config

    @classmethod
    def get_logger(cls, name: str, logger_type: Optional[Literal["file", "console"]] = None, **kwargs) -> BaseLogger:
        if cls.logger_config is None:
            raise ValueError("LoggerManager should be initialized first")

        if name in cls.logger_dict:
            return cls.logger_dict[name]
        logger_type = logger_type or cls.logger_config.type
        if logger_type == "file":
            from .file_logger import FileLogger
            if "filename" not in kwargs:
                warnings.warn("filename is not provided, using default filename")
                filename = f"{name}.log"
            else:
                filename = kwargs["filename"]
            logdir = kwargs.get("logdir", None) or cls.logger_config.logdir
            logger = FileLogger(
                name=name,
                level=kwargs.get("level", None) or cls.logger_config.level,
                fmt=cls.logger_config.fmt,
                logdir=logdir,
                filename=filename,
                timestamp_folder=TIMESTAMP_FOLDER_NAME
            )
        elif logger_type == "console":
            from .stream_logger import StreamLogger
            logger = StreamLogger(
                name=name,
                level=cls.logger_config.level,
                fmt=cls.logger_config.fmt
            )
        else:
            raise ValueError(f"unknown logger type: {logger_type}")
        cls.logger_dict[name] = logger
        return logger


def get_logger(name: str, enable_timestamp_folder: bool = True, **kwargs) -> BaseLogger:
    """根据名称获取 logger

    Args:
        name: logger 名称
        enable_timestamp_folder: 是否启用时间戳文件夹, 若启用则日志文件会存放在以时间戳命名的文件夹下
        **kwargs:

    Returns:
        BaseLogger: logger
    """
    global TIMESTAMP_FOLDER_NAME
    if enable_timestamp_folder and TIMESTAMP_FOLDER_NAME is None:
        from datetime import datetime
        TIMESTAMP_FOLDER_NAME = datetime.now().strftime(TIMESTAMP_FOLDER_FORMAT)
    return LoggerManager.get_logger(name, **kwargs, enable_timestamp_folder=enable_timestamp_folder)
