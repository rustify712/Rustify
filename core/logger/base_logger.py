from abc import ABC, abstractmethod


class BaseLogger(ABC):
    """Logger
    """

    @abstractmethod
    def debug(self, msg: str, *args, **kwargs) -> None:
        """Logs a debug message.

        Args:
            msg (str): The message to log.
        """
        raise NotImplementedError

    @abstractmethod
    def info(self, msg: str, *args, **kwargs) -> None:
        """Logs an informational message.

        Args:
            msg (str): The message to log.
        """
        raise NotImplementedError

    @abstractmethod
    def warning(self, msg: str, *args, **kwargs) -> None:
        """Logs a warning message.

        Args:
            msg (str): The message to log.
        """
        raise NotImplementedError

    @abstractmethod
    def error(self, msg: str, *args, **kwargs) -> None:
        """Logs an error message.

        Args:
            msg (str): The message to log.
        """
        raise NotImplementedError

    @abstractmethod
    def disable(self) -> None:
        """Disables logging."""
        raise NotImplementedError
