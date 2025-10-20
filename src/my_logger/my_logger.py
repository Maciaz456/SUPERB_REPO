'''Contains MyLogger'''
import logging
from pathlib import Path

from pydantic import BaseModel

from common import custom_validate_call, model_config


class LoggerException(
    Exception
):
    '''Raised when logging the output failed.'''
    pass


class MyLogger(
    BaseModel
):
    '''
    Log output messages.

    :param command_line:  Flag to log the output to the command line.
    :param log_file:      File to which to log the output.
    :param level:         Level of the logged output.
    '''
    model_config = model_config

    level: str | int = 'INFO'
    command_line: bool = True
    log_file: Path | None = None

    @custom_validate_call
    def model_post_init(
        self,
        _
    ) -> None:
        self.__logger = logging.Logger(
            'my_logger'
        )
        self.__logger.setLevel(
            self.level
        )

        self.__formatter = logging.Formatter(
            '%(asctime)s - %(levelname)-8s - %(message)s',
            datefmt='%H:%M:%S'
        )

        self.__set_logging_methods()

        if self.command_line:
            self.__add_command_line_handler()

        if self.log_file:
            if self.log_file.exists():
                self.log_file.unlink()
            self.log_file.parent.mkdir(
                parents=True,
                exist_ok=True
            )
            self.__add_file_handler()

    @custom_validate_call
    def __add_command_line_handler(
        self
    ) -> None:
        '''Add command line handler to the logger.'''
        handler = logging.StreamHandler()
        handler.setFormatter(
            self.__formatter
        )
        self.__logger.addHandler(
            handler
        )

    @custom_validate_call
    def __add_file_handler(
        self
    ) -> None:
        '''Add file handler to the logger.'''
        handler = logging.FileHandler(
            self.log_file
        )
        handler.setFormatter(
            self.__formatter
        )
        self.__logger.addHandler(
            handler
        )

    @custom_validate_call
    def __set_logging_methods(
        self,
    ) -> None:
        '''Set logging methods.'''
        self.debug = self.__logger.debug
        self.info = self.__logger.info
        self.warning = self.__logger.warning
        self.error = self.__logger.error
        self.exception = self.__logger.exception
        self.critical = self.__logger.critical
