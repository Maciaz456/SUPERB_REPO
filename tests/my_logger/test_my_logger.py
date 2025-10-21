from pathlib import Path

import pytest

from my_logger.my_logger import MyLogger


@pytest.mark.parametrize(
    (
        'command_line',
        'log_file'
    ),
    (
        (True, Path('folder/log_file.txt')),
        (False, None)
    ),
    ids=(
        'with_both_handlers',
        'with_no_handlers'
    )
)
def test_my_logger(
    tmp_path,
    command_line,
    log_file
):
    if log_file:
        log_file = Path.joinpath(
            tmp_path,
            log_file
        )

    logger = MyLogger(
        command_line=command_line,
        log_file=log_file
    )

    if command_line and log_file:
        assert len(
            logger._MyLogger__logger.handlers
        ) == 2
    elif not (
        command_line and log_file
    ):
        assert len(
            logger._MyLogger__logger.handlers
        ) == 0

    logger.info(
        'TEST'
    )
    if log_file:
        assert log_file.exists() and not Path(log_file).stat().st_size == 0
