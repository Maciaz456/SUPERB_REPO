import sys
from pathlib import Path

import pytest

sys.path.append(
    f'{Path(__file__).parent.parent.joinpath('src')}'
)
from my_logger.my_logger import MyLogger


@pytest.fixture
def dummy_my_logger():
    return MyLogger(
        command_line=False
    )


@pytest.fixture
def data_folder():
    return Path(__file__).parent.joinpath(
        'data'
    )
