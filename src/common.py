'''Contains methods and classes common for the whole source code.'''
import argparse
from logging import _levelToName
from pathlib import Path

import pydantic


model_config = pydantic.ConfigDict(
    extra='allow',
    strict=True,
    arbitrary_types_allowed=True
)


custom_validate_call = pydantic.validate_call(
    config=dict(
        strict=True,
        validate_return=True,
        arbitrary_types_allowed=True
    )
)


@custom_validate_call
def add_common_options(
    args: argparse.ArgumentParser
) -> argparse._ArgumentGroup:
    '''
    Adds common options to the argument parser.

    :param args:  ArgumentParser instance.

    :return:      Argument group with common options.
    '''
    common_args = args.add_argument_group(
        'Common options'
    )

    common_args.add_argument(
        '--log-level',
        '-ll',
        default='INFO',
        type=str,
        choices=list(
            _levelToName.values()
        ),
        help='Logging level.'
    )

    common_args.add_argument(
        '--log-file',
        '-lf',
        default=None,
        type=Path,
        help='Optional log file.'
    )

    return common_args
