'''Common methods and classes.'''
import argparse
from logging import _levelToName
from pathlib import Path

from common import custom_validate_call


class DigitsRecognizerException(
    Exception
):
    '''Raised when the model recognizing digits failed.'''
    pass


@custom_validate_call
def get_args() -> argparse.Namespace:
    '''Parse arguments passed in the CLI.'''
    args = argparse.ArgumentParser(
        description=(
            'Handle model recognizing digits learned on the MNIST dataset '
            'consisting of images with the size of 28x28 pixels.'
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    common_args = args.add_argument_group(
        'Common options'
    )
    common_args.add_argument(
        '--cuda',
        '-c',
        action='store_true',
        required=False,
        help='Use GPU for computations if possible.'
    )
    common_args.add_argument(
        '--pth-file',
        '-pf',
        default='digits_recognizer.pth',
        type=Path,
        required=False,
        help=(
            'Optional .PTH file to which the model parameters will be saved in the learning mode '
            'and/or from which they will be read in the evaluation mode.'
        )
    )
    common_args.add_argument(
        '--log-level',
        '-ll',
        default='INFO',
        type=str,
        choices=list(
            _levelToName.values()
        ),
        required=False,
        help='Logging level.'
    )
    common_args.add_argument(
        '--log-file',
        '-lf',
        default=None,
        type=Path,
        required=False,
        help='Optional log file.'
    )

    learning_args = args.add_argument_group(
        'Learning options'
    )
    learning_args.add_argument(
        '--learn',
        '-l',
        action='store_true',
        required=False,
        help='Flag to run model learning.'
    )
    learning_args.add_argument(
        '--dataset-folder',
        '-df',
        default='data',
        type=Path,
        required=False,
        help='Path to the folder where the MNIST dataset will be downloaded.'
    )
    learning_args.add_argument(
        '--batch-size',
        '-bs',
        default=100,
        type=int,
        required=False,
        help='Batch size in the data loaders.'
    )
    learning_args.add_argument(
        '--hidden-sizes',
        '-hs',
        nargs='+',
        type=int,
        default=[
            500
        ],
        required=False,
        help='Hidden layers sizes. It also defines number of the hidden layers. The input one has a size of 784.'
    )
    learning_args.add_argument(
        '--learning-rate',
        '-lr',
        default=0.01,
        type=float,
        required=False,
        help='Learning rate for the optimizer.'
    )
    learning_args.add_argument(
        '--epochs',
        '-e',
        default=2,
        type=int,
        required=False,
        help='Number of epochs in the training loop.'
    )
    learning_args.add_argument(
        '--accuracy-threshold',
        '-at',
        default=None,
        type=float,
        required=False,
        help='Acceptable accuracy threshold. If specified and met, the model parameters will be saved.'
    )

    eval_args = args.add_argument_group(
        'Evaluation options'
    )
    eval_args.add_argument(
        '--evaluate',
        '-ev',
        action='store_true',
        required=False,
        help='Flag to evaluate digits on the user-specified images.'
    )
    eval_args.add_argument(
        '--read-pth-file',
        '-rpf',
        action='store_true',
        required=False,
        help='Flag to read the model parameters from the .PTH file before evaluation.'
    )
    eval_args.add_argument(
        '--image-paths',
        '-ip',
        nargs='+',
        default=None,
        type=Path,
        required=False,
        help='Images with the size of 28x28 pixels depicting digits to recognize.'
    )

    args = args.parse_args()

    if not (
        args.learn or args.evaluate
    ):
        raise DigitsRecognizerException(
            'Neither learning nor evaluation has been selected!'
        )

    if args.evaluate and not args.image_paths:
        raise DigitsRecognizerException(
            'Evaluation has been selected but none of the image paths has been passed.'
        )

    return args
