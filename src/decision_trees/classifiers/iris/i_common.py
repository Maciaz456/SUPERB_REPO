'''Common methods and classes.'''
import argparse
from pathlib import Path

from common import custom_validate_call, add_common_options


class IrisClassifierException(
    Exception
):
    '''Raised when IrisClassifier failed.'''
    pass


@custom_validate_call
def get_args() -> argparse.Namespace:
    '''Parse arguments passed in the CLI.'''
    args = argparse.ArgumentParser(
        description=(
            'Handle the iris species classifier trained on the iris dataset.'
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    common_args = add_common_options(
        args
    )
    common_args.add_argument(
        '--pkl-file',
        '-pf',
        default='dtree.pkl',
        type=Path,
        help=(
            'Optional .PKL file to which the decision tree will be saved after learning '
            'and/or from which it will be loaded before classification.'
        )
    )

    learning_args = args.add_argument_group(
        'Learning options'
    )
    learning_args.add_argument(
        '--learn',
        '-l',
        action='store_true',
        help='Flag to run model learning.'
    )
    learning_args.add_argument(
        '--test-size',
        '-ts',
        default=0.2,
        type=float,
        help='Test size (as a fraction) for learning.'
    )
    learning_args.add_argument(
        '--accuracy-threshold',
        '-at',
        default=None,
        type=float,
        help='Acceptable accuracy threshold. If specified and met, the decision tree will be saved.'
    )

    classification_args = args.add_argument_group(
        'Classification options'
    )
    classification_args.add_argument(
        '--classify',
        '-c',
        action='store_true',
        help='Flag to classify iris species based on user-specified dimensions.'
    )
    classification_args.add_argument(
        '--read-pkl-file',
        '-rpf',
        action='store_true',
        help='Flag to load the pickled decision tree before classification.'
    )
    classification_args.add_argument(
        '--iris-dims',
        '-id',
        action='append',
        nargs='+',
        type=float,
        help=(
            'Iris dimensions (can be passed multiple times) as follows: '
            '-id sepal length [cm] sepal width [cm] petal length [cm] petal width [cm]'
        )
    )

    args = args.parse_args()

    if not (
        args.learn or args.classify
    ):
        raise IrisClassifierException(
            'Neither learning nor classification has been selected!'
        )

    if args.classify and not args.iris_dims:
        raise IrisClassifierException(
            'Classification has been selected but none of the iris dimensions have been passed!'
        )

    return args
