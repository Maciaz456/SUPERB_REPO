'''Common methods and classes.'''
import argparse

from common import custom_validate_call, add_common_options


@custom_validate_call
def get_args() -> argparse.Namespace:
    '''Parse arguments passed in the CLI.'''
    args = argparse.ArgumentParser(
        description=(
            'Handle the iris species classification based on the iris dataset.'
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    add_common_options(
        args
    )

    classification_args = args.add_argument_group(
        'Classification options'
    )

    classification_args.add_argument(
        '--test-size',
        '-ts',
        default=0.2,
        type=float,
        help='Test size (as a fraction) for learning.'
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

    return args
