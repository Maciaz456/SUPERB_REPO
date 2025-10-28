'''Main script to run the iris classification.'''
import sys
from pathlib import Path

sys.path.append(
    f'{Path(__file__).parent.parent.parent.parent}'
)
from common import custom_validate_call
from decision_trees.classifiers.iris.i_common import get_args
from decision_trees.classifiers.iris.iris import IrisClassifier
from my_logger.my_logger import MyLogger


@custom_validate_call
def main() -> None:
    '''Main function.'''
    args = get_args()

    logger = MyLogger(
        level=args.log_level,
        command_line=True,
        log_file=args.log_file
    )

    classifier = IrisClassifier(
        logger=logger,
        pkl_file=args.pkl_file
    )

    if args.learn:
        classifier.learn(
            args.test_size,
            args.accuracy_threshold
        )

    if args.classify:
        classifier.classify(
            args.iris_dims,
            args.read_pkl_file
        )


if __name__ == '__main__':
    main()
