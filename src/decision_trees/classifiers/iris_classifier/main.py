'''Main script to run the iris classification.'''
import sys
from pathlib import Path

sys.path.append(
    f'{Path(__file__).parent.parent.parent.parent}'
)
from common import custom_validate_call
from decision_trees.classifiers.iris_classifier.ic_common import get_args
from decision_trees.classifiers.iris_classifier.iris_classifier import IrisClassifier
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
        logger=logger
    )

    classifier.learn(
        args.test_size
    )

    classifier.classify(
        args.iris_dims
    )


if __name__ == '__main__':
    main()
