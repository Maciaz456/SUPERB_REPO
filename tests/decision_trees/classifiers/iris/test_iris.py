import re
from unittest.mock import Mock

import numpy as np
import pytest

from decision_trees.classifiers.iris.iris import IrisClassifier


@pytest.fixture
def iris_classifier(
    dummy_my_logger
):
    return IrisClassifier(
        logger=dummy_my_logger
    )


@pytest.mark.parametrize(
    'X',
    argvalues=[
        [
            [0.1, 0.2, 0.3],
            np.array(
                [0.2, 0.3, 0.4]
            )
        ],
        [
           (0.1, 0.2, 0.3, 0.4, 0.5)
        ],
        np.array(
            [
                [0.1, 0.2, 0.3],
                [0.2, 0.3, 0.4]
            ]
        ),
        np.array(
            [
                [0.1, 0.2, 0.3, 0.4, 0.5],
                [0.2, 0.3, 0.4, 0.5, 0.6]
            ]
        )
    ],
    ids=[
        'list_too_few_features',
        'list_too_many_features',
        'array_too_few_columns',
        'array_too_many_columns'
    ]
)
def test_classify_with_incorrect_X(
    iris_classifier,
    X
):
    iris_classifier._IrisClassifier__test = Mock()
    iris_classifier.learn()
    with pytest.raises(
        ValueError,
        match=re.compile(
            r'.*, but DecisionTreeClassifier is expecting 4 features as input.'
        )
    ):
        iris_classifier.classify(
            X
        )
