'''Contains IrisClassifier.'''
import os
import pickle
from annotated_types import Ge, Le
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
from matplotlib import pyplot
from pydantic import BaseModel, Field
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix

from common import custom_validate_call, model_config
from my_logger.my_logger import MyLogger


N_FEATURES = 4


class IrisClassifier(
    BaseModel
):
    '''Classify iris.'''
    model_config = model_config

    logger: MyLogger = Field(
        frozen=True
    )
    pkl_file: Path = Path(
        'dtree.pkl'
    )

    @custom_validate_call
    def model_post_init(
        self,
        _
    ) -> None:
        self._dtree = DecisionTreeClassifier()
        self.__prepare_learning_data()

    @custom_validate_call
    def __prepare_learning_data(
        self
    ) -> None:
        '''Prepare learning data.'''
        self.logger.info(
            'Preparing learning data.'
        )

        dataset = load_iris()
        self._X = dataset['data']
        self._feature_names = dataset['feature_names']
        self._y = dataset['target']
        self._target_names = dataset['target_names']

        self.logger.debug(
            'Preparing data completed.'
        )

    @custom_validate_call
    def learn(
        self,
        test_size: int | float = 0.2,
        accuracy_threshold: Annotated[float, Ge(0), Le(100)] | None = None
    ) -> None:
        '''
        Learn the model.

        :param test_size:           Test size.
        :param accuracy_threshold:  Acceptable accuracy threshold to save the decision tree.
                                    If not passed, the tree will not be saved.
        '''
        self.logger.info(
            'Learning the classifier.'
        )

        train_X, test_X, train_y, test_y = train_test_split(
            self._X,
            self._y,
            test_size=test_size
        )

        self.__train(
            train_X,
            train_y
        )
        self.__test(
            test_X,
            test_y,
            accuracy_threshold
        )

        self.logger.debug(
            'Learning completed.'
        )

    @custom_validate_call
    def __train(
        self,
        train_X: np.ndarray,
        train_y: np.ndarray
    ) -> None:
        '''
        Train the model.

        :param train_X:  Train X.
        :param train_y:  Train y.
        '''
        self.logger.info(
            'Training the classifier.'
        )

        self._dtree.fit(
            train_X,
            train_y
        )

        self.logger.debug(
            'Training completed.'
        )

    @custom_validate_call
    def __test(
        self,
        test_X: np.ndarray,
        test_y: np.ndarray,
        accuracy_threshold: Annotated[float, Ge(0), Le(100)] | None = None
    ):
        '''
        Test the model.

        :param test_X:              Test X.
        :param test_y:              Test y.
        :param accuracy_threshold:  Acceptable accuracy threshold to save the decision tree.
                                    If not passed, the tree will not be saved.
        '''
        self.logger.info(
            'Testing the classifier.'
        )

        y_pred = self._dtree.predict(
            test_X
        )

        conf_matrix = confusion_matrix(
            test_y,
            y_pred
        )

        display = ConfusionMatrixDisplay(
            conf_matrix,
            display_labels=self._target_names
        )
        display.plot()
        pyplot.show()

        test_accuracy = accuracy_score(
            test_y,
            y_pred
        )
        self.logger.info(
            f'Test accuracy: {(test_accuracy):.1%}'
        )
        if accuracy_threshold is not None and test_accuracy*100 >= accuracy_threshold:
            self.__handle_dtree_pickle(
                'dump'
            )

        self.logger.debug(
            'Testing completed.'
        )

    @custom_validate_call
    def __handle_dtree_pickle(
        self,
        action: Literal['dump'] | Literal['load']
    ) -> None:
        '''
        Handle self._dtree pickle.

        :param action:    Either dump to or load from the .pkl file
        '''
        if action == 'dump':
            if self.pkl_file.exists():
                os.unlink(
                    self.pkl_file
                )
            self.pkl_file.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            with open(
                self.pkl_file,
                'wb'
            ) as f:
                pickle.dump(
                    self._dtree,
                    f
                )
        elif action == 'load':
            with open(
                self.pkl_file,
                'rb'
            ) as f:
                self._dtree = pickle.load(
                    f
                )

    @custom_validate_call
    def classify(
        self,
        X: list[list | tuple | np.ndarray] | np.ndarray,
        read_pkl_file: bool = False
    ) -> list[tuple]:
        '''
        Classify the iris species.

        :param X:              X for the classification. Each row must cotain:\n
                               - sepal length [cm]
                               - sepal width [cm]
                               - petal length [cm]
                               - petal width [cm]
        :param read_pkl_file:  Flag to load the pickled decision tree before the classification.

        :return:               Iris sample features with classified names.
        '''
        self.logger.info(
            'Iris species classification.'
        )

        if read_pkl_file:
            self.__handle_dtree_pickle(
                'load'
            )

        y_pred = self._dtree.predict(
            X
        )
        result = [
            (
                X[i], f'{self._target_names[x]}'
            ) for i, x in enumerate(
                y_pred
            )
        ]

        result_as_str = '\n'.join(
            f'{x} - {name}' for x, name in result
        )
        self.logger.info(
            f'Classification outcome:\n{result_as_str}'
        )

        self.logger.debug(
            'Classification completed.'
        )

        return result
