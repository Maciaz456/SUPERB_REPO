'''Contains HouseValueRegressor.'''
import sys
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

sys.path.append(
    f'{Path(__file__).parent.parent.parent.parent}'
)
from common import custom_validate_call, model_config
from my_logger.my_logger import MyLogger


class HouseValueRegressor(
    BaseModel
):
    '''Predict the median house value in California.'''
    model_config = model_config

    logger: MyLogger = Field(
        frozen=True
    )

    @custom_validate_call
    def model_post_init(
        self,
        _
    ) -> None:
        self._dtree = DecisionTreeRegressor(
            random_state=10
        )
        self.__prepare_learning_data()

    @custom_validate_call
    def __prepare_learning_data(
        self
    ) -> None:
        '''Prepare learning data.'''
        self.logger.info(
            'Preparing learning data.'
        )

        dataset = fetch_california_housing(
            data_home=Path(__file__).parent.joinpath(
                'learning_data'
            )
        )
        self._learn_X = dataset['data']
        self._feature_names = dataset['feature_names']
        self._learn_y = dataset['target']
        self._target_names = dataset['target_names']

        self.logger.debug(
            'Preparing data completed.'
        )

    @custom_validate_call
    def learn(
        self,
        test_size: int | float = 0.2
    ) -> None:
        '''
        Learn the model.

        :param test_size:  Test size.
        '''
        self.logger.info(
            'Learning the regressor.'
        )

        train_X, test_X, train_y, test_y = train_test_split(
            self._learn_X,
            self._learn_y,
            test_size=test_size,
            random_state=10
        )

        self.__train(
            train_X,
            train_y
        )
        self.__test(
            test_X,
            test_y
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

        :param train_X:  train X.
        :param train_y:  train y.
        '''
        self.logger.info(
            'Training the regressor.'
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
        test_y: np.ndarray
    ) -> None:
        '''
        Test the model.

        :param test_X:  test X.
        :param test_y:  test y.
        '''
        self.logger.info(
            'Testing the regressor.'
        )

        y_pred = self._dtree.predict(
            test_X
        )

        mse = mean_squared_error(
            test_y,
            y_pred
        )
        self.logger.info(
            f'Mean squared error (MSE): {mse}'
        )

        r_squared = r2_score(
            test_y,
            y_pred
        )
        self.logger.info(
            f'Coefficient of determination (R^2): {r_squared}'
        )

        self.logger.debug(
            'Testing completed.'
        )

    @custom_validate_call
    def predict(
        self,
        X: list[list | tuple | np.ndarray] | np.ndarray
    ) -> list[tuple]:
        '''
        Predict the house median value.

        :param X:  X for the regression. Each row must cotain:\n
                   - median income in block
                   - median house age in block
                   - average number of rooms
                   - average number of bedrooms
                   - block population
                   - average house occupancy
                   - house block latitude
                   - house block longitude

        :return:   Median house value sample features with predicted values.
        '''
        self.logger.info(
            'Predicting the median house value.'
        )

        y_pred = self._dtree.predict(
            X
        )
        result = [
            tuple(
                sample
            ) for sample in zip(
                X,
                y_pred
            )
        ]

        result_as_str = '\n\n'.join(
            f'Features: {x}\nMedian value: {name}' for x, name in result
        )
        self.logger.info(
            f'Predicted outcome:\n{result_as_str}'
        )

        self.logger.debug(
            'Predicting completed.'
        )

        return result
