'''Contains wine clusterer.'''
import sys
from pathlib import Path
from typing import ClassVar

from matplotlib import pyplot as plt
from pydantic import BaseModel, Field
from sklearn.cluster import KMeans
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.append(
    f'{Path(__file__).parent.parent.parent}'
)
from common import custom_validate_call, model_config
from my_logger.my_logger import MyLogger


class Wine(
    BaseModel
):
    '''Cluster wine using KMeans.'''
    model_config = model_config

    __n_clusters: ClassVar[int] = 3
    logger: MyLogger = Field(
        frozen=True
    )

    @custom_validate_call
    def model_post_init(
        self,
        _
    ) -> None:
        self.__kmeans = KMeans(
            self.__n_clusters,
            random_state=10
        )
        self.__scaler = StandardScaler()
        self.__pca = PCA(
            n_components=2,
            random_state=20
        )

    @custom_validate_call
    def prepare_data(
        self
    ) -> None:
        '''Loads wine data and scale them.'''
        self.logger.info(
            'Preparing data.'
        )

        dataset = load_wine()
        samples = dataset['data']
        self._scaled_samples = self.__scaler.fit_transform(
            samples
        )

        self.logger.debug(
            'Preparing data completed.'
        )

    @custom_validate_call
    def cluster_wine(
        self
    ) -> None:
        '''Cluster wine samples.'''
        self.logger.info(
            'Clustering wine samples.'
        )

        self.__kmeans.fit(
            self._scaled_samples
        )
        for i in range(
            self.__n_clusters
        ):
            self.logger.debug(
                f'Centroid {i} coordinates: {self.__kmeans.cluster_centers_[i]}'
            )

        self.logger.debug(
            'Clustering wine samples completed.'
        )

    @custom_validate_call
    def visualize(
        self
    ) -> None:
        '''Visualize results after PCA transformation.'''
        self.logger.info(
            'Results visualization.'
        )

        figure = plt.figure(
            num='Clustering visualization'
        )
        subplot = figure.add_subplot()

        t_samples = self.__pca.fit_transform(
            self._scaled_samples
        )
        subplot.scatter(
            t_samples[:, 0],
            t_samples[:, 1],
            c=self.__kmeans.labels_
        )

        t_centroids = self.__pca.transform(
            self.__kmeans.cluster_centers_
        )
        subplot.scatter(
            t_centroids[:, 0],
            t_centroids[:, 1],
            s=50,
            color='red',
            marker='*',
            label='centroids'
        )
        subplot.legend()

        plt.show()

        self.logger.debug(
            'Results visualization completed.'
        )


@custom_validate_call
def main() -> None:
    '''Main function.'''
    logger = MyLogger(
        level='DEBUG'
    )
    clusterer = Wine(
        logger=logger
    )

    clusterer.prepare_data()
    clusterer.cluster_wine()
    clusterer.visualize()


if __name__ == '__main__':
    main()
