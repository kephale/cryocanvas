from typing import Protocol

import sys
import logging
import numpy as np
import dask.array as da
from dask import delayed
from sklearn.exceptions import NotFittedError

from cellcanvas.data.data_manager import DataManager
from tqdm import tqdm

class SegmentationModel(Protocol):
    """Protocol for semantic segmentations models that are
    compatible with the  SemanticSegmentationManager.
    """

    def fit(self, X, y): ...
    def predict(self, X): ...


class SemanticSegmentationManager:
    def __init__(self, data: DataManager, model: SegmentationModel):
        self.data = data
        self.model = model

        self._init_logging()

    def _init_logging(self):
        self.logger = logging.getLogger("cellcanvas")
        self.logger.setLevel(logging.DEBUG)
        streamHandler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        streamHandler.setFormatter(formatter)
        self.logger.addHandler(streamHandler)

    def update_data_manager(self, data: DataManager):
        self.data = data

    def fit(self):
        """Fit using the model using the data in the data manager."""
        self.logger.info("Starting to fit")
        # Get training data from the data manager
        features, labels = self.data.get_training_data()

        features_computed, labels_computed = features.compute(), labels.compute()

        self.logger.info("Starting the actual model fit")

        self.model.fit(features_computed, labels_computed)

    def predict(self, feature_image):
        """Predict using the trained model.

        Parameters
        ----------
        feature_image : np.ndarray
            (z, y, x, c) image where c is the dimensionality of the features.

        Returns
        -------
        predicted_labels : Array
            The prediction of class.
        """
        c, z, y, x = feature_image.shape
        features = feature_image.transpose(1, 2, 3, 0).reshape(-1, c)

        try:
            predicted_labels = self.model.predict(features)
        except NotFittedError:
            raise NotFittedError(
                "You must train the classifier `clf` first"
                "for example with the `fit_segmenter` function."
            ) from None

        return predicted_labels.reshape(feature_image.shape[1:])

