from typing import List, Optional, Tuple

import numpy as np
from napari.utils.events.containers import SelectableEventedList
from zarr import Array
import dask.array as da

from cellcanvas.data.data_set import DataSet


class DataManager:
    def __init__(self, datasets: Optional[List[DataSet]] = None):
        if datasets is None:
            datasets = []
        elif isinstance(datasets, DataSet):
            datasets = [datasets]
        self.datasets = SelectableEventedList(datasets)

    def get_training_data(self) -> Tuple[Array, Array]:
        """Get the pixel-wise semantic segmentation training data for datasets.

        Returns
        -------
        features : Array
            (n, d) array of features where n is the number of pixels
            and d is the number feature dimensions.
        labels : Array
            (n,) array of labels for each feature.
        """

        features = []
        labels = []
        for dataset in self.datasets:
            dataset_features = da.asarray(dataset.concatenated_features)
            dataset_labels = da.asarray(dataset.labels)
            # Flatten labels for boolean indexing
            flattened_labels = dataset_labels.flatten()

            # Compute valid_indices based on labels > 0
            valid_indices = da.nonzero(flattened_labels > 0)[0].compute()

            # Flatten only the spatial dimensions of the dataset_features while preserving the feature dimension
            c, h, w, d = dataset_features.shape
            reshaped_features = dataset_features.reshape(c, h * w * d)

            # We need to apply valid_indices for each feature dimension separately
            filtered_features_list = [da.take(reshaped_features[i, :], valid_indices, axis=0) for i in range(c)]
            filtered_features = da.stack(filtered_features_list, axis=1)

            # Adjust labels
            filtered_labels = flattened_labels[valid_indices] - 1

            features.append(filtered_features)
            labels.append(filtered_labels)
            
        return da.concatenate(features), da.concatenate(labels)
