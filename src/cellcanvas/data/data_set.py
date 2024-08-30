import os
from dataclasses import dataclass
from typing import Dict, List, Union

import numpy as np
import zarr
from zarr import Array


@dataclass
class DataSet:
    """Container for the data relating to a single tomogram"""

    image: Array
    features: Dict[str, Array]
    labels: Array
    segmentation: Array

    @property
    def concatenated_features(self) -> np.ndarray:
        """Return all of the features concatenated into a single array."""
        features_list = list(self.features.values())
        if len(features_list) == 1:
            return features_list[0]
        return np.concatenate(features_list, axis=-1)

    def __hash__(self):
        """Simple hash function, should updated in the future.

        Note: this might create issues if files are moved.
        """
        return hash(self.image.path)

    @classmethod
    def from_paths(
        cls,
        image_path: str,
        features_path: Union[List[str], str],
        labels_path: str,
        segmentation_path: str,
        make_missing_datasets: bool = False,
    ):
        """Create a DataSet from a set of paths.

        todo: add ability to create missing labels/segmentations
        """
        # get the image
        image = zarr.open(store=image_path, mode="r")

        # get the features
        if isinstance(features_path, str):
            features_path = [features_path]
        features = {path: zarr.open(store=path, mode="r") for path in features_path}

        # get the labels
        if (not os.path.isdir(labels_path)) and make_missing_datasets:
            labels = zarr.open(
                store=labels_path,
                mode="a",
                shape=image.shape,
                dtype="i4",
                dimension_separator=".",
            )
        else:
            labels = zarr.open(store=labels_path, mode="a")

        # get the segmentation
        if (not os.path.isdir(segmentation_path)) and make_missing_datasets:
            segmentation = zarr.open(
                store=segmentation_path,
                mode="a",
                shape=image.shape,
                dtype="i4",
                dimension_separator=".",
            )

        else:
            segmentation = zarr.open(store=segmentation_path, mode="a")

        return cls(
            image=image,
            features=features,
            labels=labels,
            segmentation=segmentation,
        )
