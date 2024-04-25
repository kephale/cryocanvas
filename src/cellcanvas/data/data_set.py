import os
from dataclasses import dataclass
from typing import Dict, List, Union

import numpy as np
import zarr
from zarr import Array

from ome_zarr.io import ZarrLocation
from ome_zarr.reader import Multiscales

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
        image = zarr.open(image_path, "r")

        # get the features
        if isinstance(features_path, str):
            features_path = [features_path]
        features = {path: zarr.open(path, "r") for path in features_path}

        # get the labels
        if (not os.path.isdir(labels_path)) and make_missing_datasets:
            labels = zarr.open(
                labels_path,
                mode="a",
                shape=image.shape,
                dtype="i4",
                dimension_separator=".",
            )
        else:
            if Multiscales.matches(ZarrLocation(labels_path)):
                labels = zarr.open(os.path.join(labels_path, "0"),
                                   "a")
            else:
                labels = zarr.open(labels_path, "a")

        # get the segmentation
        if (not os.path.isdir(segmentation_path)) and make_missing_datasets:
            segmentation = zarr.open(
                segmentation_path,
                mode="a",
                shape=image.shape,
                dtype="i4",
                dimension_separator=".",
            )

        else:
            segmentation = zarr.open(segmentation_path, mode="a")

        return cls(
            image=image,
            features=features,
            labels=labels,
            segmentation=segmentation,
        )

    @classmethod
    def from_stores(
        cls,
        image_store,
        features_store,
        labels_store,
        segmentation_store,
    ):
        """Create a DataSet from a set of paths.

        todo: add ability to create missing labels/segmentations
        """

        # TODO rewrite this to copy everything to be local
        
        # get the image
        # TODO fix hardcoded scale for pickathon
        image = zarr.open(zarr.storage.LRUStoreCache(image_store, None), "r")["0"]

        # get the features
        features = {"features": zarr.open(zarr.storage.LRUStoreCache(features_store, None), "r")}

        group_name = "labels"
        
        # get the labels
        labels = zarr.open_group(zarr.storage.LRUStoreCache(labels_store, None),
                                 mode="a")
        if group_name in labels:
            labels = labels[group_name]
        else:
            labels = labels.create_dataset(group_name,
                                           shape=image.shape,
                                           dtype="i4")

        # get the segmentation
        segmentation = zarr.open_group(zarr.storage.LRUStoreCache(segmentation_store, None),
                                       mode="a")
        if group_name in segmentation:
            segmentation = segmentation[group_name]
        else:
            segmentation = segmentation.create_dataset(group_name,
                                                       shape=image.shape,
                                                       dtype="i4")

        # TODO start a background thread that triggers downloads of the zarrs
            
        return cls(
            image=image,
            features=features,
            labels=labels,
            segmentation=segmentation,
        )
    
