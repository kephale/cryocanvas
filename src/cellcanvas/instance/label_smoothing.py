from typing import List, Union

import numpy as np

from pyclesperanto_prototype import erode_labels, dilate_labels, closing_labels, opening_labels
import numpy as np

from cellcanvas.instance.bounding_box_utils import (
    crop_array_with_bounding_box,
    get_expanded_bounding_box,
    insert_cropped_array_into_array,
)


def dilate_labels_with_crop(
        label_image: np.ndarray,
        label_values_to_expand: Union[int, List[int]],
        overwrite_values: Union[int, List[int]],
        expansion_amount: int = 1,
) -> np.ndarray:
    return morphological_with_crop(
        morphological_operation='dilate',
        label_image=label_image,
        label_values_to_alter=label_values_to_expand,
        overwrite_values=overwrite_values,
        operation_radius=expansion_amount,
    )


def erode_labels_with_crop(
        label_image: np.ndarray,
        label_values_to_erode: Union[int, List[int]],
        overwrite_values: Union[int, List[int]],
        erosion_amount: int = 1,
) -> np.ndarray:
    return morphological_with_crop(
        morphological_operation='erode',
        label_image=label_image,
        label_values_to_alter=label_values_to_erode,
        overwrite_values=overwrite_values,
        operation_radius=erosion_amount,
    )


def closing_labels_with_crop(
        label_image: np.ndarray,
        label_values_to_close: Union[int, List[int]],
        overwrite_values: Union[int, List[int]],
        operation_radius: int = 1,
) -> np.ndarray:
    return morphological_with_crop(
        morphological_operation='closing',
        label_image=label_image,
        label_values_to_alter=label_values_to_close,
        overwrite_values=overwrite_values,
        operation_radius=operation_radius,
    )


def opening_labels_with_crop(
        label_image: np.ndarray,
        label_values_to_open: Union[int, List[int]],
        overwrite_values: Union[int, List[int]],
        operation_radius: int = 1,
) -> np.ndarray:
    return morphological_with_crop(
        morphological_operation='opening',
        label_image=label_image,
        label_values_to_alter=label_values_to_open,
        overwrite_values=overwrite_values,
        operation_radius=operation_radius,
    )


def morphological_with_crop(
        morphological_operation: str,
        label_image: np.ndarray,
        label_values_to_alter: Union[int, List[int]],
        overwrite_values: Union[int, List[int]],
        operation_radius: int = 1,
) -> np.ndarray:
    """Dilate labels in a label image with a crop around the labels to expand.

    This function will dilate the labels in a label image within a crop around the
    labels to expand.

    Parameters
    ----------
    label_image : np.ndarray
        The label image to dilate.
    label_values_to_expand : Union[int, List[int]]
        The label values to expand.
    overwrite_values : Union[int, List[int]]
        The label values to overwrite. If no labels are to be overwritten,
        pass an empty list.
    expansion_amount : int
        The number of pixels to expand the labels by.

    Returns
    -------
    expanded_labels : np.ndarray
        The expanded label image.

    """

    label_values_to_alter = to_list(label_values_to_alter)
    overwrite_values = to_list(overwrite_values)

    # This is currently a bottleneck
    # numpy.isin seems to be slower than np.logical_or for small label sets
    label_mask = custom_isin(
        image=label_image,
        compare_values=label_values_to_alter,
    )

    expanded_bounding_box = get_expanded_bounding_box(
        label_mask=label_mask,
        expansion_amount=(operation_radius if morphological_operation != 'erode' else 0),
        image_shape=label_image.shape,
    )

    # extract a crop around the bounding box of the labels
    expansion_crop = crop_array_with_bounding_box(
        array=label_image,
        bounding_box=expanded_bounding_box,
    )

    labels_mask_crop = crop_array_with_bounding_box(
        array=label_mask,
        bounding_box=expanded_bounding_box,
    )

    # keep the locked labels
    overwrite_mask_crop = custom_isin(
        image=expansion_crop,
        compare_values=overwrite_values,
    )
    labels_mask_crop = custom_isin(
        image=expansion_crop,
        compare_values=label_values_to_alter,
    )

    # Lock all labels that are not selected and are not to be overwritten
    locked_mask_crop = np.logical_and(~labels_mask_crop, np.logical_and(expansion_crop > 0, ~overwrite_mask_crop))
    locked_labels_crop = expansion_crop[locked_mask_crop]

    # # save original labels (will be overwritten by dilated labels)
    all_labels_crop = expansion_crop.copy()

    # Set the locked labels and non-selected labels to 0 to prevent them from being expanded
    # This also allows other labels to expand into non-selected labels
    # Do we want this?
    expansion_crop[np.logical_or(locked_mask_crop, ~labels_mask_crop)] = 0

    if morphological_operation == 'dilate':
        # dilate the labels
        morphed_crop = dilate_labels(
            labeling_source=expansion_crop,
            radius=operation_radius,
        ).get()
    elif morphological_operation == 'erode':
        morphed_crop = erode_labels(
            labels_input=expansion_crop,
            radius=operation_radius,
            relabel_islands=False,  # Do we want this?
        ).get()
    elif morphological_operation == 'closing':
        morphed_crop = closing_labels(
            labels_input=expansion_crop,
            radius=operation_radius,
        ).get()
    elif morphological_operation == 'opening':
        morphed_crop = opening_labels(
            labels_input=expansion_crop,
            radius=operation_radius,
        ).get()

    # erosion and opening return a label map that is from 1 to n
    # TODO: Make sure that erased labels are still in the label numbering
    if morphological_operation in ['erode', 'opening']:
        morphed_mask = morphed_crop > 0
        morphed_crop = np.zeros_like(morphed_crop)
        morphed_crop[morphed_mask] = expansion_crop[morphed_mask]

    # combine the dilated labels with the original labels
    combined_crop = np.where(np.logical_or(morphed_crop > 0, labels_mask_crop), morphed_crop, all_labels_crop)

    # insert the locked labels back into the combined crop
    # This potentially overwrites dilated labels with locked labels
    combined_crop[locked_mask_crop] = locked_labels_crop

    # insert the expanded labels back into the original image
    expanded_labels = insert_cropped_array_into_array(
        cropped_array=combined_crop,
        array=label_image,
        bounding_box=expanded_bounding_box,
    )
    return expanded_labels


def to_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return value
    else:
        return [value]


def custom_isin(
        image: np.ndarray,
        compare_values: List[int],
):
    if len(compare_values) < 15:
        label_mask = np.zeros_like(image, dtype=bool)
        for label_value in compare_values:
            label_mask = np.logical_or(label_mask, np.asarray(image == label_value))
    else:
        label_mask = np.isin(image, compare_values)
    return label_mask
