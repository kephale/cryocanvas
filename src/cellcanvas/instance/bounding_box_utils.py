from typing import Optional, Tuple
import numpy as np


def get_expanded_bounding_box(
        label_mask: np.ndarray,
        expansion_amount: int,
        image_shape: Optional[Tuple[int]] = None,
) -> np.ndarray:
    """Expand a bounding box bidirectionally along each axis by a specified amount.

    Parameters
    ----------
    label_mask : np.ndarray
        The label mask to expand.
    expansion_amount : int
        The number of pixels to expand the bounding box by in each direction
    image_shape : Tuple[int]
        The size of the image along each axis.

    Returns
    -------
    expanded_bounding_box : np.ndarray
        The expanded bounding box.
    """
    if expansion_amount > 1:
        bounding_box_expansion = 2 * expansion_amount
    else:
        # when expansion is 1, actual radius ends up being 3
        bounding_box_expansion = 4

    bounding_box = get_mask_bounding_box(label_mask)
    expanded_bounding_box = expand_bounding_box(
        bounding_box=bounding_box,
        expansion_amount=bounding_box_expansion,
        image_shape=image_shape,
    )

    # ensure all dimensions in the bounding box have at least size 1
    singleton_dimensions = (
                                   expanded_bounding_box[:, 1] - expanded_bounding_box[:, 0]
                           ) == 0
    expanded_bounding_box[singleton_dimensions, 1] = (
            expanded_bounding_box[singleton_dimensions, 1] + 1
    )

    return expanded_bounding_box


def crop_array_with_bounding_box(
        array: np.ndarray,
        bounding_box: np.ndarray,
) -> np.ndarray:
    """Crop an array using a bounding box.

    Parameters
    ----------
    array : np.ndarray
        The array to crop.
    bounding_box : np.ndarray
        The bounding box as an array where arranged:
            [
                [0_min, _max],
                [1_min, 1_max],
                [2_min, 2_max]
            ]
        where 0, 1, and 2 are the 0th, 1st, and 2nd dimensions,
        respectively.

    Returns
    -------
    cropped_array : np.ndarray
        The cropped array.
    """
    return array[
           bounding_box[0, 0]: bounding_box[0, 1] + 1,
           bounding_box[1, 0]: bounding_box[1, 1] + 1,
           bounding_box[2, 0]: bounding_box[2, 1] + 1,
           ]


def insert_cropped_array_into_array(
        cropped_array: np.ndarray,
        array: np.ndarray,
        bounding_box: np.ndarray,
) -> np.ndarray:
    """Insert a cropped array into a larger array using a bounding box.

    Parameters
    ----------
    cropped_array : np.ndarray
        The array to insert.
    array : np.ndarray
        The array to insert into.
    bounding_box : np.ndarray
        The bounding box as an array where arranged:
            [
                [0_min, _max],
                [1_min, 1_max],
                [2_min, 2_max]
            ]
        where 0, 1, and 2 are the 0th, 1st, and 2nd dimensions,
        respectively.

    Returns
    -------
    array_with_inserted_cropped_array : np.ndarray
        The array with the cropped array inserted.
    """
    array[
    bounding_box[0, 0]: bounding_box[0, 1] + 1,
    bounding_box[1, 0]: bounding_box[1, 1] + 1,
    bounding_box[2, 0]: bounding_box[2, 1] + 1,
    ] = cropped_array
    return array


def get_mask_bounding_box(mask_image: np.ndarray) -> np.ndarray:
    """Get the axis-aligned bounding box around the True values in a 3D mask.

    Parameters
    ----------
    mask_image : BinaryImage
        The binary image from which to calculate the bounding box.

    Returns
    -------
    bounding_box : np.ndarray
        The bounding box as an array where arranged:
            [
                [0_min, _max],
                [1_min, 1_max],
                [2_min, 2_max]
            ]
        where 0, 1, and 2 are the 0th, 1st, and 2nd dimensions,
        respectively.
    """
    z = np.any(mask_image, axis=(1, 2))
    y = np.any(mask_image, axis=(0, 2))
    x = np.any(mask_image, axis=(0, 1))

    z_min, z_max = np.where(z)[0][[0, -1]]
    y_min, y_max = np.where(y)[0][[0, -1]]
    x_min, x_max = np.where(x)[0][[0, -1]]

    return np.array([[z_min, z_max], [y_min, y_max], [x_min, x_max]], dtype=int)


def expand_bounding_box(
        bounding_box: np.ndarray,
        expansion_amount: int,
        image_shape: Optional[Tuple[int]] = None,
) -> np.ndarray:
    """Expand a bounding box bidirectionally along each axis by a specified amount.

    Parameters
    ----------
    bounding_box : np.ndarray
        The bounding box as an array where arranged:
            [
                [0_min, _max],
                [1_min, 1_max],
                [2_min, 2_max]
            ]
        where 0, 1, and 2 are the 0th, 1st, and 2nd dimensions,
        respectively.
    expansion_amount : int
        The number of pixels to expand the bounding box by in each direction
    image_shape : Tuple[int]
        The size of the image along each axis.

    Returns
    -------
    expanded_bounding_box : np.ndarray
        The expanded bounding box.
    """
    expanded_bounding_box = bounding_box.copy()
    expanded_bounding_box[:, 0] = expanded_bounding_box[:, 0] - expansion_amount
    expanded_bounding_box[:, 1] = expanded_bounding_box[:, 1] + expansion_amount

    if image_shape is not None:
        # max index is image_shape - 1
        max_value = np.asarray(image_shape).reshape((len(image_shape), 1)) - 1
    else:
        max_value = None

    return np.clip(expanded_bounding_box, a_min=0, a_max=max_value)
