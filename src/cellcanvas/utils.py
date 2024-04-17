import numpy as np
from napari.layers.labels._labels_utils import (
    sphere_indices,
)

from qtpy.QtWidgets import (QApplication, QGroupBox, QVBoxLayout, QHBoxLayout,
                            QLabel, QComboBox, QPushButton, QWidget, QCheckBox)
from qtpy.QtCore import Slot, Qt

def get_labels_colormap():
    """Return a colormap for distinct label colors based on:
    Green-Armytage, P., 2010. A colour alphabet and the limits of colour coding. JAIC-Journal of the International Colour Association, 5.
    """
    colormap_22 = {
        None: np.array([1, 1, 1, 1]),  # default is white
        0: np.array([0, 0, 0, 0]),  # alpha
        1: np.array([1, 1, 0, 1]),  # yellow
        2: np.array([0.65, 0, 0.65, 1]),  # purple
        3: np.array([1, 0.65, 0, 1]),  # orange
        4: np.array([0.68, 0.85, 0.9, 1]),  # light blue
        5: np.array([1, 0, 0, 1]),  # red
        6: np.array([1, 0.94, 0.8, 1]),  # buff
        7: np.array([0.5, 0.5, 0.5, 1]),  # grey
        8: np.array([0, 0.5, 0, 1]),  # green
        9: np.array([0.8, 0.6, 0.8, 1]),  # purplish pink
        10: np.array([0, 0, 1, 1]),  # blue
        11: np.array([1, 0.85, 0.7, 1]),  # yellowish pink
        12: np.array([0.54, 0.17, 0.89, 1]),  # violet
        13: np.array([1, 0.85, 0, 1]),  # orange yellow
        14: np.array([0.65, 0.09, 0.28, 1]),  # purplish red
        15: np.array([0.68, 0.8, 0.18, 1]),  # greenish yellow
        16: np.array([0.65, 0.16, 0.16, 1]),  # reddish brown
        17: np.array([0.5, 0.8, 0, 1]),  # yellow green
        18: np.array([0.8, 0.6, 0.2, 1]),  # yellowish brown
        19: np.array([1, 0.27, 0, 1]),  # reddish orange
        20: np.array([0.5, 0.5, 0.2, 1]),  # olive green
    }
    return colormap_22


def paint_maker(logger):
    def paint(self, coord, new_label, refresh=True):
        """Paint over existing labels with a new label, using the selected
        brush shape and size, either only on the visible slice or in all
        n dimensions.

        Parameters
        ----------
        coord : sequence of int
            Position of mouse cursor in image coordinates.
        new_label : int
            Value of the new label to be filled in.
        refresh : bool
            Whether to refresh view slice or not. Set to False to batch paint
            calls.
        """
        shape, dims_to_paint = self._get_shape_and_dims_to_paint()
        paint_scale = np.array(
            [self.scale[i] for i in dims_to_paint], dtype=float
        )

        slice_coord = [int(np.round(c)) for c in coord]
        if self.n_edit_dimensions < self.ndim:
            coord_paint = [coord[i] for i in dims_to_paint]
        else:
            coord_paint = coord

        # Ensure circle doesn't have spurious point
        # on edge by keeping radius as ##.5
        radius = np.floor(self.brush_size / 2) + 0.5
        mask_indices = sphere_indices(radius, tuple(paint_scale))

        mask_indices = mask_indices + np.round(np.array(coord_paint)).astype(
            int
        )

        # logger.info("paint: label = %s, indices = %s", new_label, mask_indices)

        self._paint_indices(
            mask_indices, new_label, shape, dims_to_paint, slice_coord, refresh
        )

    return paint

def get_active_button_color():
    return "#AF8B38"

