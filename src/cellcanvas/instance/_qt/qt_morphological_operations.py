from magicgui import magicgui
import numpy as np
from qtpy.QtWidgets import QWidget, QVBoxLayout, QLabel

from cellcanvas.instance.segment_manager import SegmentManager
from cellcanvas.instance.label_smoothing import (
    dilate_labels_with_crop,
    erode_labels_with_crop,
    closing_labels_with_crop
)


class QtMorphologicalOperations(QWidget):
    def __init__(self, segment_manager: SegmentManager, parent=None):
        super().__init__(parent=parent)
        self.segment_manager = segment_manager

        # make the dilation widget
        self._dilation_widget = magicgui(
            self._dilate_selected_labels,
        )

        # make the erosion widget
        self._erosion_widget = magicgui(self._erode_selected_labels)

        # make the closing widget
        self._closing_widget = magicgui(self._closing_selected_labels)

        # make the layout
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self._dilation_widget.native)
        self.layout().addWidget(self._erosion_widget.native)
        self.layout().addWidget(self._closing_widget.native)

    def _dilate_selected_labels(
        self,
        dilation_radius: int = 1
    ):
        """Erode the selected labels"""
        selected_labels = list(self.segment_manager._selected_labels)
        labels_layer = self.segment_manager.labels_layer
        if (len(selected_labels) == 0) or (labels_layer is None):
            # return early if no labels are selected
            return

        label_image = self.segment_manager.labels_layer.data

        dilated_image = dilate_labels_with_crop(
            label_image=label_image,
            label_values_to_expand=selected_labels,
            overwrite_values=[],
            expansion_amount=dilation_radius,
        )
        labels_layer.data = dilated_image

    def _erode_selected_labels(
        self,
        erosion_radius: int = 1
    ):
        """Erode the selected labels"""
        selected_labels = list(self.segment_manager._selected_labels)
        labels_layer = self.segment_manager.labels_layer
        if (len(selected_labels) == 0) or (labels_layer is None):
            # return early if no labels are selected
            return
        label_image = self.segment_manager.labels_layer.data

        eroded_image = erode_labels_with_crop(
            label_image=label_image,
            label_values_to_erode=selected_labels,
            overwrite_values=[],
            erosion_amount=erosion_radius,
        )
        labels_layer.data = eroded_image

    def _closing_selected_labels(
        self,
        closing_radius: int = 1
    ):
        """Erode the selected labels"""
        selected_labels = list(self.segment_manager._selected_labels)
        labels_layer = self.segment_manager.labels_layer
        if (len(selected_labels) == 0) or (labels_layer is None):
            # return early if no labels are selected
            return
        label_image = self.segment_manager.labels_layer.data

        closed_image = closing_labels_with_crop(
            label_image=label_image,
            label_values_to_close=selected_labels,
            overwrite_values=[],
            operation_radius=closing_radius,
        )
        labels_layer.data = closed_image
