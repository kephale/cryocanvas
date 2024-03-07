from time import sleep
from typing import Optional, List

import numpy as np
from magicgui import magicgui, widgets
import napari
from napari.layers import Labels
from napari.types import LayerDataTuple
from napari.qt import thread_worker
from napari.qt.threading import FunctionWorker
from qtpy.QtWidgets import QWidget, QVBoxLayout, QGroupBox
from skimage.measure import label

from cellcanvas.instance.segment_manager import SegmentManager
from cellcanvas.instance.mesh import binary_mask_to_surface


class QtSegmentManager(QWidget):
    def __init__(
            self,
            viewer: napari.Viewer,
            labels_layer: Optional[Labels]  = None,
            parent:Optional[QWidget] = None
    ):
        super().__init__(parent=parent)
        self._viewer = viewer
        self._manager = SegmentManager(
            labels_layer=labels_layer,
            viewer=self._viewer
        )

        # make the label selection
        self._label_selection_widget = magicgui(
            self._select_labels_layer,
            labels_layer={"choices": self._get_valid_labels_layers},
            call_button="start curating",
        )
        self._label_selection_group = QGroupBox("Select segmentation layer")
        label_selection_layout = QVBoxLayout()
        label_selection_layout.addWidget(self._label_selection_widget.native)
        self._label_selection_group.setLayout(label_selection_layout)

        # make the mesh creation
        self._mesh_conversion_widget = magicgui(
            self._convert_segment_to_mesh,
            pbar={'visible': False, 'max': 0, 'label': 'working...'}
        )
        self._mesh_conversion_group = QGroupBox("Convert instance to mesh")
        mesh_conversion_layout = QVBoxLayout()
        mesh_conversion_layout.addWidget(self._mesh_conversion_widget.native)
        self._mesh_conversion_group.setLayout(mesh_conversion_layout)

        # set the layout
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self._label_selection_group)
        self.layout().addWidget(self._mesh_conversion_group)
        self.layout().addStretch()

    @property
    def labels_layer(self) -> Optional[Labels]:
        return self._manager.labels_layer

    @labels_layer.setter
    def labels_layer(self, labels_layer: Labels):
        self._manager.labels_layer = labels_layer

    def _select_labels_layer(
        self,
        labels_layer: Labels,
    ):
        labels_layer.visible = False

        # make a new labels layer with instances
        instance_labels = label(labels_layer.data)
        instance_labels_layer = self._viewer.add_labels(instance_labels)
        self.labels_layer = instance_labels_layer

    def _get_valid_labels_layers(self, combo_box) -> List[Labels]:
        return [
            layer
            for layer in self._viewer.layers
            if isinstance(layer, napari.layers.Labels)
        ]

    def _on_curating_change(self):
        if self._model.curating is True:
            self._label_expansion_widget.native.setVisible(True)
        else:
            self._label_expansion_widget.native.setVisible(False)

    def _toggle_curating(
        self,
        labels_layer: napari.layers.Labels,
    ):
        self.labels_layer = labels_layer

    def _convert_segment_to_mesh(
            self,
            pbar: widgets.ProgressBar,
            n_mesh_smoothing_iterations: int = 10,
            diffusion_coefficient: float = 0.5,
    ) -> FunctionWorker[LayerDataTuple]:

        @thread_worker(connect={'returned': pbar.hide})
        def convert_to_mesh() -> LayerDataTuple:
            selected_labels = list(self._manager._selected_labels)
            if self.labels_layer is None or (len(selected_labels) == 0):
                return []

            label_image = self.labels_layer.data

            # make a binary mask of the selected labels
            binary_mask = np.zeros_like(label_image, dtype=bool)
            for label_value in selected_labels:
                binary_mask[label_image == label_value] = True
            mesh = binary_mask_to_surface(
                object_mask=binary_mask,
                n_mesh_smoothing_iterations=n_mesh_smoothing_iterations,
                diffusion_coefficient=diffusion_coefficient
            )
            vertices = np.asarray(mesh.vertices)
            faces = np.asarray(mesh.faces)
            values = np.ones((vertices.shape[0]))
            mesh_data = (vertices, faces, values)
            return (mesh_data, {}, "surface")

        # show progress bar and return worker
        pbar.show()
        return convert_to_mesh()


