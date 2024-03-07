from typing import Optional, List

import napari
from magicgui import magicgui
from napari.layers import Surface, Image
import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QWidget, QVBoxLayout, QLabel, QSlider
import trimesh


class QtSurforama(QWidget):
    def __init__(
            self,
            viewer: napari.Viewer,
            surface_layer: Optional[Surface] = None,
            volume_layer: Optional[Image] = None,
            parent: Optional[QWidget] = None
    ):
        super().__init__(parent=parent)
        self.viewer = viewer


        # make the layer selection widget
        self._layer_selection_widget = magicgui(
            self._set_layers,
            surface_layer={"choices": self._get_valid_surface_layers},
            image_layer={"choices": self._get_valid_image_layers},
            call_button="start surfing",
        )

        # add callback to update choices
        self.viewer.layers.events.inserted.connect(self._on_layer_update)
        self.viewer.layers.events.removed.connect(self._on_layer_update)

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self._layer_selection_widget.native)

        label = QLabel("Extend/contract surface")
        self.layout().addWidget(label)
        self.slider = QSlider()
        self.slider.setOrientation(Qt.Horizontal)
        self.slider.setMinimum(-100)
        self.slider.setMaximum(100)
        self.slider.setValue(0)
        self.slider.valueChanged.connect(self.slide_points)
        self.slider.setVisible(False)
        self.layout().addWidget(self.slider)

        # New slider for sampling depth
        label = QLabel("Surface Thickness")
        self.layout().addWidget(label)
        self.sampling_depth_slider = QSlider()
        self.sampling_depth_slider.setOrientation(Qt.Horizontal)
        self.sampling_depth_slider.setMinimum(1)
        self.sampling_depth_slider.setMaximum(100)
        self.sampling_depth_slider.setValue(10)
        self.sampling_depth_slider.valueChanged.connect(self.update_colors_based_on_sampling)
        self.sampling_depth_slider.setVisible(False)
        self.layout().addWidget(self.sampling_depth_slider)

        self.layout().addStretch()

        # set the layers
        self._set_layers(
            surface_layer=surface_layer,
            image_layer=volume_layer

        )

    def _set_layers(
        self,
        surface_layer: Surface,
        image_layer: Image,
    ):
        self.surface_layer = surface_layer
        self.image_layer = image_layer

        if (surface_layer is None) or (image_layer is None):
            return

        # Create a mesh object using trimesh
        self.mesh = trimesh.Trimesh(vertices=surface_layer.data[0], faces=surface_layer.data[1])

        self.vertices = self.mesh.vertices
        self.faces = self.mesh.faces

        # Compute vertex normals
        self.color_values = np.ones((self.mesh.vertices.shape[0],))

        self.surface_layer.data = (self.vertices, self.faces, self.color_values)
        self.surface_layer.refresh()

        self.normals = self.mesh.vertex_normals
        self.volume = image_layer.data

        # make the widgets visible
        self.slider.setVisible(True)
        self.sampling_depth_slider.setVisible(True)

    def _get_valid_surface_layers(self, combo_box) -> List[Surface]:
        return [
            layer
            for layer in self._viewer.layers
            if isinstance(layer, napari.layers.Surface)
        ]

    def _on_layer_update(self, event=None):
        """When the model updates the selected layer, update the relevant widgets."""
        self._layer_selection_widget.reset_choices()

    def _get_valid_image_layers(self, combo_box) -> List[Image]:
        return [
            layer
            for layer in self._viewer.layers
            if isinstance(layer, napari.layers.Image)
        ]

    def get_point_colors(self, points):
        point_indices = points.astype(int)

        point_values = self.volume[point_indices[:, 0], point_indices[:, 1], point_indices[:, 2]]
        normalized_values = (point_values - point_values.min()) / (point_values.max() - point_values.min())

        return normalized_values

    def get_point_set(self):
        return self.vertices

    def slide_points(self, value):
        # Calculate the new positions of points along their normals
        shift = value / 10
        new_positions = self.get_point_set() + (self.normals * shift)
        # Update the points layer with new positions
        new_colors = self.get_point_colors(new_positions)

        self.color_values = new_colors
        self.vertices = new_positions
        self.update_mesh()

    def get_faces(self):
        return self.faces

    def update_mesh(self):
        self.surface_layer.data = (self.get_point_set(), self.get_faces(), self.color_values)

    def update_colors_based_on_sampling(self, value):
        spacing = 0.5
        sampling_depth = value / 10

        # Collect all samples for normalization calculation
        all_samples = []

        # Sample along the normal for each point
        for point, normal in zip(self.get_point_set(), self.normals):
            for depth in range(int(sampling_depth)):
                sample_point = point + normal * spacing * depth
                sample_point_clipped = np.clip(sample_point, [0, 0, 0], np.array(self.volume.shape) - 1).astype(int)
                sample_value = self.volume[sample_point_clipped[0], sample_point_clipped[1], sample_point_clipped[2]]
                all_samples.append(sample_value)

        # Calculate min and max across all sampled values
        samples_min = np.min(all_samples)
        samples_max = np.max(all_samples)

        # Normalize and update colors based on the mean value of samples for each point
        new_colors = np.zeros((len(self.get_point_set()),))
        for i, (point, normal) in enumerate(zip(self.get_point_set(), self.normals)):
            samples = []
            for depth in range(int(sampling_depth)):
                sample_point = point + normal * spacing * depth
                sample_point_clipped = np.clip(sample_point, [0, 0, 0], np.array(self.volume.shape) - 1).astype(int)
                sample_value = self.volume[sample_point_clipped[0], sample_point_clipped[1], sample_point_clipped[2]]
                samples.append(sample_value)

            # Normalize the mean of samples for this point using the min and max from all samples
            mean_value = np.mean(samples)
            normalized_value = (mean_value - samples_min) / (
                        samples_max - samples_min) if samples_max > samples_min else 0
            new_colors[i] = normalized_value

        self.color_values = new_colors
        self.update_mesh()
