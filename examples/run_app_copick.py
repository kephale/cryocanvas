"""Example of using CellCanvas to pick particles on a surface.

To use:
1. update base_file_path to point to cropped_covid.zarr example file
2. Run the script to launch CellCanvas
3. Paint/predict until you're happy with the result. The seeded labels are:
    - 1: background (including inside the capsules)
    - 2: membrane
    - 3: spike proteins
3b. You might want to switch the image layer into the plane
    depiction before doing the instance segmentation.
    Sometimes I have trouble manipulating the plane after
    the instance segmentation - need to look into this.
4. Once you're happy with the prediction, click the "instance segmentation" tab
5. Set the label value to 2. This will extract the membrane and
    make instances via connected components.
6. Remove the small objects. Suggested threshold: 100
7. Alt + left mouse button to select an instance to modify.
    Once select, you can dilate, erode, etc. to smooth it.
8. With the segment still selected, you can then mesh it
   using the mesh widget. You can play with the smoothing parameters.
9. If the mesh looks good, switch to the "geometry" tab.
    Select the mesh and start surfing!
"""
from collections import defaultdict
import os
import numpy as np
import napari
import cellcanvas
from cellcanvas._app.main_app import CellCanvasApp, QtCellCanvas
from cellcanvas.data.data_manager import DataManager
from cellcanvas.data.data_set import DataSet

import json
import copick
from copick.impl.filesystem import CopickRootFSSpec
import zarr

from qtpy.QtWidgets import QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget, QComboBox, QPushButton, QLabel
from qtpy.QtCore import Qt
import glob  # For pattern matching of file names


# Project root
root = CopickRootFSSpec.from_file("/Volumes/kish@CZI.T7/working/demo_project/copick_config_kyle.json")

## Root API
root.config # CopickConfig object
root.runs # List of run objects (lazy loading from filesystem location(s))

# TODO update to use root.config.pickable_objects


def get_copick_colormap():
    """Return a colormap for distinct label colors based on the pickable objects."""
    colormap = {obj.label: np.array(obj.color)/255.0 for obj in root.config.pickable_objects}
    colormap[None] = np.array([1, 1, 1, 1])
    colormap[9] = np.array([0, 1, 1, 1])
    return colormap

cellcanvas.utils.get_labels_colormap = get_copick_colormap

# Use the function
colormap = get_copick_colormap()

# TODO set names from copick config
# cell_canvas.semantic_segmentor.widget.class_labels_mapping = {obj.label: obj.name for obj in root.config.pickable_objects}

import napari
from qtpy.QtWidgets import QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget
from qtpy.QtCore import Qt

class NapariCopickExplorer(QWidget):
    def __init__(self, viewer: napari.Viewer, root):
        super().__init__()
        self.viewer = viewer
        self.root = root
        self.selected_run = None
        self.cell_canvas_app = None

        layout = QVBoxLayout()
        self.setLayout(layout)

        # Dropdowns for each data layer
        self.dropdowns = {}
        for layer in ["image", "features", "painting", "prediction"]:
            layout.addWidget(QLabel(f"{layer.capitalize()} Path:"))
            self.dropdowns[layer] = QComboBox()
            layout.addWidget(self.dropdowns[layer])

        # Button to update CellCanvas with the selected dataset
        self.update_button = QPushButton("Initialize/Update CellCanvas")
        self.update_button.clicked.connect(self.initialize_or_update_cell_canvas)
        layout.addWidget(self.update_button)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabel("Copick Runs")
        self.tree.itemClicked.connect(self.on_run_clicked)
        layout.addWidget(self.tree)

        self.populate_tree()

    def populate_tree(self):
        for run in self.root.runs:
            run_item = QTreeWidgetItem(self.tree, [run.name])
            run_item.setData(0, Qt.UserRole, run)

            for category in ["segmentations", "meshes", "picks", "voxel_spacings"]:
                category_item = QTreeWidgetItem(run_item, [category])
                items = getattr(run, category)
                for item in items:
                    if category == "picks":
                        item_name = item.pickable_object_name
                    else:
                        item_name = getattr(item, 'name', 'Unnamed')

                    child_item = QTreeWidgetItem(category_item, [item_name])
                    child_item.setData(0, Qt.UserRole, item)

                    # list tomograms
                    if category == "voxel_spacings":
                        for tomogram in item.tomograms:
                            tomo_item = QTreeWidgetItem(child_item, [f"Tomogram: {tomogram.tomo_type}"])
                            tomo_item.setData(0, Qt.UserRole, tomogram)

    def on_run_clicked(self, item, column):
        data = item.data(0, Qt.UserRole)
        if not isinstance(data, copick.impl.filesystem.CopickRunFSSpec):
            self.on_item_clicked(item, column)
            return

        self.selected_run = data
        static_path = self.selected_run.static_path

        # Clear existing items
        for dropdown in self.dropdowns.values():
            dropdown.clear()

        # Find VoxelSpacing directories
        voxel_spacing_dirs = glob.glob(os.path.join(static_path, "VoxelSpacing*"))

        for voxel_spacing_dir in voxel_spacing_dirs:
            # Find all Zarr datasets within the voxel spacing directory
            zarr_datasets = glob.glob(os.path.join(voxel_spacing_dir, "*_features.zarr"))

            for dataset_path in zarr_datasets:
                # Check for the existence of 'embedding' directory within each features zarr
                embedding_path = os.path.join(dataset_path, "*", "embedding")
                embedding_dirs = glob.glob(embedding_path)

                for embedding_dir in embedding_dirs:
                    # Assuming 'embedding' is the desired path for features
                    self.dropdowns["features"].addItem(embedding_dir)

            zarr_datasets = glob.glob(os.path.join(voxel_spacing_dir, "*.zarr"))
            for dataset_path in zarr_datasets:
                # For image paths, directly add non-features zarr datasets to the image dropdown
                if not "features" in os.path.basename(dataset_path).lower():
                    self.dropdowns["image"].addItem(dataset_path)


        # Set defaults for painting and prediction layers, assuming they follow a fixed naming convention
        # and are expected to be located in a specific VoxelSpacing directory, adjusting as necessary
        if voxel_spacing_dirs:  # Check if at least one VoxelSpacing directory was found
            base_voxel_dir = voxel_spacing_dirs[0]  # Assuming to use the first found directory for default paths
            self.dropdowns["painting"].addItem(os.path.join(base_voxel_dir, "painting.zarr"))
            self.dropdowns["prediction"].addItem(os.path.join(base_voxel_dir, "prediction.zarr"))
        else:
            print("No Voxel Spacing directories found. Please check the directory structure.")

                                    
    def on_item_clicked(self, item, column):
        data = item.data(0, Qt.UserRole)
        if data:
            if isinstance(data, copick.impl.filesystem.CopickPicksFSSpec):
                self.open_picks(data)
            elif isinstance(data, copick.impl.filesystem.CopickTomogramFSSpec):
                self.open_tomogram(data)

    def open_picks(self, picks):
        with open(picks.path, 'r') as f:
            points_data = json.load(f)

        # Extracting points locations
        points_locations = [
            [point['location']['z'], point['location']['y'], point['location']['x']]
            for point in points_data['points']
        ]

        # TODO hard coded scaling
        points_array = np.array(points_locations) / 10
        
        # Adding the points layer to the viewer, using the pickable_object_name as the layer name
        pickable_object = [obj for obj in root.config.pickable_objects if obj.name == picks.pickable_object_name][0]
        self.viewer.add_points(points_array, name=picks.pickable_object_name, size=25, out_of_slice_display=True, face_color=np.array(pickable_object.color)/255.0)

    def open_tomogram(self, tomogram):
        zarr_store = zarr.open(tomogram.zarr(), mode='r')
        # TODO extract scale/transform info

        # TODO scale is hard coded to 10 here
        self.viewer.add_image(zarr_store[0], name=f"Tomogram: {tomogram.tomo_type}")

    def initialize_or_update_cell_canvas(self):
        # Collect paths from dropdowns
        paths = {layer: dropdown.currentText() for layer, dropdown in self.dropdowns.items()}

        if not all(paths.values()):
            print("Please ensure all paths are selected before initializing/updating CellCanvas.")
            return

        dataset = DataSet.from_paths(
            image_path=f"{paths['image']}/0",
            features_path=paths["features"],
            labels_path=paths["painting"],
            segmentation_path=paths["prediction"],
            make_missing_datasets=True,
        )

        data_manager = DataManager(datasets=[dataset])
        
        if not self.cell_canvas_app:
            self.cell_canvas_app = CellCanvasApp(data=data_manager, viewer=self.viewer, verbose=True)
            cell_canvas_widget = QtCellCanvas(app=self.cell_canvas_app)
            self.viewer.window.add_dock_widget(cell_canvas_widget)
        else:
            # Update existing CellCanvasApp's data manager
            self.cell_canvas_app.update_data_manager(data_manager)

        # TODO this has multiple copick specific hardcoded hacks
            
        # TODO hardcoded scale factor
        # self.viewer.layers['Image'].scale = (10, 10, 10)

        # Set colormap
        # painting_layer.colormap.color_dict
        #  self.app.painting_labels
        self.cell_canvas_app.semantic_segmentor.set_colormap(get_copick_colormap())
        self.cell_canvas_app.semantic_segmentor.painting_labels = [obj.label for obj in root.config.pickable_objects] + [9]
        self.cell_canvas_app.semantic_segmentor.widget.class_labels_mapping = {obj.label: obj.name for obj in root.config.pickable_objects}

        self.cell_canvas_app.semantic_segmentor.widget.class_labels_mapping[9] = 'background'
        self.cell_canvas_app.semantic_segmentor.widget.setupLegend()

viewer = napari.Viewer()
copick_explorer_widget = NapariCopickExplorer(viewer, root)
viewer.window.add_dock_widget(copick_explorer_widget, name="Copick Explorer", area="left")


# napari.run()

# TODO finish making the prediction computation more lazy
# the strategy should be to start computing labels chunkwise
# on the zarr itself

# TODO check scaling between picks and zarrs

# TODO check why painting doesn't work when using proper scaling

# TODO add proper colormap and legend support
# - override exclusion of non-zero labels
# - consistent colormap in the charts
# - consistent colormap in the painted part of the labels image
