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
from napari.qt.threading import thread_worker

import sys
import logging
import json
import copick
from copick.impl.filesystem import CopickRootFSSpec
import zarr

from qtpy.QtWidgets import QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget, QComboBox, QPushButton, QLabel
from qtpy.QtCore import Qt
import glob  # For pattern matching of file names

from sklearn.ensemble import RandomForestClassifier

from cellcanvas.semantic.segmentation_manager import (
    SemanticSegmentationManager,
)
from cellcanvas.utils import get_active_button_color

import dask.array as da

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

        self._init_logging()

        # Adding new buttons for "Fit on all" and "Predict for all"
        self.fit_all_button = QPushButton("Fit on all")
        self.fit_all_button.clicked.connect(self.fit_on_all)
        layout.addWidget(self.fit_all_button)

        self.predict_all_button = QPushButton("Predict for all")
        self.predict_all_button.clicked.connect(self.predict_for_all)
        layout.addWidget(self.predict_all_button)
        
        # Dropdowns for each data layer
        self.dropdowns = {}
        self.layer_buttons = {}
        for layer in ["image", "features", "painting", "prediction"]:
            # Make layer button
            button = QPushButton(f"Select {layer.capitalize()} Layer")
            button.clicked.connect(lambda checked, layer=layer: self.activate_layer(layer))
            layout.addWidget(button)
            self.layer_buttons[layer] = button
            # Make layer selection dropdown
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

        # Monkeypatch
        cellcanvas.utils.get_labels_colormap = self.get_copick_colormap

    def get_copick_colormap(self):
        """Return a colormap for distinct label colors based on the pickable objects."""
        colormap = {obj.label: np.array(obj.color)/255.0 for obj in self.root.config.pickable_objects}
        colormap[None] = np.array([1, 1, 1, 1])
        colormap[9] = np.array([0, 1, 1, 1])
        return colormap
    
    def get_voxel_spacing(self):
        return 10
        
    def _init_logging(self):
        self.logger = logging.getLogger("cellcanvas")
        self.logger.setLevel(logging.DEBUG)
        streamHandler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        streamHandler.setFormatter(formatter)
        self.logger.addHandler(streamHandler)
        

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

    def activate_layer(self, layer):
        print(f"Activating layer {layer}")
        if layer == "image":
            layer = self.cell_canvas_app.semantic_segmentor.data_layer
        elif layer == "painting":
            layer = self.cell_canvas_app.semantic_segmentor.painting_layer
        elif layer == "prediction":
            layer = self.cell_canvas_app.semantic_segmentor.prediction_layer
        else:
            return
        layer.visible = True
        layer.editable = True
        self.viewer.layers.selection.active = layer

    def get_complete_data_manager(self, all_pairs=False):
        datasets = []
        for run in self.root.runs:
            run_dir = run.static_path
            voxel_spacing_dir = self.get_default_voxel_spacing_directory(run_dir)
            segmentation_dir = self.get_segmentations_directory(run_dir)

            if not voxel_spacing_dir:
                print(f"No Voxel Spacing directory found for run {run.name}.")
                continue

            os.makedirs(segmentation_dir, exist_ok=True)

            voxel_spacing = self.get_voxel_spacing()
            
            # Reused paths for all datasets in a run
            painting_path = os.path.join(segmentation_dir, f'{voxel_spacing:.3f}_cellcanvas-painting_0_all-multilabel.zarr')
            prediction_path = os.path.join(segmentation_dir, f'{voxel_spacing:.3f}_cellcanvas-prediction_0_all-multilabel.zarr')

            zarr_datasets = glob.glob(os.path.join(voxel_spacing_dir, "*.zarr"))
            image_feature_pairs = {}

            # Locate all images and corresponding features
            for dataset_path in zarr_datasets:
                dataset_name = os.path.basename(dataset_path)
                if dataset_name.endswith(".zarr") and not dataset_name.endswith("_features.zarr"):
                    base_image_name = dataset_name.replace(".zarr", "")
                    # Find corresponding feature files
                    feature_files = [path for path in zarr_datasets if base_image_name in path and "_features.zarr" in path]
                    for feature_path in feature_files:
                        features_base_name = os.path.basename(feature_path).replace("_features.zarr", "")
                        # Check if the image base name matches the start of the feature base name
                        if features_base_name.startswith(base_image_name):
                            image_feature_pairs[features_base_name] = {
                                'image': os.path.join(dataset_path, "0"),  # Assuming highest resolution
                                'features': feature_path
                            }

            # Handle either all pairs or only those specified by the configuration
            config_path = os.path.join(run_dir, "dataset_config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r') as file:
                    config = json.load(file)
                    if 'painting' in config:
                        painting_path = os.path.join(segmentation_dir, config['painting'])
                    if 'prediction' in config:
                        prediction_path = os.path.join(segmentation_dir, config['prediction'])

            if not all_pairs:
                with open(config_path, 'r') as file:
                    config = json.load(file)
                    image_path = os.path.join(voxel_spacing_dir, config['image'])
                    features_path = os.path.join(voxel_spacing_dir, config['features'])
                    if 'painting' in config:
                        painting_path = os.path.join(segmentation_dir, config['painting'])
                    if 'prediction' in config:
                        prediction_path = os.path.join(segmentation_dir, config['prediction'])

                    # Load dataset with specific config paths
                    dataset = DataSet.from_paths(
                        image_path=image_path,
                        features_path=features_path,
                        labels_path=painting_path,
                        segmentation_path=prediction_path,
                        make_missing_datasets=True
                    )
                    datasets.append(dataset)
            else:
                # Load all available pairs
                for base_name, paths in image_feature_pairs.items():
                    dataset = DataSet.from_paths(
                        image_path=paths['image'],
                        features_path=paths['features'],
                        labels_path=painting_path,
                        segmentation_path=prediction_path,
                        make_missing_datasets=True
                    )
                    datasets.append(dataset)

            print(f"Loaded datasets for run {run.name}")

        return DataManager(datasets=datasets)        

    # Only train on config pairs
    # def get_complete_data_manager(self, all_pairs=False):
    #     datasets = []
    #     for run in self.root.runs:
    #         run_dir = run.static_path
    #         config_path = os.path.join(run_dir, "dataset_config.json")

    #         voxel_spacing_dir = self.get_default_voxel_spacing_directory(run_dir)
    #         segmentation_dir = self.get_segmentations_directory(run_dir)

    #         if not voxel_spacing_dir:
    #             print(f"No Voxel Spacing directory found for run {run.name}.")
    #             continue

    #         os.makedirs(segmentation_dir, exist_ok=True)
            
    #         if os.path.exists(config_path):
    #             with open(config_path, 'r') as file:
    #                 config = json.load(file)
    #                 image_path = os.path.join(voxel_spacing_dir, config['image'])
    #                 features_path = os.path.join(voxel_spacing_dir, config['features'])
    #                 painting_path = os.path.join(segmentation_dir, config['painting'])
    #                 prediction_path = os.path.join(segmentation_dir, config['prediction'])
    #         else:
    #             # Existing logic to find paths                
    #             voxel_spacing = self.get_voxel_spacing()

    #             zarr_datasets = glob.glob(os.path.join(voxel_spacing_dir, "*.zarr"))
    #             image_path = None
    #             features_path = None
    #             painting_path = os.path.join(segmentation_dir, f'{voxel_spacing:.3f}_cellcanvas-painting_0_all-multilabel.zarr')
    #             prediction_path = os.path.join(segmentation_dir, f'{voxel_spacing:.3f}_cellcanvas-prediction_0_all-multilabel.zarr')

    #             for dataset_path in zarr_datasets:
    #                 dataset_name = os.path.basename(dataset_path).lower()
    #                 if "_features.zarr" in dataset_name:
    #                     features_path = dataset_path
    #                 elif "painting" in dataset_name:
    #                     painting_path = dataset_path
    #                 elif "prediction" in dataset_name:
    #                     prediction_path = dataset_path
    #                 else:
    #                     # TODO hard coded to use highest resolution
    #                     image_path = os.path.join(dataset_path, "0")

    #             # Save paths to JSON
    #             config = {
    #                 'image': os.path.relpath(image_path, voxel_spacing_dir),
    #                 'features': os.path.relpath(features_path, voxel_spacing_dir),
    #                 'painting': os.path.relpath(painting_path, segmentation_dir),
    #                 'prediction': os.path.relpath(prediction_path, segmentation_dir)
    #             }
    #             with open(config_path, 'w') as file:
    #                 json.dump(config, file)

    #         print(f"Fitting on paths:")
    #         print(f"Image: {image_path}")
    #         print(f"Features: {features_path}")
    #         print(f"Painting: {painting_path}")
    #         print(f"Prediction: {prediction_path}")
                    
    #         # Load dataset with paths
    #         if image_path and features_path:
    #             dataset = DataSet.from_paths(
    #                 image_path=image_path,
    #                 features_path=features_path,
    #                 labels_path=painting_path,
    #                 segmentation_path=prediction_path,
    #                 make_missing_datasets=True
    #             )
    #             datasets.append(dataset)

    #     return DataManager(datasets=datasets)

    def get_default_voxel_spacing_directory(self, static_path):
        # Find VoxelSpacing directories, assuming a hard coded match for now
        voxel_spacing = self.get_voxel_spacing()
        voxel_spacing_dirs = glob.glob(os.path.join(static_path, f'VoxelSpacing{voxel_spacing:.3f}'))
        if voxel_spacing_dirs:
            return voxel_spacing_dirs[0]
        return None

    def get_segmentations_directory(self, static_path):
        segmentation_dir = os.path.join(static_path, "Segmentations")
        return segmentation_dir

    def change_button_color(self, button, color):
        button.setStyleSheet(f"background-color: {color};")

    def reset_button_color(self, button):
        self.change_button_color(button, "")
    
    def fit_on_all(self):
        if not self.cell_canvas_app:
            print("Initialize cell canvas first")
            return
        
        print("Fitting all models to the selected dataset.")

        self.change_button_color(
            self.fit_all_button, get_active_button_color()
        )
        
        self.model_fit_worker = self.threaded_fit_on_all()
        self.model_fit_worker.returned.connect(self.on_model_fit_completed)
        self.model_fit_worker.start()

    @thread_worker
    def threaded_fit_on_all(self):
        # Fit model on all pairs
        data_manager = self.get_complete_data_manager(all_pairs=True)

        clf = RandomForestClassifier(
            n_estimators=50,
            n_jobs=-1,
            max_depth=10,
            max_samples=0.05,
        )
        
        segmentation_manager = SemanticSegmentationManager(
            data=data_manager, model=clf
        )
        segmentation_manager.fit()

        return segmentation_manager        

    def on_model_fit_completed(self, segmentation_manager):
        self.logger.debug("on_model_fit_completed")

        self.cell_canvas_app.semantic_segmentor.segmentation_manager = segmentation_manager

        # Reset color
        self.reset_button_color(self.fit_all_button)
        
    def predict_for_all(self):
        if not self.cell_canvas_app:
            print("Initialize cell canvas first")
            return
        
        print("Fitting all models to the selected dataset.")

        self.change_button_color(
            self.predict_all_button, get_active_button_color()
        )
        
        self.predict_worker = self.threaded_predict_for_all()
        self.predict_worker.returned.connect(self.on_predict_completed)
        self.predict_worker.start()

    def on_predict_completed(self, result):
        self.logger.debug("on_predict_completed")

        # Reset color
        self.reset_button_color(self.predict_all_button)
        
    @thread_worker
    def threaded_predict_for_all(self):
        print("Running predictions on all datasets.")

        # Check if segmentation manager is properly initialized
        if not hasattr(self.cell_canvas_app.semantic_segmentor, 'segmentation_manager') or self.cell_canvas_app.semantic_segmentor.segmentation_manager is None:
            print("Segmentation manager is not initialized.")
            return

        # Retrieve the complete data manager that includes all runs
        data_manager = self.get_complete_data_manager()

        # Iterate through each dataset within the data manager
        for dataset in data_manager.datasets:
            dataset_features = da.asarray(dataset.concatenated_features)
            chunk_shape = dataset_features.chunksize
            shape = dataset_features.shape
            dtype = dataset_features.dtype

            # Iterate over chunks
            for z in range(0, shape[1], chunk_shape[1]):
                for y in range(0, shape[2], chunk_shape[2]):
                    for x in range(0, shape[3], chunk_shape[3]):
                        # Compute the slice for the current chunk
                        # in feature,z,y,x order
                        chunk_slice = (
                            slice(None),
                            slice(z, min(z + chunk_shape[1], shape[1])),
                            slice(y, min(y + chunk_shape[2], shape[2])),
                            slice(x, min(x + chunk_shape[3], shape[3])),                        
                        )
                        print(f"Predicting on chunk {chunk_slice}")

                        # Extract the current chunk
                        chunk = dataset_features[chunk_slice].compute()

                        # Predict on the chunk (adding 1 to each prediction)
                        predicted_chunk = self.cell_canvas_app.semantic_segmentor.segmentation_manager.predict(chunk) + 1

                        # Write the prediction to the corresponding region in the Zarr array
                        dataset.segmentation[chunk_slice[1:]] = predicted_chunk

            print(f"Predictions written")

    def on_run_clicked(self, item, column):
        data = item.data(0, Qt.UserRole)
        if not isinstance(data, copick.impl.filesystem.CopickRunFSSpec):
            self.on_item_clicked(item, column)
            return

        self.selected_run = data
        static_path = self.selected_run.static_path
        self.logger.info(f"Selected {static_path}")

        # Clear existing items
        for dropdown in self.dropdowns.values():
            dropdown.clear()

        # Define directories
        voxel_spacing_dirs = glob.glob(os.path.join(static_path, "VoxelSpacing10*"))
        segmentation_dir = self.get_segmentations_directory(static_path)
        os.makedirs(segmentation_dir, exist_ok=True)

        # Initialize dictionary to hold default selections from config
        default_selections = {}

        # Check for config file and load selections if present
        config_path = os.path.join(static_path, "dataset_config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as file:
                config = json.load(file)
            default_selections = {
                'image': os.path.join(voxel_spacing_dirs[0], config.get('image')),
                'features': os.path.join(voxel_spacing_dirs[0], config.get('features')),
                'painting': os.path.join(segmentation_dir, config.get('painting')),
                'prediction': os.path.join(segmentation_dir, config.get('prediction'))
            }

        # Helper function to add items if not already in dropdown
        def add_item_if_not_exists(dropdown, item_name, item_data):
            if dropdown.findData(item_data) == -1:
                dropdown.addItem(item_name, item_data)

        # Load all zarr datasets from voxel spacing directories
        if voxel_spacing_dirs:
            for voxel_spacing_dir in voxel_spacing_dirs:
                zarr_datasets = glob.glob(os.path.join(voxel_spacing_dir, "*.zarr"))
                for dataset_path in zarr_datasets:
                    dataset_name = os.path.basename(dataset_path)
                    if "_features.zarr" in dataset_name.lower():
                        add_item_if_not_exists(self.dropdowns["features"], dataset_name, dataset_path)
                    else:
                        add_item_if_not_exists(self.dropdowns["image"], dataset_name + "/0", dataset_path + "/0")

        # Load all zarr datasets from segmentation directory
        zarr_datasets = glob.glob(os.path.join(segmentation_dir, "*.zarr"))
        for dataset_path in zarr_datasets:
            dataset_name = os.path.basename(dataset_path)
            if "painting" not in dataset_name.lower():
                add_item_if_not_exists(self.dropdowns["prediction"], dataset_name, dataset_path)
            if "prediction" not in dataset_name.lower():
                add_item_if_not_exists(self.dropdowns["painting"], dataset_name, dataset_path)

        # Set default selections in dropdowns if specified in the config
        for key, dropdown in self.dropdowns.items():
            if default_selections.get(key):
                index = dropdown.findData(default_selections[key])
                if index != -1:
                    dropdown.setCurrentIndex(index)


    def on_item_clicked(self, item, column):
        data = item.data(0, Qt.UserRole)
        if data:
            if isinstance(data, copick.impl.filesystem.CopickPicksFSSpec):
                self.open_picks(data)
            elif isinstance(data, copick.impl.filesystem.CopickTomogramFSSpec):
                self.open_tomogram(data)
            elif isinstance(data, copick.models.CopickSegmentation):
                self.open_labels(data)

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
        pickable_object = [obj for obj in self.root.config.pickable_objects if obj.name == picks.pickable_object_name][0]
        self.viewer.add_points(points_array, name=picks.pickable_object_name, size=25, out_of_slice_display=True, face_color=np.array(pickable_object.color)/255.0)

    def open_tomogram(self, tomogram):
        zarr_store = zarr.open(tomogram.zarr(), mode='r')
        print(f"open_tomogram {tomogram.zarr()}")
        # TODO extract scale/transform info

        # TODO scale is hard coded to 10 here
        self.viewer.add_image(zarr_store[0], name=f"Tomogram: {tomogram.tomo_type}")

    def open_labels(self, tomogram):
        zarr_store = zarr.open(tomogram.zarr(), mode='r')
        print(f"open_labels {tomogram.zarr()}")
        # TODO extract scale/transform info

        # TODO scale is hard coded to 10 here
        self.viewer.add_image(zarr_store[0], name=f"Tomogram: {tomogram.name}")

    def initialize_or_update_cell_canvas(self):
        # Collect paths from dropdowns
        paths = {layer: dropdown.currentText() for layer, dropdown in self.dropdowns.items()}
        
        if not paths["image"] or not paths["features"]:
            print("Please ensure image and feature paths are selected before initializing/updating CellCanvas.")
            return        

        run_dir = self.selected_run.static_path
        segmentation_dir = self.get_segmentations_directory(self.selected_run.static_path)
        voxel_spacing_dir = self.get_default_voxel_spacing_directory(self.selected_run.static_path)

        voxel_spacing = self.get_voxel_spacing()

        # Ensure segmentations directory exists
        os.makedirs(segmentation_dir, exist_ok=True)
        
        default_painting_path = os.path.join(segmentation_dir, f'{voxel_spacing:.3f}_cellcanvas-painting_0_all-multilabel.zarr')
        default_prediction_path = os.path.join(segmentation_dir, f'{voxel_spacing:.3f}_cellcanvas-prediction_0_all-multilabel.zarr')

        painting_path = default_painting_path if not paths["painting"] else os.path.join(segmentation_dir, paths["painting"])
        prediction_path = default_prediction_path if not paths["prediction"] else os.path.join(segmentation_dir, paths["prediction"])
        image_path = os.path.join(voxel_spacing_dir, paths['image'])
        features_path = os.path.join(voxel_spacing_dir, paths["features"])
        
        # TODO note this is hard coded to use the highest resolution of a multiscale zarr
        print(f"Opening paths:")
        print(f"Image: {image_path}")
        print(f"Features: {features_path}")
        print(f"Painting: {painting_path}")
        print(f"Prediction: {prediction_path}")
        try:
            dataset = DataSet.from_paths(
                image_path=image_path,
                features_path=features_path,
                labels_path=painting_path,
                segmentation_path=prediction_path,
                make_missing_datasets=True,
            )
        except FileNotFoundError:
            print(f"File {path} not found!", file=sys.stderr)
            return

        config_path = os.path.join(run_dir, "dataset_config.json")

        config = {
            'image': os.path.relpath(os.path.join(voxel_spacing_dir, f"{paths['image']}"), voxel_spacing_dir),
            'features': os.path.relpath(os.path.join(voxel_spacing_dir, paths["features"]), voxel_spacing_dir),
            'painting': os.path.relpath(painting_path, segmentation_dir),
            'prediction': os.path.relpath(prediction_path, segmentation_dir)
        }

        with open(config_path, 'w') as file:
            json.dump(config, file)
        
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
        self.cell_canvas_app.semantic_segmentor.set_colormap(self.get_copick_colormap())
        self.cell_canvas_app.semantic_segmentor.painting_labels = [obj.label for obj in self.root.config.pickable_objects]
        self.cell_canvas_app.semantic_segmentor.widget.class_labels_mapping = {obj.label: obj.name for obj in self.root.config.pickable_objects}

#        self.cell_canvas_app.semantic_segmentor.widget.class_labels_mapping[9] = 'background'
        self.cell_canvas_app.semantic_segmentor.widget.setupLegend()

if __name__ == "__main__":
    # Project root
    root = CopickRootFSSpec.from_file("/Volumes/kish@CZI.T7/demo_project/copick_config_kyle.json")
    # root = CopickRootFSSpec.from_file("/Volumes/kish@CZI.T7/chlamy_copick/copick_config_kyle.json")

    ## Root API
    root.config # CopickConfig object
    root.runs # List of run objects (lazy loading from filesystem location(s))
        
    viewer = napari.Viewer()

    # Hide layer list and controls
    # viewer.window.qt_viewer.dockLayerList.setVisible(False)
    # viewer.window.qt_viewer.dockLayerControls.setVisible(False)

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
