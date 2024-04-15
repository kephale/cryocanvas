import contextlib
import logging
import sys
from typing import Optional

import joblib
import matplotlib.pyplot as plt
import napari
import numpy as np
import dask.array as da
import toolz as tz
import zarr
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
)
from matplotlib.figure import Figure
from matplotlib.path import Path
from matplotlib.widgets import LassoSelector
from napari.qt.threading import thread_worker
from napari.utils import DirectLabelColormap
from psygnal import debounced
from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QColor, QPainter, QPixmap
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
from superqt import ensure_main_thread

from cellcanvas.data.data_manager import DataManager
from cellcanvas.data.data_set import DataSet
from cellcanvas.semantic.segmentation_manager import (
    SemanticSegmentationManager,
)
from cellcanvas.utils import get_labels_colormap, paint_maker

import xgboost as xgb

ACTIVE_BUTTON_COLOR = "#AF8B38"


class EmbeddingPaintingApp:
    def __init__(
        self,
        data_manager: DataManager,
        viewer: Optional[napari.Viewer] = None,
        extra_logging=False,
    ):

        self.extra_logging = extra_logging
        self.data = data_manager
        self.colormap = get_labels_colormap()
        # clf = RandomForestClassifier(
        #     n_estimators=25,
        #     n_jobs=-1,
        #     max_depth=10,
        #     max_samples=0.05,
        #     max_features='sqrt',
        #     class_weight='balanced'
        # )

        clf = xgb.XGBClassifier(
            objective='multi:softmax',
            num_class=10,  # Specify number of classes if using softmax
            n_estimators=200,
            max_depth=20,
            learning_rate=0.1,
            scale_pos_weight='balanced'  # For handling imbalance
        )
        self.segmentation_manager = SemanticSegmentationManager(
            data=self.data, model=clf
        )

        # get the image and features
        # todo this is temporarily assuming a single dataset
        # need to generalize
        self.image_data = self.data.datasets[0].image
        self.features = self.data.datasets[0].features

        self.corner_pixels = None
        self.model_type = None
        self.prediction_labels = None
        self.prediction_counts = None
        self.painting_labels = None
        self.painting_counts = None
        if viewer is None:
            viewer = napari.Viewer()
        self.viewer = viewer
        self._init_logging()

        # if self.extra_logging:
        #     self.logger.info(f"zarr_path: {zarr_path}")

        self._add_threading_workers()
        self.update_data_manager(self.data)
        self._add_widget()
        self.model = None

        # Initialize plots
        self.start_computing_embedding_plot()
        self.update_class_distribution_charts()

    def set_colormap(self, colormap):
        self.colormap = colormap
        
        self.prediction_layer.colormap = DirectLabelColormap(color_dict=colormap)
        self.painting_layer.colormap = DirectLabelColormap(color_dict=colormap)
        self.update_class_distribution_charts()

    def update_data_manager(self, data: DataManager):
        self.data = data
        self.segmentation_manager.update_data_manager(data)

        # get the image and features
        # todo this is temporarily assuming a single dataset
        # need to generalize
        self.image_data = self.data.datasets[0].image
        self.features = self.data.datasets[0].features

        # TODO remove old layers
        self.viewer.layers.clear()
        self._init_viewer_layers()

    def reshape_features(self, arr):
        return arr.reshape(-1, arr.shape[-1])

    def _add_threading_workers(self):
        # Model fitting worker
        self.model_fit_worker = None
        # Prediction worker
        self.prediction_worker = None
        self.embedding_worker = None

    def _init_viewer_layers(self):
        self.data_layer = self.viewer.add_image(
            self.image_data, name="Image", projection_mode="mean"
        )
        self.data_layer._keep_auto_contrast = True
        self.data_layer.refresh()
        # self.prediction_data = zarr.open(
        #     f"{self.zarr_path}/prediction",
        #     mode="a",
        #     shape=self.image_data.shape,
        #     dtype="i4",
        #     dimension_separator=".",
        # )
        self.prediction_data = self.data.datasets[0].segmentation
        self.prediction_layer = self.viewer.add_labels(
            self.prediction_data,
            name="Prediction",
            scale=self.data_layer.scale,
            opacity=0.1,
            colormap=DirectLabelColormap(color_dict=self.colormap),
        )

        # self.painting_data = zarr.open(
        #     f"{self.zarr_path}/painting",
        #     mode="a",
        #     shape=self.image_data.shape,
        #     dtype="i4",
        #     dimension_separator=".",
        # )
        
        self.painting_data = self.data.datasets[0].labels.astype(int)
        self.painting_layer = self.viewer.add_labels(
            self.painting_data,
            name="Painting",
            scale=self.data_layer.scale,
            colormap=DirectLabelColormap(color_dict=self.colormap),
        )

        # Set up painting logging
        if self.extra_logging:
            logging_paint = paint_maker(self.logger)
            self.painting_layer.paint = logging_paint.__get__(
                self.painting_layer, napari.layers.labels.Labels
            )

        self.painting_labels, self.painting_counts = np.unique(
            self.painting_data[:], return_counts=True
        )

        # Set defaults for layers
        self.get_painting_layer().brush_size = 2
        self.get_painting_layer().n_edit_dimensions = 3

    def _init_logging(self):
        self.logger = logging.getLogger("cellcanvas")
        self.logger.setLevel(logging.DEBUG)
        streamHandler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        streamHandler.setFormatter(formatter)
        self.logger.addHandler(streamHandler)

    def _add_widget(self):
        self.widget = EmbeddingPaintingWidget(self)
        # self.viewer.window.add_dock_widget(self.widget, name="CellCanvas")
        self._connect_events()

    def _connect_events(self):
        # Use a partial function to pass additional arguments to the event handler
        on_data_change_handler = tz.curry(self.on_data_change)(app=self)
        # self.viewer.camera.events, self.viewer.dims.events,
        for listener in [
            self.painting_layer.events.labels_update,
        ]:
            listener.connect(
                debounced(
                    ensure_main_thread(on_data_change_handler),
                    timeout=1000,
                )
            )

        # connect start_computing_embedding_plot to self.create_embedding_plot()

        # TODO other events:
        # - paint layer data change triggers live model fit, class distributions, embeddings
        # - model fit triggers live prediction
        # - prediction triggers class distributions, embeddings

    def get_data_layer(self):
        return self.viewer.layers["Image"]

    def get_prediction_layer(self):
        return self.viewer.layers["Prediction"]

    def get_painting_layer(self):
        return self.viewer.layers["Painting"]

    def on_data_change(self, event, app):
        self.logger.debug("on_data_change")
        # Define corner_pixels based on the current view or other logic
        self.corner_pixels = self.viewer.layers["Image"].corner_pixels

        # TODO check if this is stalling things
        # TODO recheck this after copick
        # self.painting_labels, self.painting_counts = np.unique(
        _, self.painting_counts = np.unique(
            self.painting_data[:], return_counts=True
        )

        # Ensure the prediction layer visual is updated
        self.get_prediction_layer().refresh()

        # Update class distribution charts
        self.update_class_distribution_charts()

        # Update projection
        self.start_computing_embedding_plot()

        self.widget.setupLegend()

    @thread_worker
    def threaded_on_data_change(
        self,
        event,
        corner_pixels,
        dims,
        model_type,
        feature_params,
        live_fit,
        live_prediction,
    ):
        self.logger.info(f"Labels data has changed! {event}")  # noqa: G004

        # Update stats
        # TODO check after copick
        # self.painting_labels, self.painting_counts = np.unique(
        _, self.painting_counts = np.unique(
            self.painting_data[:], return_counts=True
        )

        if live_fit:
            # Pass features and labels along with the model_type
            self.start_model_fit()
        if live_prediction and self.model is not None:
            # For prediction, ensure there's an existing model
            self.start_prediction()

    def get_model_type(self):
        if not self.model_type:
            self.model_type = self.widget.model_dropdown.currentText()
        return self.model_type

    def get_corner_pixels(self):
        if self.corner_pixels is None:
            self.corner_pixels = self.viewer.layers["Image"].corner_pixels
        return self.corner_pixels

    # def compute_features(self):
    #     # TODO: this introduces a slowdown so we should do all this when features are added
    #     features = []
    #
    #     for name, array in self.features.items():
    #         arr = array[:]
    #
    #         features.append(arr)
    #
    #     if features:
    #         return np.concatenate(features, axis=1)

    # def prepare_data_for_model(self):
    #     corner_pixels = self.get_corner_pixels()
    #
    #     # Find a mask of indices we will use for fetching our data
    #     mask_idx = (
    #         slice(
    #             self.viewer.dims.current_step[0],
    #             self.viewer.dims.current_step[0] + 1,
    #         ),
    #         slice(corner_pixels[0, 1], corner_pixels[1, 1]),
    #         slice(corner_pixels[0, 2], corner_pixels[1, 2]),
    #     )
    #
    #     mask_idx = tuple(
    #         [slice(0, sz) for sz in self.get_data_layer().data.shape]
    #     )
    #
    #     self.logger.info(
    #         f"mask idx {mask_idx}, image {self.get_data_layer().data.shape}"
    #     )
    #     active_image = self.get_data_layer().data[mask_idx]
    #     self.logger.info(
    #         f"active image shape {active_image.shape} painting_data {self.painting_data.shape} mask_idx {mask_idx}"
    #     )
    #
    #     training_features = self.compute_features()
    #     training_labels = np.array(self.painting_data)
    #
    #     if (training_labels is None) or np.any(training_labels.shape == 0):
    #         self.logger.info("No training data yet. Skipping model update")
    #         return
    #
    #     return training_features, training_labels

    @thread_worker
    def model_fit_thread(self, model_type):
        return self.update_model(model_type)

    def update_model(self, model_type):
        # Flatten labels
        labels = np.asarray(self.data.datasets[0].labels).flatten()

        # Filter features where labels are greater than 0
        valid_labels = labels > 0
        filtered_labels = labels[valid_labels] - 1  # Adjust labels

        if filtered_labels.size == 0:
            self.logger.info("No labels present. Skipping model update.")
            return None
        
        # Calculate class weights
        unique_labels = np.unique(filtered_labels)
        class_weights = compute_class_weight(
            "balanced", classes=unique_labels, y=filtered_labels
        )
        weight_dict = dict(zip(unique_labels, class_weights))
        self.logger.info(f"Class balance calculated {class_weights}")

        # Apply weights
        # sample_weights = np.vectorize(weight_dict.get)(filtered_labels)

        # Model fitting
        if model_type == "Random Forest":
            clf = RandomForestClassifier(
                n_estimators=50,
                n_jobs=-1,
                max_depth=10,
                max_samples=0.05,
                class_weight=weight_dict,
            )
            self.segmentation_manager.model = clf
            # self.segmentation_manager.fit()
            self.logger.info(f"Starting model fitting")
            self.segmentation_manager.fit()
            return self.segmentation_manager.model
        elif model_type == "XGBoost":
            raise NotImplementedError("XGBoost temporarily disabled")
            # clf = xgb.XGBClassifier(
            #     n_estimators=100, learning_rate=0.1, use_label_encoder=False
            # )
            # clf.fit(
            #     filtered_features,
            #     filtered_labels,
            #     sample_weight=sample_weights,
            # )
            # return clf
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def predict(self):
        dataset_features = da.asarray(self.data.datasets[0].concatenated_features)
        chunk_shape = dataset_features.chunksize
        shape = dataset_features.shape
        dtype = dataset_features.dtype
        
        # Placeholder for aggregated labels and counts
        all_labels = []
        all_counts = []
        
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
                    predicted_chunk = self.segmentation_manager.predict(chunk) + 1
                    
                    # Write the prediction to the corresponding region in the Zarr array
                    self.prediction_data[chunk_slice[1:]] = predicted_chunk
                    
                    # Aggregate labels and counts
                    labels, counts = np.unique(predicted_chunk, return_counts=True)
                    all_labels.append(labels)
                    all_counts.append(counts)

        # Combine all_labels and all_counts
        unique_labels, inverse = np.unique(np.concatenate(all_labels), return_inverse=True)
        total_counts = np.bincount(inverse, weights=np.concatenate(all_counts))

        # Now, self.prediction_data should contain the predicted labels
        return self.prediction_data, unique_labels, total_counts

    @thread_worker
    def prediction_thread(self):
        # The prediction logic
        return self.predict()

    def get_features(self):
        features, _ = self.data.get_training_data()
        return features

    def start_prediction(self):
        if self.prediction_worker is not None:
            self.prediction_worker.quit()

        if self.extra_logging:
            self.logger.info("started prediction")

        # Change button color with status
        self.widget.change_button_color(
            self.widget.live_pred_button, ACTIVE_BUTTON_COLOR
        )

        # features = self.get_features()

        # TODO use a yielded connect worker
        
        self.prediction_worker = self.prediction_thread()
        self.prediction_worker.returned.connect(self.on_prediction_completed)
        self.prediction_worker.start()

    def on_prediction_completed(self, result):
        prediction, prediction_labels, prediction_counts = result
        self.logger.debug("on_prediction_completed")
        self.prediction_data = prediction

        self.prediction_labels = prediction_labels
        self.prediction_counts = prediction_counts

        self.get_prediction_layer().refresh()

        self.update_class_distribution_charts()

        # Reset color
        self.widget.reset_button_color(self.widget.live_pred_button)

        # self.create_embedding_plot()

    def start_model_fit(self):
        # Change button color with status
        self.widget.change_button_color(
            self.widget.live_fit_button, ACTIVE_BUTTON_COLOR
        )

        if self.model_fit_worker is not None:
            self.model_fit_worker.quit()

        if self.extra_logging:
            self.logger.info("started model fit")

        # features, labels = self.prepare_data_for_model()

        self.model_fit_worker = self.model_fit_thread(self.get_model_type())
        self.model_fit_worker.returned.connect(self.on_model_fit_completed)
        # TODO update UI to indicate that model training has started
        self.model_fit_worker.start()

    def on_model_fit_completed(self, model):
        self.logger.debug("on_model_fit_completed")
        self.model = model

        # Reset color
        self.widget.reset_button_color(self.widget.live_fit_button)

        if (
            self.widget.live_pred_checkbox.isChecked()
            and self.model is not None
        ):
            self.logger.debug(
                "live prediction is active, prediction triggered by model fit completion."
            )
            self.start_prediction()

    def update_class_distribution_charts(self):
        total_pixels = (
            np.product(self.painting_data.shape)
            if self.painting_data is not None
            else 1
        )

        # Initialize counts for all labels in painting_labels with zero
        if self.painting_labels is not None:
            unique_labels = np.unique(self.painting_labels)
            painting_counts_dict = {label: 0 for label in unique_labels}
        else:
            unique_labels = np.array([0])
            painting_counts_dict = {0: 0}

        # Update counts from existing painting_counts if available
        if self.painting_counts is not None and self.painting_labels is not None:
            for label, count in zip(self.painting_labels, self.painting_counts):
                painting_counts_dict[label] = count

        # Create arrays from the dictionary
        painting_labels = np.array(list(painting_counts_dict.keys()))
        painting_counts = np.array(list(painting_counts_dict.values()))

        prediction_counts = (
            self.prediction_counts
            if self.prediction_counts is not None
            else np.array([0])
        )
        prediction_labels = (
            self.prediction_labels
            if self.prediction_labels is not None
            else np.array([0])
        )

        if self.extra_logging:
            self.logger.info(
                f"update_class_distribution_charts: painting_counts = {painting_counts}, painting_labels = {painting_labels}, prediction_counts = {prediction_counts}, prediction_labels = {prediction_labels}"  # noqa G004
            )
            self.logger.info(
                f"painting layer: opacity = {self.get_painting_layer().opacity}, brush_size = {self.get_painting_layer().brush_size}, num edit dimensions = {self.get_painting_layer().n_edit_dimensions}"  # noqa G004
            )
            self.logger.info(
                f"prediction layer: opacity = {self.get_prediction_layer().opacity}, brush_size = {self.get_prediction_layer().brush_size}, num edit dimensions = {self.get_prediction_layer().n_edit_dimensions}"  # noqa G004
            )
            self.logger.info(
                f"image layer: contrast_limits = {self.viewer.layers['Image'].contrast_limits}, opacity = {self.viewer.layers['Image'].opacity}, gamma = {self.viewer.layers['Image'].gamma}"  # noqa G004
            )
            self.logger.info(
                f"Current model type: {self.widget.model_dropdown.currentText()}"  # noqa G004
            )

        # Calculate percentages instead of raw counts
        painting_percentages = (painting_counts / total_pixels) * 100
        prediction_percentages = (prediction_counts / total_pixels) * 100

        # Handle cases where there are no valid painting or prediction labels yet
        if not np.any(painting_labels > 0) and not np.any(
            prediction_labels > 0
        ):
            painting_percentages = np.array([0])
            prediction_percentages = np.array([0])
            valid_painting_labels = np.array([0])
            valid_prediction_labels = np.array([0])
            valid_painting_percentages = np.array([0])
            valid_prediction_percentages = np.array([0])

            unpainted_percentage = 0
        else:
            # Exclude class 0 for both painting and prediction layers
            valid_painting_indices = painting_labels > 0
            valid_painting_labels = painting_labels[valid_painting_indices]
            valid_painting_percentages = painting_percentages[
                valid_painting_indices
            ]

            valid_prediction_indices = prediction_labels > 0
            valid_prediction_labels = prediction_labels[
                valid_prediction_indices
            ]
            valid_prediction_percentages = prediction_percentages[
                valid_prediction_indices
            ]

            unpainted_percentage = (
                painting_percentages[painting_labels == 0]
                if 0 in painting_labels
                else [0]
            )

        # Example class to color mapping
        class_color_mapping = {
            label: f"#{int(rgba[0] * 255):02x}{int(rgba[1] * 255):02x}{int(rgba[2] * 255):02x}"
            for label, rgba in self.colormap.items()
        }

        self.widget.figure.clear()

        napari_charcoal_hex = "#262930"

        # Custom style adjustments for dark theme
        dark_background_style = {
            "figure.facecolor": napari_charcoal_hex,
            "axes.facecolor": napari_charcoal_hex,
            "axes.edgecolor": "white",
            "axes.labelcolor": "white",
            "text.color": "white",
            "xtick.color": "white",
            "ytick.color": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }

        with plt.style.context(dark_background_style):
            # Create subplots with adjusted heights
            gs = self.widget.figure.add_gridspec(3, 1, height_ratios=[1, 4, 4])
            ax0 = self.widget.figure.add_subplot(gs[0])
            ax1 = self.widget.figure.add_subplot(gs[1])
            ax2 = self.widget.figure.add_subplot(gs[2])

            # Plot for unpainted pixels
            ax0.barh(
                0, unpainted_percentage, color="#AAAAAA", edgecolor="white"
            )
            # ax0.set_title("Unpainted", loc='left')
            ax0.set_xlabel("% of Image")
            ax0.set_yticks([])

            # Ensure to handle the case where valid_painting_labels or valid_prediction_labels are empty or only contain zeros
            if len(valid_painting_labels) == 0:
                valid_painting_labels = np.array([0])  # Default label
                valid_painting_percentages = np.array(
                    [0]
                )  # Default percentage

            # Horizontal bar plots for painting and prediction layers
            ax1.barh(
                valid_painting_labels,
                valid_painting_percentages,
                color=[
                    class_color_mapping.get(x, "#FFFFFF")
                    for x in valid_painting_labels
                ],
                edgecolor="white",
            )
            # ax1.set_title("Painting", loc='left')

            ax1.set_xlabel("% of Image")
            # ax1.set_yticks(valid_painting_labels)
            ax1.set_yticks([])
            ax1.invert_yaxis()  # Invert y-axis to have labels in ascending order from top to bottom

            if len(valid_prediction_labels) == 0:
                valid_prediction_labels = np.array([0])  # Default label
                valid_prediction_percentages = np.array(
                    [0]
                )  # Default percentage

            ax2.barh(
                valid_prediction_labels,
                valid_prediction_percentages,
                color=[
                    class_color_mapping.get(x, "#FFFFFF")
                    for x in valid_prediction_labels
                ],
                edgecolor="white",
            )
            # ax2.set_title("Prediction", loc='left')
            ax2.set_xlabel("% of Image")
            # ax2.set_yticks(valid_prediction_labels)
            ax2.set_yticks([])
            ax2.invert_yaxis()

            # Use set_ylabel to position the titles outside and to the left of the y-axis labels
            ax0.set_ylabel(
                "Unpainted",
                labelpad=20,
                fontsize=12,
                rotation=0,
                ha="right",
                va="center",
            )
            ax1.set_ylabel(
                "Painting",
                labelpad=20,
                fontsize=12,
                rotation=0,
                ha="right",
                va="center",
            )
            ax2.set_ylabel(
                "Prediction",
                labelpad=20,
                fontsize=12,
                rotation=0,
                ha="right",
                va="center",
            )

            self.widget.figure.subplots_adjust(
                left=0.33, right=0.9, top=0.95, bottom=0.05
            )

        # Adjust the left margin to make space for the y-axis labels (titles)
        plt.subplots_adjust(left=0.25)

        # Automatically adjust subplot params so that the subplot(s) fits into the figure area
        self.widget.figure.tight_layout(pad=3.0)

        # Explicitly set figure background color again to ensure it
        self.widget.figure.patch.set_facecolor(napari_charcoal_hex)

        self.widget.canvas.draw()

    @thread_worker
    def compute_embedding_projection(self):
        # Filter out entries where the label is 0
        filtered_features, filtered_labels = self.data.get_training_data()
        filtered_features = filtered_features.compute()
        filtered_labels = filtered_labels.compute()

        # label values are offset by 1 for training,
        # undo the offset.
        filtered_labels += 1

        # Check if there are enough samples to proceed
        if filtered_features.shape[0] < 2:
            return (
                None,
                None,
                "Not enough labeled data to create an embedding plot. Need at least 2 samples.",
            )

        # Proceed with PLSRegression as there's enough data
        pls = PLSRegression(n_components=2)
        pls_embedding = pls.fit_transform(filtered_features, filtered_labels)[
            0
        ]

        self.pls = pls

        return pls_embedding, filtered_labels, None

    def start_computing_embedding_plot(self):
        if self.embedding_worker is not None:
            self.embedding_worker.quit()

        if self.extra_logging:
            self.logger.info("start computing embedding plot")

        self.embedding_worker = self.compute_embedding_projection()
        self.embedding_worker.returned.connect(self.create_embedding_plot)
        # TODO update UI to indicate that model training has started
        self.embedding_worker.start()

    def create_embedding_plot(self, result):
        pls_embedding, filtered_labels, error_message = result
        if error_message:
            print(error_message)
            return

        self.logger.info("done computing embedding plot")

        self.widget.embedding_figure.clear()

        self.pls_embedding = pls_embedding
        labels = self.painting_data[:].flatten()

        # Original image coordinates
        z_dim, y_dim, x_dim = self.image_data.shape
        X, Y, Z = np.meshgrid(
            np.arange(x_dim), np.arange(y_dim), np.arange(z_dim), indexing="ij"
        )
        original_coords = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
        # Filter coordinates using the same mask applied to the features
        self.filtered_coords = original_coords[labels > 0]

        class_color_mapping = {
            label: "#{:02x}{:02x}{:02x}".format(
                int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)
            )
            for label, rgba in self.colormap.items()
        }

        # Convert filtered_labels to a list of colors for each point
        point_colors = [
            class_color_mapping[label] for label in filtered_labels
        ]

        # Custom style adjustments for dark theme
        napari_charcoal_hex = "#262930"
        plt.style.use("dark_background")
        self.widget.embedding_figure.patch.set_facecolor(napari_charcoal_hex)

        ax = self.widget.embedding_figure.add_subplot(
            111, facecolor=napari_charcoal_hex
        )
        _ = ax.scatter(
            self.pls_embedding[:, 0],
            self.pls_embedding[:, 1],
            s=0.1,
            c=point_colors,
            alpha=1.0,
        )

        # Hide the axes ticks
        ax.set_xticks([])
        ax.set_yticks([])

        # Set spines to be visible to create a border effect
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("white")  # Set the color of the border here

        plt.title(
            "PLS-DA Embedding Using Labels from Painting Layer", color="white"
        )

        def onclick(event):
            if event.inaxes == ax:
                clicked_embedding = np.array([event.xdata, event.ydata])
                distances = np.sqrt(
                    np.sum(
                        (self.pls_embedding - clicked_embedding) ** 2, axis=1
                    )
                )
                nearest_point_index = np.argmin(distances)
                nearest_image_coordinates = self.filtered_coords[
                    nearest_point_index
                ]
                print(
                    f"Clicked embedding coordinates: ({event.xdata}, {event.ydata}), Image space coordinate: {nearest_image_coordinates}"
                )

        def onselect(verts):
            path = Path(verts)
            self.update_painting_layer(path)

        # Create the LassoSelector
        self.lasso = LassoSelector(ax, onselect, useblit=True)

        _ = self.widget.embedding_canvas.mpl_connect(
            "button_press_event", onclick
        )
        self.widget.embedding_canvas.draw()

    def update_painting_layer(self, path):
        # Fetch the currently active label from the painting layer
        target_label = self.get_painting_layer().selected_label
        # Start a new thread to update the painting layer with the current target label

        self.logger.info("start update painting layer")
        self.widget.change_embedding_label_color(ACTIVE_BUTTON_COLOR)

        self.painting_worker = self.paint_thread(path, target_label)
        self.painting_worker.returned.connect(self.on_embedding_paint_complete)
        self.painting_worker.start()

    def on_embedding_paint_complete(self):
        self.widget.change_embedding_label_color("#262930")

    @thread_worker
    def paint_thread(self, lasso_path, target_label):
        # Ensure we're working with the full feature dataset
        all_features_flat = self.get_features()

        # Use the PLS model to project these features into the embedding space
        all_embeddings = self.pls.transform(all_features_flat)

        # Determine which points fall within the lasso path
        contained = np.array(
            [
                lasso_path.contains_point(point)
                for point in all_embeddings[:, :2]
            ]
        )

        # The shape of the original image data, to map flat indices back to spatial coordinates
        shape = self.image_data.shape

        # Iterate over all points to update the painting data where contained is True
        paint_indices = np.where(contained)[0]
        for idx in paint_indices:
            # Map flat index back to spatial coordinates
            z, y, x = np.unravel_index(idx, shape)
            # Update the painting data
            self.painting_data[z, y, x] = target_label

        if self.extra_logging:
            self.logger.info(
                f"lasso paint: label = {target_label}, indices = {paint_indices}"  # noqa G004
            )

        # print(f"Painted {np.sum(contained)} pixels with label {target_label}")

class ClickableLabel(QLabel):
    clicked = Signal(int)  # Emits the label ID

    def __init__(self, label_id, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_id = label_id

    def mousePressEvent(self, event):
        self.clicked.emit(self.label_id)        

class EmbeddingPaintingWidget(QWidget):
    def __init__(self, app, parent=None):
        super().__init__(parent=parent)
        self.app = app
        self.label_edits = {}
        self.initUI()

    def initUI(self):
        main_layout = QVBoxLayout()

        self.legend_placeholder_index = 0

        # Settings Group
        settings_group = QGroupBox("Settings")
        settings_layout = QVBoxLayout()

        model_layout = QHBoxLayout()
        model_label = QLabel("Select Model")
        self.model_dropdown = QComboBox()
        self.model_dropdown.addItems(["Random Forest", "XGBoost"])
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_dropdown)
        settings_layout.addLayout(model_layout)

        self.add_features_button = QPushButton("Add Features")
        self.add_features_button.clicked.connect(self.add_features)
        settings_layout.addWidget(self.add_features_button)

        thickness_layout = QHBoxLayout()
        thickness_label = QLabel("Adjust Slice Thickness")
        self.thickness_slider = QSlider(Qt.Horizontal)
        self.thickness_slider.setMinimum(0)
        self.thickness_slider.setMaximum(50)
        self.thickness_slider.valueChanged.connect(self.on_thickness_changed)
        self.thickness_slider.setValue(10)
        thickness_layout.addWidget(thickness_label)
        thickness_layout.addWidget(self.thickness_slider)
        settings_layout.addLayout(thickness_layout)

        # Update layer contrast limits after thick slices has effect
        self.app.viewer.layers["Image"].reset_contrast_limits()

        settings_group.setLayout(settings_layout)
        main_layout.addWidget(settings_group)

        # Controls Group
        controls_group = QGroupBox("Controls")
        controls_layout = QVBoxLayout()

        # Live Model Fitting
        live_fit_layout = QHBoxLayout()
        self.live_fit_checkbox = QCheckBox("Live Model Fitting")
        self.live_fit_checkbox.setChecked(False)
        self.live_fit_button = QPushButton("Fit Model Now")
        live_fit_layout.addWidget(self.live_fit_checkbox)
        live_fit_layout.addWidget(self.live_fit_button)
        controls_layout.addLayout(live_fit_layout)

        # Live Prediction
        live_pred_layout = QHBoxLayout()
        self.live_pred_checkbox = QCheckBox("Live Prediction")
        self.live_pred_checkbox.setChecked(False)
        self.live_pred_button = QPushButton("Predict Now")
        live_pred_layout.addWidget(self.live_pred_checkbox)
        live_pred_layout.addWidget(self.live_pred_button)
        controls_layout.addLayout(live_pred_layout)

        # Export model
        self.export_model_button = QPushButton("Export Model")
        controls_layout.addWidget(self.export_model_button)
        self.export_model_button.clicked.connect(self.export_model)

        # Import model
        self.import_model_button = QPushButton("Import Model")
        controls_layout.addWidget(self.import_model_button)
        self.import_model_button.clicked.connect(self.import_model)        

        controls_group.setLayout(controls_layout)
        main_layout.addWidget(controls_group)

        # Stats Summary Group
        stats_summary_group = QGroupBox("Stats Summary")
        self.stats_summary_layout = QVBoxLayout()

        self.stats_summary_layout.insertStretch(self.legend_placeholder_index)

        self.setupLegend()
        # Connect legend updates
        with contextlib.suppress(AttributeError):
            self.app.painting_layer.events.selected_label.connect(
                self.updateLegendHighlighting
            )

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.stats_summary_layout.addWidget(self.canvas)

        self.embedding_label = QLabel(
            "Painting Embedding (Draw to label in embedding)"
        )
        self.stats_summary_layout.addWidget(self.embedding_label)

        self.embedding_figure = Figure()
        self.embedding_canvas = FigureCanvas(self.embedding_figure)
        self.stats_summary_layout.addWidget(self.embedding_canvas)

        stats_summary_group.setLayout(self.stats_summary_layout)
        main_layout.addWidget(stats_summary_group)

        self.setLayout(main_layout)

        # Connect checkbox signals to actions
        self.live_fit_checkbox.stateChanged.connect(self.on_live_fit_changed)
        self.live_pred_checkbox.stateChanged.connect(self.on_live_pred_changed)

        # Connect button clicks to actions
        self.live_fit_button.clicked.connect(self.app.start_model_fit)
        self.live_pred_button.clicked.connect(self.app.start_prediction)

    def add_features(self):
        zarr_path = QFileDialog.getExistingDirectory(self, "Select Directory")

        if zarr_path:
            try:
                new_features = zarr.open_array(zarr_path, mode="r")

                self.app.features[zarr_path] = self.app.reshape_features(
                    new_features[:]
                )
            except FileNotFoundError as e:

                print(f"Error loading features from zarr array: {e}")

    def export_model(self):
        model = self.app.model
        if model is not None:
            filePath, _ = QFileDialog.getSaveFileName(
                self, "Save Model", "", "Joblib Files (*.joblib)"
            )
            if filePath:
                joblib.dump(model, filePath)
                QMessageBox.information(
                    self, "Model Export", "Model exported successfully!"
                )
                print(f"Wrote model file to: {filePath}")
        else:
            QMessageBox.warning(
                self, "Model Export", "No model available to export."
            )

    def import_model(self):
        filePath, _ = QFileDialog.getOpenFileName(
            self, "Open Model", "", "Joblib Files (*.joblib)"
        )
        if filePath:
            try:
                model = joblib.load(filePath)
                self.app.model = model
                QMessageBox.information(
                    self, "Model Import", "Model imported successfully!"
                )
                print(f"Loaded model file from: {filePath}")
            except Exception as e:
                QMessageBox.warning(
                    self, "Model Import", f"Failed to import model. Error: {str(e)}"
                )
            
    def change_embedding_label_color(self, color):
        """Change the background color of the embedding label."""
        self.embedding_label.setStyleSheet(f"background-color: {color};")

    def change_button_color(self, button, color):
        button.setStyleSheet(f"background-color: {color};")

    def reset_button_color(self, button):
        self.change_button_color(button, "")

    def on_live_fit_changed(self, state):
        if state == Qt.Checked:
            self.app.start_model_fit()

    def on_live_pred_changed(self, state):
        if state == Qt.Checked:
            # TODO might need to check if this is safe to do, e.g. if a model exists
            self.app.start_prediction()

    def fit_model_now(self):
        self.app.start_model_fit()

    def predict_now(self):
        self.app.start_prediction()

    def setupLegend(self):
        if not hasattr(self, "class_labels_mapping"):
            # Initialize class labels
            self.class_labels_mapping = {}

        if hasattr(self, "legend_group"):
            self.stats_summary_layout.takeAt(
                self.legend_placeholder_index
            ).widget().deleteLater()

        painting_layer = self.app.get_painting_layer()
        self.legend_layout = QVBoxLayout()
        self.legend_group = QGroupBox("Class Labels Legend")
        # Track label edits
        self.label_edits = {}

        active_labels = self.app.painting_labels

        if active_labels is not None:
            for label_id in active_labels:
                color = painting_layer.colormap.color_dict[label_id]

                # Create a QLabel for color swatch
                color_swatch = ClickableLabel(label_id)
                pixmap = QPixmap(16, 16)

                if color is None:
                    pixmap = self.createCheckerboardPattern()
                else:
                    pixmap.fill(QColor(*[int(c * 255) for c in color]))

                color_swatch.setPixmap(pixmap)
                color_swatch.clicked.connect(self.activateLabel)

                # Update the mapping with new classes or use the existing name
                if label_id not in self.class_labels_mapping:
                    self.class_labels_mapping[label_id] = (
                        f"Class {label_id if label_id is not None else 0}"
                    )

                # Use the name from the mapping
                label_name = self.class_labels_mapping[label_id]
                label_edit = QLineEdit(label_name)

                # Highlight the label if it is currently being used
                if label_id == painting_layer.selected_label:
                    self.highlightLabel(label_edit)

                # Save changes to class labels back to the mapping
                label_edit.textChanged.connect(
                    lambda text, l_id=label_id: self.updateClassLabelName(
                        l_id, text
                    )
                )

                # Layout for each legend entry
                entry_layout = QHBoxLayout()
                entry_layout.addWidget(color_swatch)
                entry_layout.addWidget(label_edit)
                self.legend_layout.addLayout(entry_layout)
                self.label_edits[label_id] = label_edit

        self.legend_group.setLayout(self.legend_layout)
        self.stats_summary_layout.insertWidget(
            self.legend_placeholder_index, self.legend_group
        )

    def activateLabel(self, label_id):
        painting_layer = self.app.get_painting_layer()
        painting_layer.selected_label = label_id
        
    def updateLegendHighlighting(self, selected_label_event):
        """Update highlighting of legend"""
        current_label_id = selected_label_event.source._selected_label

        for label_id, label_edit in self.label_edits.items():
            if label_id == current_label_id:
                self.highlightLabel(label_edit)
            else:
                self.removeHighlightLabel(label_edit)

    def highlightLabel(self, label_edit):
        label_edit.setStyleSheet("QLineEdit { background-color: #3D6A88; }")

    def removeHighlightLabel(self, label_edit):
        label_edit.setStyleSheet("")

    def updateClassLabelName(self, label_id, name):
        self.class_labels_mapping[label_id] = name

    def createCheckerboardPattern(self):
        """Creates a QPixmap with a checkerboard pattern."""
        pixmap = QPixmap(16, 16)
        pixmap.fill(Qt.white)
        painter = QPainter(pixmap)
        painter.setPen(Qt.NoPen)

        # Define the colors for the checkerboard squares
        color1 = Qt.lightGray
        color2 = Qt.darkGray
        size = 4

        for x in range(0, pixmap.width(), size):
            for y in range(0, pixmap.height(), size):
                if (x + y) // size % 2 == 0:
                    painter.fillRect(x, y, size, size, color1)
                else:
                    painter.fillRect(x, y, size, size, color2)

        painter.end()
        return pixmap

    def on_thickness_changed(self, value):
        self.app.viewer.dims.thickness = (value,) * self.app.viewer.dims.ndim
        self.app.logger.info(
            f"Thickness changes: {self.app.viewer.dims.thickness}"  # noqa G004
        )


if __name__ == "__main__":
    # zarr_path = "/Users/kharrington/Data/CryoCanvas/cryocanvas_crop_007.zarr"
    # zarr_path = "/Users/kharrington/Data/CryoCanvas/cryocanvas_crop_007_v2.zarr/cryocanvas_crop_007.zarr"
    # zarr_path = "/Users/kharrington/Data/cellcanvas/cellcanvas_crop_007.zarr/"
    # zarr_path = "/Users/kharrington/Data/cellcanvas/cellcanvas_crop_009.zarr/"
    # zarr_path = "/Users/kharrington/Data/cellcanvas/cellcanvas_crop_010.zarr/"
    # zarr_path = "/Users/kharrington/Data/cellcanvas/cryocanvas_crop_008.zarr/"
    # zarr_path = "/Users/kharrington/Data/cellcanvas/cryocanvas_crop_011.zarr/"
    # zarr_path = "/Users/kyamauch/Documents/cellcanvas/examples/embedding_seg/cropped_covid.zarr"

    base_file_path = (
        "/Users/kyamauch/Documents/cellcanvas/examples/seg/cropped_covid.zarr"
    )

    dataset = DataSet.from_paths(
        image_path=f"{base_file_path}/image/cropped_covid",
        features_path=f"{base_file_path}/embedding/swin_unetr",
        labels_path=f"{base_file_path}/painting",
        segmentation_path=f"{base_file_path}/prediction",
        make_missing_datasets=True,
    )

    data_manager = DataManager(datasets=[dataset])
    app = EmbeddingPaintingApp(data_manager=data_manager, extra_logging=True)

    app.viewer.window.add_dock_widget(app.widget, name="CellCanvas")
    napari.run()
