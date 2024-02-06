import napari
import zarr
import numpy as np
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from functools import partial
from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QComboBox,
    QHBoxLayout,
    QCheckBox,
    QGroupBox,
    QPushButton,
)
from skimage.feature import multiscale_basic_features
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
from skimage import future
import toolz as tz
from psygnal import debounced
from superqt import ensure_main_thread
import logging
import sys
import xgboost as xgb
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
)
from matplotlib.figure import Figure
from matplotlib.colors import to_rgba

# from https://github.com/napari/napari/issues/4384


# Define a class to encapsulate the Napari viewer and related functionalities
class CryoCanvasApp:
    def __init__(self, zarr_path):
        self.zarr_path = zarr_path
        self.dataset = zarr.open(zarr_path, mode="r")
        self.image_data = self.dataset["crop/original_data"]
        self.feature_data_skimage = self.dataset["features/skimage"]
        self.feature_data_tomotwin = self.dataset["features/tomotwin"]
        self.viewer = napari.Viewer()
        self._init_viewer_layers()
        self._init_logging()
        self._add_widget()
        self.model = None
        self.executor = ThreadPoolExecutor(max_workers=1)

    def get_labels_colormap(self):
        """Return a colormap for distinct label colors based on:
        Green-Armytage, P., 2010. A colour alphabet and the limits of colour coding. JAIC-Journal of the International Colour Association, 5.
        """
        colormap_22 = {
            0: np.array([0, 0, 0, 0]),  # alpha
            1: np.array([1, 1, 0, 1]),  # yellow
            2: np.array([0.5, 0, 0.5, 1]),  # purple
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

    def _init_viewer_layers(self):
        self.data_layer = self.viewer.add_image(self.image_data, name="Image")
        self.prediction_data = zarr.open(
            f"{self.zarr_path}/prediction",
            mode="a",
            shape=self.image_data.shape,
            dtype="i4",
            dimension_separator=".",
        )
        self.prediction_layer = self.viewer.add_labels(
            self.prediction_data,
            name="Prediction",
            scale=self.data_layer.scale,
            opacity=0.1,
            color=self.get_labels_colormap(),
        )
        self.painting_data = zarr.open(
            f"{self.zarr_path}/painting",
            mode="a",
            shape=self.image_data.shape,
            dtype="i4",
            dimension_separator=".",
        )
        self.painting_layer = self.viewer.add_labels(
            self.painting_data,
            name="Painting",
            scale=self.data_layer.scale,
            color=self.get_labels_colormap(),
        )

        # Set defaults for layers
        self.get_painting_layer().brush_size = 2
        self.get_painting_layer().n_edit_dimensions = 3

    def _init_logging(self):
        self.logger = logging.getLogger("cryocanvas")
        self.logger.setLevel(logging.DEBUG)
        streamHandler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(threadName)s - %(levelname)s - %(message)s"
        )
        streamHandler.setFormatter(formatter)
        self.logger.addHandler(streamHandler)

    def _add_widget(self):
        self.widget = CryoCanvasWidget()
        self.viewer.window.add_dock_widget(self.widget, name="CryoCanvas")
        self.widget.estimate_background_button.clicked.connect(
            self.estimate_background
        )
        self._connect_events()

    def _connect_events(self):
        for listener in [
            self.viewer.camera.events,
            self.viewer.dims.events,
        ]:
            listener.connect(
                debounced(
                    self.on_view_change,
                    timeout=1000,
                )
            )
        self.painting_layer.events.paint.connect(
            debounced(
                self.on_data_change,
                timeout=1000,
            )
        )

    def get_data_layer(self):
        return self.viewer.layers["Image"]

    def get_prediction_layer(self):
        return self.viewer.layers["Prediction"]

    def get_painting_layer(self):
        return self.viewer.layers["Painting"]

    @ensure_main_thread
    def on_view_change(self, event):
        self.logger.info("on_view_change")
        data_choice = self.widget.data_dropdown.currentText()
        if data_choice != "Whole Image":
            self.on_data_change(event=None)

    @ensure_main_thread
    def on_data_change(self, event):
        self.logger.info("on_data_change")
        data_choice = self.widget.data_dropdown.currentText()
        live_fit = self.widget.live_fit_checkbox.isChecked()
        live_prediction = self.widget.live_pred_checkbox.isChecked()
        model_type = self.widget.model_dropdown.currentText()
        use_skimage_features = False
        use_tomotwin_features = True

        self.logger.info("getting training features and labels")
        training_features, training_labels = self._get_training_features_and_labels(
            data_choice=data_choice,
            use_skimage_features=use_skimage_features,
            use_tomotwin_features=use_tomotwin_features,
        )

        if np.any(training_labels.shape == 0):
            self.logger.info("No training data yet. Skipping model update")
        elif live_fit:
            self.widget.status.setText("Fitting model ...")
            self.fit_model_task = self.executor.submit(
                self.fit_model,
                training_labels,
                training_features,
                model_type,
            )
            on_model_fit = partial(
                self.on_model_fit,
                live_prediction=live_prediction,
                use_skimage_features=use_skimage_features,
            )
            self.fit_model_task.add_done_callback(on_model_fit)

    def _get_training_features_and_labels(self, *, data_choice: str, use_skimage_features: bool, use_tomotwin_features: bool) -> tuple[np.ndarray, np.ndarray]:
        mask_idx = self._get_mask_idx(data_choice)
        
        features = []
        if use_skimage_features:
            features.append(
                self.feature_data_skimage[mask_idx].reshape(
                    -1, self.feature_data_skimage.shape[-1]
                )
            )
        if use_tomotwin_features:
            features.append(
                self.feature_data_tomotwin[mask_idx].reshape(
                    -1, self.feature_data_tomotwin.shape[-1]
                )
            )
        if features:
            features = np.concatenate(features, axis=1)
        else:
            raise ValueError("No features selected for computation.")

        if data_choice == "Current Displayed Region":
            # Use only the currently displayed region.
            training_features = self._get_features(mask_idx=mask_idx, use_skimage_features=use_skimage_features, use_tomotwin_features=use_tomotwin_features)
            _, active_labels = self._get_active_image_and_labels(data_choice=data_choice, mask_idx=mask_idx)
            training_labels = np.squeeze(active_labels)
        elif data_choice == "Whole Image":
            if use_skimage_features:
                training_features = np.array(self.feature_data_skimage)
            else:
                training_features = np.array(self.feature_data_tomotwin)
            training_labels = np.squeeze(np.array(self.painting_data))
        else:
            raise ValueError(f"Invalid data choice: {data_choice}")
        return training_features, training_labels

    def _get_mask_idx(self, data_choice: str) -> tuple[slice, slice, slice]:
        # Find a mask of indices we will use for fetching our data
        if data_choice == "Whole Image":
            return tuple(
                [slice(0, sz) for sz in self.get_data_layer().data.shape]
            )
        else:
            current_step = self.viewer.dims.current_step
            corner_pixels = self.viewer.layers["Image"].corner_pixels
            # TODO: handle view order permutations
            return (
                slice(current_step[0], current_step[0] + 1),
                slice(corner_pixels[0, 1], corner_pixels[1, 1]),
                slice(corner_pixels[0, 2], corner_pixels[1, 2]),
            )

    def _get_active_image_and_labels(self, *, data_choice: str, mask_idx: tuple[slice, slice, slice]) -> tuple[np.ndarray, np.ndarray]:
        self.logger.info(
            f"mask idx {mask_idx}, image {self.get_data_layer().data.shape}"
        )
        active_image = self.get_data_layer().data[mask_idx]
        self.logger.info(
            f"active image shape {active_image.shape} data choice {data_choice} painting_data {self.painting_data.shape} mask_idx {mask_idx}"
        )
        active_labels = self.painting_data[mask_idx]
        return active_image, active_labels

    def _get_features(self, *, mask_idx: tuple[slice, slice, slice], use_skimage_features: bool, use_tomotwin_features: bool) -> np.ndarray:
        features = []
        if use_skimage_features:
            features.append(
                self.feature_data_skimage[mask_idx].reshape(
                    -1, self.feature_data_skimage.shape[-1]
                )
            )
        if use_tomotwin_features:
            features.append(
                self.feature_data_tomotwin[mask_idx].reshape(
                    -1, self.feature_data_tomotwin.shape[-1]
                )
            )
        if features:
            return np.concatenate(features, axis=1)
        else:
            raise ValueError("No features selected for computation.")

    @ensure_main_thread
    def on_model_fit(self, task: Future, *, live_prediction: bool, use_skimage_features: bool) -> None:
        self.logger.info("on_model_fit")
        model = task.result()
        self.model = model
        if live_prediction and self.model:
            # Update prediction_data
            if use_skimage_features:
                prediction_features = np.array(self.feature_data_skimage)
            else:
                prediction_features = np.array(self.feature_data_tomotwin)
            
            self.widget.status.setText("Predicting labels ...")
            self.predict_task = self.executor.submit(self.predict, prediction_features)
            self.predict_task.add_done_callback(self.on_prediction)

    @ensure_main_thread
    def on_prediction(self, task: Future[np.ndarray]) -> None:
        prediction = task.result()
        layer = self.get_prediction_layer()
        self.logger.info(
            f"prediction {prediction.shape} prediction layer {layer.data.shape} prediction {np.transpose(prediction).shape}"# features {prediction_features.shape}"
        )
        layer.data = np.transpose(prediction)
        # Ensure the prediction layer visual is updated
        # layer.refresh()
        self.logger.info("update charts")
        self.update_class_distribution_charts()
        self.logger.info("finished")
        self.widget.status.setText("Ready")

    def fit_model(self, labels, features, model_type):
        self.logger.info("fit_model")
        # Retrain model
        self.logger.info(
            f"training model with labels {labels.shape} features {features.shape} unique labels {np.unique(labels[:])}"
        )

        # Flatten labels
        labels = labels.flatten()
        reshaped_features = features.reshape(-1, features.shape[-1])

        # Filter features where labels are greater than 0
        valid_labels = labels > 0
        filtered_features = reshaped_features[valid_labels, :]
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

        # Apply weights
        sample_weights = np.vectorize(weight_dict.get)(filtered_labels)

        # Model fitting
        if model_type == "Random Forest":
            clf = RandomForestClassifier(
                n_estimators=50,
                n_jobs=-1,
                max_depth=10,
                max_samples=0.05,
                class_weight=weight_dict,
            )
            clf.fit(filtered_features, filtered_labels)
            return clf
        elif model_type == "XGBoost":
            clf = xgb.XGBClassifier(
                n_estimators=100, learning_rate=0.1, use_label_encoder=False
            )
            clf.fit(
                filtered_features,
                filtered_labels,
                sample_weight=sample_weights,
            )
            return clf
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def predict(self, features):
        # We shift labels + 1 because background is 0 and has special meaning
        prediction = (
            future.predict_segmenter(
                features.reshape(-1, features.shape[-1]), self.model
            ).reshape(features.shape[:-1])
            + 1
        )
        return np.transpose(prediction)

    def update_class_distribution_charts(self):
        # Example class to color mapping, this needs to match your label colors
        class_color_mapping = {
            label: "#{:02x}{:02x}{:02x}".format(
                int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)
            )
            for label, rgba in self.get_labels_colormap().items()
        }

        painting_labels, painting_counts = np.unique(
            self.painting_data[:], return_counts=True
        )
        prediction_labels, prediction_counts = np.unique(
            self.get_prediction_layer().data[:], return_counts=True
        )

        # Include class 0 for prediction if it's missing
        if 0 not in prediction_labels:
            prediction_labels = np.insert(prediction_labels, 0, 0)
            prediction_counts = np.insert(prediction_counts, 0, 0)

        # Align classes between painting and prediction
        all_labels = np.union1d(painting_labels, prediction_labels)

        # Get counts for all classes, filling in zeros where a class doesn't exist
        aligned_painting_counts = [
            painting_counts[np.where(painting_labels == label)][0]
            if label in painting_labels
            else 0
            for label in all_labels
        ]
        aligned_prediction_counts = [
            prediction_counts[np.where(prediction_labels == label)][0]
            if label in prediction_labels
            else 0
            for label in all_labels
        ]

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
            ax1 = self.widget.figure.add_subplot(211)
            ax2 = self.widget.figure.add_subplot(212)

            # Plot the bars with the correct color mapping
            ax1.bar(
                all_labels,
                aligned_painting_counts,
                color=[
                    class_color_mapping.get(x, "#FFFFFF") for x in all_labels
                ],
                edgecolor="white",
            )
            ax1.set_title("Painting Layer")
            ax1.set_xlabel("Class")
            ax1.set_ylabel("Count")
            ax1.set_xticks(all_labels)  # Ensure only integer ticks are shown

            ax2.bar(
                all_labels,
                aligned_prediction_counts,
                color=[
                    class_color_mapping.get(x, "#FFFFFF") for x in all_labels
                ],
                edgecolor="white",
            )
            ax2.set_title("Prediction Layer")
            ax2.set_xlabel("Class")
            ax2.set_ylabel("Count")
            ax2.set_xticks(all_labels)  # Ensure only integer ticks are shown

        # Automatically adjust subplot params so that the subplot(s) fits into the figure area
        self.widget.figure.tight_layout(pad=3.0)

        # Explicitly set figure background color again to ensure it
        self.widget.figure.patch.set_facecolor(napari_charcoal_hex)

        self.widget.canvas.draw()

    def estimate_background(self):
        # Start the background painting in a new thread
        threading.Thread(target=self._paint_background_thread).start()

    def _paint_background_thread(self):
        print("Estimating background label")
        embedding_data = self.feature_data_tomotwin[:]

        # Compute the median of the embeddings
        median_embedding = np.median(embedding_data, axis=(0, 1, 2))

        # Compute the Euclidean distance from the median for each embedding
        distances = np.sqrt(
            np.sum((embedding_data - median_embedding) ** 2, axis=-1)
        )

        # Define a threshold for background detection
        # TODO note this is hardcoded
        threshold = np.percentile(distances.flatten(), 1)

        # Identify background pixels (where distance is less than the threshold)
        background_mask = distances < threshold
        indices = np.where(background_mask)

        print(
            f"Distance distribution: min {np.min(distances)} max {np.max(distances)} mean {np.mean(distances)} median {np.median(distances)} threshold {threshold}"
        )

        print(f"Labeling {np.sum(background_mask)} pixels as background")

        # TODO: optimize this because it is wicked slow
        #       once that is done the threshold can be increased
        # Update the painting data with the background class (1)
        for i in range(len(indices[0])):
            self.painting_data[indices[0][i], indices[1][i], indices[2][i]] = 1

        # Refresh the painting layer to show the updated background
        self.get_painting_layer().refresh()

class CryoCanvasWidget(QWidget):
    def __init__(self, parent=None):
        super(CryoCanvasWidget, self).__init__(parent)
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # Dropdown for selecting the model
        model_label = QLabel("Select Model")
        self.model_dropdown = QComboBox()
        self.model_dropdown.addItems(["Random Forest", "XGBoost"])
        model_layout = QHBoxLayout()
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_dropdown)
        layout.addLayout(model_layout)

        # Boolean options for features
        self.basic_checkbox = QCheckBox("Basic")
        self.basic_checkbox.setChecked(True)
        self.embedding_checkbox = QCheckBox("Embedding")
        self.embedding_checkbox.setChecked(True)

        features_group = QGroupBox("Features")
        features_layout = QVBoxLayout()
        features_layout.addWidget(self.basic_checkbox)
        features_layout.addWidget(self.embedding_checkbox)
        features_group.setLayout(features_layout)
        layout.addWidget(features_group)

        # Button for estimating background
        self.estimate_background_button = QPushButton("Estimate Background")
        layout.addWidget(self.estimate_background_button)

        # Dropdown for data selection
        data_label = QLabel("Select Data for Model Fitting")
        self.data_dropdown = QComboBox()
        self.data_dropdown.addItems(
            ["Current Displayed Region", "Whole Image"]
        )
        self.data_dropdown.setCurrentText("Whole Image")
        data_layout = QHBoxLayout()
        data_layout.addWidget(data_label)
        data_layout.addWidget(self.data_dropdown)
        layout.addLayout(data_layout)

        # Checkbox for live model fitting
        self.live_fit_checkbox = QCheckBox("Live Model Fitting")
        self.live_fit_checkbox.setChecked(True)
        layout.addWidget(self.live_fit_checkbox)

        # Checkbox for live prediction
        self.live_pred_checkbox = QCheckBox("Live Prediction")
        self.live_pred_checkbox.setChecked(True)
        layout.addWidget(self.live_pred_checkbox)

        # Add class distribution plot
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # Add status log
        self.status = QLabel("Ready")
        layout.addWidget(self.status)

        self.setLayout(layout)


# Initialize your application
if __name__ == "__main__":
    zarr_path = "/Users/asweet/data/cryocanvas/cryocanvas_crop_006.zarr"
    app = CryoCanvasApp(zarr_path)
    napari.run()
