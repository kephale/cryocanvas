from enum import Enum

import napari
from qtpy.QtWidgets import QTabWidget, QVBoxLayout, QWidget

from cellcanvas.data.data_manager import DataManager
from cellcanvas.geometry.surforama import QtSurforama
from cellcanvas.instance._qt.qt_segment_manager import QtSegmentManager
from cellcanvas.semantic._embedding_segmentor import EmbeddingPaintingApp


class AppMode(Enum):
    SEMANTIC = "semantic"
    INSTANCE = "instance"


class CellCanvasApp:
    def __init__(
        self, data: DataManager, viewer: napari.Viewer, verbose: bool = False
    ):
        self.data = data
        self.viewer = viewer
        self.verbose = verbose

        self.mode = AppMode.SEMANTIC

        self.semantic_segmentor = EmbeddingPaintingApp(
            data_manager=self.data,
            viewer=self.viewer,
            extra_logging=self.verbose,
        )


    def update_data_manager(self, data: DataManager):
        self.data = data
        self.semantic_segmentor.update_data_manager(data)
        
    @property
    def mode(self) -> AppMode:
        return self._mode

    @mode.setter
    def mode(self, mode: AppMode):
        self._mode = mode


class QtCellCanvas(QWidget):
    def __init__(self, app: CellCanvasApp, parent=None):
        super().__init__(parent=parent)
        self.app = app
        self.viewer = self.app.viewer

        self.tab_widget = QTabWidget(self)

        # make the instance segmentation widget
        self.instance_widget = QtSegmentManager(
            labels_layer=None, viewer=self.viewer, parent=self.tab_widget
        )

        # make the surforama widget
        self.surforama = QtSurforama(
            viewer=self.viewer, surface_layer=None, volume_layer=None
        )
        # add the tabs
        self.tab_widget.addTab(
            self.app.semantic_segmentor.widget, "semantic segmentation"
        )
        self.tab_widget.addTab(self.instance_widget, "instance segmentation")
        self.tab_widget.addTab(self.surforama, "geometry builder")

        # set the layout
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.tab_widget)
