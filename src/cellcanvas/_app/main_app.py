from enum import Enum

import napari
from qtpy.QtWidgets import QWidget, QTabWidget, QVBoxLayout, QLineEdit

from cellcanvas.instance._qt.qt_segment_manager import QtSegmentManager
from cellcanvas.geometry.surforama import QtSurforama

class AppMode(Enum):
    SEMANTIC = "semantic"
    INSTANCE = "instance"


class CellCanvasApp:
    def __init__(self, viewer: napari.Viewer):
        self.viewer = viewer

        self.mode = AppMode.SEMANTIC

    @property
    def mode(self) -> AppMode:
        return self._mode

    @mode.setter
    def mode(self, mode: AppMode):
        self._mode = mode


class QtCellCanvas(QWidget):
    def __init__(self, app: CellCanvasApp, parent=None):
        super(QtCellCanvas, self).__init__(parent)
        self.app = app
        self.viewer = self.app.viewer

        self.tab_widget = QTabWidget(self)

        # make the instance segmentation widget
        self.instance_widget = QtSegmentManager(
            labels_layer=None, viewer=self.viewer, parent=self.tab_widget
        )

        # make the surforama widget
        self.surforama = QtSurforama(
            viewer=self.viewer,
            surface_layer=None,
            volume_layer=None
        )
        # add the tabs
        self.tab_widget.addTab(QLineEdit("semantic"), "semantic segmentation")
        self.tab_widget.addTab(self.instance_widget, "instance segmentation")
        self.tab_widget.addTab(self.surforama, "geometry builder")

        # set the layout
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.tab_widget)
