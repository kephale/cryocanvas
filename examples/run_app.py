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

import napari
from cellcanvas._app.main_app import CellCanvasApp, QtCellCanvas
from cellcanvas.data.data_manager import DataManager
from cellcanvas.data.data_set import DataSet

base_file_path = "./seg/cropped_covid.zarr"

dataset = DataSet.from_paths(
    image_path=f"{base_file_path}/image/cropped_covid",
    features_path=f"{base_file_path}/embedding/swin_unetr",
    labels_path=f"{base_file_path}/painting",
    segmentation_path=f"{base_file_path}/prediction",
    make_missing_datasets=True,
)

data_manager = DataManager(datasets=[dataset])
viewer = napari.Viewer()
cell_canvas = CellCanvasApp(data=data_manager, viewer=viewer, verbose=True)
cell_canvas_widget = QtCellCanvas(app=cell_canvas)
viewer.window.add_dock_widget(cell_canvas_widget)

napari.run()
