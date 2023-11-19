import open3d as o3d
from dataset import KittiSequenceDataset
import numpy as np
from tqdm import tqdm

basedir = "/home/pcktm/inzynierka/kitti/dataset"
selected_sequence = "02"

dataset = KittiSequenceDataset(basedir, selected_sequence, load_images=False)

vis = o3d.visualization.Visualizer()

vis.create_window()
vis.get_render_option().background_color = np.asarray([0.6, 0.6, 0.6])

all_geometry = []
max_timestamp = dataset.dataset.timestamps[-1]

for i, data in tqdm(
    enumerate(dataset), desc="Loading sequence frames", total=len(dataset)
):
    mesh_box = o3d.geometry.TriangleMesh.create_cone()
    mesh_box.compute_vertex_normals()

    # paint it according to the timestamp
    mesh_box.paint_uniform_color(
        np.asarray([data["timestamp"] / max_timestamp, 0, 1 - data["timestamp"] / max_timestamp])
    )

    # draw position and rotation as a coordinate frame
    position = data["pose"]["position"]

    # rotation is in quaternion format so we need to convert it to rotation matrix
    rotation = data["pose"]["rotation"]
    rotation = o3d.geometry.get_rotation_matrix_from_quaternion(rotation)

    mesh_box.rotate(rotation)
    mesh_box.translate(position)

    all_geometry.append(mesh_box)
    vis.poll_events()

for index, geometry in tqdm(
    enumerate(all_geometry),
    desc="Adding geometry to visualizer",
    total=len(all_geometry),
):
    vis.add_geometry(geometry)
    if index % 500 == 0:
        vis.poll_events()

vis.get_render_option().background_color = np.asarray([1, 1, 1])
vis.update_renderer()
vis.run()
