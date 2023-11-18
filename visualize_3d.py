import open3d as o3d
from dataset import KittiSequenceDataset
import numpy as np
from tqdm import tqdm

basedir = "/home/pcktm/inzynierka/kitti/dataset"
selected_sequence = "02"

dataset = KittiSequenceDataset(basedir, selected_sequence, load_images=False)

vis = o3d.visualization.Visualizer()
vis.create_window()

for i, data in tqdm(
    enumerate(dataset), desc="Loading sequence frames", total=len(dataset)
):
    mesh_box = o3d.geometry.TriangleMesh.create_cone()
    mesh_box.compute_vertex_normals()
    mesh_box.paint_uniform_color([0.1, 0.1, 0.7])

    # draw position and rotation as a coordinate frame
    position = data["pose"]["position"]

    # rotation is in quaternion format so we need to convert it to rotation matrix
    rotation = data["pose"]["rotation"]
    rotation = o3d.geometry.get_rotation_matrix_from_quaternion(rotation)

    mesh_box.rotate(rotation)
    mesh_box.translate(position)

    # draw the box
    vis.add_geometry(mesh_box)
    vis.poll_events()

vis.update_renderer()
vis.run()
