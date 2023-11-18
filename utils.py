import numpy as np
from tqdm import tqdm
import os
from scipy.spatial.transform import Rotation as R


def find_nearest_neighbors(query: np.ndarray, database, K=5) -> list[int]:
    """
    Finds the K nearest neighbors of a query in a database of vectors.
    """
    distances = np.linalg.norm(database - query, axis=1)
    return np.argsort(distances)[:K]


def extract_position_rotation(transform):
    """
    Extract position and rotation from a 4x4 transformation matrix.

    Args:
        transform (numpy.ndarray): a 4x4 transformation matrix.

    Returns:
        dict: a dictionary with keys 'position' and 'rotation', where
              'position' is a 3D vector representing the position and
              'rotation' is a quaternion representing the rotation.
    """

    assert transform.shape == (4, 4), "Input should be a 4x4 matrix."

    # Position is the last column of the matrix
    position = transform[:3, 3]

    # Rotation matrix is the first 3 columns and rows
    rotation_matrix = transform[:3, :3]

    # Convert rotation matrix to quaternion
    rotation = R.from_matrix(rotation_matrix).as_quat()

    return {"position": position, "rotation": rotation}
