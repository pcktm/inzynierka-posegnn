import numpy as np
from scipy.spatial.transform import Rotation as R
import torch


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
    rotation = R.from_matrix(rotation_matrix)

    return {"position": position, "rotation": rotation}


def normalize_position_and_rotation(samples: list[torch.tensor]) -> torch.tensor:
    # position and rotation are encoded [x, y, z, w, x, y, z]
    pos = torch.stack([s[0] for s in samples])
    rot = torch.stack([s[1] for s in samples])
    # normalize position to the first sample
    pos = pos - pos[0]

    # normalize rotation to the first sample (remember quaternion rotations)
    new_rot = []
    for i in range(rot.shape[0]):
        new_rot.append(R.from_quat(rot[i]).inv() * R.from_quat(rot[0]))

    new_rot = torch.tensor([r.as_quat() for r in new_rot])

    return torch.concatenate((pos, new_rot), axis=1)


def tensor_to_position_and_rotation(tensor):
    pos = tensor[:, :3]
    rot = tensor[:, 3:]

    rot = R.from_quat(rot)

    return pos, rot
