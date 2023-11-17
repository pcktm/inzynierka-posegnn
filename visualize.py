import pykitti
import numpy as np
import matplotlib.pyplot as plt
import colorcet
from PIL import ImageColor
import cv2

basedir = "/home/pcktm/inzynierka/kitti/dataset"
sequence = "07"
np.set_printoptions(precision=4, suppress=True)

dataset = pykitti.odometry(basedir, sequence)

trajectory_image = np.zeros((1000, 1000, 3), dtype=np.uint8)

def get_pose_and_absolute_scale(gt, frame_id):
    prev = gt[frame_id - 1]
    curr = gt[frame_id]
    x_prev, y_prev, z_prev = prev[:3, 3]
    x_curr, y_curr, z_curr = curr[:3, 3]
    absolute_scale = np.sqrt(
        (x_curr - x_prev) ** 2 + (y_curr - y_prev) ** 2 + (z_curr - z_prev) ** 2
    )
    x, y, z = curr[:3, 3]
    return x, y, z, absolute_scale


for index, img in enumerate(dataset.cam2):
    print(index)
    image = np.array(img)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    try:
        features = np.load(f"features/{sequence}/{index}.npy")
    except FileNotFoundError:
        features = np.zeros(2048)

    x, y, z, absolute_scale = get_pose_and_absolute_scale(dataset.poses, index)
    # scale it down by half to fit on the screen
    x, y, z = x / 2, y , z / 2
    # normalize y to [0, 255]
    # y = (y - min_y) / (max_y - min_y) * 255
    # map height y to continous colormap
    # color = colorcet.b_isoluminant_cm_70_c39[int(y)]
    frame_id = str(index).zfill(6)
    color = colorcet.b_cyclic_ymcgy_60_90_c67[int(frame_id) % 255]

    color = ImageColor.getcolor(color, "RGB")
    color = (int(color[2]), int(color[1]), int(color[0]))
    cv2.circle(trajectory_image, (int(x) + 500, int(z) + 500), 1, color, 1)
    text = "Coordinates: x=%2fm y=%2fm z=%2fm" % (x, y, z)

    string_features = np.array2string(features, precision=2, separator=", ", max_line_width=1000)

    image = cv2.putText(image, text, (20, 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)

    cv2.imshow("Road facing camera", image)
    cv2.imshow("Trajectory", trajectory_image)
    cv2.waitKey(1 if index != dataset.__len__() - 1 else 0)
