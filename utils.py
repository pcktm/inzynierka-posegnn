import os
import pykitti


basedir = "/home/pcktm/inzynierka/kitti/dataset"
sequence = "05"

# create directory for features
os.makedirs(f"models/{sequence}", exist_ok=True)

kitti_dataset = pykitti.odometry(basedir, sequence)
