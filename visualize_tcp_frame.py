import sys
import os
import pickle
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from umi.common.pose_util import pose_to_mat
import cv2
import json

print("Loading in dataset plan.")
# Load your dataset plan
with open('/home/armstrong/umi/first_demo_session/dataset_plan.pkl', 'rb') as f:
    all_plans = pickle.load(f)
print("Dataset plan loaded.")
import cv2
import json

with open('/home/armstrong/umi/example/calibration/gopro_intrinsics_2_7k.json', 'r') as f:
    intr = json.load(f)
intrinsics = intr['intrinsics']
fx = fy = intrinsics['focal_length']
cx = intrinsics['principal_pt_x']
cy = intrinsics['principal_pt_y']
dist_coeffs = [
    intrinsics['radial_distortion_1'],
    intrinsics['radial_distortion_2'],
    intrinsics['radial_distortion_3'],
    intrinsics['radial_distortion_4']
]
K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0,  0,  1]], dtype=np.float64)
D = np.array(dist_coeffs, dtype=np.float64)

def project_points_fisheye(points_3d):
    points_3d = np.asarray(points_3d).reshape(-1, 1, 3)
    rvec = np.zeros((3, 1))  # no rotation (already in camera frame)
    tvec = np.zeros((3, 1))  # no translation
    points_2d, _ = cv2.fisheye.projectPoints(points_3d, rvec, tvec, K, D)
    return points_2d.reshape(-1, 2)

    v = fy * y / z + cy
    return int(round(u)), int(round(v))

# Example: for the first frame of the first episode, first camera
plan = all_plans[0]
video_rel_path = plan['cameras'][0]['video_path']
video_full_path = os.path.join('/home/armstrong/umi/first_demo_session/demos', video_rel_path)
start_frame, _ = plan['cameras'][0]['video_start_end']


# Use the static tcp-to-camera transform from calibration
cam_to_center_height = 0.086 # constant for UMI
cam_to_mount_offset = 0.01465 # constant for GoPro Hero 9,10,11
tcp_offset = 0.205 # or use your actual value
cam_to_tip_offset = cam_to_mount_offset + tcp_offset
pose_cam_tcp = np.array([0, cam_to_center_height, cam_to_tip_offset, 0,0,0])
tx_cam_tcp = pose_to_mat(pose_cam_tcp)

# The TCP frame in camera coordinates is simply tx_cam_tcp
mat_cam_tcp = tx_cam_tcp

axis_length = 0.05  # 5 cm
orig = mat_cam_tcp[:3, 3]
x_axis = orig + mat_cam_tcp[:3, 0] * axis_length
y_axis = orig + mat_cam_tcp[:3, 1] * axis_length
z_axis = orig + mat_cam_tcp[:3, 2] * axis_length

# Project to image
orig_uv = tuple(map(int, project_points_fisheye([orig])[0]))
x_uv = tuple(map(int, project_points_fisheye([x_axis])[0]))
y_uv = tuple(map(int, project_points_fisheye([y_axis])[0]))
z_uv = tuple(map(int, project_points_fisheye([z_axis])[0]))

# Read the frame
cap = cv2.VideoCapture(video_full_path)
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
ret, frame = cap.read()
cap.release()

if ret:
    # Draw axes
    cv2.line(frame, orig_uv, x_uv, (0,0,255), 2)  # X - red
    cv2.line(frame, orig_uv, y_uv, (0,255,0), 2)  # Y - green
    cv2.line(frame, orig_uv, z_uv, (255,0,0), 2)  # Z - blue
    cv2.imshow('TCP Frame Overlay', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Failed to read frame.")