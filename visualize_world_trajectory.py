import sys
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from umi.common.pose_util import pose_to_mat

# Load the dataset plan (contains episode info)
with open('/home/armstrong/umi/simple_demo_session/dataset_plan.pkl', 'rb') as f:
    all_plans = pickle.load(f)

# Choose an episode (e.g., the first one)
episode_idx = 0
plan = all_plans[episode_idx]
print("Camera video path:", plan['cameras'][0]['video_path'])
# Extract TCP poses directly from the plan
print(plan["grippers"][0])
tcp_poses = plan["grippers"][0]["tcp_pose"]
start_pose = plan["grippers"][0]["demo_start_pose"]
goal_pose = plan["grippers"][0]["demo_end_pose"]
# Print the first 3 elements of each pose in tcp_pose
for i, pose in enumerate(tcp_poses):
    print(f"Pose {i} first 3 elements:", pose[:3])
print("Distance to goal:", np.linalg.norm(np.array(start_pose[:3]) - np.array(goal_pose[:3])))
# Each pose: [x, y, z, ...]
pos_list = []
for pose in tcp_poses:
    if len(pose) >= 3:
        pos_list.append(pose[:3])
pos_arr = np.array(pos_list)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(pos_arr[:,0], pos_arr[:,1], pos_arr[:,2], marker='o', label='Trajectory')
ax.scatter(pos_arr[0,0], pos_arr[0,1], pos_arr[0,2], color='g', s=50, label='Start')
ax.scatter(pos_arr[-1,0], pos_arr[-1,1], pos_arr[-1,2], color='r', s=50, label='End')

# Plot orientation axes for each pose
axis_length = 0.03  # 3 cm
for pose in tcp_poses[::max(1, len(tcp_poses)//50)]:  # subsample for clarity if many poses
    mat = pose_to_mat(pose)
    origin = mat[:3, 3]
    x_axis = origin + mat[:3, 0] * axis_length
    y_axis = origin + mat[:3, 1] * axis_length
    z_axis = origin + mat[:3, 2] * axis_length
    ax.plot([origin[0], x_axis[0]], [origin[1], x_axis[1]], [origin[2], x_axis[2]], color='r')  # X - red
    ax.plot([origin[0], y_axis[0]], [origin[1], y_axis[1]], [origin[2], y_axis[2]], color='g')  # Y - green
    ax.plot([origin[0], z_axis[0]], [origin[1], z_axis[1]], [origin[2], z_axis[2]], color='b')  # Z - blue

ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title(f'Episode {episode_idx} Trajectory in World Frame')
ax.legend()
plt.show()
