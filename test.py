import numpy as np
import pybullet as p

def project_world_pixel(world_coords, view_mat, proj_mat, img_height, img_width):
    coords_1d = np.asarray(world_coords).ravel()
    wc_arr = np.append(coords_1d, 1.0)  # Shape (4,)
    
    # Combined View-Projection
    VP = proj_mat @ view_mat
    clip = VP @ wc_arr
    
    # Perspective divide
    w_clip = float(clip[3])
    ndc_x = float(clip[0]) / w_clip
    ndc_y = float(clip[1]) / w_clip

    # Map NDC [-1, 1] to Pixel Space
    pix_width  = (ndc_x + 1.0) / 2.0 * img_width   # Column (x)
    pix_height = (1.0 - ndc_y) / 2.0 * img_height  # Row (y)

    return int(pix_height), int(pix_width)

# Scene Setup
img_w, img_h = 640, 480
target_pos = [0.0, 0.0, 0.0]
dist = 0.4
yaw = 45.
pitch = -90.0

# Exact top-down eye & up vector derivation
yaw_rad = np.radians(yaw)
eye_pos = [target_pos[0], target_pos[1], target_pos[2] + dist]
up_vector = [np.cos(yaw_rad), np.sin(yaw_rad), 0.0] 

# Matrices with .T transpose
view_mat_flat = p.computeViewMatrix(cameraEyePosition=eye_pos, cameraTargetPosition=target_pos, cameraUpVector=up_vector)
proj_mat_flat = p.computeProjectionMatrixFOV(fov=60.0, aspect=float(img_w)/img_h, nearVal=0.01, farVal=10.0)

print(view_mat_flat)
print("\n")
print(proj_mat_flat)

view_mat_arr = np.array(view_mat_flat).reshape(4, 4).T
proj_mat_arr = np.array(proj_mat_flat).reshape(4, 4).T

# Your test points
pt_start = [-0.1, -0.1, 0.01]
pt_end   = [0.1, 0.1, 0.01]

res_start = project_world_pixel(pt_start, view_mat_arr, proj_mat_arr, img_h, img_w)
res_end   = project_world_pixel(pt_end, view_mat_arr, proj_mat_arr, img_h, img_w)

print("start point (row, col):", res_start)
print("end point   (row, col):", res_end)