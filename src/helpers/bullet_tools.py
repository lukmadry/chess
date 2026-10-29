import pybullet as p
import numpy as np
import os
import tempfile
import glob
import matplotlib

from src.helpers.vis import load_images, display_grid_with_labels

matplotlib.use('Agg')

from PIL import Image


def pan_camera_relative(dx_local, dy_local, current_target, yaw_deg):
    """
    Pans the camera target on the XY plane relative to the camera's current view.
    dx_local: slide right (+) or left (-)
    dy_local: slide forward (+) or backward (-)
    """
    yaw_rad = np.radians(yaw_deg)
    
    # Rotate the local pan offset into world coordinates
    dx_world = dx_local * np.cos(yaw_rad) - dy_local * np.sin(yaw_rad)
    dy_world = dx_local * np.sin(yaw_rad) + dy_local * np.cos(yaw_rad)
    
    new_target = [
        current_target[0] + dx_world,
        current_target[1] + dy_world,
        current_target[2]  # Keep Z height unchanged
    ]
    return new_target


def load_tiled_texture(image_path, scale_factor=3, target_size=(1024, 1024)):
    """
    Tiles an image `scale_factor` times along each axis to prevent stretching
    when applied to large visual shapes in PyBullet.
    """
    # mean_pixel = 0
    with Image.open(image_path) as img:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # 1. Tile the image across a grid (e.g., 3x3)
        w, h = img.size
        tiled_img = Image.new('RGB', (w * scale_factor, h * scale_factor))
        # mean_pixel = tiled_img.mean()
        for i in range(scale_factor):
            for j in range(scale_factor):
                tiled_img.paste(img, (i * w, j * h))
                
        # 2. Optionally resize to standard dimension (e.g. 1024x1024)
        tiled_img = tiled_img.resize(target_size, Image.Resampling.LANCZOS)

        temp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        temp_path = temp_file.name
        tiled_img.save(temp_path, "JPEG", quality=90)
        temp_file.close()
        
        # 3. Extract raw pixel bytes
        raw_bytes = tiled_img.tobytes()
        width, height = tiled_img.size

    # Load directly into PyBullet's texture memory
    txtr = p.loadTexture(temp_path)
    os.remove(temp_path)
    return txtr#, mean_pixel


def get_coordinates_from_half_extents(half_extent):
    bl = [-half_extent[0], -half_extent[1], half_extent[2]]
    tl = [-half_extent[0], half_extent[1], half_extent[2]]
    tr = [half_extent[0], half_extent[1], half_extent[2]]
    br = [half_extent[0], -half_extent[1], half_extent[2]]
    return (bl, tl, tr, br)


def project_world_pixel(world_coords, proj_mat, view_mat, img_height, img_width):
    wc_arr = np.array(np.append(world_coords, 1.))

    VP = proj_mat @ view_mat
    clip = VP @ wc_arr
    
    w_clip = float(clip[3])
    ndc_x = float(clip[0]) / w_clip
    ndc_y = float(clip[1]) / w_clip

    pix_width  = (ndc_x + 1.0) / 2.0 * img_width   # Column (x)
    pix_height = (1.0 - ndc_y) / 2.0 * img_height  # Row (y)

    return int(pix_height), int(pix_width)


def compute_robust_view_matrix(target_pos, dist, yaw_deg, pitch_deg):
    if abs(pitch_deg + 90) > 1:
        raise NotImplementedError
    yaw_rad = np.radians(yaw_deg)
    pitch_rad = np.radians(pitch_deg)
    
    # 1. Compute 3D camera eye position
    # eye_x = target_pos[0] + dist * np.cos(pitch_rad) * np.cos(yaw_rad)
    # eye_y = target_pos[1] + dist * np.cos(pitch_rad) * np.sin(yaw_rad)
    # eye_z = target_pos[2] - dist * np.sin(pitch_rad)
    eye_pos = [target_pos[0], target_pos[1], target_pos[2] + dist]
    
    # 2. Singular case check (top-down view where pitch is -90 degrees)
    # if abs(pitch_deg + 90.0) < 1e-3:
    #     # Derived formula for top-down camera up vector
    up_vector = [np.cos(yaw_rad), np.sin(yaw_rad), 0.0]
        
    return p.computeViewMatrix(
        cameraEyePosition=eye_pos,
        cameraTargetPosition=target_pos,
        cameraUpVector=up_vector
    )

def turn_image_list_into_textures(image_list=None, glob_query=None):
    if glob_query is not None and image_list is None:
        image_list = glob.glob(glob_query)

    return [p.loadTexture(x) for x in image_list]


# split the loading part and the visualisation part
# maybe irrelevant, just use Dataset ?

# probably to be rewritten in the future

# wrapper for visualisation direct from the hard drive
def sample_imgs_and_display(model, image_dir, image_grid_size=3, epoch=-1, figsize=10., with_ground_truths=True, 
    savefig=True, fig_dir=None, show=True):

    display_images, image_array, gt_list = load_images(
        image_dir=image_dir, 
        image_grid_size=image_grid_size, 
        with_ground_truths=True)

    display_grid_with_labels(
        model=model, 
        display_images=display_images,
        image_array=image_array, 
        gt_list=gt_list, 
        savefig=savefig, 
        image_grid_size=image_grid_size,
        fig_dir=fig_dir, 
        show=show,
        epoch=epoch)