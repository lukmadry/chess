import pybullet as p
import pybullet_data
import time
import numpy as np
import os
import tempfile
import glob
import torch
import cv2
import matplotlib.pyplot as plt
import json
import datetime
import functools
from collections.abc import Iterable
from mpl_toolkits.axes_grid1 import ImageGrid

from random import choices
from PIL import Image

def none_sentinel(**kwargs):
    non_none_keys = [key for key, val in kwargs.items() if val is not None]
    if len(non_none_keys) != 1:
        all_keys = ", ".join(kwargs.keys())
        raise ValueError(f"Expected exactly one non-None argument among ({all_keys})"
            f"but found non-None values for {non_none_keys}")

def canonical_corner_order(pts, dtype=np.int32):
    """
    pts: array-like of 4 points [[y, x], ...]
    Returns array of 4 points ordered [Top-Left, Top-Right, Bottom-Right, Bottom-Left].
    """
    pts = np.array(pts, dtype=np.float32).reshape(4, 2)
    center = pts.mean(axis=0)
    
    # Compute relative vectors from center: [dy, dx]
    rel = pts - center
    
    dy = rel[:, 0]
    dx = rel[:, 1]
    
    # Quadrant masks
    tl_mask = (dy < 0) & (dx < 0)  # Top-Left (-y, -x)
    tr_mask = (dy < 0) & (dx > 0)  # Top-Right (-y, +x)
    br_mask = (dy > 0) & (dx > 0)  # Bottom-Right (+y, +x)
    bl_mask = (dy > 0) & (dx < 0)  # Bottom-Left (+y, -x)
    
    canonical_pts = np.zeros_like(pts)
    canonical_pts[0] = pts[tl_mask]
    canonical_pts[1] = pts[tr_mask]
    canonical_pts[2] = pts[br_mask]
    canonical_pts[3] = pts[bl_mask]
    
    return canonical_pts.astype(dtype)

def vectorise(depth_check = 2):
    if depth_check == 2:

        def decorator(func):
            @functools.wraps(func)
            def wrapper(first_arg, *args, **kwargs):
                if isinstance(first_arg, Iterable) and isinstance(first_arg[0], Iterable):
                    return np.array([func(f, *args, **kwargs) for f in first_arg])
                return func(first_arg)
            return wrapper

    elif depth_check == 3:

        def decorator(func):
            @functools.wraps(func)
            def wrapper(first_arg, *args, **kwargs):
                if isinstance(first_arg, Iterable) and isinstance(first_arg[0], Iterable) and isinstance(first_arg[0][0], Iterable):
                    return np.array([func(f, *args, **kwargs) for f in first_arg])
                return func(first_arg)
    
            return wrapper

    else:
        raise NotImplementedError("depth_check other than 2 or 3 not supported")
        
    return decorator

def model_path_from_date(last_model_date, training_epoch=None, no_model=None, delimiter="-"):
    print('no_model arg ', no_model)
    model_day = datetime.datetime.strptime(last_model_date, f"%d{delimiter}%m{delimiter}%Y")
    formatted_date = model_day.strftime("%A, %d. %B %Y")
    matching_checkpoints_by_date = glob.glob(os.path.join("checkpoints", formatted_date + "*"))
    matching_checkpoints_by_date = sorted(matching_checkpoints_by_date, key=lambda x: datetime.datetime.strptime(x.split("/")[-1], "%A, %d. %B %Y %I:%M%p"))
    if not matching_checkpoints_by_date:
        raise ValueError("no matching checkpoints. You gave the date: {}".format(last_model_date))
    if no_model is not None:
        chckpt_path = matching_checkpoints_by_date[no_model]
    else:
        chckpt_path = matching_checkpoints_by_date[-1]

    nr_epochs = len(glob.glob(os.path.join(chckpt_path, "epoch_*.pth"))) - 1
    if training_epoch is not None and training_epoch != -1:
        nr_epochs = training_epoch
    chckpt_path = os.path.join(chckpt_path, f"epoch_{nr_epochs}.pth")
    return chckpt_path

@vectorise(3)
def convert_boxes_to_cas(gt_input, height_first=True, normalised_labels=True, height=640, width=480, max_angle_rad=0.349):
    # max angle corresponds to 20 degrees
    # print(gt_input)
    # print([c for c in gt_input])
    center_x, center_y = sum([corner[0] for corner in gt_input])/4, sum([corner[1] for corner in gt_input])/4
    size = np.sqrt((gt_input[0][0]-gt_input[1][0])**2 + (gt_input[1][1]-gt_input[0][1])**2) 

    idx = 1-int(height_first)
    gt_input = sorted(gt_input, key=lambda x: x[0])
    angle = np.arctan((gt_input[0][idx] - gt_input[1][idx]) / (gt_input[0][1-idx] - gt_input[1][1-idx]))
    if normalised_labels:
        return [center_x / width, center_y / height, size / min(height, width), angle / max_angle_rad]
    return [center_x, center_x, size, angle]

@vectorise(2)
def cas_to_boxes(cas, normalised_labels=True, height=640, width=480, max_angle_rad=0.349):
    m = min(height, width)
    center_x = cas[0] * width
    center_y = cas[1] * height
    halfsize = cas[2] * m / 2
    offsets = [
        [ - halfsize,  - halfsize],
        [  halfsize,  - halfsize],
        [  halfsize, halfsize],
        [ - halfsize, halfsize]
    ]
    offsets = np.array(offsets, dtype=np.float32)
    angle = -cas[3] * max_angle_rad
    c, s = np.cos(angle), np.sin(angle)
    rot = np.array([[c, s], [-s, c]])
    rot_offsets = offsets @ rot
    return rot_offsets + np.array([center_x, center_y])

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

color_dict = {'r': (255, 0, 0), 'g': (0, 255, 0), 'b': (0, 0, 255)}

def labels_to_pixels(img, label_set, color='r'):
    h, w = img.shape[:2]
    pixels = [(int(y*w), int(x*h)) for x, y in zip(label_set[0::2], label_set[1::2])]
    pixels = np.array(pixels).reshape((-1,1,2))
    cv2.polylines(img, [pixels], isClosed=True, color=color_dict[color], thickness=2)

def gt_to_pixels(img, label_gt, color='g'):
    label_gt = [(row, col) for (col, row) in label_gt]
    pixels = np.array(label_gt).reshape((-1,1,2))
    cv2.polylines(img, [pixels], isClosed=True, color=color_dict[color], thickness=2)

# split the loading part and the visualisation part
# maybe irrelevant, just use Dataset ?
def load_images(image_dir, image_grid_size, with_ground_truths=True):
    image_list = glob.glob(os.path.join(image_dir, "*.jpg"))

    tot_nr_images = int(image_grid_size ** 2)
    images = choices(image_list, k=tot_nr_images)

    if with_ground_truths:
        gt_list = [os.path.join(image_dir, "ground_truths", x.split("/")[-1].replace("_bright", "").replace(".jpg", ".json")) for x in images]

    norm_mean=[0.485, 0.456, 0.406]
    norm_std=[0.229, 0.224, 0.225]
    image_array = []
    display_images = []

    for im in images:
        pil_im = Image.open(im)
        display_img = np.ascontiguousarray(np.array(pil_im, dtype=np.uint8))
        display_images.append(display_img)

        im_array = np.array(pil_im.resize((224, 224)), dtype=np.float32) / 255.
        for j in range(3):
            im_array[:, :, j] = (im_array[:, :, j] - norm_mean[j])/norm_std[j]

        im_array = np.transpose(im_array, (2, 0, 1))

        image_array.append(im_array)
    image_array = np.array(image_array)
    if with_ground_truths:
        return display_images, image_array, gt_list
    return display_images, image_array

# probably to be rewritten in the future
def display_grid_with_labels(model, display_images, image_array, gt_list, image_grid_size, epoch=None,
    savefig=False, fig_dir=None, show=True, figsize=9, with_ground_truths=True, model_for_cas=True):

    if savefig and fig_dir is None:
        raise ValueError

    regresor = model
    image_tensor = torch.tensor(image_array)

    print("max value of image tensor that gets passed to the model: ", image_tensor.max())
    regresor.eval()
    with torch.no_grad():
        predictions = regresor(image_tensor)

    fig = plt.figure(figsize=(figsize, figsize))
    grid = ImageGrid(fig, 111, nrows_ncols=(image_grid_size, image_grid_size), axes_pad=0.1)
    image_array = np.transpose(image_array, (0, 2, 3, 1))

    if isinstance(gt_list[0], str):
        gt = [json.load(open(g, "r")) for g in gt_list]
        gt = [[[y, x] for y, x in g["inner_board"]] for g in gt]
    else:
        gt = gt_list

    for k, (ax, im) in enumerate(zip(grid, display_images)):

        im = np.array(im, dtype=np.uint8)
        h, w = im.shape[:2]
        if with_ground_truths and not model_for_cas:
            # print(gt[k])
            labels_to_pixels(im, predictions[k].detach().numpy())
            pred_canonical = predictions[k].detach().numpy()
            pred_canonical = list(zip(pred_canonical[0::2], pred_canonical[1::2]))
            gt_to_pixels(im, gt[k])
            axis_one_error = sum([abs(x[0]-y[0]*h) for x, y in zip(gt[k], pred_canonical)])/4
            axis_two_error = sum([abs(x[1]-y[1]*w) for x, y in zip(gt[k], pred_canonical)])/4
            print(f"Pixel error: axis one {axis_one_error}, axis two {axis_two_error}")
        if with_ground_truths and model_for_cas:

            pred_canonical = cas_to_boxes(predictions[k].detach().tolist())
            pred_canonical = np.array(pred_canonical, dtype=np.int32)
            gt_to_pixels(im, gt[k], color='g')
            gt_to_pixels(im, pred_canonical, color='r')
            gt_canonical = np.roll(gt[k], shift=1, axis=0)[::-1, :] # needs some shifting

            axis_one_error = sum([abs(x[0] - y[0]) for x, y in zip(gt_canonical, pred_canonical)])/4
            axis_two_error = sum([abs(x[1] - y[1]) for x, y in zip(gt_canonical, pred_canonical)])/4
            print(f"Pixel error: axis one {axis_one_error}, axis two {axis_two_error}")

        im = cv2.resize(im, (224, 224))
        ax.imshow(im)
    if savefig:
        fig.savefig(fname=fig_dir)
    if show:
        plt.show()
    plt.close()

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