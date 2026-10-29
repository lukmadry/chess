import datetime
import functools
import glob
import os
from typing import Iterable

import numpy as np


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


def _get_min_max_of_label(lb):
    mmin, mmax = np.inf, -np.inf
    for x, y in lb:
        if mmin < min(x, y):
            mmin = min(x, y)
        if mmax > max(x, y):
            mmax = max(x, y)
    return mmin, mmax


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


def _update_stats_of_gradients(model, grad_dict):
	for name, param in model.named_parameters():
		if param.requires_grad and param.grad is not None:
			grad_mean = param.grad.abs().mean().item()
			grad_dict[name]['mean'].append(grad_mean)

			grad_std = param.grad.abs().std().item()
			grad_dict[name]['std'].append(grad_std)

			grad_max = param.grad.abs().max().item()
			grad_dict[name]['max'].append(grad_max)

			grad_min = param.grad.abs().min().item()
			grad_dict[name]['min'].append(grad_min)

	return grad_dict


def _print_stats_of_gradient(grad_dict):
	for name in list(grad_dict.keys()):
		mean_mean_grad = np.mean(grad_dict[name]['mean'])
		max_mean_grad = np.max(grad_dict[name]['mean'])
		max_max_grad = np.max(grad_dict[name]['max'])

		print(f"Layer: {name}\nMean grad: {mean_mean_grad:.5f}\nMax value: {max_max_grad:.5f}")
