import glob
import os
from random import choices

import cv2
import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

from src.helpers.tools import cas_to_boxes

color_dict = {'r': (255, 0, 0), 'g': (0, 255, 0), 'b': (0, 0, 255)}


def labels_to_pixels(img, label_set, color='r'):
    h, w = img.shape[:2]
    pixels = [(int(y*w), int(x*h)) for x, y in zip(label_set[0::2], label_set[1::2])]
    pixels = np.array(pixels).reshape((-1,1,2))
    cv2.polylines(img, [pixels], isClosed=True, color=color_dict[color], thickness=2)


def gt_to_pixels(img, label_gt, color='g'):
    img = np.ascontiguousarray(img)
    label_gt = [(row, col) for (col, row) in label_gt]
    pixels = np.array(label_gt).reshape((-1,1,2))
    cv2.polylines(img, [pixels], isClosed=True, color=color_dict[color], thickness=2)


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


def rescale_normalised_bb(h, w, normalised_bb, final_dtype=None):
    if len(normalised_bb) == 8:
        bb_rect_format = list(zip(normalised_bb[0::2], normalised_bb[1::2]))
    else:
        bb_rect_format = normalised_bb
    bb_rect_format = np.array(bb_rect_format)
    bb_rect_format[:, 0] *= h
    bb_rect_format[:, 1] *= w
    if final_dtype is not None:
        return bb_rect_format.astype(final_dtype)
    return bb_rect_format


def display_grid_with_labels(model, image_array, raw_images, gt_list, image_grid_size, epoch=None,
    savefig=False, fig_dir=None, show=True, figsize=9, with_ground_truths=True, model_for_cas=True):

    if savefig and fig_dir is None:
        raise ValueError

    if model_for_cas:
        raise NotImplementedError("conversion for cas not implemented (there are attempts below but it's buggy)")

    regresor = model
    if not isinstance(image_array, torch.FloatTensor):
        image_tensor = torch.tensor(np.array(image_array)) # for some reason torch.tensor doesn't work for list of tensors
    else:
        image_tensor = image_array

    regresor.eval()
    with torch.no_grad():
        predictions = regresor(image_tensor)

    fig = plt.figure(figsize=(figsize, figsize))
    grid = ImageGrid(fig, 111, nrows_ncols=(image_grid_size, image_grid_size), axes_pad=0.1)
    image_array = np.transpose(image_array, (0, 2, 3, 1))

    if isinstance(gt_list[0], str):
        raise ValueError("Function display_grid_with_labels demands that gt_list is a list of suitably formatted normalised labels")

    for gt in gt_list:
        gt = np.array(gt)
        if np.max(gt) > 1.01:
            raise ValueError("Function display_grid_with_labels accepts only normalised ground truths")

    for k, (ax, im, gt, pred) in enumerate(zip(grid, raw_images, gt_list, predictions)):
        im = np.array(im, dtype=np.uint8)
        h, w = im.shape[:2]
        gt_image_scale = rescale_normalised_bb(h, w, gt, np.int32)
        pred_image_scale = rescale_normalised_bb(h, w, pred, np.int32)

        if with_ground_truths and not model_for_cas:
            # print(gt[k])
            gt_to_pixels(im, pred_image_scale, color='g')
            gt_to_pixels(im, gt_image_scale, color='r')
            axis_one_error = np.abs(gt_image_scale[:, 0] - pred_image_scale[:, 0]).sum()/4
            axis_two_error = np.abs(gt_image_scale[:, 1] - pred_image_scale[:, 1]).sum()/4
            print(f"Pixel error: axis one {axis_one_error}, axis two {axis_two_error} on a picture of size {h} x {w}")

        if with_ground_truths and model_for_cas:

            pred_canonical = cas_to_boxes(predictions[k].detach().tolist())
            pred_canonical = np.array(pred_canonical, dtype=np.int32)
            gt_to_pixels(im, gt[k], color='g')
            gt_to_pixels(im, pred_canonical, color='r')
            gt_canonical = np.roll(gt[k], shift=1, axis=0)[::-1, :] # needs some shifting

            axis_one_error = sum([abs(x[0] - y[0]) for x, y in zip(gt_canonical, pred_canonical)])/4
            axis_two_error = sum([abs(x[1] - y[1]) for x, y in zip(gt_canonical, pred_canonical)])/4
            print(f"Pixel error: axis one {axis_one_error}, axis two {axis_two_error}")

        # im = cv2.resize(im, (224, 224))
        ax.imshow(im)
    if savefig:
        fig.savefig(fname=fig_dir)
    if show:
        plt.show()
    plt.close()
