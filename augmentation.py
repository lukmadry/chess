import time
import numpy as np
import os
import tempfile
import cv2
import argparse
import glob

from random import choice, sample, choices
from functools import partial
from PIL import Image

from src.helpers.tools import none_sentinel


# base functions
def apply_rgb_shift(img, max_rgb_shift=3):
	noise = np.random.randint(-max_rgb_shift, max_rgb_shift + 1, size=(1, 1, 3), dtype=np.int16)
	img_rgb = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
	return img_rgb


def apply_hue_shift(img, max_hue_shift=3, max_value_shift=20):
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)
	hsv[:, :, 0] = (hsv[:, :, 0] + np.random.randint(-max_hue_shift, max_hue_shift)) % 180
	hsv[:, :, 1] = np.clip(hsv[:, :, 1] + np.random.randint(-20, 20), 0, 255)

	# if max_value_shift:
	#     hsv[:, :, 2] = np.clip(hsv[:, :, 2] + np.random.randint(-max_value_shift, max_value_shift), 0, 255)
	return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

def apply_value_shift_deterministic(img, value_shift):
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)
	hsv[:, :, 2] = np.clip(hsv[:, :, 2] + value_shift, 0, 255)
	return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

def convert_scale_abs(img, alpha_range=(0.7, 1.3), beta_range=(-30, 30)):
	alph = np.random.uniform(*alpha_range)
	bet = np.random.uniform(*beta_range)
	# if isinstance(img, torch.Tensor):
	# 	out = torch.abs(img.float() * alph + bet)
	# 	return torch.clamp(out, 0, 255).to(torch.uint8)
	# else:
	return cv2.convertScaleAbs(img, alpha=alph, beta=bet)


def apply_rotation(img, angle_range=(-180, 180), scale_range=(0.9, 1.2), 
                   scale=None, return_matrix=False, rotation_center=None):
	none_sentinel(scale_range=scale_range, scale=scale)
    
	angle = np.random.uniform(*angle_range)
    # print(angle)
	h, w = img.shape[:2]
	center = (float(h) / 2.0, float(w) / 2.0) if rotation_center is None else rotation_center
	if scale_range is not None:
		scale = np.random.uniform(*scale_range)
	M = cv2.getRotationMatrix2D(center, angle, scale)
	rot = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
	if return_matrix:
		return rot, M
	return rot


def apply_random_crop(img, fraction_range=(0.6, 0.8)):
	fraction = np.random.uniform(*fraction_range)
	h, w = img.shape[:2]
	shift_h = int(h * fraction)
	shift_w = int(w * fraction)
	start_idx_h = np.random.randint(0, int(h*(1-fraction)))
	start_idx_w = np.random.randint(0, int(w*(1-fraction)))

	crop = img[start_idx_h:start_idx_h + shift_h, start_idx_w:start_idx_w + shift_w]
	return cv2.resize(crop, (h, w))


def add_gaussian_noise(img, std_range=(3, 9)):
	std = np.random.uniform(*std_range)
	noise = np.random.normal(0, std, img.shape).astype(np.float32)
	noisy_img = np.clip(img.astype(np.float32) + noise, 0, 255)
	return noisy_img.astype(np.uint8)


def apply_gaussian_blur(img, ksize_range=(3, 9)):
	# note : kernel size must be odd
	ksize = np.random.choice(range(ksize_range[0], ksize_range[1] + 1, 2))
	return cv2.GaussianBlur(img, (ksize, ksize), 0)


def apply_local_blur(img, max_patches=3, patch_size=50):
	out = img.copy()
	h, w = out.shape[:2]
	for _ in range(np.random.randint(1, max_patches + 1)):
		x = np.random.randint(0, w - patch_size)
		y = np.random.randint(0, h - patch_size)
		patch = out[y:y+patch_size, x:x+patch_size]
		out[y:y+patch_size, x:x+patch_size] = cv2.GaussianBlur(patch, (15, 15), 0)
	return out


def apply_random_cutout(img, max_cutouts=3, size_range=(20, 60)):
	out = img.copy()
	h, w = out.shape[:2]
	for _ in range(np.random.randint(1, max_cutouts + 1)):
		size = np.random.randint(*size_range)
		x = np.random.randint(0, w - size)
		y = np.random.randint(0, h - size)
		out[y:y+size, x:x+size] = np.random.randint(0, 255, size=(3,), dtype=np.uint8)
	return out


def apply_mean_cutout(img, max_cutouts=3, size_range=(20,60), std_perturb=20):
	out = img.copy()
	h, w = out.shape[:2]
	mean_r, mean_g, mean_b = np.mean(img, axis=(0,1))
	if std_perturb > 0:
		mean_r += np.random.normal(scale=std_perturb)
		mean_g += np.random.normal(scale=std_perturb)
		mean_b += np.random.normal(scale=std_perturb)

	mean_r = int(mean_r)
	mean_g = int(mean_g)
	mean_b = int(mean_b)
	
	for _ in range(np.random.randint(1, max_cutouts + 1)):
		size = np.random.randint(*size_range)
		x = np.random.randint(0, w - size)
		y = np.random.randint(0, h - size)
		out[y:y+size, x:x+size] = [mean_r, mean_g, mean_b]
	return out

def generate_soft_blob_shadow(shape, num_blobs=3, max_radius=150, blur_ksize=101):
	h, w = shape[:2]
	mask = np.zeros((h, w), dtype=np.uint8)

	for _ in range(num_blobs):
		# Pick random center and radius
		cx = np.random.randint(0, w)
		cy = np.random.randint(0, h)
		radius = np.random.randint(max_radius // 2, max_radius)
		
		# Draw solid white circle
		cv2.circle(mask, (cx, cy), radius, 255, -1)

	# Blur heavily to create soft radial falloff (kernel size must be odd)
	if blur_ksize % 2 == 0:
		blur_ksize += 1
	soft_mask = cv2.GaussianBlur(mask, (blur_ksize, blur_ksize), 0)

	return soft_mask

def apply_blob_shadows(image, num_blobs=3, max_shadow_strength=0.5):
	"""
	max_shadow_strength: 0.0 (no shadow) to 1.0 (pitch black at center)
	"""
	h, w = image.shape[:2]
	soft_mask = generate_soft_blob_shadow((h, w), num_blobs=num_blobs)

	# Normalize mask to range [0.0, 1.0]
	normalized_mask = soft_mask.astype(np.float32) / 255.0

	# Map shadow intensity: 1.0 = untouched, (1.0 - max_shadow_strength) = darkest point
	shadow_map = 1.0 - (normalized_mask * max_shadow_strength)

	# Expand dims for 3-channel RGB broadcasting
	shadow_map = shadow_map[:, :, np.newaxis]

	# Apply shadow to image
	shadowed_image = (image.astype(np.float32) * shadow_map).clip(0, 255).astype(np.uint8)

	return shadowed_image

def add_synthetic_shadow(img, max_shadows=2, blur_shadows=True, kernel_blur=400, opacity=0.25):
	out = img.copy()
	h, w = out.shape[:2]
	for _ in range(np.random.randint(1, max_shadows + 1)):
		# Create a random convex polygon for the shadow
		pt1 = (np.random.randint(0, w), np.random.randint(0, h))
		pt2 = (np.random.randint(0, w), np.random.randint(0, h))
		pt3 = (np.random.randint(0, w), np.random.randint(0, h))
		pts = np.array([pt1, pt2, pt3], dtype=np.int32)
		
		mask = np.zeros((h, w), dtype=np.uint8)
		cv2.fillConvexPoly(mask, pts, 255)

		if blur_shadows:
			mask = apply_gaussian_blur(mask, ksize_range=(1+kernel_blur // 4,kernel_blur))
			mask = mask.astype(float) / 255.
		
		# Darken region inside mask
		shadow_factor = np.random.uniform(0.4, 0.7)
		out[mask > 0] = np.clip(np.multiply(out[mask > 0], 1-opacity*mask[:, :, None][mask > 0]), 0, 255).astype(np.uint8)
	return out

def add_directional_board_shadow(background, board_mask, offset=(15, 20), blur_ksize=31, opacity=0.3):
	dx, dy = offset
	M = np.float32([[1, 0, dx], [0, 1, dy]])
	shadow_mask = cv2.warpAffine(board_mask, M, (background.shape[1], background.shape[0]))

	if blur_ksize % 2 == 0:
		blur_ksize += 1
	soft_shadow = cv2.GaussianBlur(shadow_mask, (blur_ksize, blur_ksize), 0)

	shadow_factor = 1.0 - (soft_shadow.astype(np.float32) / 255.0) * opacity
	shadow_factor = shadow_factor[:, :, np.newaxis]

	shadowed_bg = (background.astype(np.float32) * shadow_factor).clip(0, 255).astype(np.uint8)
	return shadowed_bg

def add_objects_to_the_image(whole_image, max_obj=4, max_corners=7, max_side_length=100):
	nr_obj = np.random.randint(max_obj)
	h, w = whole_image.shape[:2]
	new_obj = np.zeros_like(whole_image)

	for i in range(max_obj):
		poly = np.zeros_like(whole_image)

		no_corners = np.random.randint(3, max(3, max_corners))
		points = [(np.random.randint(0, h), np.random.randint(0, w))]
		for pt in range(1, no_corners):

			pt_horizontal = np.random.randint(points[pt-1][0]-max_side_length, points[pt-1][0]+max_side_length)
			pt_horizontal = min(h, max(0, pt_horizontal))
			pt_vertical = np.random.randint(points[pt-1][1]-max_side_length, points[pt-1][1]+max_side_length)
			pt_vertical = min(w, max(0, pt_vertical))

			points.append((pt_horizontal, pt_vertical))
							
		cv2.fillConvexPoly(poly, np.array(points, dtype=np.int32), (np.random.randint(255), np.random.randint(255), np.random.randint(255)))
		blur = 2*np.random.randint(7)+1
		poly = cv2.GaussianBlur(poly, (blur, blur), 0)
		new_obj[poly > 0] = poly[poly > 0]

	whole_image_upd = whole_image.copy()
	whole_image_upd[new_obj > 0] = new_obj[new_obj > 0]
	return whole_image_upd
	

raw_image_shifts_fns = [apply_rgb_shift, apply_hue_shift, convert_scale_abs, apply_rotation, apply_random_crop, apply_gaussian_blur]
final_image_shifts_fns = [apply_rgb_shift, apply_hue_shift, convert_scale_abs, apply_gaussian_blur, apply_random_cutout, apply_mean_cutout, apply_blob_shadows, add_gaussian_noise]

raw_image_shifts_names = ["rgb_shift", "hue_shift", "scale_abs", "rotation", "random_crop", "blur"]
final_image_shifts_names = ["rgb_shift", "hue_shift", "scale_abs", "blur", "random_cutout", "mean_cutout", "synth_shadow", "gaussian_noise"]

raw_image_shift_dict = dict(zip(raw_image_shifts_names, raw_image_shifts_fns))
final_image_shifts_dict = dict(zip(final_image_shifts_names, final_image_shifts_fns))

roughening_final_image_shifts_names = ["blur", "synth_shadow", "gaussian_noise"]

# function that prepares images. old version is in the gemini history, you can compare performance one day, for now I take its word for granted
def _prepare_images_for_tiles(
	image_list, 
	images_per_batch, 
	transformation_list, 
	nr_batches=None, 
	transformation_params=None, 
	subsample_transformations=True,
	verbose=True,
):
	"""
	Creates nr_batches lists of images, each containing images_per_batch items.
	Applies a series of transformations to each image in place/per batch.
	"""

	if transformation_params is None:
		transformation_params = {k: {} for k in raw_image_shifts_names}

	# Pre-fetch function pointers & parameters to avoid dictionary lookups in loops
	tf_funcs = [raw_image_shift_dict[tr] for tr in transformation_list]
	tf_kwargs = [transformation_params[tr] for tr in transformation_list]
	num_tfs = len(transformation_list)

	# Directly sample a 2D matrix of indices
	if nr_batches is None:
		batch_indices = np.arange(len(image_list))
		nr_batches = 1
	else:
		batch_indices = np.random.choice(len(image_list), size=(nr_batches, images_per_batch))

	new_images = []
	
	for b in range(nr_batches):
		# Fetch the base images for this batch
		batch_imgs = [image_list[i] for i in batch_indices[b]]
		
		# Determine active transformations for this batch
		if subsample_transformations and num_tfs > 0:
			# Pick k random transformations for this batch
			k = np.random.randint(0, num_tfs + 1)
			# Use zip to pair selected functions with their kwargs
			# (Note: use 'sample' if you want unique transforms, 'choices' if duplicates are allowed)
			active_tfs = sample(list(zip(tf_funcs, tf_kwargs)), k)
		else:
			active_tfs = zip(tf_funcs, tf_kwargs)

		# Apply active transformations across all images in the batch
		for func, kwargs in active_tfs:
			batch_imgs = [func(img, **kwargs) for img in batch_imgs]
			
		new_images += batch_imgs

	return new_images

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--glob_query", type=str, help="glob query that is going to be used to pick images that we are going to transform")
	parser.add_argument("--output_folder", type=str, help="folder where the final batch is going to be dumped")
	parser.add_argument("--augment_for_tiles", action=argparse.BooleanOptionalAction, help="special mode for producing augmentation of tiles.")
	parser.add_argument("--nr_batches", type=int, help="number of images.")
	parser.add_argument("--grid_size", type=int, help="grid size of a chessboard. For each choice we will produce grid_size^2 / 2 images per batch. Only relevant if augment_for_tiles is on.")

	args = parser.parse_args()
	im_list = glob.glob(args.glob_query)
	print(args)

	images_list = []
	for im_name in im_list:
		with Image.open(im_name, "r") as im:
			if im.mode != "RGB":
				im = im.convert("RGB")
			im_arr = np.array(im)
		images_list.append(im_arr)

	if args.augment_for_tiles:
		imgs_per_batch = int(args.grid_size ** 2 / 2)
		new_images = _prepare_images_for_tiles(
			image_list=images_list, 
			nr_batches=args.nr_batches, 
			images_per_batch=imgs_per_batch,
			transformation_list=raw_image_shifts_names
			)
	else:
		raise NotImplementedError

	path = args.output_folder
	os.makedirs(path, exist_ok=True)
	for k, im in enumerate(new_images):
		loc_path = os.path.join(path, f"img_{k}.jpg")
		im = Image.fromarray(im)
		im.save(loc_path)