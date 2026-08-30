import pybullet as p
import pybullet_data
import time
import numpy as np
import os
import tempfile
import argparse
import glob
import json
import io
import tempfile

from copy import deepcopy
from random import choice, choices
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import cv2

from helpers import project_world_pixel, get_coordinates_from_half_extents, compute_robust_view_matrix, pan_camera_relative, canonical_corner_order, none_sentinel
from augmentation import roughening_final_image_shifts_names, final_image_shifts_dict, apply_rotation, add_directional_board_shadow, apply_hue_shift, apply_value_shift_deterministic
from augmentation import add_objects_to_the_image

post_processing_fns = [final_image_shifts_dict[fn] for fn in roughening_final_image_shifts_names]
MAX_WORKERS = 4

PIECES = ["pawn", "rook", "knight", "bishop", "queen", "king"]
PIECES_STRATIFIED = []

for loc_piece, nr in zip(PIECES, [8,2,2,2]): # we will always add a king
	PIECES_STRATIFIED += [loc_piece] * nr

scale_param = 0.015

white_rgba = (0.9, 0.9, 0.9, 1.0)
black_rgba = (0.1, 0.1, 0.1, 1.0)

# TODO this file can be massively shortened by putting shared chunks of code together

def create_mesh_piece(mesh_filepath, scale=[scale_param, scale_param, scale_param], color=[0.1, 0.1, 0.1, 1]):
	visual_id = p.createVisualShape(
		shapeType=p.GEOM_MESH,
		fileName=mesh_filepath,
		meshScale=scale,
		rgbaColor=color
	)
	collision_id = p.createCollisionShape(
		shapeType=p.GEOM_CYLINDER,
		fileName=mesh_filepath,
		meshScale=scale
	)
	return visual_id, collision_id


def initialise_shapes_cache(scale_parameter=0.015):

	shape_cache = {'w': {}, 'b': {}}
	mesh_scale = [scale_parameter, scale_parameter, scale_parameter]

	# height_cache = {}

	for loc_piece in PIECES:
		fname = os.path.join(f"chess_models/{loc_piece}.obj")
		col_id = p.createCollisionShape(
			shapeType=p.GEOM_CYLINDER,
			radius=0.015,
			height=0.05
		)

		vis_id_black = p.createVisualShape(
			shapeType=p.GEOM_MESH,
			fileName=fname,
			meshScale=mesh_scale,
			rgbaColor=black_rgba
			)

		vis_id_white = p.createVisualShape(
			shapeType=p.GEOM_MESH,
			fileName=fname,
			meshScale=mesh_scale,
			rgbaColor=white_rgba
			)

		shape_cache['w'][loc_piece] = (vis_id_white, col_id)
		shape_cache['b'][loc_piece] = (vis_id_black, col_id)

	return shape_cache


def place_piece_on_square(board_center, square_size, row, col, height, visual_id, collision_id):
	"""Calculates board coordinates (row, col) and spawns a piece."""
	# Compute relative offset from board center (0,0 is corner)
	x = board_center[0] + (col - 3.5) * square_size
	y = board_center[1] + (row - 3.5) * square_size
	z = board_center[2] + (height / 2.0)  # Offset by half-height so base rests on board
	
	body_id = p.createMultiBody(
		baseMass=0.05,  # Non-zero mass allows physics interaction
		baseCollisionShapeIndex=collision_id,
		baseVisualShapeIndex=visual_id,
		basePosition=[x, y, z]
	)
	return body_id

def postprocess_img(img):
	for func in post_processing_fns:
		img = func(img) 
	return img

def save_frame_and_gt(img, img_path, gt_json, gt_json_path):
	cv2.imwrite(img_path, img)

	with open(gt_json_path, "w") as f:
		json.dump(gt_json, f, default=str)

# this doesn't work super well and leaves white edges around the image, to be reworked or scrapped altogether
def composite_with_soft_edges(fg_board, bg_img, board_mask, blur_ksize=3):
	"""
	fg_board: (H, W, 3) uint8 PyBullet render of the board
	bg_img: (H, W, 3) uint8 background image
	board_mask: (H, W) uint8 binary mask (255 inside board, 0 outside)
	blur_ksize: Odd integer (3 or 5 works best for a subtle 1-2px feather)
	"""
	# 1. Blur the binary mask to create a smooth alpha gradient along the border
	if blur_ksize % 2 == 0:
		blur_ksize += 1
	alpha = cv2.GaussianBlur(board_mask.astype(np.float32), (blur_ksize, blur_ksize), 0) / 255.0
	alpha = alpha[:, :, np.newaxis] # Expand for RGB channels

	# 2. Linear interpolation (lerp) between background and foreground
	composite = (fg_board.astype(np.float32) * alpha) + (bg_img.astype(np.float32) * (1.0 - alpha))
	
	return composite.clip(0, 255).astype(np.uint8)


def introduce_pieces_ids_on_board(tile_size, board_center, frame_thickness, return_row_col=False):
	global shape_cache

	N_pieces_white = choice(range(4, 16))
	N_pieces_black = choice(range(4, 16))
	upright_orientation = p.getQuaternionFromEuler([np.radians(90), 0., 0.])

	pieces_on_board_white = choices(PIECES_STRATIFIED, k=N_pieces_white) + ['king']
	pieces_on_board_black = choices(PIECES_STRATIFIED, k=N_pieces_black) + ['king']
	pieces_on_board = pieces_on_board_white + pieces_on_board_black
	N = N_pieces_black + N_pieces_white + 2 # +2 because kings

	occupied_spots = choices([(x,y) for x in range(8) for y in range(8)], k=N)
	ids_of_pieces_on_board = []
	piece_names = []
	for j_piece, (row, col) in enumerate(occupied_spots):
		loc_color = 'w' if j_piece < N_pieces_white + 1 else 'b'
		x = board_center[0] + (col - 3.5) * tile_size
		y = board_center[1] + (row - 3.5) * tile_size
		z = board_center[2] + (frame_thickness / 2.0)

		loc_vis_id, loc_col_id = shape_cache[loc_color][pieces_on_board[j_piece]]
		piece_id = p.createMultiBody(
			baseMass=0.5,
			baseVisualShapeIndex=loc_vis_id,
			baseCollisionShapeIndex=loc_col_id,
			basePosition=[x,y,z],
			baseOrientation=upright_orientation,
			)
		ids_of_pieces_on_board.append(piece_id)
		piece_names.append(pieces_on_board[j_piece])

	if return_row_col:
		return ids_of_pieces_on_board, piece_names, occupied_spots
	return ids_of_pieces_on_board, piece_names

def create_chessboard_images(nr_chessboards, nr_permutations, bg_images, white_file_list, dark_file_list, output_dir,
	tile_size=0.05, nr_rows=4, img_h=480, img_w=640, verbose=1, panning_range=(0, 0.1), overwrite=False, yaw_range=(-90, 90),
	random_positions=20, dist_range=(0.6, 1.0), border_margin_range=(0.1, 0.75)):

	# physics_client = p.connect(p.DIRECT)
	p.setAdditionalSearchPath(pybullet_data.getDataPath())

	batch_outputs = []
	p.resetSimulation()

	proj_mat = p.computeProjectionMatrixFOV(fov=60, aspect=float(img_w)/img_h, nearVal=0.1, farVal=3.0)
	proj_mat_arr = np.array(proj_mat).reshape(4,4).T

	frame_thickness = 0.02

	# TODO create textures of all images in the file list named white_batch, dark_batch
	textures_white = [p.loadTexture(x) for x in white_file_list]
	textures_dark = [p.loadTexture(x) for x in dark_file_list]

	os.makedirs(output_dir, exist_ok=True)
	output_dir_data = os.path.join(output_dir, "ground_truths")
	os.makedirs(output_dir_data, exist_ok=True)

	# load background for later
	if isinstance(bg_images[0], str):
		bg_images_list = deepcopy(bg_images)
		bg_images_textures = [p.loadTexture(b) for b in bg_images_list]
		bg_img = [cv2.cvtColor(cv2.imread(x), cv2.COLOR_BGR2RGB) for x in bg_images] # not sure if this is correct
		bg_images = [cv2.resize(x, (img_w, img_h), interpolation=cv2.INTER_AREA) for x in bg_img]
	else:
		raise NotImplementedError # make sure that they are of correct size if not rescale them

	colors = [[0.9, 0.9, 0.8, 1.0], [0.2, 0.2, 0.2, 1.0]]
	nr_cols = nr_rows

	progress = 0.
	divisor = 1+nr_chessboards // 10

	start_idx = 0
	# if not overwrite:
	#   print("chose not to overwrite pictures")
	#   start_idx = max([int(x.split("_")[1]) for x in os.listdir(output_dir)]) + 1

	# shape_cache = initialise_shapes_cache(scale_param)
	board_center = [0., 0., 0.]

	with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:

		for k in range(start_idx, start_idx + nr_chessboards):

			target_pos = [0.0, 0.0, 0.0]
			white_batch = choices(textures_white, k=int(nr_rows ** 2 / 2))
			dark_batch = choices(textures_dark, k=int(nr_rows ** 2 / 2))

			grid_size = tile_size * nr_rows
			tiles = []

			progress = k / float(nr_chessboards)
			if verbose and k % divisor == 0:
				print(f"Progress made: {100*progress:.3f}%", end="\r")

			for row in range(nr_rows):
				tiles.append([])
				for col in range(nr_rows):
					vis_id = p.createVisualShape(
						p.GEOM_BOX, 
						halfExtents=[tile_size / 2., tile_size / 2., 0.002], 
						rgbaColor=colors[(row + col) % 2]
						)
					pos = [(col - (nr_cols - 1) / 2. ) * tile_size, (row - (nr_rows - 1) / 2.) * tile_size, frame_thickness / 2]
					tile_body = p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis_id, basePosition=pos)
					tiles[row].append(tile_body)

					is_white = ((row + col) % 2 == 0)
					txtr = choice(textures_white) if is_white else choice(textures_dark)
					# txtr = p.loadTexture(txtr_file)
					p.changeVisualShape(objectUniqueId=tile_body, linkIndex=-1, textureUniqueId=txtr) 

			for i in range(nr_permutations):
				shadow_x_direction = 2*(np.random.uniform() > 0.5 - 1) * 10
				shadow_y_direction = 2*(np.random.uniform() > 0.5 - 1) * 10

				dist = np.random.uniform(*dist_range)
				yaw = np.random.uniform(*yaw_range)
				pitch = -89.5
				shift_x = np.random.uniform(*panning_range)
				shift_y = np.random.uniform(*panning_range)
				target_pos = pan_camera_relative(shift_x, shift_y, target_pos, yaw)

				border_margin = np.random.uniform(*border_margin_range) * tile_size
				frame_width = grid_size + (2 * border_margin)

				inner_board_half_extents = [grid_size / 2., grid_size / 2., frame_thickness / 2.]
				frame_half_extents = [frame_width / 2., frame_width / 2., frame_thickness / 2.]

				inner_board_coordinates = get_coordinates_from_half_extents(inner_board_half_extents)
				frame_coordinates = get_coordinates_from_half_extents(frame_half_extents)

				texture_for_frame = choice(bg_images)

				p.resetDebugVisualizerCamera(
					cameraDistance=dist,
					cameraYaw=yaw,
					cameraPitch=pitch if abs(pitch+90) > 1 else -89,
					cameraTargetPosition=target_pos
				)
				view_mat = compute_robust_view_matrix(target_pos, dist, yaw, pitch)
				view_mat_arr = np.array(view_mat).reshape(4, 4).T

				frame_vis = p.createVisualShape(
					shapeType=p.GEOM_BOX,
					halfExtents=[frame_width / 2., frame_width / 2., frame_thickness / 2],
					rgbaColor=[0.5, 0.4, 0.35, 1.0])

				frame_col = p.createCollisionShape(
					shapeType=p.GEOM_BOX,
					halfExtents=[frame_width / 2., frame_width / 2., frame_thickness / 2.]
					)

				frame_id = p.createMultiBody(
					baseMass=0.0,
					baseCollisionShapeIndex=frame_col,
					baseVisualShapeIndex=frame_vis,
					basePosition=[0.0, 0.0, 0]
					)

				# TODO to be tested if this didn't break anything
				ids_of_pieces_on_board, _ = introduce_pieces_ids_on_board(tile_size=tile_size, board_center=board_center, frame_thickness=frame_thickness)

				frame_texture = choice(bg_images_textures)
				p.changeVisualShape(objectUniqueId=frame_id, linkIndex=-1, specularColor=[0.4, 0.4, 0.4], textureUniqueId=frame_texture)
				# TODO relegate this part to a separate function
				width, height, rgb_pixels, depth_pixels, segmentation_mask = p.getCameraImage(
					img_w, img_h, 
					view_mat, proj_mat, 
					lightDirection=[1.0, 1.5, 0.3],     # Direction of light source
					lightColor=[1.0, 0.98, 0.92],        # Warm white light color
					lightDistance=2.5,
					lightAmbientCoeff=0.35,              # <-- BOOST THIS (e.g. 0.35 -> 0.50) to lighten dark areas
					lightDiffuseCoeff=0.65,              # Softens direct harsh highlights
					lightSpecularCoeff=0.2,             # Regulates surface shine/varnish
					shadow=1,
					renderer=p.ER_BULLET_HARDWARE_OPENGL
				)

				rgb_array = np.reshape(rgb_pixels, (height, width, 4)).astype(np.uint8)[:, :, :3]
				# print("height and width of camera image: ", height, width)
				# bgr_img = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)

				seg_mask = np.array(segmentation_mask, dtype=np.int32).reshape((img_h, img_w))
				board_mask = (seg_mask >= 0).astype(np.uint8)

				bg = choice(bg_images) # we should rotate and/or zoom it ! maybe we can also change color
				bg = apply_rotation(bg)
				bg = apply_hue_shift(bg)
				bg = add_directional_board_shadow(bg, board_mask * 255, offset=(0,0), opacity=0.65, blur_ksize=45)
				bg = add_directional_board_shadow(bg, board_mask * 255, offset=(shadow_x_direction, shadow_y_direction), opacity=0.4, blur_ksize=12)

				# final_img = composite_with_soft_edges(rgb_array, bg, board_mask * 255, blur_ksize=11)

				# print((board_mask[:, :, None]).shape, bgr_img.shape, bg_resized.shape)
				final_img = np.where(board_mask[:, :, None] == 1, rgb_array, bg)

				# computing positions of pixels         
				inner_board_projections = [project_world_pixel(x, proj_mat_arr, view_mat_arr, img_h, img_w) for x in inner_board_coordinates]

				final_img = cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR)
				final_img = postprocess_img(final_img)
				final_img_bright = apply_value_shift_deterministic(final_img, 20)

				ground_truths = {
					"inner_board": inner_board_projections,
					"tile_size": tile_size,
					"border_margin": border_margin,
					"yaw": yaw,
					"pitch": pitch,
					"target_position": target_pos,
				}

				executor.submit(save_frame_and_gt, 
					img=final_img.copy(), 
					img_path=os.path.join(output_dir, f"chessboard_{k}_{i}.jpg"), 
					gt_json=ground_truths, 
					gt_json_path=os.path.join(output_dir_data, f"chessboard_{k}_{i}.json"))

				executor.submit(cv2.imwrite,
					os.path.join(output_dir, f"chessboard_{k}_{i}_bright.jpg"),
					final_img_bright.copy())

				p.removeBody(frame_id)
				target_pos = [0.0, 0.0, 0.0]

				for idx in ids_of_pieces_on_board:
					p.removeBody(idx)

			for row in range(nr_rows):
				for col in range(nr_cols):
					p.removeBody(tiles[row][col])

def paint_shadows(display_image, occupied_spots, tile_size, init_shift_x, init_shift_y, shadow_angle=30, opacity=0.2):
	"""
	takes the original image, spots describing the location of chess pieces and attaches shadows

	"""

	start_angle = 0
	end_angle = 180
	axes = (40, 15) # to modify depending on the figure
	shadow_base = np.zeros_like(display_image)
	color = (255, 255, 255)
	for (col, row) in occupied_spots:
		# code for Gaussian blob goes here
		pos_x = int(init_shift_x + tile_size * (8-col))
		pos_y = int(init_shift_y + tile_size * (8-row))

		cv2.ellipse(shadow_base, (pos_x, pos_y), axes, shadow_angle, start_angle, end_angle, color, -1)

	shadow_soft = cv2.GaussianBlur(shadow_base, (15, 15), 0)
	shadow_soft = shadow_soft / 255.
	mask = shadow_soft > 0
	fin_img = display_image.copy()
	fin_img[mask] = fin_img[mask] * ( 1 - opacity * shadow_soft[mask] )
	return np.array(fin_img, dtype=np.uint8)


def create_segmentation_mask(corners, height=None, width=None, image=None):
	# H, W = image.shape[:2]
	if height is None or width is None:
		height, width = image.shape[:2]
	mask = np.zeros((height, width), dtype=np.uint8)

	pts = np.array(corners).astype(np.int32).reshape((-1, 1, 2))
	cv2.fillpoly(mask, pts, color=1)
	return mask

# TODO I think it may turn out to be a bit useless...
def load_cropped_texture(image_filepath, corners=None, corner_filepath=None, 
	reorient_corners=False, height_axis=1, target_size=(480, 640)):

	none_sentinel(corners=corners, corner_filepath=corner_filepath)
	# will check if cropped variant exists, if not then will crop and load

	basedir = os.path.dirname(image_filepath)
	img_file = image_filepath.split("/")[-1]
	cropped_img_file = img_file.replace(".jpg", "_cropped.jpg")
	cropped_img_path = os.path.join(basedir, cropped_img_file)
	if not os.path.exists(cropped_img_path):
		image = Image.open(image_filepath)

		if corner_filepath is not None:
			with open(corner_filepath, "r") as f:
				corners = json.load(f)['inner_board']

		if reorient_corners:
			corners = canonical_corner_order(corners)

		width_axis = 1 - height_axis

		print(corners)
		top = int(corners[0][height_axis] / 2.0 + corners[1][height_axis] / 2.0) # or change to max?
		bottom = int(corners[2][height_axis] / 2.0 + corners[2][height_axis] / 2.0)

		left = int(corners[0][width_axis] / 2.0 + corners[3][width_axis] / 2.0)
		right = int(corners[1][width_axis] / 2.0 + corners[2][width_axis] / 2.0)

		# turn into a cropbox
		img_cropped = image.crop((left, top, right, bottom))

		# dump cropped image
		img_cropped.save(os.path.join(basedir, cropped_img_file), format="PNG")
	else:
		img_cropped = Image.open(cropped_img_path)

	img_cropped = np.array(img_cropped)
	return cv2.resize(img_cropped, target_size)


def create_one_halfsynth(rim_to_tile, 
	center=[0., 0.], tile_size=0.05, frame_thickness=0):

	half_extent = 4 * tile_size + rim_to_tile * tile_size

	# they should be equal if all goes well

	visual_id = p.createVisualShape(
		shapeType=p.GEOM_BOX,
		halfExtents=[half_extent, half_extent, 0.],
		rgbaColor=[0,1,0,1], specularColor=[0, 0, 0])

	body_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=visual_id, basePosition=center+[0.])
	# p.changeVisualShape(body_id, -1, textureUniqueId=chessboard_txture)

	ids_of_pieces_on_board, names_pieces_on_board, occupied_spots = introduce_pieces_ids_on_board(tile_size, center + [frame_thickness], 
		frame_thickness, 
		return_row_col=True)

	return ids_of_pieces_on_board, names_pieces_on_board, body_id, occupied_spots


def _rotate_coords(cb_coords, rot_matrix):
	raw_pts = np.array(cb_coords, dtype=np.float32)
	# swaps are here due to opencv coordinates ordering
	pts_xy = raw_pts[:, [1, 0]] if raw_pts.shape[1] == 2 else raw_pts

	pts_homog = np.hstack([pts_xy, np.ones((len(pts_xy), 1), dtype=np.float32)])
	new_pts_xy = pts_homog @ rot_matrix.T
	new_cb = new_pts_xy[:, [1, 0]]
	return new_cb

def rotate_and_crop(img, playable_cb, whole_cb, angle_range=(-5, 5), max_attempts=10):
	h, w = img.shape[:2]
	waiting_for_correct_rotation = True
	k = 0
	while waiting_for_correct_rotation:
		rotated_im, rot_matrix = apply_rotation(img, angle_range=angle_range, scale=1.0, 
												scale_range=None, return_matrix=True)

		new_whole_cb = _rotate_coords(whole_cb, rot_matrix)
		new_playable_cb = _rotate_coords(playable_cb, rot_matrix)

		min_horizontal = new_whole_cb[:, 0].min()
		max_horizontal = new_whole_cb[:, 0].max()
		min_vertical = new_whole_cb[:, 1].min()
		max_vertical = new_whole_cb[:, 1].max()

		if min_horizontal < 0 or max_horizontal > h or min_vertical < 0 or max_vertical > w:
			waiting_for_correct_rotation = True
		else:
			waiting_for_correct_rotation = False

		k += 1

		if k > max_attempts:
			print(f"tried finding a good rotation {max_attempts} times, will return unmodified image")
			return img, whole_cb, playable_cb

	new_min_horizontal = np.random.randint(int(min_horizontal/2))
	new_max_horizontal = np.random.randint(max_horizontal + int((h-max_horizontal)/2), h)
	new_min_vertical = np.random.randint(int(min_vertical/2))
	new_max_vertical = np.random.randint(max_vertical + int((w-max_vertical)/2), w)

	new_whole_cb[:, 0] -= new_min_horizontal
	new_whole_cb[:, 1] -= new_min_vertical

	return rotated_im[new_min_horizontal:new_max_horizontal, new_min_vertical:new_max_vertical], new_whole_cb.astype(np.int32), new_playable_cb.astype(np.int32)

# TODO add permutations ?
def create_half_synth_chessboard_images(nr_chessboards, bg_images,
	chessboard_images_dir, output_dir, playable_folder="playable_chessboard", whole_folder="whole_chessboard", target_w=640, target_h=480,
	dist_range=(0.4, 1.0), yaw_range=(-5,5), pitch=-89.5, shadow_angle=30):

	chessboard_images_filelist = glob.glob(os.path.join(chessboard_images_dir, "*.jpg")) # note that we are only considering non-cropped versions !
	chessboard_images_filelist = [x for x in chessboard_images_filelist if not x.endswith("_cropped.jpg")]
	whole_cb_files = [os.path.join(chessboard_images_dir, whole_folder, x.split("/")[-1].replace(".jpg", ".json")) for x in chessboard_images_filelist]
	playable_cb_files = [os.path.join(chessboard_images_dir, playable_folder, x.split("/")[-1].replace(".jpg", ".json")) for x in chessboard_images_filelist]

	chessboard_images_textures = []
	for img_filepath, whole_crn_fpath in zip(chessboard_images_filelist, whole_cb_files):
		cropped_texture = load_cropped_texture(
			image_filepath=img_filepath, corner_filepath=whole_crn_fpath, height_axis=0)
		chessboard_images_textures.append(cropped_texture)

	N_cb = len(whole_cb_files)

	# p.resetSimulation()

	img_w = int( 1.2 * target_w )
	img_h = int( 1.2 * target_h )

	proj_mat = p.computeProjectionMatrixFOV(fov=60, aspect=float(img_w)/img_h, nearVal=0.1, farVal=3.0)
	proj_mat_arr = np.array(proj_mat).reshape(4,4).T

	target_pos = [0., 0., 0.]

	bg_images_arrs = []
	for fname in bg_images:
		loc_bg = Image.open(fname)
		loc_bg = np.array(loc_bg)
		loc_bg = cv2.resize(loc_bg, (img_w, img_h), interpolation=cv2.INTER_AREA)
		bg_images_arrs.append(loc_bg)

	os.makedirs(os.path.join(output_dir, "ground_truths"), exist_ok=True)

	for i in range(nr_chessboards):

		shadow_angle = np.random.uniform(10, 180)

		chessboard_idx = choice(range(N_cb))
		with open(whole_cb_files[chessboard_idx], "r") as f:
			whole_cb = json.load(f)['inner_board']
			whole_cb = canonical_corner_order(whole_cb)

		with open(playable_cb_files[chessboard_idx], "r") as f:
			playable_cb = json.load(f)['inner_board']
			playable_cb = canonical_corner_order(playable_cb)

		whole_cb_width = whole_cb[1][1] - whole_cb[0][1] # TODO to be double checked
		playable_cb_width = playable_cb[1][1] - playable_cb[0][1]

		outer_rim = (whole_cb_width - playable_cb_width) / 2.
		tile_size = playable_cb_width / 8.
		rim_to_tile = outer_rim / tile_size

		chessboard_texture_filepath = chessboard_images_filelist[chessboard_idx]
		chessboard_img_cropped = load_cropped_texture(chessboard_texture_filepath, corners=whole_cb, target_size=(img_w, img_h))

		bg_img = choice(bg_images_arrs)

		ids_pieces, names_pieces, body_id, occupied_spots = create_one_halfsynth(rim_to_tile)

		print("body_id: ", body_id)
		dist = np.random.uniform(*dist_range)
		yaw = 0

		# this part could be relegated to a separate function
		p.resetDebugVisualizerCamera(
			cameraDistance=dist,
			cameraYaw=yaw,
			cameraPitch=pitch if abs(pitch+90) > 1 else -89,
			cameraTargetPosition=target_pos
		)
		view_mat = compute_robust_view_matrix(target_pos, dist, yaw, pitch)
		view_mat_arr = np.array(view_mat).reshape(4, 4).T

		width, height, rgb_pixels, depth_pixels, segmentation_mask = p.getCameraImage(
			img_w, img_h, 
			view_mat, proj_mat, 
			lightDirection=[1.0, 1.5, 0.3],     # Direction of light source
			lightColor=[1.0, 0.98, 0.92],        # Warm white light color
			lightDistance=2.5,
			lightAmbientCoeff=0.35,              # <-- BOOST THIS (e.g. 0.35 -> 0.50) to lighten dark areas
			lightDiffuseCoeff=0.65,              # Softens direct harsh highlights
			lightSpecularCoeff=0.2,             # Regulates surface shine/varnish
			shadow=0,
			renderer=p.ER_BULLET_HARDWARE_OPENGL
		)

		rgb_array = np.reshape(rgb_pixels, (height, width, 4)).astype(np.uint8)[:, :, :3]
		# print("height and width of camera image: ", height, width)
		# bgr_img = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)

		# identify where the green part with a chessboard actually is
		seg_mask = np.array(segmentation_mask, dtype=np.int32).reshape((img_h, img_w))
		green_mask = (seg_mask == body_id)
		# print(segmentation_mask.shape)
		board_mask = (seg_mask >= 0).astype(np.uint8)

		y_indices, x_indices = np.where(board_mask)

		if len(y_indices) > 0 and len(x_indices) > 0:
			y_min, y_max = y_indices.min(), y_indices.max()
			x_min, x_max = x_indices.min(), x_indices.max()

			target_width = x_max - x_min + 1
			target_height = y_max - y_min + 1

			resized_cb = cv2.resize(chessboard_img_cropped, (target_height, target_width))
			full_canvas = np.zeros_like(rgb_array)
			full_canvas[y_min:y_max+1, x_min:x_max+1] = np.transpose(resized_cb, (1,0,2))

			rgb_array[green_mask] = full_canvas[green_mask]

			tile_width_rescaled = target_width / (8 + 2 * rim_to_tile)
			rim_width_rescaled = rim_to_tile * tile_width_rescaled

			tile_height_rescaled = target_height / (8 + 2 * rim_to_tile)
			rim_height_rescaled = rim_to_tile * tile_height_rescaled

			playable_cb_rescaled = [
				[y_min+rim_height_rescaled, x_min + rim_width_rescaled], 
				[y_min+rim_height_rescaled, x_min + rim_width_rescaled + 8 * tile_width_rescaled],
				[y_min + rim_height_rescaled + 8 * tile_height_rescaled, x_min + rim_width_rescaled],
				[y_min + rim_height_rescaled + 8 * tile_height_rescaled, x_min+  rim_width_rescaled + 8 * tile_width_rescaled]
			]
			playable_cb_rescaled = canonical_corner_order(playable_cb_rescaled)
			playable_cb_rescaled = [[int(x), int(y)] for x, y in playable_cb_rescaled]

			whole_cb_rescaled = [
				[y_min, x_min], 
				[y_min, x_min + 2* rim_width_rescaled + 8 * tile_width_rescaled],
				[y_min + 2* rim_height_rescaled + 8 * tile_height_rescaled, x_min],
				[y_min + 2*rim_height_rescaled + 8 * tile_height_rescaled, x_min + 2 *rim_width_rescaled + 8 * tile_width_rescaled]
			]

			whole_cb_rescaled = canonical_corner_order(whole_cb_rescaled)
			whole_cb_rescaled = [[int(x), int(y)] for x, y in whole_cb_rescaled]
		else:
			raise ValueError("board_mask after getCameraImage is empty")

		# rgb_array = np.where(green_mask[:, :, None], resized_cb, rgb_array)

		shadow_x_direction = 2*(np.random.uniform() > 0.5 - 1) * 10
		shadow_y_direction = 2*(np.random.uniform() > 0.5 - 1) * 10

		bg_img = choice(bg_images_arrs).copy()
		bg_img = add_objects_to_the_image(bg_img)
		bg_img = add_directional_board_shadow(bg_img, board_mask * 255, offset=(0,0), opacity=0.65, blur_ksize=45)
		bg_img = add_directional_board_shadow(bg_img, board_mask * 255, offset=(shadow_x_direction, shadow_y_direction), opacity=0.4, blur_ksize=12)

		final_img = np.where(board_mask[:, :, None] == 1, rgb_array, bg_img)
		final_img = paint_shadows(final_img, occupied_spots, tile_width_rescaled, x_min, y_min, shadow_angle=shadow_angle, opacity=0.4)
		final_img = cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR)

		final_img, whole_cb_rescaled, playable_cb_rescaled = rotate_and_crop(final_img, 
			playable_cb=playable_cb_rescaled, 
			whole_cb=whole_cb_rescaled, 
			angle_range=yaw_range)

		# final_img, playable_cb_rescaled = crop_and_rotate(final_img, playable_cb_rescaled, whole_cb_rescaled)

		fname = os.path.join(output_dir, f"chessboard_synth_{i}.jpg")
		cv2.imwrite(fname, final_img)

		with open(os.path.join(output_dir, "ground_truths", f"chessboard_synth_{i}.json"), "w") as f:
			# print(playable_cb_rescaled)
			playable_cb_rescaled = playable_cb_rescaled.tolist()
			json.dump({"inner_board" :playable_cb_rescaled}, f)

		with open(os.path.join(output_dir, "ground_truths", f"chessboard_synth_{i}_whole_cb.json"), "w") as f:
			# print(whole_cb_rescaled)
			whole_cb_rescaled = whole_cb_rescaled.tolist()
			json.dump({"inner_board": whole_cb_rescaled}, f)

		p.removeBody(body_id)
		for ids in ids_pieces:
			p.removeBody(ids)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--nr_chessboards", type=int)
	parser.add_argument("--half_synth", action=argparse.BooleanOptionalAction)
	parser.add_argument("--white_tiles", type=str, default=os.path.join("temp", "white_augment"))
	parser.add_argument("--dark_tiles", type=str, default=os.path.join("temp", "brown_augment"))
	parser.add_argument("--bg_folder", type=str, default=os.path.join("wood_frame"))
	parser.add_argument("--full_cbs_folder", type=str, default="empty_chessboards")
	parser.add_argument("--nr_permutations", type=int, default=5)
	parser.add_argument("--output_dir", type=str, required=True)
	parser.add_argument("--nr_rows", type=int, default=8)
	parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction)
	parser.add_argument("--yaw_range", nargs=2, type=int, default=(-90,90))

	parser.add_argument("--test_run", action=argparse.BooleanOptionalAction)
	parser.add_argument("--test_run_marker", type=str)

	args = parser.parse_args()

	if args.test_run and args.test_run_marker is None:
		raise ValueError("To do a test run you need to give test_run_marker, otherwise the script will not be able to check if it passed correctly.")

	white_tiles = glob.glob(os.path.join(args.white_tiles, "*.jpg"))
	dark_tiles = glob.glob(os.path.join(args.dark_tiles, "*.jpg"))
	bg_imgs = glob.glob(os.path.join(args.bg_folder, "*.jpg"))
	nr_chessboards = args.nr_chessboards
	nr_permutations = args.nr_permutations
	output_dir = args.output_dir

	physics_client = p.connect(p.DIRECT)

	global shape_cache
	shape_cache = initialise_shapes_cache()

	os.makedirs(output_dir, exist_ok=True)

	if args.half_synth:
		create_half_synth_chessboard_images(
			nr_chessboards=nr_chessboards,
			bg_images=bg_imgs,
			output_dir=args.output_dir,
			chessboard_images_dir=args.full_cbs_folder,
			yaw_range=args.yaw_range)   
	else:
		create_chessboard_images(
			nr_chessboards=nr_chessboards, 
			nr_permutations=nr_permutations, 
			bg_images=bg_imgs, 
			white_file_list=white_tiles, 
			dark_file_list=dark_tiles, 
			output_dir=args.output_dir,
			nr_rows=args.nr_rows,
			overwrite=args.overwrite,
			yaw_range=args.yaw_range,
			)

	if args.test_run:
		with open(args.test_run_marker, "w") as f:
			f.write("SUCCESS")