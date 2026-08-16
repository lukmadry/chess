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

from copy import deepcopy
from random import choice, choices
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import cv2

from helpers import project_world_pixel, get_coordinates_from_half_extents, compute_robust_view_matrix, pan_camera_relative, canonical_corner_order
from augmentation import roughening_final_image_shifts_names, final_image_shifts_dict, apply_rotation, add_directional_board_shadow, apply_hue_shift, apply_value_shift_deterministic

post_processing_fns = [final_image_shifts_dict[fn] for fn in roughening_final_image_shifts_names]
MAX_WORKERS = 4

PIECES = ["pawn", "rook", "knight", "bishop", "queen", "king"]
PIECES_STRATIFIED = []

for loc_piece, nr in zip(PIECES, [8,2,2,2]): # we will always add a king
	PIECES_STRATIFIED += [loc_piece] * nr

scale_param = 0.015

white_rgba = (0.9, 0.9, 0.9, 1.0)
black_rgba = (0.1, 0.1, 0.1, 1.0)

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


def introduce_pieces_ids_on_board(tile_size, board_center, frame_thickness):
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

	return ids_of_pieces_on_board, piece_names

def create_chessboard_images(nr_chessboards, nr_permutations, bg_images, white_file_list, dark_file_list, output_dir,
	tile_size=0.05, nr_rows=4, img_h=480, img_w=640, verbose=1, panning_range=(0, 0.1), overwrite=False, yaw_range=(-90, 90),
	random_positions=20, dist_range=(0.6, 1.0), border_margin_range=(0.1, 0.75)):

	physics_client = p.connect(p.DIRECT)
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
	# 	print("chose not to overwrite pictures")
	# 	start_idx = max([int(x.split("_")[1]) for x in os.listdir(output_dir)]) + 1

	shape_cache = initialise_shapes_cache(scale_param)
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


				ids_of_pieces_on_board, _ = introduce_pieces_ids_on_board(tile_size=tile_size, board_center=board_center, frame_thickness=frame_thickness)

				frame_texture = choice(bg_images_textures)
				p.changeVisualShape(objectUniqueId=frame_id, linkIndex=-1, specularColor=[0.4, 0.4, 0.4], textureUniqueId=frame_texture)

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

def paint_shadows(display_image, ids_of_pieces):

	for piece_id in ids_of_pieces:
		base_position, _ = p.getBasePositionAndOrientation(piece_id)

		# code for Gaussian blob goes here

def rescale_corners(corners, orig_image, reference_image_shape):
	pass

def create_segmentation_mask(corners, height=None, width=None, image=None):
	# H, W = image.shape[:2]
	if height is None or width is None:
		height, width = image.shape[:2]
	mask = np.zeros((height, width), dtype=np.uint8)

	pts = np.array(corners).astype(np.int32).reshape((-1, 1, 2))
	cv2.fillpoly(mask, pts, color=1)
	return mask

def load_cropped_texture(image_filepath, corner_filepath, reorient_corners=False, height_axis=1):

	image = Image.open(image_filepath)
	with open(corner_filepath, "r") as f:
		corners = json.load(f)['inner_board']

	if reorient_corners:
		corners = canonical_corner_order(corners)

	width_axis = 1 - height_axis

	top = int(corners[0][height_axis] / 2.0 + corners[1][height_axis] / 2.0) # or change to max?
	bottom = int(corners[2][height_axis] / 2.0 + corners[2][height_axis] / 2.0)

	left = int(corners[0][width_axis] / 2.0 + corners[3][width_axis] / 2.0)
	right = int(corners[1][width_axis] / 2.0 + corners[2][width_axis] / 2.0)

	# turn into a cropbox
	img_cropped = image.crop((left, top, right, bottom))
	buffer = io.BytesIO()
	img_cropped.save(buffer, format="PNG")

	texture_id = p.loadTexture(buffer.getvalue())
	return texture_id


def create_half_synthetic_chessboard(chessboard_image, playable_chessboard_location, whole_chessboard_location,
	background_image, nr_pieces_variants, target_size, frame_thickness=0):
	"""
	chessboard location - a list of corners of a chessboard in a format [top-left, top-right, bottom-right, bottom-left]
	"""

	tile_size_x = np.mean([playable_chessboard_location[1][0] - playable_chessboard_location[0][0], playable_chessboard_location[3][0] - playable_chessboard_location[2][0]])/8
	tile_size_y = np.mean([playable_chessboard_location[2][1] - playable_chessboard_location[1][1], playable_chessboard_location[3][1] - playable_chessboard_location[0][1]])/8

	center = np.mean(playable_chessboard_location, axis=0).tolist()

	# they should be equal if all goes well
	square_size = ??
	half_extent = square_size/2.0
	visual_id = p.createVisualShape(
		shapeType=p.GEOM_BOX,
		halfExtents=[half_extent, half_extent, 0.001],
		rgbaColor=[1,1,1,1])


	# shadows go here
	ids_of_pieces_on_board, names_pieces_on_board = introduce_pieces_ids_on_board((tile_size_x + tile_size_y) / 2, center + [frame_thickness], frame_thickness)

def create_half_synth_chessboard_images(nr_chessboards, nr_permutations, 
	bg_images, chessboard_images_dir, playable_folder="playable_chessboard", whole_folder="whole_chessboard"):

	chessboard_images_filelist = glob.glob(os.path.join(chessboard_images_dir, "*.jpg"))
	chessboard_images_filelist.sort()

	corners_chessboards_filelist = [os.path.join(chessboard_images_dir, whole_folder, x.split("/")[-1].replace(".jpg", ".json")) for x in chessboard_images_filelist]

	chessboard_images_textures = []
	for img_filepath, crn_fpath in zip(chessboard_images_filelist, corners_chessboards_filelist):
		cropped_texture = load_cropped_texture(
			image_filepath=filepath, corner_filepath=crn_fpath)
		chessboard_images_textures.append(cropped_texture)

	for i in range(nr_chessboards):


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--nr_chessboards", type=int)
	parser.add_argument("--white_tiles", type=str, default=os.path.join("temp", "white_augment"))
	parser.add_argument("--dark_tiles", type=str, default=os.path.join("temp", "brown_augment"))
	parser.add_argument("--bg_folder", type=str, default=os.path.join("wood_frame"))
	parser.add_argument("--nr_permutations", type=int, default=5)
	parser.add_argument("--output_dir", type=str)
	parser.add_argument("--nr_rows", type=int, default=8)
	parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction)
	parser.add_argument("--yaw_range", nargs=2, type=int, default=(-90,90))

	args = parser.parse_args()

	white_tiles = glob.glob(os.path.join(args.white_tiles, "*.jpg"))
	dark_tiles = glob.glob(os.path.join(args.dark_tiles, "*.jpg"))
	bg_imgs = glob.glob(os.path.join(args.bg_folder, "*.jpg"))
	nr_chessboards = args.nr_chessboards
	nr_permutations = args.nr_permutations
	output_dir = args.output_dir
	
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