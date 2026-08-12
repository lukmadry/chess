import pybullet as p
import pybullet_data
import time
import numpy as np
import os
import tempfile
import glob
from random import choice
from PIL import Image
import cv2
import urllib.request

GITHUB_RAW_BASE = "https://raw.githubusercontent.com/scenevr/chess/master/models/"
PIECES = ["pawn", "rook", "knight", "bishop", "queen", "king"]

os.makedirs("chess_models", exist_ok=True)
scale_param = 0.015

# 1. Download .obj files if they don't exist locally
for piece in PIECES:
    filename = f"{piece}.obj"
    local_path = os.path.join("chess_models", filename)
    if not os.path.exists(local_path):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(GITHUB_RAW_BASE + filename, local_path)

def create_simple_piece(radius=0.018, height=0.05, color=(0.1, 0.1, 0.1, 1.0)):
    visual_shape = p.createVisualShape(
        shapeType=p.GEOM_CYLINDER,
        radius=radius,
        length=height,
        rgbaColor=color
    )
    collision_shape = p.createCollisionShape(
        shapeType=p.GEOM_CYLINDER,
        radius=radius,
        height=height
    )
    return visual_shape, collision_shape

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
    
    # Perspective divide
    w_clip = float(clip[3])
    ndc_x = float(clip[0]) / w_clip
    ndc_y = float(clip[1]) / w_clip

    # Map NDC [-1, 1] to Pixel Space
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

chess_pieces = glob.glob("chess_models/*.obj")

# 1. Setup Simulation
p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Camera Setup
img_w, img_h = 640, 480

target_pos = [0.0, 0.0, 0.0]  # Looking at board center
dist = 0.8                     # Distance in metres
yaw = 34.                    # Orbit angle around Z-axis
pitch = -89.5

# target_pos = pan_camera_relative(0.1, 0.03, target_pos, yaw)

p.resetDebugVisualizerCamera(
    cameraDistance=dist,
    cameraYaw=yaw,
    cameraPitch=pitch if abs(pitch+90) > 1 else -89,
    cameraTargetPosition=target_pos
)

view_mat = compute_robust_view_matrix(target_pos, dist, yaw, pitch)
view_mat_arr = np.array(view_mat).reshape(4, 4).T

print("view matrix flat: ", view_mat)
# if abs(pitch + 90) > 1:
#     view_mat = p.computeViewMatrixFromYawPitchRoll(
#         cameraTargetPosition=target_pos,
#         distance=dist,
#         yaw=yaw,
#         pitch=pitch,
#         roll=0.0,
#         upAxisIndex=2  # Z-axis up
#     )
# else:
#     eye_position = [0.0, 0.0, dist]     # Directly above the origin
#     yaw_rad = np.radians(yaw)
#     up_vector = [0, 1, 0.0]
#     view_mat = p.computeViewMatrix(
#         cameraEyePosition=eye_position,
#         cameraTargetPosition=target_pos,
#         cameraUpVector=up_vector,
#         )

proj_mat = p.computeProjectionMatrixFOV(fov=60, aspect=float(img_w)/img_h, nearVal=0.1, farVal=3.0)
print("proj matrix flat: ", proj_mat)
proj_mat_arr = np.array(proj_mat).reshape(4,4).T

p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
light_direction = [0.5, 0.5, 1.0]

# 2. Build a 2x2 Chessboard Grid Programmatically
tile_size = 0.05
colors = [[0.9, 0.9, 0.8, 1.0], [0.2, 0.2, 0.2, 1.0]] # Light & Dark
tiles = []

nr_rows = 8
nr_cols = nr_rows

# turn these to glob
all_white_textures = [os.path.join("temp", x) for x in os.listdir("temp") if "white" in x]
all_white_textures = [p.loadTexture(x) for x in all_white_textures]
all_brown_textures = [os.path.join("temp", x) for x in os.listdir("temp") if "brown" in x]
all_brown_textures = [p.loadTexture(x) for x in all_brown_textures]
all_wood_textures_names = [os.path.join("wood_frame", x) for x in os.listdir("wood_frame") if os.path.isfile(os.path.join("wood_frame", x)) and x.endswith(".jpg")]
all_backgrounds = [os.path.join("wood_frame", "background", "compressed", x) for x in os.listdir(os.path.join("wood_frame", "background", "compressed")) if x.endswith(".jpg")]

print(f"Possible choices of textures:\nWhite tiles: {len(all_white_textures)}\nBrown tiles: {len(all_brown_textures)}\nFrame wood textures: {len(all_wood_textures_names)}")

grid_size = tile_size * nr_rows
border_margin = 0.25 * tile_size
frame_width = grid_size + (2 * border_margin)
frame_thickness = 0.02

inner_board_half_extents = [grid_size / 2., grid_size / 2., frame_thickness / 2.]
frame_half_extents = [frame_width / 2., frame_width / 2., frame_thickness / 2.]

inner_board_coordinates = get_coordinates_from_half_extents(inner_board_half_extents)
frame_coordinates = get_coordinates_from_half_extents(frame_half_extents)

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

p.changeVisualShape(objectUniqueId=frame_id, linkIndex=-1, specularColor=[0.4, 0.4, 0.4])

wood_txtr_name = choice(all_wood_textures_names)
wood_txtr = load_tiled_texture(wood_txtr_name, scale_factor=3, target_size=[img_w, img_h])


p.changeVisualShape(objectUniqueId=frame_id, linkIndex=-1, textureUniqueId=wood_txtr)

print("chosen wood texture: {}".format(wood_txtr_name))

# bg_id = add_background_texture(background_txtr_name, scale_factor=20, z_offset=0)

for row in range(nr_rows):
    tiles.append([])
    for col in range(nr_cols):
        vis_id = p.createVisualShape(
            p.GEOM_BOX, 
            halfExtents=[tile_size / 2., tile_size / 2., 0.002], 
            rgbaColor=colors[(row + col) % 2]
            )
        pos = [(col - (nr_cols - 1) / 2. ) * tile_size, (row - (nr_rows - 1) / 2.) * tile_size, frame_thickness / 2]
        tile_body = p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis_id, basePosition=pos)
        tiles[row].append(tile_body)

        is_white = ((row + col) % 2 == 0)
        txtr = choice(all_white_textures) if is_white else choice(all_brown_textures)
        # txtr = p.loadTexture(txtr_file)
        p.changeVisualShape(objectUniqueId=tile_body, linkIndex=-1, textureUniqueId=txtr) 


        if np.random.uniform() > 0.75:
            print("will generate a piece")
            color = 'w' if np.random.uniform() > 0.5 else 'b'
            piece_color = (0.9, 0.9, 0.9, 1.0) if color == 'w' else (0.2, 0.1, 0.1, 1.0)
            # visual_shape_piece, collision_shape_piece = create_simple_piece(color=piece_color)
            # board_center, square_size, row, col, height, visual_id, collision_id
            piece = choice(chess_pieces)
            visual_shape_piece, collision_shape_piece = create_mesh_piece(piece, color=piece_color)
            piece_id = place_piece_on_square(target_pos, tile_size, row, col, frame_thickness, visual_shape_piece, collision_shape_piece)

            current_pos, _ = p.getBasePositionAndOrientation(piece_id)
            new_orientation = p.getQuaternionFromEuler([np.radians(90), 0, 0])
            p.resetBasePositionAndOrientation(piece_id, current_pos, new_orientation)


# p.setConfigureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
# p.setConfigureDebugVisualizer(p.COV_ENABLE_GUI, 0)

p.resetDebugVisualizerCamera(
    cameraDistance=dist,
    cameraYaw=yaw,
    cameraPitch=pitch,
    cameraTargetPosition=target_pos
)


width, height, rgb_pixels, depth_pixels, segmentation_mask = p.getCameraImage(
    img_w, img_h, 
    view_mat, proj_mat, 
    lightDirection=[0.5, 0.5, 1.0],     # Direction of light source
    lightColor=[1.0, 0.98, 0.92],        # Warm white light color
    lightDistance=2.5,
    lightAmbientCoeff=0.1,              # <-- BOOST THIS (e.g. 0.35 -> 0.50) to lighten dark areas
    lightDiffuseCoeff=0.75,              # Softens direct harsh highlights
    lightSpecularCoeff=0.5,             # Regulates surface shine/varnish
    renderer=p.ER_BULLET_HARDWARE_OPENGL,
    shadow=1,
)

rgb_array = np.reshape(rgb_pixels, (height, width, 4)).astype(np.uint8)[:, :, :3]
print("height and width of camera image: ", height, width)

bgr_img = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)

seg_mask = np.array(segmentation_mask, dtype=np.int32).reshape((img_h, img_w))
board_mask = (seg_mask >= 0).astype(np.uint8)

background_txtr_name = choice(all_backgrounds)
print("chosen background texture: {}".format(background_txtr_name))

bg_img = cv2.imread(background_txtr_name)
bg_resized = cv2.resize(bg_img, (img_w, img_h), interpolation=cv2.INTER_AREA)
# print((board_mask[:, :, None]).shape, bgr_img.shape, bg_resized.shape)
final_img = np.where(board_mask[:, :, None] == 1, bgr_img, bg_resized)

print("view matrix just before projecting pixels: \n", view_mat_arr)
print("projection matrix just before projecting pixels: \n", proj_mat_arr)

inner_board_projections = [project_world_pixel(x, proj_mat_arr, view_mat_arr, img_h, img_w) for x in inner_board_coordinates]
pts_2d = np.array([(row, col) for (col, row) in inner_board_projections], dtype=np.int32)
pts_2d = pts_2d.reshape((-1,1,2))
cv2.polylines(final_img, [pts_2d], isClosed=True, color=(0,0,255), thickness=2)

print(f"Tile size: {tile_size}")
print(f"Calculated inner grid width (4 * tile_size): {tile_size * nr_rows}")
print(f"Inner board 3D corners (World Space): {inner_board_coordinates[0]}")
print(f"Projected 2D pixel corners: {inner_board_projections[0]}")

frame_projections = [project_world_pixel(x, proj_mat_arr, view_mat_arr, img_h, img_w) for x in frame_coordinates]
pts_2d = np.array([(row, col) for (col, row) in frame_projections], dtype=np.int32)
pts_2d = pts_2d.reshape((-1,1,2))
cv2.polylines(final_img, [pts_2d], isClosed=True, color=(255,0,0), thickness=2)

print("Actual OpenCV image shape:", final_img.shape)  # Should show (height, width, channels)
print("img_h passed to project:", img_h)
print("img_w passed to project:", img_w)


cv2.imwrite("annotated_chessboard.jpg", final_img)

p.disconnect()  