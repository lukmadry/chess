import os
import cv2
import json
import pandas as pd

# Configuration
IMAGE_DIR = "images_to_annotate"       # Folder containing your images
OUTPUT_CSV = "images_to_annotate/labels.csv"  # Output file for ground-truth coordinates
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

# Collect image paths

image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(IMAGE_EXTS)]
image_files.sort()

annot = pd.read_csv(OUTPUT_CSV)
image_files = [f for f in image_files if f not in annot["filename"].values] # without values it's undefined behaviour...

# Mouse callback state
points = []

def mouse_callback(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:
            points.append((x, y))

# Setup window and mouse hook
cv2.namedWindow("Chessboard Annotator")
cv2.setMouseCallback("Chessboard Annotator", mouse_callback)

annotations = []

print("will not convert to json files that are already in .csv")

for fname in annot["filename"]:
    print(fname)
    print(f"Coordinates of {fname} are already in .csv, but not in .json, converting.")
    ext = fname.split(".")[-1]
    json_fname = os.path.join(IMAGE_DIR, fname.replace("." + ext, ".json"))
    flat_coords = annot[annot['filename'] == fname].iloc[0, 1:].values
    flat_coords = [int(f) for f in flat_coords]
    flat_coords = list(zip(flat_coords[1::2], flat_coords[0::2]))
    flat_coords = [list(f) for f in flat_coords]
    with open(json_fname, "w") as f:
        json.dump({"inner_board": flat_coords}, f)
    

for filename in image_files:
    print(filename)
    
    filepath = os.path.join(IMAGE_DIR, filename)
    raw_img = cv2.imread(filepath)
    if raw_img is None:
        continue

    points = []
    
    while True:
        display_img = raw_img.copy()
        
        # Draw recorded points and bounding lines
        for idx, pt in enumerate(points):
            cv2.circle(display_img, pt, 5, (0, 255, 0), -1)
            cv2.putText(display_img, str(idx + 1), (pt[0] + 8, pt[1] - 8), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Connect points in order as you click
            if idx > 0:
                cv2.line(display_img, points[idx - 1], pt, (255, 0, 0), 2)
            if idx == 3:
                cv2.line(display_img, points[3], points[0], (255, 0, 0), 2)

        # Status overlay
        status = f"{filename} | Points: {len(points)}/4 | [Space/Enter]: Save & Next | [r]: Reset | [q]: Quit"
        cv2.putText(display_img, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        cv2.imshow("Chessboard Annotator", display_img)
        key = cv2.waitKey(20) & 0xFF

        # Reset points for current image
        if key == ord('r'):
            points = []

        elif key == ord('o'):
            print("flipping image by 90 degrees")
            display_img = cv2.rotate(display_img, cv2.ROTATE_90_CLOCKWISE)
            cv2.imshow("Chessboard Annotator", display_img)
            
        # Quit script
        elif key == ord('q'):
            cv2.destroyAllWindows()
            exit()
            
        # Save & Next image (only allows proceeding if all 4 points are clicked)
        elif key in (13, 32):  # Enter or Space
            if len(points) == 4:
                # Flatten (x, y) pairs: x1, y1, x2, y2, x3, y3, x4, y4
                flat_coords = [coord for pt in points for coord in pt]
                flat_coords = list(zip(flat_coords[0::2], flat_coords[1::2]))
                flat_coords = [list(f) for f in flat_coords]
                annotations.append({"image_path": filename, "points": flat_coords})
                print(f"Saved {filename}: {flat_coords}")
                # json_fname = filename.replace(".jpg", ".json")
                with open(json_fname, "w") as f:
                    json.dump({"inner_board": flat_coords})
                break
            else:
                print("Please click all 4 corners before saving!")

cv2.destroyAllWindows()

# Export annotations to CSV
if annotations:
    df = pd.DataFrame([
        {"filename": a["image_path"], **{f"c{i}": v for i, v in enumerate(a["points"])}} 
        for a in annotations
    ])
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved {len(annotations)} annotations to {OUTPUT_CSV}")