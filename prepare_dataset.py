import json
import os
import shutil
import random
from pathlib import Path

# Paths
SOURCE_DIR = "C:\\Users\\user\\Desktop\\train"
ANNOTATION_FILE = os.path.join(SOURCE_DIR, "_annotations.coco.json")
DATASET_DIR = "d:/datasets/tomato"
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "val")

# Ensure clean directories
if os.path.exists(DATASET_DIR):
    shutil.rmtree(DATASET_DIR)

for p in [os.path.join(TRAIN_DIR, "images"), os.path.join(TRAIN_DIR, "labels"),
          os.path.join(VAL_DIR, "images"), os.path.join(VAL_DIR, "labels")]:
    os.makedirs(p, exist_ok=True)

print("Loading annotations...")
with open(ANNOTATION_FILE, 'r') as f:
    data = json.load(f)

# Map categories
# COCO IDs might not be continuous 0,1,2..., so we map them to YOLO 0-indexed
# Current data: 0: green...?, 1: green, 2: half, 3: ripened.
# Let's check the previous analysis output or just rely on name mapping.
# 'green-fully_ripened-half_ripened' seems to be id 0 (supercategory?) or a mistake?
# I will map specific names to IDs.
target_categories = ["green", "half-ripened", "ripened"]
cat_name_to_id = {cat['name']: cat['id'] for cat in data['categories']}
print(f"Categories in json: {cat_name_to_id}")

# We want: green->0, half-ripened->1, ripened->2
# The JSON has specific IDs. We need a map from JSON_ID -> YOLO_ID
json_id_to_yolo_id = {}
for cid, name in zip(range(3), target_categories):
    # Find the json id for this name
    jid = next((c['id'] for c in data['categories'] if c['name'] == name), None)
    if jid is not None:
        json_id_to_yolo_id[jid] = cid
    else:
        print(f"Warning: Category {name} not found in JSON categories!")

print(f"Mapping JSON ID -> YOLO ID: {json_id_to_yolo_id}")

# Group annotations by image
image_annotations = {} # image_id -> list of anns
for ann in data['annotations']:
    img_id = ann['image_id']
    if img_id not in image_annotations:
        image_annotations[img_id] = []
    image_annotations[img_id].append(ann)

# Process images
images = data['images']
# Shuffle for split
random.shuffle(images)

split_idx = int(len(images) * 0.8)
train_images = images[:split_idx]
val_images = images[split_idx:]

def process_set(image_set, base_dir):
    count = 0
    for img in image_set:
        img_id = img['id']
        file_name = img['file_name']
        
        # Source file path
        src_path = os.path.join(SOURCE_DIR, file_name)
        if not os.path.exists(src_path):
            print(f"Missing file: {src_path}")
            continue
            
        # Destination Paths
        dst_img_path = os.path.join(base_dir, "images", file_name)
        dst_lbl_path = os.path.join(base_dir, "labels", os.path.splitext(file_name)[0] + ".txt")
        
        # Copy image
        shutil.copy2(src_path, dst_img_path)
        
        # Create YOLO label file
        if img_id in image_annotations:
            with open(dst_lbl_path, 'w') as lf:
                for ann in image_annotations[img_id]:
                    cat_id = ann['category_id']
                    if cat_id not in json_id_to_yolo_id:
                        continue # Skip filtered categories if any
                        
                    yolo_cat = json_id_to_yolo_id[cat_id]
                    
                    # Convert bbox [x_min, y_min, w, h] to [x_center, y_center, w, h] normalized
                    img_w = img['width']
                    img_h = img['height']
                    bbox = ann['bbox']
                    
                    x_center = (bbox[0] + bbox[2]/2) / img_w
                    y_center = (bbox[1] + bbox[3]/2) / img_h
                    w = bbox[2] / img_w
                    h = bbox[3] / img_h
                    
                    # Clip values to [0, 1] just in case
                    x_center = max(0, min(1, x_center))
                    y_center = max(0, min(1, y_center))
                    w = max(0, min(1, w))
                    h = max(0, min(1, h))
                    
                    lf.write(f"{yolo_cat} {x_center} {y_center} {w} {h}\n")
        count += 1
    return count

print(f"Processing Train Set ({len(train_images)} images)...")
c_train = process_set(train_images, TRAIN_DIR)
print(f"Processing Val Set ({len(val_images)} images)...")
c_val = process_set(val_images, VAL_DIR)

print(f"Done! Train: {c_train}, Val: {c_val}")
