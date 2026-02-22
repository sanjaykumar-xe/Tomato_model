import json
import os
import collections
import numpy as np

annotation_file = 'd:/train/_annotations.coco.json'

with open(annotation_file, 'r') as f:
    data = json.load(f)

print(f"Dataset Info: {data['info']}")
print(f"Licenses: {len(data['licenses'])}")

categories = {cat['id']: cat['name'] for cat in data['categories']}
print("\nCategories:")
for result_id, name in categories.items():
    print(f"ID {result_id}: {name}")

image_count = len(data['images'])
print(f"\nTotal Images: {image_count}")

annotations = data['annotations']
print(f"Total Annotations: {len(annotations)}")

# Class distribution
class_counts = collections.Counter([ann['category_id'] for ann in annotations])
print("\nClass Distribution:")
for class_id, count in class_counts.items():
    print(f"{categories[class_id]}: {count} ({count/len(annotations)*100:.2f}%)")

# Bounding Box Analysis
bbox_areas = []
for ann in annotations:
    # bbox is [x, y, width, height]
    bbox_areas.append(ann['bbox'][2] * ann['bbox'][3])

bbox_areas = np.array(bbox_areas)
print(f"\nBBox Area Stats (pixels^2):")
print(f"Min: {bbox_areas.min()}")
print(f"Max: {bbox_areas.max()}")
print(f"Mean: {bbox_areas.mean():.2f}")
print(f"Median: {np.median(bbox_areas)}")

# Images without annotations
annotated_images = set(ann['image_id'] for ann in annotations)
all_images = set(img['id'] for img in data['images'])
empty_images = all_images - annotated_images
print(f"\nImages without annotations: {len(empty_images)}")
