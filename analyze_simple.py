import json
import os
import collections
import sys

# Force flush
sys.stdout.reconfigure(encoding='utf-8')

annotation_file = 'd:/train/_annotations.coco.json'

try:
    with open(annotation_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Categories
    categories = {cat['id']: cat['name'] for cat in data['categories']}
    print("CATEGORIES_START")
    for result_id, name in categories.items():
        print(f"ID {result_id}: {name}")
    print("CATEGORIES_END")

    # Counts
    image_count = len(data['images'])
    annotations = data['annotations']
    print(f"Total Images: {image_count}")
    print(f"Total Annotations: {len(annotations)}")

    # Distribution
    class_counts = collections.Counter([ann['category_id'] for ann in annotations])
    print("DISTRIBUTION_START")
    for class_id, count in class_counts.items():
        name = categories.get(class_id, "Unknown")
        print(f"{name}: {count}")
    print("DISTRIBUTION_END")

except Exception as e:
    print(f"ERROR: {e}")
