import json

# Initialize counters
total_block_boxes = 0
total_line_boxes = 0
total_word_boxes = 0
total_points = 0
total_qa = 0
total_images = 0

JSON_FILE = "/data/BADRI/FINAL/THESIS/GRVQA/main/outputs/json/doctr_grounding_annotations.json"

# Read the JSON file
with open(JSON_FILE, 'r') as f:
    data = json.load(f)

# Iterate through each image and its annotations
for image_name, annotations in data.items():
    total_images += 1
    for annotation in annotations:
        total_qa += 1
        total_block_boxes += len(annotation.get('blockBoxes', []))
        total_line_boxes += len(annotation.get('lineBoxes', []))
        total_word_boxes += len(annotation.get('wordBoxes', []))
        total_points += len(annotation.get('points', []))

# Print the results
print(f"Total number of blockBoxes: {total_block_boxes}")
print(f"Total number of lineBoxes: {total_line_boxes}")
print(f"Total number of wordBoxes: {total_word_boxes}")
print(f"Total number of points: {total_points}")
print(f"Total number of QA: {total_qa}")
print(f"Total number of images: {total_images}")
