import os
import json
from PIL import Image, ImageDraw

INPUT_ANNOTATION_JSON_FILE = "/data/BADRI/FINAL/THESIS/GRVQA/ANNOTATION/code/grounding_annotations.json"
OUTPUT_ANNOTATION_JSON_FILE = "/data/BADRI/FINAL/THESIS/GRVQA/ANNOTATION/data/filtered_grounding_annotations.json"

IMAGES_INPUT_DIR = "/data/BADRI/FINAL/THESIS/GRVQA/ANNOTATION/final/"
IMAGES_OUTPUT_DIR = "/data/BADRI/FINAL/THESIS/GRVQA/main/outputs/images/"

# Load the annotations from the JSON file
with open(INPUT_ANNOTATION_JSON_FILE, 'r') as file:
    annotations = json.load(file)

# Function to check if all boxes are empty
def are_all_boxes_empty(qa):
    return (not qa.get('blockBoxes') or len(qa['blockBoxes']) == 0) and \
           (not qa.get('lineBoxes') or len(qa['lineBoxes']) == 0) and \
           (not qa.get('wordBoxes') or len(qa['wordBoxes']) == 0)


def draw_bounding_boxes(image_name, qa_id, boxes_type, boxes, color):
    # Load the image
    image_path = os.path.join(IMAGES_INPUT_DIR, image_name)
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    output_path = os.path.join(IMAGES_OUTPUT_DIR, boxes_type, f'{qa_id}.png')

    if boxes_type == 'points':
        for point in boxes:
            draw.ellipse([point['x'] - 5, point['y'] - 5, point['x'] + 5, point['y'] + 5], fill=color)
        print(f"Saved image with bounding boxes: {output_path}")

    else:
        for box in boxes:
            draw.rectangle([box['x'], box['y'], box['x'] + box['width'], box['y'] + box['height']], outline=color, width=2)


    # Save the image with the specified naming convention
    
    image.save(output_path)
    
if not os.path.exists(os.path.join(IMAGES_OUTPUT_DIR, 'blocks')):
    os.makedirs(os.path.join(IMAGES_OUTPUT_DIR, 'blocks'), exist_ok=True)
    os.makedirs(os.path.join(IMAGES_OUTPUT_DIR, 'lines'), exist_ok=True)
    os.makedirs(os.path.join(IMAGES_OUTPUT_DIR, 'points'), exist_ok=True)
    os.makedirs(os.path.join(IMAGES_OUTPUT_DIR, 'words'), exist_ok=True)

# Remove images with all empty boxes
filtered_annotations = {}
for image_name, qa_pairs in annotations.items():
    # Filter out QA pairs where all boxes are empty
    filtered_qa_pairs = [qa for qa in qa_pairs if not are_all_boxes_empty(qa)]

    # save qna_ids for each image
    for i, qa in enumerate(filtered_qa_pairs):
        image_id = image_name.split('.')[0]
        qa['id'] = f"{image_id}_{i}"

        if os.path.exists(os.path.join(IMAGES_OUTPUT_DIR, 'blocks', f'{qa["id"]}.png')):
            print(f'{qa["id"]} already exists')
            continue
        else:
            draw_bounding_boxes(image_name, qa['id'], 'blocks', qa['blockBoxes'], 'red')
            draw_bounding_boxes(image_name, qa['id'], 'lines', qa['lineBoxes'], 'blue')
            draw_bounding_boxes(image_name, qa['id'], 'words', qa['wordBoxes'], 'green')
            draw_bounding_boxes(image_name, qa['id'], 'points', qa['points'], 'orange')

    
    # Only add to filtered_annotations if there are remaining QA pairs
    if filtered_qa_pairs:
        filtered_annotations[image_name] = filtered_qa_pairs

# Save the filtered annotations back to a new JSON file
with open(OUTPUT_ANNOTATION_JSON_FILE, 'w') as file:
    json.dump(filtered_annotations, file, indent=2)

print(f"Filtered annotations saved to {OUTPUT_ANNOTATION_JSON_FILE}")