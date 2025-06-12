import json
from PIL import Image, ImageDraw
import os
import re

# def extract_json_from_string(s):
#     # Find the JSON part between [ and ]
#     match = re.search(r'\[\s*{.*', s, re.DOTALL)
#     if not match:
#         raise ValueError("No JSON array found in the string.")
    
#     json_str = match.group(0)
    
#     # Try to fix incomplete JSON by trimming after the last complete object
#     objects = []
#     brace_count = 0
#     current_obj = ""
#     in_object = False

#     for char in json_str:
#         current_obj += char
#         if char == '{':
#             brace_count += 1
#             in_object = True
#         elif char == '}':
#             brace_count -= 1
        
#         if in_object and brace_count == 0:
#             try:
#                 obj = json.loads(current_obj.strip().rstrip(','))
#                 objects.append(obj)
#             except json.JSONDecodeError:
#                 pass  # Skip invalid object
#             current_obj = ""
#             in_object = False

#     return objects

def extract_json_from_string(s):
    # Find the JSON part using a regex that matches stuff between [ and ]
    
    match = re.search(r'\[\s*{.*?}\s*\]', s, re.DOTALL)
    if not match:
        print(s)
        return []
        # raise ValueError("No JSON array found in the string.")
    
    json_str = match.group(0)
    try:
        data = json.loads(json_str)
        return data
    except json.JSONDecodeError as e:
        print(json_str)
        return []
        raise ValueError(f"Found JSON, but couldn't decode it: {e}")

def calculate_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    Each box is [x1, y1, x2, y2] where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner.
    """
    # Get the coordinates of the intersection rectangle
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    
    # Check if there is an intersection
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    # Calculate intersection area
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate area of each bounding box
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Calculate Union area
    union_area = box1_area + box2_area - intersection_area
    
    # Calculate IoU
    iou = intersection_area / union_area
    
    return iou

def evaluate_detections(predictions, ground_truths, iou_threshold=0.5):
    """
    Evaluate the predictions against ground truths using IoU threshold.
    
    Args:
        predictions: Dictionary with image_id as key and list of prediction bboxes as value
        ground_truths: Dictionary with image_id as key and list of ground truth bboxes as value
        iou_threshold: IoU threshold for considering a prediction as correct
        
    Returns:
        precision, recall, f1_score
    """
    total_predictions = 0
    total_ground_truths = 0
    true_positives = 0
    
    # For each image/ID
    for image_id in ground_truths:
        gt_boxes = ground_truths[image_id]
        total_ground_truths += len(gt_boxes)
        
        # Skip if no predictions for this image
        if image_id not in predictions:
            continue
            
        pred_boxes = predictions[image_id]
        total_predictions += len(pred_boxes)
        
        # Mark ground truths as matched or not
        gt_matched = [False] * len(gt_boxes)
        
        # For each prediction, find the best matching ground truth
        for pred_box in pred_boxes:
            best_iou = 0
            best_gt_idx = -1
            
            # Find the ground truth with the highest IoU
            for gt_idx, gt_box in enumerate(gt_boxes):
                if gt_matched[gt_idx]:
                    continue
                    
                iou = calculate_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            # If the IoU is above threshold, it's a true positive
            if best_iou >= iou_threshold and best_gt_idx != -1:
                true_positives += 1
                gt_matched[best_gt_idx] = True
    
    # Calculate metrics
    precision = true_positives / total_predictions if total_predictions > 0 else 0
    recall = true_positives / total_ground_truths if total_ground_truths > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1_score

def get_processed_prediction_bboxes(predictions, evaluation_level):
    bboxes = []
    if type(predictions) != list:
        predictions = extract_json_from_string(predictions)
    for pred in predictions:
        if pred['relevance'] >= RELEVANCE_THRESHOLD:
            if evaluation_level == 'line':
                data = pred['line_bbox'].split()
            elif evaluation_level == 'block':
                data = pred['block_bbox'].split()
            elif evaluation_level == 'word':
                data = pred['word_bbox'].split()
            try:
                x1 = int(data[0])
                y1 = int(data[1])
                x2 = int(data[2])
                y2 = int(data[3])
                bboxes.append([x1, y1, x2, y2])
            except:
                continue
    
    return bboxes





evaluation_level = 'line'

IMG_DIR = "/data/BADRI/FINAL/THESIS/GRVQA/ANNOTATION/final/"
INPUT_JSON_FILE = "/data/BADRI/FINAL/THESIS/GRVQA/main/outputs/json/qwen_grounding_annotations.json"
OUT_IMG_DIR = "/data/BADRI/FINAL/THESIS/GRVQA/main/outputs/images/qwen/" + evaluation_level

if not os.path.exists(OUT_IMG_DIR):
    os.makedirs(OUT_IMG_DIR)

RELEVANCE_THRESHOLD = 0.8

with open(INPUT_JSON_FILE, 'r') as f:
    data = json.load(f)

predictions = {}
ground_truths = {}

for image_name, qa_data in data.items():
    # question = qa_data['question']
    # answer = qa_data['answer']
    
    for qa in qa_data:
        img = Image.open(os.path.join(IMG_DIR, image_name))
        draw = ImageDraw.Draw(img)
        if evaluation_level == 'word':
            gts = qa['wordBoxes']
        elif evaluation_level == 'block':
            gts = qa['blockBoxes']
        elif evaluation_level == 'line':
            gts = qa['lineBoxes']

        img_id = qa["id"]
        ground_truths[img_id] = []
        for gt in gts:
            x1 = gt['x']
            y1 = gt['y']
            x2 = x1 + gt['width']
            y2 = y1 + gt['height']
            ground_truths[img_id].append([x1, y1, x2, y2])
            draw.rectangle([x1, y1, x2, y2], outline='red', width=2)

        if evaluation_level == 'word':
            predictions[img_id] = get_processed_prediction_bboxes(qa['word_level_matches'], 'word') #qa['top_k_matches']
        elif evaluation_level == 'block':
            predictions[img_id] = get_processed_prediction_bboxes(qa['block_level_matches'], 'block')
        elif evaluation_level == 'line':
            predictions[img_id] = get_processed_prediction_bboxes(qa['line_level_predictions'], 'line')

        for pred in predictions[img_id]:
            try:
                draw.rectangle(pred, outline='blue', width=2)
            except:
                print(pred)
                continue

        # save the image
        img.save(os.path.join(OUT_IMG_DIR, qa["id"] + '.png'))
        

# Evaluate the predictions
precision, recall, f1_score = evaluate_detections(predictions, ground_truths, iou_threshold=0.5)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1_score:.4f}")





