import numpy as np
import json
from tqdm import tqdm


def calculate_midpoint_of_bboxes(bboxes):
    """
    Calculate the midpoint of all bounding boxes combined.
    
    Args:
        bboxes: List of bounding boxes in format [x1, y1, x2, y2]
        
    Returns:
        Tuple (x, y) representing the midpoint
    """
    if not bboxes:
        return None
    
    # Convert to numpy array for easier manipulation
    bboxes = np.array(bboxes)
    
    # Find the extreme points of all bboxes combined
    min_x = np.min(bboxes[:, 0])
    min_y = np.min(bboxes[:, 1])
    max_x = np.max(bboxes[:, 2])
    max_y = np.max(bboxes[:, 3])
    
    # Calculate midpoint
    midpoint_x = (min_x + max_x) / 2
    midpoint_y = (min_y + max_y) / 2
    
    return round(midpoint_x, 2), round(midpoint_y, 2)

# Example usage
if __name__ == "__main__":
    # Example with the bboxes from the JSON file

    JSON_FILE = "/data/BADRI/FINAL/THESIS/GRVQA/main/outputs/json/doctr_grounding_annotations.json"

    with open(JSON_FILE, 'r') as f:
        data = json.load(f)

    for image_name, qa_data in tqdm(data.items()):
        for qa in qa_data:
            bboxes = qa['word_level_matches']
            len_of_block_bboxes = len(qa['blockBoxes'])
            if len_of_block_bboxes ==1:
                try:
                    x, y = calculate_midpoint_of_bboxes(bboxes)
                    qa['point_level_matches'] = [[x, y]]
                    # print(x, y)
                except:
                    try:
                        x, y = calculate_midpoint_of_bboxes(qa['line_level_matches'])
                        qa['point_level_matches'] = [[x, y]]
                    except:
                        qa['point_level_matches'] = []
            else:
                points = []
                for block_bbox in qa['block_level_matches']:
                    try:
                        x, y = calculate_midpoint_of_bboxes(block_bbox)
                        points.append([x, y])
                    except:
                        continue
                qa['point_level_matches'] = points

    with open(JSON_FILE, 'w') as f:
        json.dump(data, f, indent=4)
