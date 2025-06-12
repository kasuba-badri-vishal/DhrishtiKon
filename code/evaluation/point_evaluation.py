import numpy as np
from typing import List, Tuple, Dict, Union, Any
import json
from tqdm import tqdm
from PIL import Image, ImageDraw
import os

def calculate_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
    """
    Calculate Euclidean distance between two points.
    
    Args:
        point1: Tuple (x, y) for first point
        point2: Tuple (x, y) for second point
        
    Returns:
        Euclidean distance between the points
    """
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def evaluate_samples(
    samples: List[Dict[str, Any]],
    distance_thresholds: Union[float, List[float]] = [0.01, 0.025, 0.05]
) -> Dict:
    """
    Evaluate point predictions for a list of samples, each with its own image size.
    
    Args:
        samples: List of dictionaries, each containing:
            - qid: Question/sample ID
            - predictions: List of predicted points as (x, y) tuples
            - groundtruths: List of ground truth points as (x, y) tuples
            - img_size: Tuple (width, height) of the image
        distance_thresholds: Normalized distance threshold(s) to consider a match
                           (can be single value or list of thresholds)
                           These values are relative to the image diagonal length
    
    Returns:
        Dictionary with precision, recall, and F1 score for each threshold and per-sample metrics
    """
    if isinstance(distance_thresholds, (int, float)):
        distance_thresholds = [distance_thresholds]
    
    # Initialize results dictionary
    results = {threshold: {
        'precision': 0,
        'recall': 0,
        'f1': 0,
        'true_positives': 0,
        'false_positives': 0,
        'false_negatives': 0,
        'per_sample_metrics': []
    } for threshold in distance_thresholds}
    
    # Process each sample
    for sample in samples:
        qid = sample['qid']
        predictions = sample['predictions']
        ground_truths = sample['groundtruths']
        img_size = sample['img_size']
        
        # Calculate image diagonal length for this specific image
        image_diagonal = np.sqrt(img_size[0]**2 + img_size[1]**2)
        
        # Evaluate this sample individually
        for normalized_threshold in distance_thresholds:
            # Convert normalized threshold to absolute pixels for this image
            absolute_threshold = normalized_threshold * image_diagonal
            
            # Initialize counters for this sample
            true_positives = 0
            false_positives = 0
            false_negatives = 0
            
            # Track which ground truths have been matched
            matched_gt = set()
            
            # For each prediction, find the closest ground truth
            for pred in predictions:
                min_distance = float('inf')
                closest_gt_idx = -1
                
                # Find closest ground truth
                for i, gt in enumerate(ground_truths):
                    if i in matched_gt:
                        continue  # Skip already matched ground truths
                        
                    dist = calculate_distance(pred, gt)
                    if dist < min_distance:
                        min_distance = dist
                        closest_gt_idx = i
                
                # Normalize the minimum distance by this image's diagonal
                normalized_min_distance = min_distance / image_diagonal
                
                # Check if the closest ground truth is within threshold
                if closest_gt_idx != -1 and normalized_min_distance <= normalized_threshold:
                    true_positives += 1
                    matched_gt.add(closest_gt_idx)
                else:
                    false_positives += 1
            
            # Count unmatched ground truths as false negatives
            false_negatives = len(ground_truths) - len(matched_gt)
            
            # Calculate metrics for this sample
            sample_precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            sample_recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            sample_f1 = 2 * sample_precision * sample_recall / (sample_precision + sample_recall) if (sample_precision + sample_recall) > 0 else 0
            
            # Add to overall metrics
            results[normalized_threshold]['true_positives'] += true_positives
            results[normalized_threshold]['false_positives'] += false_positives
            results[normalized_threshold]['false_negatives'] += false_negatives
            
            # Store per-sample metrics
            results[normalized_threshold]['per_sample_metrics'].append({
                'qid': qid,
                'precision': sample_precision,
                'recall': sample_recall,
                'f1': sample_f1,
                'true_positives': true_positives,
                'false_positives': false_positives,
                'false_negatives': false_negatives,
                'img_size': img_size,
                'absolute_threshold_px': absolute_threshold
            })
    
    # Calculate overall metrics
    for threshold in distance_thresholds:
        tp = results[threshold]['true_positives']
        fp = results[threshold]['false_positives']
        fn = results[threshold]['false_negatives']
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        results[threshold]['precision'] = precision
        results[threshold]['recall'] = recall
        results[threshold]['f1'] = f1
    
    return results

def get_points(points: List[List[float]]) -> List[Tuple[float, float]]:
    final_points = []
    for point in points:
        current_point = []
        current_point.append(point['x'])
        current_point.append(point['y'])
        final_points.append(current_point)
    return final_points

def get_overall_scores(sample_results: Dict) -> Dict:
    """
    Extract just the overall scores from sample results.
    
    Args:
        sample_results: Results dictionary from evaluate_samples
        
    Returns:
        Dictionary with just the overall metrics for each threshold
    """
    overall_scores = {}
    
    for threshold, metrics in sample_results.items():
        overall_scores[threshold] = {
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1'],
            'true_positives': metrics['true_positives'],
            'false_positives': metrics['false_positives'],
            'false_negatives': metrics['false_negatives']
        }
    
    return overall_scores

# Example usage
if __name__ == "__main__":


    JSON_FILE = "/data/BADRI/FINAL/THESIS/GRVQA/main/outputs/json/doctr_grounding_annotations.json"
    IMG_DIR = "/data/BADRI/FINAL/THESIS/GRVQA/ANNOTATION/final/"
    OUT_IMG_DIR = "/data/BADRI/FINAL/THESIS/GRVQA/main/outputs/images/algorithm/point/"
    if not os.path.exists(OUT_IMG_DIR):
        os.makedirs(OUT_IMG_DIR)

    with open(JSON_FILE, 'r') as f:
        data = json.load(f)

    samples = []

    for image_name, qa_data in tqdm(data.items()):
        
        for qa in qa_data:
            img = Image.open(os.path.join(IMG_DIR, image_name))
            width, height = img.size
            draw = ImageDraw.Draw(img)
            point_value = {
                'qid': qa['id'],
                'predictions': qa['point_level_matches'],
                'groundtruths': get_points(qa['points']),
                'img_size': (width, height)
            }
            samples.append(point_value)
            for point in point_value['groundtruths']:
                draw.ellipse([point[0] - 5, point[1] - 5, point[0] + 5, point[1] + 5], fill='red')

            for point in point_value['predictions']:
                draw.ellipse([point[0] - 5, point[1] - 5, point[0] + 5, point[1] + 5], fill='blue')

            img.save(os.path.join(OUT_IMG_DIR, qa["id"] + '.png'))

    distance_thresholds = [0.05, 0.07, 0.1]

    sample_results = evaluate_samples(samples, distance_thresholds)
     # Extract and print just the overall scores
    overall_scores = get_overall_scores(sample_results)
    
    print("Overall evaluation results:")
    for threshold, metrics in overall_scores.items():
        print(f"\nResults for normalized distance threshold {threshold}:")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        print(f"TP: {metrics['true_positives']}, FP: {metrics['false_positives']}, FN: {metrics['false_negatives']}")  