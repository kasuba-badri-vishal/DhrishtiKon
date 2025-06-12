import os
import json
from fuzzywuzzy import fuzz
from tqdm import tqdm

from doctr.io import DocumentFile
from doctr.models import ocr_predictor



stop_words = {'what', 'is', 'the', 'this', 'that', 'these', 'those', 'which', 'how', 'why', 'where', 'when', 'who', 'will', 'be', 'and', 'or', 'in', 'at', 'to', 'for', 'of', 'with', 'by'}


def get_matched_regions(question_text, target_text, predictions):

    question_terms = [word.lower() for word in question_text.split() if word.lower() not in stop_words]
    matched_regions = []
    for region in predictions:
        region_text = region['text']
        region_copy = region.copy()

        if target_text.lower() in region_text.lower():
            region_copy['match_score'] = 100
            region_copy['match_details'] = {
                    'exact_match': True,
                    'answer_score': 100,
                    'question_score': 100
                }
            matched_regions.append(region_copy)
            continue

        partial_score = fuzz.partial_ratio(target_text.lower(), region_text.lower())
        token_score = fuzz.token_set_ratio(target_text.lower(), region_text.lower())
        
        # Calculate length factor (preference for longer matches that contain meaningful content)
        target_len = len(target_text)
        region_len = len(region_text)
        length_factor = min(1.0, region_len / min(50, target_len))  # Cap at 1.0, adapt based on target length
        
        # Combine scores for answer with weights
        # Higher weight to token matching for longer texts, higher weight to partial matching for shorter texts
        if region_len > 10:
            answer_score = (partial_score * 0.3) + (token_score * 0.5) + (length_factor * 100 * 0.2)
        else:
            # For very short texts, reduce their overall score unless they're exact matches
            answer_score = (partial_score * 0.3) + (token_score * 0.4) + (length_factor * 100 * 0.3)
            if region_len < 5 and partial_score < 100:
                answer_score *= 0.5  # Penalize very short inexact matches

        # penalize shorter region_texts
        if region_len < 5:
            answer_score *= 0.5
        
        # Calculate fuzzy match scores for question terms using both methods
        partial_question_scores = [fuzz.partial_ratio(term, region_text.lower()) for term in question_terms]
        token_question_scores = [fuzz.token_set_ratio(term, region_text.lower()) for term in question_terms]
        
        # Get best scores for question terms
        best_partial_question = max(partial_question_scores) if partial_question_scores else 0
        best_token_question = max(token_question_scores) if token_question_scores else 0
        
        # Combine question scores
        question_score = (best_partial_question * 0.4) + (best_token_question * 0.6)
        
        # Combine scores (giving more weight to answer matches)
        combined_score = (answer_score * ANSWER_WEIGHT) + (question_score * QUESTION_WEIGHT)

        # print(combined_score)
        
        if combined_score >= CUT_OFF_THRESHOLD:
            region_copy['match_score'] = combined_score
            region_copy['match_details'] = {
                'exact_match': False,
                'answer_score': answer_score,
                'question_score': question_score,
                'answer_weight': ANSWER_WEIGHT,
                'question_weight': QUESTION_WEIGHT
            }
            matched_regions.append(region_copy)


    matched_regions.sort(key=lambda x: x['match_score'], reverse=True)
    top_matches = matched_regions[:MAX_LINE_MATCHES]
    return top_matches
        
        
def longest_consecutive_range(indices):
    if not indices:
        return []

    indices = sorted(set(indices))
    longest = []
    current = [indices[0]]

    for i in range(1, len(indices)):
        if indices[i] == indices[i - 1] + 1:
            current.append(indices[i])
        else:
            if len(current) > len(longest):
                longest = current
            current = [indices[i]]

    if len(current) > len(longest):
        longest = current

    return longest


def get_word_level_matches(answer_text, top_k_matches):
    bboxes = []
    for match in top_k_matches:
        indices = []
        for index, word in enumerate(match['words']):
            if word['text'].lower() in answer_text.lower():
                # bboxes.append(word['bbox'])
                indices.append(index)
        longest_indices = longest_consecutive_range(indices)
        for index in longest_indices:
            bboxes.append(match['words'][index]['bbox'])
    return bboxes

# model = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True)

MAX_LINE_MATCHES = 10
CUT_OFF_THRESHOLD = 70
QUESTION_WEIGHT = 0.2
ANSWER_WEIGHT = 0.8
LEVEL = "line"


for i in range(1, 11):

    MAX_LINE_MATCHES = i

    JSON_FILE = "/data/BADRI/FINAL/THESIS/GRVQA/main/outputs/json/filtered_grounding_annotations.json"
    IMG_DIR = "/data/BADRI/FINAL/THESIS/GRVQA/ANNOTATION/final/"

    OUTPUT_FILE = f"/data/BADRI/FINAL/THESIS/GRVQA/main/outputs/json/doctr_grounding_annotations_{i}.json"

    OCR_FILE = "/data/BADRI/FINAL/THESIS/GRVQA/main/outputs/intermediate/doctr_line_ocr_store.json"

    with open(OCR_FILE, 'r') as f:
        ocr_data = json.load(f)


    with open(JSON_FILE, 'r') as f:
        data = json.load(f)

    for image_name, qa_data in tqdm(data.items()):
        IMG_PATH = os.path.join(IMG_DIR, image_name)

        predictions = ocr_data[image_name]


        for qa in tqdm(qa_data):
            question = qa['question']
            answer = qa['answer']
            top_k_matches = get_matched_regions(question, answer, predictions)

            matched_bboxes = []
            for match in top_k_matches:
                matched_bboxes.append(match['bbox'])

            word_level_matches = get_word_level_matches(answer, top_k_matches=top_k_matches)

            qa['line_level_matches'] = matched_bboxes
            qa['word_level_matches'] = word_level_matches


    with open(OUTPUT_FILE, 'w') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
        


        

