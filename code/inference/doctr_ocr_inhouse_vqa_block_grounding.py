import os
import json
from PIL import ImageDraw, Image
from fuzzywuzzy import fuzz
from tqdm import tqdm

from surya.layout import LayoutPredictor

from doctr.io import DocumentFile
from doctr.models import ocr_predictor

import requests

# MAX_MATCHES = 4
CUT_OFF_THRESHOLD = 70
QUESTION_WEIGHT = 0.2
ANSWER_WEIGHT = 0.8
LEVEL = "line"

stop_words = {'what', 'is', 'the', 'this', 'that', 'these', 'those', 'which', 'how', 'why', 'where', 'when', 'who', 'will', 'be', 'and', 'or', 'in', 'at', 'to', 'for', 'of', 'with', 'by'}

def call_vision_language_model(
    api_key: str,
    text: str,
    image_path: str,
    max_tokens: int = 50,
    temperature: float = 0.7,
    endpoint: str = "http://103.207.148.38:9000/api/v1/chat/upload"
):
    headers = {
    "x-api-key": api_key  # or whatever the Swagger UI says
}

    files = {
        "image": open(image_path, "rb")
    }

    data = {
        "text": text,
        "max_tokens": str(max_tokens),
        "temperature": str(temperature)
    }

    try:
        response = requests.post(endpoint, headers=headers, files=files, data=data)
        response.raise_for_status()
        result = response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}
    
    return result['response']['choices'][0]['message']['content']

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
    top_matches = matched_regions[:MAX_MATCHES]
    return top_matches


for i in range(1, 6):

    MAX_MATCHES = i


    # layout_predictor = LayoutPredictor()
    # model = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True)

    JSON_FILE = f"/data/BADRI/FINAL/THESIS/GRVQA/main/outputs/json/filtered_grounding_annotations.json"
    IMG_DIR = "/data/BADRI/FINAL/THESIS/GRVQA/ANNOTATION/final/"

    OUTPUT_FILE = f"/data/BADRI/FINAL/THESIS/GRVQA/main/outputs/json/doctr_inhouse_vqa_grounding_annotations_{i}.json"

    OCR_FILE = "/data/BADRI/FINAL/THESIS/GRVQA/main/outputs/intermediate/doctr_block_ocr_store.json"



    with open(JSON_FILE, 'r') as f:
        data = json.load(f)


    with open(OCR_FILE, 'r') as f:
        ocr_data = json.load(f)


    for image_name, qa_data in tqdm(data.items()):
        IMG_PATH = os.path.join(IMG_DIR, image_name)


        image = Image.open(IMG_PATH)


        predictions = ocr_data[image_name]


        for qa in tqdm(qa_data):
            question = qa['question']
            
            answer = call_vision_language_model("VISION-TEAM", question, IMG_PATH, max_tokens=256, temperature=0.7)
            top_k_matches = get_matched_regions(question, answer, predictions)

            matched_bboxes = []
            for match in top_k_matches:
                matched_bboxes.append(match['bbox'])
            qa['block_level_matches'] = matched_bboxes


        with open(OUTPUT_FILE, 'w') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)



    with open(OUTPUT_FILE, 'w') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)