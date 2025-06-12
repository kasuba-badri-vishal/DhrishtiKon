import os
import shutil
from fuzzywuzzy import fuzz
from tqdm import tqdm
from PIL import Image
import requests
# from surya.layout import LayoutPredictor

from doctr.io import DocumentFile
from pdf2image import convert_from_path
import pymupdf
# from doctr.models import ocr_predictor
import numpy as np
from time import time
import streamlit as st

pipe = None
layout_predictor = None

MAX_BLOCK_MATCHES = 2
MAX_LINE_MATCHES = 5
CUT_OFF_THRESHOLD = 70
QUESTION_WEIGHT = 0.2
ANSWER_WEIGHT = 0.8
LEVEL = "line"

jpg_options = {
    "quality"    : 100,
    "progressive": True,
    "optimize"   : False
}

stop_words = {'what', 'is', 'the', 'this', 'that', 'these', 'those', 'which', 'how', 'why', 'where', 'when', 'who', 'will', 'be', 'and', 'or', 'in', 'at', 'to', 'for', 'of', 'with', 'by'}

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


def get_matched_regions(question_text, target_text, predictions, level):

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
    
    # If no matches, reduce threshold by 20 and get the topmost single output
    if not matched_regions:
        new_threshold = max(CUT_OFF_THRESHOLD - 30, 0)  # Prevent negative threshold
        matched_regions = [region for region in matched_regions if region['match_score'] >= new_threshold]
        matched_regions.sort(key=lambda x: x['match_score'], reverse=True)
        if matched_regions:
            matched_regions = [matched_regions[0]]  # Only keep the topmost single output

    if level == "block":
        top_matches = matched_regions[:MAX_BLOCK_MATCHES]
    elif level == "line":
        top_matches = matched_regions[:MAX_LINE_MATCHES]
    return top_matches


def get_processed_text_for_llm(block_predictions, gap):
    final_text = ""
    for block_data in block_predictions:
        final_text += block_data['text'] + gap
    return final_text


def get_page_number(block_bboxes):
    pages = {}
    for block in block_bboxes:
        if block['page'] not in pages:
            pages[block['page']] = 1
        else:
            pages[block['page']] += 1

    print(pages)
    try : 
        max_page = max(pages, key=pages.get)
    except:
        max_page = 0
    return max_page


def predict_output(document_path, question, _pipe, _layout_predictor, _model, model_type, document_type="image"):
    """Main prediction function that coordinates all predictions."""
    predicted_answer = None
    block_box_predictions = None
    line_box_predictions = None
    word_box_predictions = None
    point_box_predictions = None

    curr_time = time()
    line_predictions, pages_count = cached_line_predictions(document_path, _model, document_type)
    line_time = time()
    print(f"Done with line predictions in {line_time - curr_time} seconds")
    
    curr_time = time()
    if(document_type == "pdf" and pages_count < 3):
        block_predictions = cached_block_predictions(document_path, _layout_predictor, _model, document_type)
        gap = '\n\n\n'
    else:
        block_predictions = line_predictions
        gap = '\n'
    block_time = time()
    print(f"Done with block predictions in {block_time - line_time} seconds")

    curr_time = time()
    if model_type == "Drishtikon" or document_type=="pdf":
        processed_text_for_llm = get_processed_text_for_llm(block_predictions, gap=gap)
        predicted_answer = generate_llm_answer(question, processed_text_for_llm, _pipe)
    elif model_type == "Param":
        predicted_answer = generate_via_inhouse_model_answer(question, document_path)
    llm_time = time()
    print(f"Done with LLM in {llm_time - curr_time} seconds")

    print("LLM Answer: ", predicted_answer)

    total_algo_time = time()
    curr_time = time()
    
    line_matches = get_matched_regions(question, predicted_answer, line_predictions, "line")
    block_bboxes = get_matched_regions(question, predicted_answer, block_predictions, "block")
    match_time = time()
    print(f"Done with match in {match_time - curr_time} seconds")

    if document_type == "pdf":
        current_page = get_page_number(block_bboxes)
    else:
        current_page = -1

    if(current_page != -1):
        predicted_answer = "Answer predicted from page: " + str(current_page+1) + "\n" + predicted_answer

    block_box_predictions = []
    for match in block_bboxes:
        block_box_predictions.append(match['bbox'])

    line_box_predictions = []
    for match in line_matches:
        if current_page == -1 or match['page'] == current_page:
            line_box_predictions.append(match['bbox'])

    curr_time = time()
    word_box_predictions = get_word_level_matches(predicted_answer, top_k_matches=line_matches)
    word_time = time()
    print(f"Done with word in {word_time - curr_time} seconds")

    curr_time = time()
    point_box_predictions = get_point_level_matches(block_box_predictions, line_box_predictions, word_box_predictions)
    point_time = time()
    print(f"Done with point in {point_time - curr_time} seconds")
    
    print(f"Total algo time: {time() - total_algo_time} seconds")

    return predicted_answer, block_box_predictions, line_box_predictions, word_box_predictions, point_box_predictions, current_page


def calculate_midpoint_of_bboxes(bboxes):

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


def get_point_level_matches(block_box_predictions, line_box_predictions, word_box_predictions):

    point_box_predictions = []

    if len(block_box_predictions) ==1:
        try:
            x, y = calculate_midpoint_of_bboxes(block_box_predictions)
            point_box_predictions = [[x, y]]
            # print(x, y)
        except:
            try:
                x, y = calculate_midpoint_of_bboxes(line_box_predictions)
                point_box_predictions = [[x, y]]
            except:
                point_box_predictions = []
    else:
        points = []
        for block_bbox in block_box_predictions:
            try:
                x, y = calculate_midpoint_of_bboxes(block_bbox)
                points.append([x, y])
            except:
                continue
        point_box_predictions = points
    
    return point_box_predictions


def generate_via_inhouse_model_answer(question, image_path, api_key="VISION-TEAM", max_tokens=512, temperature=0.7, endpoint="http://103.207.148.38:9000/api/v1/chat/upload"):
    headers = {
        "x-api-key": api_key  # or whatever the Swagger UI says
    }

    files = {
        "image": open(image_path, "rb")
    }

    data = {
        "text": question,
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

def generate_llm_answer(question, context, pipe):

    prompt = f"""You are given a question and context. Your task is to find and return the best possible answer to the question using only the context as it is. 
Do not generate summaries, paraphrased content, or any additional explanations including any preamble and postamble. 
Return only the exact phrase or sentence fragment from the context that answers the question. 
If the answer is not found in the context, return: Answer not found in context.

Question: {question}
Context: {context}
Answer:
"""

    messages = [ {"role": "user", "content": prompt}]
    result = pipe(messages, max_new_tokens=512, do_sample=True, temperature=0.7)
    ans = result[0]["generated_text"][1]['content']
    return ans
    

@st.cache_data(show_spinner="Running OCR for lines...")
def cached_line_predictions(document_path, _model, document_type):
    """Get line predictions from OCR model."""
    current_dir = os.getcwd()
    if document_type == "pdf":
        output_file = simple_counter_generator("page", ".jpg")
        current_dir = os.getcwd()
        temp_output_folder = os.path.join(current_dir, "temp_output_folder/")

        # delete the temp_output_folder
        if os.path.exists(temp_output_folder):
            shutil.rmtree(temp_output_folder)

        if not os.path.exists(temp_output_folder):
            os.makedirs(temp_output_folder)

        doc = pymupdf.open(document_path)  # open document
        for page in doc:  # iterate through the pages
            pix = page.get_pixmap()  # render page to an image
            pix.save(f"{temp_output_folder}/{page.number}.png")  

        images_path = sorted(os.listdir(temp_output_folder))
    else:
        images_path = [os.path.join(current_dir, document_path)]

    line_predictions = []
    pages_count = -1
    for image_path in images_path:
        pages_count += 1

        if(len(images_path) > 1):
            doc = DocumentFile.from_images(os.path.join(temp_output_folder, image_path))
        else:
            doc = DocumentFile.from_images(image_path)

        result = _model(doc)
        for page in result.pages:     
            dim = tuple(reversed(page.dimensions))
            for block in page.blocks:
                for line in block.lines:
                    output = {}
                    geo = line.geometry
                    a = list(a*b for a,b in zip(geo[0],dim))
                    b = list(a*b for a,b in zip(geo[1],dim))
                    x1 = round(a[0], 2).astype(float)
                    y1 = round(a[1], 2).astype(float)
                    x2 = round(b[0], 2).astype(float)
                    y2 = round(b[1], 2).astype(float)
                    line_bbox = [x1, y1, x2, y2]
                    
                    sent = []
                    words_data = []
                    for word in line.words:
                        word_data = {}
                        sent.append(word.value)
                        geo = word.geometry
                        a = list(a*b for a,b in zip(geo[0],dim))
                        b = list(a*b for a,b in zip(geo[1],dim))
                        x1 = round(a[0], 2).astype(float)
                        y1 = round(a[1], 2).astype(float)
                        x2 = round(b[0], 2).astype(float)
                        y2 = round(b[1], 2).astype(float)
                        bbox = [x1, y1, x2, y2]
                        
                        word_data['bbox'] = bbox
                        word_data['text'] = word.value
                        words_data.append(word_data)
                    output['bbox'] = line_bbox
                    output['text'] = " ".join(sent)
                    output['words'] = words_data
                    output['page'] = pages_count
                    line_predictions.append(output)

    return line_predictions, pages_count


@st.cache_data(show_spinner="Running OCR for blocks...")
def cached_block_predictions(document_path, _layout_predictor, _model, document_type):
    """Get block predictions from layout predictor and OCR model."""
    current_dir = os.getcwd()
    if document_type == "pdf":
        output_file = simple_counter_generator("page", ".jpg")
        current_dir = os.getcwd()
        temp_output_folder = os.path.join(current_dir, "temp_output_folder/")

        if os.path.exists(temp_output_folder):
            shutil.rmtree(temp_output_folder)

        if not os.path.exists(temp_output_folder):
            os.makedirs(temp_output_folder)

        doc = pymupdf.open(document_path)
        for page in doc:
            pix = page.get_pixmap()
            pix.save(f"{temp_output_folder}/{page.number}.png")  

        images_path = sorted(os.listdir(temp_output_folder))
    else:
        images_path = [os.path.join(current_dir, document_path)]

    block_predictions = []
    page_count = -1
    for image_path in images_path:
        page_count += 1

        if(len(images_path) > 1):
            image = Image.open(os.path.join(temp_output_folder, image_path))
        else:
            image = Image.open(os.path.join(current_dir, document_path))

        layout_predictions = _layout_predictor([image])

        for block in layout_predictions[0].bboxes:
            output = {}
            bbox = [int(x) for x in block.bbox]
            
            cropped_image = image.crop(bbox)
            cropped_image.save(f'temp.png')
            doc = DocumentFile.from_images('temp.png')
            result = _model(doc)

            text = []
            for page in result.pages:
                for block in page.blocks:
                    for line in block.lines:
                        for word in line.words:
                            text.append(word.value)

            output['bbox'] = bbox
            output['text'] = " ".join(text)
            output['page'] = page_count
            block_predictions.append(output)
    
    return block_predictions

def simple_counter_generator(prefix="", suffix=""):
    while True:
        yield 'p'

def clear_prediction_caches():
    """Clear all cached predictions."""
    cached_line_predictions.clear()
    cached_block_predictions.clear()



# from doctr.models import ocr_predictor
# model = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True)


# # from transformers import pipeline
# # def load_llm_model(device):
# #     pipe = pipeline("text-generation", model="meta-llama/Meta-Llama-3.1-8B-Instruct", device=device)
# #     return pipe

# # pipe = load_llm_model("cuda")
# pipe = None

# # from surya.layout import LayoutPredictor
# # layout_predictor = LayoutPredictor()
# layout_predictor = None

# document_path = "sample.pdf"
# question = "What is the subject of the circular?"

# answer, block_box_predictions, line_box_predictions, word_box_predictions, point_box_predictions = predict_output(document_path, question, pipe, layout_predictor, model, "Inhouse", document_type="pdf")

# print(answer)
# print(block_box_predictions)
# print(line_box_predictions)
# print(word_box_predictions)
# print(point_box_predictions)
    
    
    
