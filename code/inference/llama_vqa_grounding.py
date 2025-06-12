from transformers import pipeline
import pytesseract
from PIL import Image
import json
import os
from tqdm import tqdm

from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from surya.layout import LayoutPredictor

model = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True)
layout_predictor = LayoutPredictor()

def perform_block_level_ocr(image_path):


    image = Image.open(image_path)
    predictions = []

    # layout_predictions is a list of dicts, one per image
    layout_predictions = layout_predictor([image])

    for block in layout_predictions[0].bboxes:
        output = {}
        bbox = [int(x) for x in block.bbox]
        

        cropped_image = image.crop(bbox)

        cropped_image.save(f'temp.png')
        doc = DocumentFile.from_images('temp.png')
        result = model(doc)

        text = []
        for page in result.pages:
            for block in page.blocks:
                for line in block.lines:
                    for word in line.words:
                        text.append(word.value)


        output['bbox'] = bbox
        output['text'] = " ".join(text)
        predictions.append(output)
    return predictions

def perform_ocr(image_path):
    
    doc = DocumentFile.from_images(image_path)
    result = model(doc)

    predictions = []

    for page in result.pages:     
        dim = tuple(reversed(page.dimensions))
        for block in page.blocks:
            for line in block.lines:
                output = {}
                geo = line.geometry
                a = list(a*b for a,b in zip(geo[0],dim))
                b = list(a*b for a,b in zip(geo[1],dim))
                x1 = str(int(round(a[0], 2).astype(float)))
                y1 = str(int(round(a[1], 2).astype(float)))
                x2 = str(int(round(b[0], 2).astype(float)))
                y2 = str(int(round(b[1], 2).astype(float)))
                # line_bbox = [x1, y1, x2, y2]
                bbox = x1 + " " + y1 + " " + x2 + " " + y2
                
                sent = [bbox]
                for word in line.words:
                    sent.append(word.value)
                # words_data = []
                # for word in line.words:
                #     word_data = {}
                #     sent.append(word.value)
                #     geo = word.geometry
                #     a = list(a*b for a,b in zip(geo[0],dim))
                #     b = list(a*b for a,b in zip(geo[1],dim))
                #     x1 = round(a[0], 2).astype(float)
                #     y1 = round(a[1], 2).astype(float)
                #     x2 = round(b[0], 2).astype(float)
                #     y2 = round(b[1], 2).astype(float)
                #     bbox = [x1, y1, x2, y2]
                    
                #     word_data['bbox'] = bbox
                #     word_data['text'] = word.value
                #     words_data.append(word_data)
                # output['bbox'] = line_bbox
                output = " ".join(sent)
                # output['words'] = words_data
                predictions.append(output)
    return predictions

def load_llm_model(device):
    pipe = pipeline("text-generation", model="meta-llama/Meta-Llama-3.1-8B-Instruct", device=device)
    return pipe

def generate_llm_answer(question, answer, context, pipe):
    
    # # content = f"Question: {question}\n\nContext: {context}\n\n Give your answer in a crisp manner. Do not add any preamble or postamble."
    # prompt = f"""You are analyzing text extracted from an image. Find the line(s) of text that best matches the question and answer.


    # Question: {question}
    # Answer: {answer}
    # Context: {context}

    # Analyze the document text and determine which lines are most relevant to answering the question with respect to the provided answer.
    # For each line that's relevant, provide a relevance score from 0 to 1 where 1 means highly relevant.
    # Strictly ensure to only provide the line bbox of the line and the relevance score in JSON-like format and no other textual content, for example:
    # [
    # {{"line_bbox": "1 1 2 2", "relevance": 0.9}},
    # {{"line_bbox": "2 3 4 4", "relevance": 0.4}}
    # ]
    # """
    prompt = f"""You are analyzing text extracted from an image. Find the block(s) of text that best matches the answer to the question.


    Question: {question}
    Answer: {answer}
    Context: {context}

    Analyze the document text and determine which block(s) of text are most relevant to answering the question with respect to the answer based on the context.
    For each block that's relevant, provide a relevance score from 0 to 1 where 1 means highly relevant.
    Also the blocks are at max 3 blocks.
    Strictly ensure to only provide the block bbox of the relavant blocks and the relevance score in JSON-like format and no other textual content, for example:
    [
    {{"block_bbox": "1 1 2 2", "relevance": 0.9}},
    {{"block_bbox": "2 3 4 4", "relevance": 0.4}}
    ]
    """

    messages = [ {"role": "user", "content": prompt}]
    result = pipe(messages, max_new_tokens=512, do_sample=True, temperature=0.7)
    # print(result[0]["generated_text"][1])
    # exit()
    ans = result[0]["generated_text"][1]['content']
    # print(question)
    # print(ans)
    return ans


INPUT_JSON_FILE = "/data/BADRI/FINAL/THESIS/GRVQA/main/outputs/json/doctr_llama_grounding_annotations.json"
OUTPUT_JSON_FILE = "/data/BADRI/FINAL/THESIS/GRVQA/main/outputs/json/doctr_llama_grounding_annotations.json"
IMG_DIR = "/data/BADRI/FINAL/THESIS/GRVQA/ANNOTATION/final/"

LEVEL = "block"

with open(INPUT_JSON_FILE, 'r') as f:
    data = json.load(f)

    


pipe = load_llm_model("cuda")


for image_name, qa_data in tqdm(data.items()):
    image_path  = os.path.join(IMG_DIR, image_name)
    if LEVEL == "block":
        predictions = perform_block_level_ocr(image_path)
    else:
        predictions = perform_ocr(image_path)

    for qa in tqdm(qa_data):

        if LEVEL == "block":
            if 'block_level_predictions' in qa and qa['block_level_predictions'] is not None:
                continue
            question = qa['question']
            answer = qa['answer']
            grounding_answer = generate_llm_answer(question, answer, predictions, pipe)
            try:
                grounding_answer = json.loads(grounding_answer)
                qa['block_level_predictions'] = grounding_answer
            except:
                qa['block_level_predictions'] = grounding_answer
                print(qa['id'])
        else:
            if 'line_level_predictions' in qa and qa['line_level_predictions'] is not None:
                continue
            question = qa['question']
            answer = qa['answer']
            grounding_answer = generate_llm_answer(question, answer, predictions, pipe)
            try:
                grounding_answer = json.loads(grounding_answer)
                qa['line_level_predictions'] = grounding_answer
            except:
                qa['line_level_predictions'] = grounding_answer
                print(qa['id'])

        # save json file dynamically
        with open(OUTPUT_JSON_FILE, 'w') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
    
with open(OUTPUT_JSON_FILE, 'w') as f:
    json.dump(data, f, indent=4, ensure_ascii=False)
