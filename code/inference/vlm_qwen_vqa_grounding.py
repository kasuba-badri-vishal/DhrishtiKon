from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import json
from tqdm import tqdm
import os

from doctr.io import DocumentFile
from doctr.models import ocr_predictor

model = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True)


device = "cuda" if torch.cuda.is_available() else "cpu"

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    torch_dtype=torch.bfloat16,
    # attn_implementation="flash_attention_2",
    device_map="auto",
)

# model.to(device)
model.bfloat16()
model.eval()

min_pixels = 256*28*28
max_pixels = 1280*28*28
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-72B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)



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
                output = " ".join(sent)
                predictions.append(output)
    return predictions

def custom_hf_inference(messages):
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt",)

    # Ensure all inputs are moved to the same device

    inputs = inputs.to(device)


    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=256)
    generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    return output_text




INPUT_JSON_FILE = "/data/BADRI/FINAL/THESIS/GRVQA/main/outputs/json/filtered_grounding_annotations.json"
IMG_DIR = "/data/BADRI/FINAL/THESIS/GRVQA/ANNOTATION/final/"
OUT_JSON_FILE = "/data/BADRI/FINAL/THESIS/GRVQA/main/outputs/json/qwen_grounding_annotations.json"

with open(INPUT_JSON_FILE, 'r') as f:
    data = json.load(f)


for image_name, qa_data in tqdm(data.items()):
    image_path  = os.path.join(IMG_DIR, image_name)
    # predictions = perform_ocr(image_path)

    for qa in tqdm(qa_data):

        if 'line_level_predictions' in qa and qa['line_level_predictions'] is not None:
            continue
        question = qa['question']
        answer = qa['answer']

        prompt = f"""You are analyzing text extracted from an image. Find the line(s) of text that best matches the question and answer.
            Question: {question}
            Answer: {answer}

            Analyze the document text and determine which lines are most relevant to answering the question with respect to the provided answer.
            For each line that's relevant, provide a relevance score from 0 to 1 where 1 means highly relevant.
            The lines are at max 10 lines.
            Strictly ensure to only provide the line bbox of the line and the relevance score in JSON-like format and no other textual content, for example:
            [
            {{"line_bbox": "1 1 2 2", "relevance": 0.9}},
            {{"line_bbox": "2 3 4 4", "relevance": 0.4}}
            ]
            """

        message = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

       


        grounding_answer = custom_hf_inference(message)
        print(grounding_answer)
        grounding_answer = grounding_answer[0]
        
        try:
            grounding_answer = json.loads(grounding_answer)
            qa['line_level_predictions'] = grounding_answer
        except:
            qa['line_level_predictions'] = grounding_answer
            print(grounding_answer)
            print(qa['id'])
            # exit()

        # save json file dynamically
        with open(OUT_JSON_FILE, 'w') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
    
with open(OUT_JSON_FILE, 'w') as f:
    json.dump(data, f, indent=4, ensure_ascii=False)
