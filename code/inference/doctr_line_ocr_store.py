import os
import json
from fuzzywuzzy import fuzz
from tqdm import tqdm

from doctr.io import DocumentFile
from doctr.models import ocr_predictor


model = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True)


JSON_FILE = "/data/BADRI/FINAL/THESIS/GRVQA/main/outputs/json/filtered_grounding_annotations.json"
IMG_DIR = "/data/BADRI/FINAL/THESIS/GRVQA/ANNOTATION/final/"

OUTPUT_FILE = "/data/BADRI/FINAL/THESIS/GRVQA/main/outputs/intermediate/doctr_line_ocr_store.json"




with open(JSON_FILE, 'r') as f:
    data = json.load(f)


output_data = {}

for image_name, qa_data in tqdm(data.items()):
    IMG_PATH = os.path.join(IMG_DIR, image_name)
    doc = DocumentFile.from_images(IMG_PATH)
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
                predictions.append(output)

    output_data[image_name] = predictions


    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)


with open(OUTPUT_FILE, 'w') as f:
    json.dump(output_data, f, indent=4, ensure_ascii=False)