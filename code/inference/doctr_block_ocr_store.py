import os
import json
from PIL import ImageDraw, Image
from fuzzywuzzy import fuzz
from tqdm import tqdm

from surya.layout import LayoutPredictor

from doctr.io import DocumentFile
from doctr.models import ocr_predictor


layout_predictor = LayoutPredictor()
model = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True)

JSON_FILE = "/data/BADRI/FINAL/THESIS/GRVQA/main/outputs/json/doctr_grounding_annotations.json"
IMG_DIR = "/data/BADRI/FINAL/THESIS/GRVQA/ANNOTATION/final/"

OUTPUT_FILE = "/data/BADRI/FINAL/THESIS/GRVQA/main/outputs/intermediate/doctr_block_ocr_store.json"



with open(JSON_FILE, 'r') as f:
    data = json.load(f)

output_data = {}


for image_name, qa_data in tqdm(data.items()):
    IMG_PATH = os.path.join(IMG_DIR, image_name)


    image = Image.open(IMG_PATH)

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


    output_data[image_name] = predictions
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)


with open(OUTPUT_FILE, 'w') as f:
    json.dump(output_data, f, indent=4, ensure_ascii=False)




