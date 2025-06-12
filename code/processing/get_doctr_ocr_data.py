import os
from PIL import Image

from surya.layout import LayoutPredictor

from doctr.io import DocumentFile
from doctr.models import ocr_predictor

IMG_DIR = "/data/BADRI/FINAL/THESIS/GRVQA/ANNOTATION/final/"
OUT_JSON_FILE = "/data/BADRI/FINAL/THESIS/GRVQA/main/outputs/ocr/doctr_ocr_data.json"


layout_predictor = LayoutPredictor()
model = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True)


final_data = {}

for image_name in os.listdir(IMG_DIR):
    IMG_PATH = os.path.join(IMG_DIR, image_name)
    image = Image.open(IMG_PATH)

    block_predictions = []
    line_predictions = []

    
    layout_predictions = layout_predictor([image])
    doc = DocumentFile.from_images(IMG_PATH)
    result = model(doc)

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
        block_predictions.append(output)

    

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
                line_predictions.append(output)

    final_data[image_name] = {
        'block_predictions': block_predictions,
        'line_predictions': line_predictions
    }

with open(OUT_JSON_FILE, 'w') as f:
    json.dump(final_data, f, indent=4)
