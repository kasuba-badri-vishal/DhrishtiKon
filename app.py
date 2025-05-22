import streamlit as st
import time
import os
import shutil
import pymupdf
import json

st.set_page_config(
    page_title="Grounding Task Demo",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="logo.png"
)

# --- Simple Authentication ---
import streamlit as st
import time

# Define your valid credentials
VALID_USERS = {
    "iitb": "iitb123",
    "badri": "badri123"
}

def login():
    # Set a professional background for the whole app
    st.markdown(
        '''
        <style>
        body, .stApp {
            background: linear-gradient(120deg, #e0eafc 0%, #cfdef3 100%) !important;
        }
        .login-box {
            background: #fff;
            padding: 2.5em 2em 2em 2em;
            border-radius: 16px;
            box-shadow: 0 4px 24px rgba(80, 120, 200, 0.12);
            min-width: 320px;
            max-width: 90vw;
            margin: auto;
        }
        </style>
        ''', unsafe_allow_html=True
    )
    # Center the login box using columns
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        # st.markdown('<div class="login-box">', unsafe_allow_html=True)
        # image at center
        st.image("logo.png", width=800, use_container_width=False)
        st.markdown('<h2 style="text-align:center; color:#2b6cb0; margin-bottom:1.5em;">🔒 Please log in to access the app</h2>', unsafe_allow_html=True)
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        login_btn = st.button("Login")
        if login_btn:
            if username in VALID_USERS and VALID_USERS[username] == password:
                st.session_state["authenticated"] = True
                st.success("Login successful!")
                st.session_state["show_continue"] = True
            else:
                st.error("Invalid username or password")
        if st.session_state.get("show_continue", False):
            if st.button("Continue to App"):
                st.session_state["show_continue"] = False
                st.experimental_rerun() if hasattr(st, "experimental_rerun") else None
        st.markdown('</div>', unsafe_allow_html=True)

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    login()
    st.stop()
# --- End Authentication ---

# st.image("logo.png", width=250)

from PIL import Image, ImageDraw
import io
# from st_audiorec import st_audiorec

from surya.layout import LayoutPredictor
from doctr.models import ocr_predictor
from transformers import pipeline

@st.cache_resource
def get_layout_predictor():
    return LayoutPredictor()

@st.cache_resource
def get_ocr_model():
    return ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True)

@st.cache_resource
def get_llm_model(device):
    return pipeline("text-generation", model="meta-llama/Meta-Llama-3.1-8B-Instruct", device=device)

from predict_output import predict_output


layout_predictor = get_layout_predictor()
model = get_ocr_model()
pipe = get_llm_model("cuda")

print("Models loaded")

# --- Placeholder function for demo ---
def get_corresponding_bboxes(image, question):
    # Returns dummy bounding boxes and answer for demo
    # Each bbox: (x1, y1, x2, y2)
    w, h = image.size
    block_bboxes = [(w//8, h//8, w//2, h//2)]
    line_bboxes = [(w//4, h//4, w//2, h//3)]
    word_bboxes = [(w//3, h//3, w//2, h//2)]
    point_bboxes = [(w//2, h//2, w//2+5, h//2+5)]
    answer = "This is a demo answer."
    return block_bboxes, line_bboxes, word_bboxes, point_bboxes, answer

# --- Helper to draw bboxes ---
def draw_bboxes(image, bboxes, color):
    img = image.copy()
    draw = ImageDraw.Draw(img)
    for bbox in bboxes:
        draw.rectangle(bbox, outline=color, width=4)
    return img

def draw_points(image, bboxes, color):
    img = image.copy()
    draw = ImageDraw.Draw(img)
    for bbox in bboxes:
        # x1, y1, x2, y2 = bbox
        cx, cy = bbox[0], bbox[1]
        r = 6
        draw.ellipse((cx-r, cy-r, cx+r, cy+r), outline=color, width=4, fill=color)
    return img

# model_type = st.sidebar.checkbox("Use LLM Model", value=False)
# model_type = "llm" if model_type else "inhouse"

st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #f8fafc 0%, #e0e7ef 100%);
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stButton>button {
        background-color: #4F8BF9;
        color: white;
        border-radius: 8px;
        font-size: 1.1rem;
        padding: 0.5em 2em;
    }
    .stTextInput>div>input {
        border-radius: 8px;
        border: 1px solid #4F8BF9;
    }
    .stFileUploader>div>div {
        border-radius: 8px;
        border: 2px dashed #4F8BF9;
    }
    .stAudio>audio {
        width: 100% !important;
    }
    </style>
""", unsafe_allow_html=True)

col_logo, col_title = st.columns([1, 8])
with col_logo:
    st.image("logo.png", width=180)
with col_title:
    st.markdown("<h1 style='margin-bottom: 0;'>Question Answering with Visual Grounding</h1>", unsafe_allow_html=True)

# List of quotes (HTML formatted)
QUOTES = [
    '''<div style="color: #2b6cb0; font-size: 1.3em; font-weight: 500; margin-bottom: 1em;">
        "प्रत्यक्षं किं प्रमाणं?" <span style="font-size:0.9em; color:#444;">(<i>What better proof is there than direct perception?)</i></span>
    </div>''',
    '''<div style="color: #2b6cb0; font-size: 1.3em; font-weight: 500; margin-bottom: 1em;">
        <i>"Truth is not told—it is seen."</i>
    </div>'''
]

# Initialize session state for quote index and last update time
if "quote_index" not in st.session_state:
    st.session_state.quote_index = 0
    st.session_state.last_quote_time = time.time()

# Check if 5 seconds have passed
if time.time() - st.session_state.last_quote_time > 5:
    st.session_state.quote_index = (st.session_state.quote_index + 1) % len(QUOTES)
    st.session_state.last_quote_time = time.time()
    # Rerun the app to update the quote
    if hasattr(st, "experimental_rerun"):
        st.experimental_rerun()

# Display the current quote
st.markdown(QUOTES[st.session_state.quote_index], unsafe_allow_html=True)

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("1. Upload Image or pdf document")
    image = "Not Uploaded"
    uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg", "pdf"])
    if uploaded_file:
        current_dir = os.getcwd()
        temp_output_folder = os.path.join(current_dir, "temp_output_folder/")
        # delete the temp_output_folder
        if os.path.exists(temp_output_folder):
            shutil.rmtree(temp_output_folder)

        document_type = "image"
        if uploaded_file.type == "application/pdf":


            # save the uploaded file to a temp file
            temp_file_path = os.path.join(current_dir, "temp_file.pdf")

            # delete the temp_file_path
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            if not os.path.exists(temp_output_folder):
                os.makedirs(temp_output_folder)
            # output_file = simple_counter_generator("page", ".jpg")
            # convert_from_path(document_path, output_folder=temp_output_folder, dpi=300, fmt='jpeg', jpegopt= jpg_options, output_file=output_file)

            pages = 0
            doc = pymupdf.open(temp_file_path)  # open document
            for page in doc:  # iterate through the pages
                pages += 1
                pix = page.get_pixmap()  # render page to an image
                pix.save(f"{temp_output_folder}/{page.number}.png")

            if(pages == 1):
                document_type = "image"
                document_path = os.path.join(temp_output_folder, "0.png")
                uploaded_file = os.path.join(temp_output_folder, "0.png")
                image = Image.open(uploaded_file).convert("RGB")
            else:
                document_type = "pdf"
                # image = Image.open(uploaded_file).convert("RGB")
        
        if document_type == "image":
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_container_width=True)
            # Save uploaded image to a temp file for predict_output
            temp_file_path = "sample.png"
            image.save(temp_file_path)
        else:
            document_type = "pdf"
            document_path = uploaded_file.name
            image = "Uploaded PDF"
            # st.image(uploaded_file, caption="Uploaded PDF", use_container_width=True)
    else:
        image = "Not Uploaded"
        temp_output_folder = None
        st.image("https://placehold.co/400x300?text=Upload+Image", caption="Uploaded Image", use_container_width=True)

    st.subheader("2. Ask a question")
    question = st.text_input("Type your question here")
    
    # Add radio button for model selection
    model_type = st.radio(
        "Select Model Type:",
        options=["MGVG", "IndoDocs"],
        index=1,
        horizontal=True
    )

    run_demo = st.button("Run Grounding Demo", use_container_width=True)

# --- Output placeholders ---
with col2:
    st.subheader("3. Visual Grounding Outputs")
    if image!="Not Uploaded" and (question):
        print(image)
        print(question)
    if run_demo and image!="Not Uploaded" and (question):
        # Use text input only
        q = question
        answer, block_bboxes, line_bboxes, word_bboxes, point_bboxes, current_page = predict_output(
            temp_file_path, q, pipe, layout_predictor, model, model_type, document_type
        )


        # print(block_bboxes)
        # print(line_bboxes)
        # print(word_bboxes)
        # print(point_bboxes)
        print(answer)

        if(current_page != -1):
            image = Image.open(os.path.join(temp_output_folder, f"{current_page}.png")).convert("RGB")
        print("--------------------------------")
        print(image)

        block_img = draw_bboxes(image, block_bboxes, color="#4F8BF9")
        line_img = draw_bboxes(image, line_bboxes, color="#F97B4F")
        word_img = draw_bboxes(image, word_bboxes, color="#4FF9B2")
        point_img = draw_points(image, point_bboxes, color="#F94F8B")
        imgs = [block_img, line_img, word_img, point_img]
        labels = ["Block Level", "Line Level", "Word Level", "Point Level"]
        cols = st.columns(4)
        for i, (img, label) in enumerate(zip(imgs, labels)):
            with cols[i]:
                st.image(img, caption=label, use_container_width=True)
        st.markdown("""
        <div style='background: #f1f5fa; border-radius: 10px; padding: 1em 2em; border: 1.5px solid #4F8BF9;'>
            <h4 style='color: #4F8BF9;'>Predicted Answer:</h4>
            <p style='font-size: 1.2em; color: #222;'>""" + answer + """</p>
        </div>
        """, unsafe_allow_html=True)

        # --- Centered Save Results Button ---
        result_data = {
            "question": q,
            "answer": answer,
            "block_bboxes": block_bboxes,
            "line_bboxes": line_bboxes,
            "word_bboxes": word_bboxes,
            "point_bboxes": point_bboxes,
            "current_page": current_page
        }
        json_str = json.dumps(result_data, indent=2)
        col_left, col_center, col_right = st.columns([2, 3, 2])
        with col_center:
            st.download_button(
                label="Save Results as JSON",
                data=json_str,
                file_name="grounding_results.json",
                mime="application/json"
            )
    else:
        st.markdown("""
        <div style='display: flex; gap: 2em; flex-wrap: wrap;'>
            <div style='flex: 1; min-width: 220px;'>
                <img src='https://placehold.co/220x180?text=Block+Level' style='width:100%; border-radius: 10px; border: 2px solid #4F8BF9;'>
                <p style='text-align:center; font-weight:600;'>Block Level</p>
            </div>
            <div style='flex: 1; min-width: 220px;'>
                <img src='https://placehold.co/220x180?text=Line+Level' style='width:100%; border-radius: 10px; border: 2px solid #4F8BF9;'>
                <p style='text-align:center; font-weight:600;'>Line Level</p>
            </div>
            <div style='flex: 1; min-width: 220px;'>
                <img src='https://placehold.co/220x180?text=Word+Level' style='width:100%; border-radius: 10px; border: 2px solid #4F8BF9;'>
                <p style='text-align:center; font-weight:600;'>Word Level</p>
            </div>
            <div style='flex: 1; min-width: 220px;'>
                <img src='https://placehold.co/220x180?text=Point+Level' style='width:100%; border-radius: 10px; border: 2px solid #4F8BF9;'>
                <p style='text-align:center; font-weight:600;'>Point Level</p>
            </div>
        </div>
        <br>
        <div style='background: #f1f5fa; border-radius: 10px; padding: 1em 2em; border: 1.5px solid #4F8BF9;'>
            <h4 style='color: #4F8BF9;'>Predicted Answer:</h4>
            <p style='font-size: 1.2em; color: #222;'>[Answer will appear here]</p>
        </div>
        """, unsafe_allow_html=True) 
