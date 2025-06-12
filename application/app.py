import streamlit as st
import time
import os
import shutil
import pymupdf
import json
import datetime
import logging
import sys

from predict_output import clear_prediction_caches, predict_output

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="DrishtiKon Grounding Demo",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="data/logo.png"
)

# Load credentials from secrets
try:
    VALID_USERS = st.secrets["credentials"]
except Exception as e:
    logger.error(f"Error loading credentials: {e}")
    st.error("Error loading credentials. Please check your secrets configuration.")
    st.stop()

# Import ML-related libraries with error handling
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    from PIL import Image, ImageDraw
    from surya.layout import LayoutPredictor
    from doctr.models import ocr_predictor
except ImportError as e:
    logger.error(f"Error importing ML libraries: {e}")
    st.error("Error loading required ML libraries. Please check the installation.")
    st.stop()

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
        st.image("data/logo.png", width=800, use_container_width=False)
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

# st.image("data/logo.png", width=250)

@st.cache_resource
def get_layout_predictor():
    try:
        return LayoutPredictor()
    except Exception as e:
        logger.error(f"Error loading layout predictor: {e}")
        st.error("Error loading layout predictor model. Please try again later.")
        return None

@st.cache_resource
def get_ocr_model():
    try:
        return ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True)
    except Exception as e:
        logger.error(f"Error loading OCR model: {e}")
        st.error("Error loading OCR model. Please try again later.")
        return None

@st.cache_resource
def get_llm_model(device):
    try:
        model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        # First try to load tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                token=st.secrets["hf_token"],
                trust_remote_code=True
            )
        except Exception as e:
            logger.error(f"Error loading tokenizer: {e}")
            st.error("Error loading tokenizer. Please check your Hugging Face token.")
            return None

        # Then try to load model
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                load_in_4bit=True,
                torch_dtype="auto",
                token=st.secrets["hf_token"],
                trust_remote_code=True
            )
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            st.error("Error loading language model. Please check your Hugging Face token.")
            return None

        return pipeline("text-generation", model=model, tokenizer=tokenizer)
    except Exception as e:
        logger.error(f"Error in get_llm_model: {e}")
        st.error("Error initializing language model pipeline.")
        return None

# Initialize models with error handling
try:
    with st.spinner("Loading models... This might take a few minutes."):
        layout_predictor = get_layout_predictor()
        model = get_ocr_model()
        pipe = get_llm_model("cuda")
        
        if not all([layout_predictor, model, pipe]):
            st.error("Failed to initialize one or more models. Please try again later.")
            st.stop()
except Exception as e:
    logger.error(f"Error initializing models: {e}")
    st.error("Error initializing models. Please try again later.")
    st.stop()

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
    # width proportional to the image size
    width = int(img.width/100)
    draw = ImageDraw.Draw(img)
    for bbox in bboxes:
        draw.rectangle(bbox, outline=color, width=width)
    return img

def draw_points(image, bboxes, color):
    img = image.copy()
    width = int(img.width)
    draw = ImageDraw.Draw(img)
    for bbox in bboxes:
        # x1, y1, x2, y2 = bbox
        cx, cy = bbox[0], bbox[1]
        # r being relative to the image size
        r = int(img.width/100)
        draw.ellipse((cx-r, cy-r, cx+r, cy+r), outline=color, width=width, fill=color)
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
        transition: background 0.2s;
        box-shadow: 0 2px 8px rgba(80, 120, 200, 0.08);
    }
    .stButton>button:hover {
        background-color: #2b6cb0;
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
    .stDownloadButton>button {
        background-color: #4FF9B2;
        color: #222;
        border-radius: 8px;
        font-size: 1.1rem;
        padding: 0.5em 2em;
        margin-top: 1em;
        transition: background 0.2s;
    }
    .stDownloadButton>button:hover {
        background-color: #2b6cb0;
        color: #fff;
    }
    </style>
""", unsafe_allow_html=True)

# Center the title image
col_left, col_center, col_right = st.columns([1, 4, 1])
with col_center:
    st.image("data/title.png", width=800, use_container_width=True)

# List of quotes (HTML formatted)
QUOTES = [
    '''<div style="color: #2b6cb0; font-size: 1.3em; font-weight: 500; margin-bottom: 1em;">
        "प्रत्यक्षं किं प्रमाणं?" <span style="font-size:0.9em; color:#444;">(<i>What better proof is there than direct perception?)</i></span>
    </div>''',
    '''<div style="color: #2b6cb0; font-size: 1.3em; font-weight: 500; margin-bottom: 1em;">
        <i>"Truth is not told—it is seen."</i>
    </div>''',
    '''<div style="color: #2b6cb0; font-size: 1.3em; font-weight: 500; margin-bottom: 1em;">
        "दृष्टया बुद्धिर्जायते" <span style="font-size:0.9em; color:#444;">(<i>From vision arises true understanding.</i>)</span>
    </div>''',

    '''<div style="color: #2b6cb0; font-size: 1.3em; font-weight: 500; margin-bottom: 1em;">
        "तत्त्वमस्यि" — <i>That thou art</i>, seen not in words but in awakened awareness.
    </div>''',

    '''<div style="color: #2b6cb0; font-size: 1.3em; font-weight: 500; margin-bottom: 1em;">
        "न चक्षुषा गृह्यते" <span style="font-size:0.9em; color:#444;">(<i>Not grasped by the eye, yet vision leads the way to it.)</i></span>
    </div>''',

    '''<div style="color: #2b6cb0; font-size: 1.3em; font-weight: 500; margin-bottom: 1em;">
        <i>"To see rightly is to know rightly. The seer and the seen are not two."</i>
    </div>''',

    '''<div style="color: #2b6cb0; font-size: 1.3em; font-weight: 500; margin-bottom: 1em;">
        "दर्शने दर्शनमेव साधनम्" <span style="font-size:0.9em; color:#444;">(<i>In vision alone is the path of vision fulfilled.)</i></span>
    </div>''',

    '''<div style="color: #2b6cb0; font-size: 1.3em; font-weight: 500; margin-bottom: 1em;">
        <i>"What the eye of wisdom sees, no scripture can fully tell."</i>
    </div>'''
]

# Initialize session state for quote index and last update time
if "quote_index" not in st.session_state:
    st.session_state.quote_index = 0
    st.session_state.last_quote_time = time.time()

# Check if 2 seconds have passed
if time.time() - st.session_state.last_quote_time > 2:
    st.session_state.quote_index = (st.session_state.quote_index + 1) % len(QUOTES)
    st.session_state.last_quote_time = time.time()
    # Rerun the app to update the quote
    if hasattr(st, "experimental_rerun"):
        st.experimental_rerun()

# Centered quote with animation
col_left, col_center, col_right = st.columns([1, 3, 1])
with col_center:
    st.markdown(
        f"""
        <div style='
            display: flex; 
            justify-content: center; 
            align-items: center; 
            min-height: 80px; 
            font-size: 1em; 
            font-weight: 300; 
            color: #2b6cb0; 
            text-align: center;
            animation: fadeIn 1s;
        '>
            {QUOTES[st.session_state.quote_index]}
        </div>
        <style>
        @keyframes fadeIn {{
            from {{ opacity: 0; }}
            to {{ opacity: 1; }}
        }}
        </style>
        """, unsafe_allow_html=True
    )

quote_progress = min((time.time() - st.session_state.last_quote_time) / 2, 1.0)
st.progress(quote_progress)

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("<h2 style='color:#4F8BF9; font-size:2em; margin-bottom:0.5em;'>📤 1. Upload Image or PDF</h2>", unsafe_allow_html=True)
    image = "Not Uploaded"
    uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg", "pdf"])
    show_uploaded = False
    if uploaded_file:
        current_dir = os.getcwd()
        temp_output_folder = os.path.join(current_dir, "temp_output_folder/")
        # delete the temp_output_folder
        if os.path.exists(temp_output_folder):
            shutil.rmtree(temp_output_folder)
        
        clear_prediction_caches()  # Clear caches when new file is uploaded

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
        if document_type == "image":
            image = Image.open(uploaded_file).convert("RGB")
            show_uploaded = st.checkbox("Show Uploaded Image", value=True)
            if show_uploaded:
                st.image(image, caption="Uploaded Image", use_container_width=True)
            # Save uploaded image to a temp file for predict_output
            temp_file_path = "sample.png"
            image.save(temp_file_path)
        else:
            document_type = "pdf"
            document_path = uploaded_file.name
            show_uploaded = st.checkbox("Show Uploaded PDF Pages", value=True)
            if show_uploaded and temp_output_folder is not None:
                page_files = sorted([f for f in os.listdir(temp_output_folder) if f.endswith('.png')], key=lambda x: int(x.split('.')[0]))
                if page_files:
                    st.image([os.path.join(temp_output_folder, f) for f in page_files], caption=[f"Page {i}" for i in range(len(page_files))], use_container_width=True)
                else:
                    st.info("No PDF pages found.")
            image = "Uploaded PDF"
    else:
        image = "Not Uploaded"
        temp_output_folder = None
        st.image("https://placehold.co/400x300?text=Upload+Image", caption="Uploaded Image", use_container_width=True)

    st.subheader("2. Ask a question")
    question = st.text_input("Type your question here", help="Ask a question about the uploaded document.")
    
    # # Add radio button for model selection
    # model_type = st.radio(
    #     "Select Model Type:",
    #     options=["Drishtikon", "Param"],
    #     index=1,
    #     horizontal=True
    # )

    model_type = "Drishtikon"

    run_demo = st.button("Run Grounding Demo", use_container_width=True)

# --- Chat Management ---

# Initialize session state for chat management
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []  # List of dicts: {question, answer, timestamp}
if 'all_chats' not in st.session_state:
    st.session_state['all_chats'] = []  # List of chat histories
if 'current_chat_index' not in st.session_state:
    st.session_state['current_chat_index'] = None  # None means current chat

# --- Sidebar: Chat Controls and History ---
with st.sidebar:
    st.markdown("<h3 style='color:#4F8BF9;'>💬 Chat Controls</h3>", unsafe_allow_html=True)
    if st.button('🆕 New Chat', key='sidebar_new_chat'):
        if st.session_state['chat_history']:
            st.session_state['all_chats'].append(list(st.session_state['chat_history']))
        st.session_state['chat_history'] = []
        st.session_state['current_chat_index'] = None
        clear_prediction_caches()  # Clear caches when new chat is started
    if st.button('🗑️ Clear Chat', key='sidebar_clear_chat'):
        st.session_state['chat_history'] = []
        st.session_state['current_chat_index'] = None
    if st.button('🚪 Logout', key='sidebar_logout'):
        st.session_state['authenticated'] = False
        st.experimental_rerun() if hasattr(st, "experimental_rerun") else None
    st.markdown("---")
    st.markdown("<h4 style='color:#4F8BF9;'>🗂️ Chat History</h4>", unsafe_allow_html=True)
    if st.session_state['all_chats']:
        chat_options = [f"Chat {i+1} ({len(chat)} Q&A)" for i, chat in enumerate(st.session_state['all_chats'])]
        selected = st.selectbox('View Old Chats', options=["Current Chat"] + chat_options, key='chat_select_sidebar')
        if selected != "Current Chat":
            idx = chat_options.index(selected)
            st.session_state['current_chat_index'] = idx
        else:
            st.session_state['current_chat_index'] = None
    else:
        st.session_state['current_chat_index'] = None

# Function to get the chat to display
if st.session_state['current_chat_index'] is not None:
    chat_to_display = st.session_state['all_chats'][st.session_state['current_chat_index']]
else:
    chat_to_display = st.session_state['chat_history']

# --- Main Area: Image/PDF View Option ---
with col2:
    st.subheader("3. Visual Grounding Outputs")
    # Display chat history as Q&A cards
    if chat_to_display:
        for i, qa in enumerate(chat_to_display):
            st.markdown(f"""
            <div style='background: #f8fafc; border-radius: 10px; padding: 1em 1.5em; margin-bottom: 0.7em; border: 1.5px solid #4F8BF9;'>
                <div style='color:#4F8BF9; font-weight:600;'>Q{i+1}: {qa['question']}</div>
                <div style='color:#222; margin-top:0.5em;'><b>Answer:</b> {qa['answer']}</div>
                <div style='font-size:0.9em; color:#888; margin-top:0.3em;'>🕒 {qa['timestamp']}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No chat history yet. Ask a question to get started!")

    if image!="Not Uploaded" and (question):
        print(image)
        print(question)
    if run_demo and image!="Not Uploaded" and (question):
        # Use text input only
        q = question
        with st.spinner("Running Visual Grounding..."):
            answer, block_bboxes, line_bboxes, word_bboxes, point_bboxes, current_page = predict_output(
                temp_file_path, q, pipe, layout_predictor, model, model_type, document_type
            )

        # Append Q&A to chat history
        st.session_state['chat_history'].append({
            'question': q,
            'answer': answer,
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'block_bboxes': block_bboxes,
            'line_bboxes': line_bboxes,
            'word_bboxes': word_bboxes,
            'point_bboxes': point_bboxes,
            'current_page': current_page
        })

        print(answer)

        if(current_page != -1):
            image = Image.open(os.path.join(temp_output_folder, f"{current_page}.png")).convert("RGB")
        print("--------------------------------")
        print(image)

        block_img = draw_bboxes(image, block_bboxes, color="#4F8BF9")
        line_img = draw_bboxes(image, line_bboxes, color="#F97B4F")
        word_img = draw_bboxes(image, word_bboxes, color="#4FF9B2")
        point_img = draw_points(image, point_bboxes, color="#FFFF00")
        imgs = [block_img, line_img, word_img, point_img]
        labels = ["Block Level", "Line Level", "Word Level", "Point Level"]
        cols = st.columns(4)
        for i, (img, label) in enumerate(zip(imgs, labels)):
            with cols[i]:
                st.image(img, caption=label, use_container_width=True)
        answer_lines = answer.splitlines()
        st.markdown("""
        <div style='background: #f1f5fa; border-radius: 10px; padding: 1em 2em; border: 1.5px solid #4F8BF9;'>
            <h4 style='color: #4F8BF9;'>Predicted Answer:</h4>
            <p style='font-size: 1.2em; color: #222;'>""" + "<br>".join(answer_lines) + """</p>
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
        <div style='display: flex; gap: 1em; justify-content: space-between;'>
            <div style='flex: 1; min-width: 0;'>
                <img src='https://placehold.co/220x180?text=Block+Level' style='width:100%; border-radius: 10px; border: 2px solid #4F8BF9;'>
                <p style='text-align:center; font-weight:600;'>Block Level</p>
            </div>
            <div style='flex: 1; min-width: 0;'>
                <img src='https://placehold.co/220x180?text=Line+Level' style='width:100%; border-radius: 10px; border: 2px solid #4F8BF9;'>
                <p style='text-align:center; font-weight:600;'>Line Level</p>
            </div>
            <div style='flex: 1; min-width: 0;'>
                <img src='https://placehold.co/220x180?text=Word+Level' style='width:100%; border-radius: 10px; border: 2px solid #4F8BF9;'>
                <p style='text-align:center; font-weight:600;'>Word Level</p>
            </div>
            <div style='flex: 1; min-width: 0;'>
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