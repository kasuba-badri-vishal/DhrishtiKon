# DrishtiKon Grounding Demo

A Streamlit-based web application for visual grounding and document understanding. This application allows users to upload images or PDFs and ask questions about their content, with the system providing answers along with visual grounding at different levels (block, line, word, and point).

## Features

- Image and PDF document upload support
- Multi-level visual grounding (Block, Line, Word, Point)
- Interactive chat interface
- Chat history management
- Secure login system
- Export results as JSON

## Setup

1. Clone the repository:
```bash
git clone https://github.com/kasuba-badri-vishal/DhrishtiKon.git
cd DhrishtiKon/application
# If you are using a virtual environment, activate it first
# For example, if using venv:
source venv/bin/activate  # On Linux/Mac
# or
venv\Scripts\activate  # On Windows
# If you don't have a virtual environment set up, you can create one:
python -m venv venv
# Then activate it as shown above
# If you are using conda, you can create an environment with:
conda create -n dhrishti_env python=3.8
conda activate dhrishti_env
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up secrets:
   - Create a `.streamlit/secrets.toml` file in the root directory of the application with the following content:
```toml
# Hugging Face API Token
hf_token = "YOUR_HUGGING_FACE_API_TOKEN"

# User Credentials
[credentials]
USERNAME1 = "PASSWORD1"
USERNAME2 = "PASSWORD2"
```
   - Ensure that the secrets are kept secure and not shared publicly.

4. Run the application:
```bash
streamlit run app.py
```

## Usage

1. Log in using the provided credentials
2. Upload an image or PDF document
3. Type your question about the document
4. View the results with visual grounding at different levels
5. Save results as JSON if needed

## Dependencies

All required dependencies are listed in `requirements.txt`. The main dependencies include:

## Deployment

To deploy this application on Streamlit Cloud:

1. Push your code to GitHub
2. Go to https://share.streamlit.io/
3. Connect your GitHub repository
4. Add your secrets in the Streamlit Cloud dashboard:
   - Go to your app's settings
   - Add the secrets from your `.streamlit/secrets.toml` file
5. Deploy the application

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

Badri Vishal Kasuba
22M2119