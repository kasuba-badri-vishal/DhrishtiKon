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
git clone <your-repo-url>
cd <repo-name>
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up secrets:
   - Copy `secrets.toml.template` to `.secrets.toml`
   - Get your Hugging Face token from https://huggingface.co/settings/tokens
   - Update the credentials in `.secrets.toml` with your desired username/password combinations
   ```bash
   cp secrets.toml.template .secrets.toml
   # Edit .secrets.toml with your credentials
   ```

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
- Streamlit
- PyTorch
- Transformers
- OpenCV
- PyMuPDF
- and more...

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

[Your License Here]

## Contact

[Your Contact Information] 