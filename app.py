from flask import Flask, request, jsonify
from flask_cors import CORS  # To allow React to connect
import os
import PyPDF2
import requests
import logging

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

logging.basicConfig(level=logging.INFO)

# Constants
BASE_FOLDER_PATH = "documents/"
API_KEY = os.getenv("GEMINI_API_KEY")  # Store in environment variable
MODEL_NAME = "gemini-2.0-flash"
API_URL = f"https://generativelanguage.googleapis.com/v1/models/{MODEL_NAME}:generateContent?key={API_KEY}"

def extract_text_from_pdf_path(file_path):
    """Extract text from a PDF file."""
    try:
        with open(file_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = "\n".join([page.extract_text() or "" for page in pdf_reader.pages])
            return text.strip()
    except Exception as e:
        logging.error(f"Error extracting text from {file_path}: {e}")
        return ""

@app.route('/load_documents', methods=['GET'])
def load_documents():
    """Load all PDFs from the folder and extract text."""
    documents = {}
    for filename in os.listdir(BASE_FOLDER_PATH):
        if filename.lower().endswith('.pdf'):
            file_path = os.path.join(BASE_FOLDER_PATH, filename)
            text = extract_text_from_pdf_path(file_path)
            if text:
                documents[filename] = text
    return jsonify(documents)

@app.route('/chat', methods=['POST'])
def query_gemini():
    """Query the Gemini AI API with context and user question."""
    data = request.json
    context = data.get("context", "")
    prompt = data.get("prompt", "")

    full_prompt = f"Context:\n{context}\n\nQuestion: {prompt}"
    payload = {"contents": [{"parts": [{"text": full_prompt}]}]}
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(API_URL, json=payload, headers=headers)
        if response.status_code == 200:
            data = response.json()
            return jsonify({"response": data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "No response")})
        else:
            logging.error(f"API error: {response.status_code}, {response.text}")
            return jsonify({"error": f"API returned {response.status_code}"}), response.status_code
    except Exception as e:
        logging.error(f"Error making API request: {e}")
        return jsonify({"error": "Unable to connect to the API"}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
