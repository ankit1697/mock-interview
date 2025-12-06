"""
resume_parser.py
----------------
This module securely fetches the OpenAI API key from GCP Secret Manager,
reads a PDF/DOCX resume, extracts text, and uses GPT-4o to parse it into structured JSON.

Usage:
----------------------------
from resume_parser import read_resume, parse_resume_fields

file_path = "../Ankit_Agrawal_Data_Scientist.pdf"
resume_text = read_resume(file_path)
parsed_resume = parse_resume_fields(resume_text)
print(parsed_resume)
"""

import os
import json
try:
    import fitz  # PyMuPDF
    FITZ_AVAILABLE = True
except ImportError:
    FITZ_AVAILABLE = False
try:
    import docx
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv(), override=True)

# Get OpenAI API key from environment
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise RuntimeError("No OpenAI API key found in environment variables. Please set OPENAI_API_KEY.")

client = OpenAI(api_key=openai_api_key)


# ============================================================
# 3. Resume Text Extraction
# ============================================================
def read_resume(file_path: str) -> str:
    """
    Reads a PDF or DOCX file and extracts text.
    """
    if file_path.endswith(".pdf"):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith(".docx"):
        return extract_text_from_docx(file_path)
    else:
        raise ValueError("Unsupported file format. Please upload a .pdf or .docx file.")


def extract_text_from_pdf(file_path: str) -> str:
    """
    Extracts text from a PDF using PyMuPDF.
    """
    if not FITZ_AVAILABLE:
        raise ImportError("PyMuPDF (fitz) is not installed. Please install it with: pip install PyMuPDF")
    text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            text += page.get_text()
    return text.strip()


def extract_text_from_docx(file_path: str) -> str:
    """
    Extracts text from a DOCX file using python-docx.
    """
    if not DOCX_AVAILABLE:
        raise ImportError("python-docx is not installed. Please install it with: pip install python-docx")
    document = docx.Document(file_path)
    return "\n".join([para.text for para in document.paragraphs]).strip()


def parse_resume_fields(resume_text: str) -> dict:
    """
    Sends extracted resume text to GPT-4o and returns structured JSON fields.
    """
    system_prompt = """
You are an AI resume parser. Extract the following fields from the given resume text:

1. Name
2. Email
3. Phone Number
4. Professional Summary
5. Professional Projects
6. Skills
7. Academic Projects
8. Work Experience
9. Total Experience (in years in a numerical format)
10. List of Core skills (e.g., “XGBoost”, “Forecasting”, “Optimization”)
11. List of Project domains (e.g., “Donor forecasting model”, “Route optimization”)
12. List of Work context (e.g., “BioLife Plasma Services”, “Healthcare analytics”)

Return the result as a JSON object with these keys.
"""

    # Truncate overly long resumes (approx 8000 chars)
    MAX_CHARS = 8000
    resume_text = resume_text[:MAX_CHARS]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": resume_text},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
    )

    try:
        result = json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"⚠️ Failed to parse model output: {e}")
        print(response.choices[0].message.content)
        result = {}

    return result


def parse_resume_from_file(file_path: str) -> dict:
    """
    High-level convenience function:
    1. Reads the resume file (PDF or DOCX)
    2. Sends extracted text to GPT-4o for parsing
    3. Returns the structured JSON result
    """
    resume_text = read_resume(file_path)
    parsed_resume = parse_resume_fields(resume_text)
    return parsed_resume