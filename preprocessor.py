import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import fitz  # PyMuPDF

# Download required NLTK data (only runs once)
nltk.download('stopwords', quiet=True)

# Initialize stemmer and stopwords once (reused across calls)
stemmer = PorterStemmer()
STOPWORDS = set(stopwords.words('english'))


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract raw text from a PDF file using PyMuPDF.
    Returns empty string if extraction fails.
    """
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        print(f"[ERROR] Could not read PDF: {pdf_path} — {e}")
        return ""


def extract_name_from_pdf(pdf_path: str) -> str:
    """
    Extract the candidate's name from a resume PDF.
    Looks for capitalized words at the beginning of the document.
    Returns empty string if extraction fails.
    """
    try:
        raw_text = extract_text_from_pdf(pdf_path)
        if not raw_text:
            return ""
        
        # Get first 500 characters (name is typically at the top)
        first_section = raw_text[:500]
        
        # Split into lines and get the first non-empty line
        lines = first_section.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Look for a line with 1-4 capitalized words (typical name format)
            words = line.split()
            if words:
                # Filter words that are NOT all caps (avoid headers) and not too short
                name_words = [
                    w for w in words[:4]  # Max 4 words in a name
                    if w and (w[0].isupper() or w[0].isdigit()) and len(w) > 1
                ]
                
                if name_words:
                    candidate_name = ' '.join(name_words)
                    # Only return if it looks like a name (not all numbers, reasonable length)
                    if not candidate_name.isdigit() and len(candidate_name) > 2 and len(candidate_name) < 60:
                        return candidate_name
        
        return ""
    except Exception as e:
        print(f"[ERROR] Could not extract name from PDF: {pdf_path} — {e}")
        return ""


def clean_text(text: str) -> str:
    """
    Clean and normalize raw resume text.
    Steps:
      1. Lowercase
      2. Remove URLs
      3. Remove emails
      4. Remove phone numbers
      5. Remove special characters and punctuation
      6. Remove extra whitespace
      7. Remove stopwords
      8. Apply stemming
    """
    if not isinstance(text, str) or text.strip() == "":
        return ""

    # 1. Lowercase
    text = text.lower()

    # 2. Remove URLs (http, https, www)
    text = re.sub(r'http\S+|www\.\S+', ' ', text)

    # 3. Remove email addresses
    text = re.sub(r'\S+@\S+', ' ', text)

    # 4. Remove phone numbers (various formats)
    text = re.sub(r'(\+?\d[\d\s\-().]{7,}\d)', ' ', text)

    # 5. Remove special characters, punctuation, and digits
    text = re.sub(r'[^a-z\s]', ' ', text)

    # 6. Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # 7. Tokenize, remove stopwords, and apply stemming
    tokens = text.split()
    tokens = [
        stemmer.stem(word)
        for word in tokens
        if word not in STOPWORDS and len(word) > 2
    ]

    return ' '.join(tokens)


def preprocess_resume_text(raw_text: str) -> str:
    """
    Full preprocessing pipeline for a resume given as plain text.
    Use this for resumes coming from the CSV (Resume_str column).
    """
    return clean_text(raw_text)


def preprocess_resume_pdf(pdf_path: str) -> str:
    """
    Full preprocessing pipeline for a resume given as a PDF file.
    Use this when a recruiter uploads a PDF via the Streamlit interface.
    """
    raw_text = extract_text_from_pdf(pdf_path)
    return clean_text(raw_text)


# ── Quick test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import pandas as pd

    df = pd.read_csv('data/Resume.csv')
    sample_raw = df['Resume_str'].iloc[0]
    sample_category = df['Category'].iloc[0]

    print("=== CATEGORY ===")
    print(sample_category)

    print("\n=== RAW TEXT (first 300 chars) ===")
    print(sample_raw[:300])

    cleaned = preprocess_resume_text(sample_raw)
    print("\n=== CLEANED TEXT (first 300 chars) ===")
    print(cleaned[:300])

    print("\n=== TOKEN COUNT ===")
    print(f"Before: {len(sample_raw.split())} words")
    print(f"After:  {len(cleaned.split())} words")