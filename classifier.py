import torch
import torch.nn as nn
import numpy as np
import joblib
import os
from sentence_transformers import SentenceTransformer
from preprocessor import preprocess_resume_text, preprocess_resume_pdf

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_DIR          = "model"
BERT_MODEL_PT      = os.path.join(MODEL_DIR, "bert_full_model.pt")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")
BERT_NAME_PATH     = os.path.join(MODEL_DIR, "bert_model_name.txt")

# ── Confidence threshold for rejection ───────────────────────────────────────
REJECTION_THRESHOLD = 0.40   # below 40% confidence → rejected


# ── Neural classification head (must match Colab architecture) ────────────────
class ResumeClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)


# ── Load everything once at import time ──────────────────────────────────────
def _load_pipeline():
    """
    Load BERT encoder + neural classifier + label encoder.
    Called once when the module is first imported.
    """
    print("[Classifier] Loading models...")

    # 1. Load label encoder
    le = joblib.load(LABEL_ENCODER_PATH)
    num_classes = len(le.classes_)
    print(f"[Classifier] {num_classes} categories loaded.")

    # 2. Load BERT sentence encoder name
    with open(BERT_NAME_PATH, "r") as f:
        bert_name = f.read().strip()
    print(f"[Classifier] Loading BERT encoder: {bert_name}")
    bert_encoder = SentenceTransformer(bert_name)

    # 3. Load PyTorch neural head
    checkpoint  = torch.load(BERT_MODEL_PT, map_location=torch.device('cpu'))
    state_dict  = checkpoint['state_dict']
    input_dim   = checkpoint['input_dim']
    num_classes = checkpoint['num_classes']

    model = ResumeClassifier(input_dim, num_classes)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"[Classifier] Neural head loaded ({input_dim}→{num_classes}).")

    return bert_encoder, model, le


# Load once globally
_bert_encoder, _neural_model, _label_encoder = _load_pipeline()


# ── Core prediction function ──────────────────────────────────────────────────
def predict(cleaned_text: str) -> dict:
    """
    Predict the job category for a preprocessed resume text.

    Returns a dict:
    {
        'category':   str   — predicted job category or 'Rejected',
        'confidence': float — confidence score (0.0 to 1.0),
        'rejected':   bool  — True if below threshold,
        'all_scores': dict  — confidence for every category
    }
    """
    if not cleaned_text or cleaned_text.strip() == "":
        return {
            'category':   'Rejected',
            'confidence': 0.0,
            'rejected':   True,
            'all_scores': {}
        }

    # 1. Generate BERT embedding
    embedding = _bert_encoder.encode(
        [cleaned_text],
        batch_size=1,
        show_progress_bar=False
    )

    # 2. Run through neural classifier
    tensor = torch.tensor(embedding, dtype=torch.float32)
    with torch.no_grad():
        logits      = _neural_model(tensor)
        probs       = torch.softmax(logits, dim=1).numpy()[0]

    # 3. Get predicted class and confidence
    pred_idx    = int(np.argmax(probs))
    confidence  = float(probs[pred_idx])
    category    = _label_encoder.inverse_transform([pred_idx])[0]

    # 4. Build all scores dict (sorted by confidence)
    all_scores = {
        _label_encoder.inverse_transform([i])[0]: float(probs[i])
        for i in range(len(probs))
    }
    all_scores = dict(sorted(all_scores.items(), key=lambda x: x[1], reverse=True))

    # 5. Apply rejection threshold
    rejected = confidence < REJECTION_THRESHOLD

    return {
        'category':   'Rejected' if rejected else category,
        'confidence': confidence,
        'rejected':   rejected,
        'all_scores': all_scores
    }


def classify_from_text(raw_text: str) -> dict:
    """
    Full pipeline for a resume given as raw plain text.
    Handles preprocessing + prediction.
    """
    cleaned = preprocess_resume_text(raw_text)
    result  = predict(cleaned)
    result['input_type'] = 'text'
    return result


def classify_from_pdf(pdf_path: str) -> dict:
    """
    Full pipeline for a resume given as a PDF file path.
    Handles extraction + preprocessing + prediction.
    """
    cleaned = preprocess_resume_pdf(pdf_path)
    result  = predict(cleaned)
    result['input_type'] = 'pdf'
    return result


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import pandas as pd

    df = pd.read_csv('data/Resume.csv')

    print("\n" + "=" * 55)
    print("  CLASSIFIER TEST — 5 random resumes")
    print("=" * 55)

    sample = df.sample(5, random_state=99)

    correct = 0
    for _, row in sample.iterrows():
        result = classify_from_text(row['Resume_str'])
        actual = row['Category']
        match  = "✅" if result['category'] == actual else "❌"
        print(f"\n  Actual   : {actual}")
        print(f"  Predicted: {result['category']}  {match}")
        print(f"  Confidence: {result['confidence']*100:.1f}%")
        print(f"  Rejected : {result['rejected']}")
        if result['category'] == actual:
            correct += 1

    print(f"\n  Accuracy on sample: {correct}/5")
    print("=" * 55)