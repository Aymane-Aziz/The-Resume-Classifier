import pandas as pd
import numpy as np
import joblib
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

from preprocessor import preprocess_resume_text

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

# ── Config ───────────────────────────────────────────────────────────────────
DATA_PATH   = "data/Resume.csv"
MODEL_DIR   = "model"
BEST_MODEL  = os.path.join(MODEL_DIR, "best_model.pkl")
RESULTS_DIR = "results"
TEST_SIZE   = 0.20
RANDOM_SEED = 42

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── 1. Load & preprocess data ─────────────────────────────────────────────────
print("=" * 60)
print("  RESUME CLASSIFIER — TRAINING PIPELINE")
print("=" * 60)

print("\n[1/5] Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"      {len(df)} resumes | {df['Category'].nunique()} categories")

print("\n[2/5] Preprocessing resumes (this may take ~1 min)...")
df['cleaned'] = df['Resume_str'].apply(preprocess_resume_text)
print(f"      Done. Sample: '{df['cleaned'].iloc[0][:80]}...'")

X = df['cleaned']
y = df['Category']

# ── 2. Train / test split ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
)
print(f"\n      Train: {len(X_train)} samples | Test: {len(X_test)} samples")

# ── 3. Define the 5 models ────────────────────────────────────────────────────
tfidf = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)

MODELS = {
    "Logistic Regression": LogisticRegression(
    max_iter=1000,
    random_state=RANDOM_SEED,
    solver='lbfgs'
    ),
    "SVM": SVC(
    kernel='linear',
    probability=True,
    random_state=RANDOM_SEED,
    max_iter=2000,
    cache_size=500
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=50,
        random_state=RANDOM_SEED
    ),
    "KNN": KNeighborsClassifier(
        n_neighbors=5,
        metric='cosine'
    ),
    "MLP Neural Network": MLPClassifier(
    hidden_layer_sizes=(256, 128),
    max_iter=300,
    random_state=RANDOM_SEED,
),
}

# ── 4. Train, evaluate, and compare all models ────────────────────────────────
print("\n[3/5] Training and evaluating all 5 models...")
print("-" * 60)

results = {}

for name, clf in MODELS.items():
    print(f"\n  → {name}")

    pipeline = Pipeline([
        ('tfidf', tfidf),
        ('clf',   clf)
    ])

    # Train
    start = time.time()
    pipeline.fit(X_train, y_train)
    train_time = time.time() - start

    # Predict
    y_pred = pipeline.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    precision = report['weighted avg']['precision']
    recall    = report['weighted avg']['recall']
    f1        = report['weighted avg']['f1-score']

    results[name] = {
        'pipeline':  pipeline,
        'accuracy':  acc,
        'precision': precision,
        'recall':    recall,
        'f1':        f1,
        'time':      train_time,
        'y_pred':    y_pred
    }

    print(f"     Accuracy  : {acc:.4f} ({acc*100:.2f}%)")
    print(f"     Precision : {precision:.4f}")
    print(f"     Recall    : {recall:.4f}")
    print(f"     F1-Score  : {f1:.4f}")
    print(f"     Train time: {train_time:.2f}s")

# ── 5. Summary comparison table ───────────────────────────────────────────────
print("\n" + "=" * 60)
print("  COMPARISON SUMMARY")
print("=" * 60)
print(f"{'Model':<22} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Time(s)':>9}")
print("-" * 60)

best_name = max(results, key=lambda k: results[k]['f1'])

for name, r in results.items():
    marker = " ◄ BEST" if name == best_name else ""
    print(
        f"{name:<22} {r['accuracy']:>10.4f} {r['precision']:>10.4f} "
        f"{r['recall']:>10.4f} {r['f1']:>10.4f} {r['time']:>9.2f}{marker}"
    )

print(f"\n  Best model: {best_name} (F1 = {results[best_name]['f1']:.4f})")

# ── 6. Save the best model ────────────────────────────────────────────────────
print(f"\n[4/5] Saving best model → {BEST_MODEL}")
joblib.dump(results[best_name]['pipeline'], BEST_MODEL)

# Also save model name for reference
with open(os.path.join(MODEL_DIR, "best_model_name.txt"), "w") as f:
    f.write(best_name)

print(f"      Saved successfully.")

# ── 7. Generate plots ─────────────────────────────────────────────────────────
print(f"\n[5/5] Generating comparison charts → results/")

# --- Plot 1: Accuracy bar chart ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Model Comparison — Resume Classifier", fontsize=14, fontweight='bold')

names     = list(results.keys())
accs      = [results[n]['accuracy']  for n in names]
f1s       = [results[n]['f1']        for n in names]
colors    = ['#2E75B6' if n == best_name else '#A0C4E8' for n in names]

axes[0].bar(names, accs, color=colors, edgecolor='white', linewidth=0.5)
axes[0].set_title("Accuracy by Model")
axes[0].set_ylabel("Accuracy")
axes[0].set_ylim(0, 1)
axes[0].set_xticklabels(names, rotation=20, ha='right', fontsize=9)
for i, v in enumerate(accs):
    axes[0].text(i, v + 0.01, f"{v:.3f}", ha='center', fontsize=9, fontweight='bold')

axes[1].bar(names, f1s, color=colors, edgecolor='white', linewidth=0.5)
axes[1].set_title("F1-Score by Model")
axes[1].set_ylabel("F1-Score (weighted)")
axes[1].set_ylim(0, 1)
axes[1].set_xticklabels(names, rotation=20, ha='right', fontsize=9)
for i, v in enumerate(f1s):
    axes[1].text(i, v + 0.01, f"{v:.3f}", ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "model_comparison.png"), dpi=150, bbox_inches='tight')
plt.close()
print("      Saved: results/model_comparison.png")

# --- Plot 2: Confusion matrix for best model ---
cm = confusion_matrix(y_test, results[best_name]['y_pred'])
categories = sorted(y.unique())

fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=categories,
    yticklabels=categories,
    ax=ax,
    linewidths=0.3
)
ax.set_title(f"Confusion Matrix — {best_name}", fontsize=13, fontweight='bold', pad=15)
ax.set_xlabel("Predicted", fontsize=11)
ax.set_ylabel("Actual", fontsize=11)
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"), dpi=150, bbox_inches='tight')
plt.close()
print("      Saved: results/confusion_matrix.png")

# --- Plot 3: Multi-metric comparison ---
metrics      = ['accuracy', 'precision', 'recall', 'f1']
metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
x     = np.arange(len(names))
width = 0.2

fig, ax = plt.subplots(figsize=(14, 6))
for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
    vals = [results[n][metric] for n in names]
    ax.bar(x + i * width, vals, width, label=label, edgecolor='white', linewidth=0.4)

ax.set_title("All Metrics — Model Comparison", fontsize=13, fontweight='bold')
ax.set_ylabel("Score")
ax.set_ylim(0, 1.1)
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(names, rotation=15, ha='right', fontsize=9)
ax.legend(fontsize=9)
ax.grid(axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "all_metrics.png"), dpi=150, bbox_inches='tight')
plt.close()
print("      Saved: results/all_metrics.png")

print("\n" + "=" * 60)
print(f"  TRAINING COMPLETE")
print(f"  Best model : {best_name}")
print(f"  Model saved: {BEST_MODEL}")
print(f"  Charts     : results/")
print("=" * 60)