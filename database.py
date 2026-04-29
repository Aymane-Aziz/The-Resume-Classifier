import sqlite3
import os
import shutil
from datetime import datetime

# ── Config ────────────────────────────────────────────────────────────────────
DB_PATH     = "resume_classifier.db"
UPLOADS_DIR = "uploads"

os.makedirs(UPLOADS_DIR, exist_ok=True)


# ── Connection helper ─────────────────────────────────────────────────────────
def _get_connection():
    """Get a SQLite connection with row factory for dict-like access."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ── Initialize database ───────────────────────────────────────────────────────
def init_db():
    """
    Create the candidates table if it doesn't exist.
    Safe to call multiple times — won't overwrite existing data.
    """
    conn = _get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS candidates (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            name             TEXT    NOT NULL,
            filename         TEXT,
            category         TEXT    NOT NULL,
            confidence       REAL    NOT NULL,
            similarity_score REAL    NOT NULL,
            rank             INTEGER NOT NULL,
            resume_text      TEXT,
            rejected         INTEGER NOT NULL DEFAULT 0,
            uploaded_at      TEXT    NOT NULL
        )
    """)

    conn.commit()
    conn.close()
    print(f"[Database] Initialized at: {DB_PATH}")


# ── Save a candidate ──────────────────────────────────────────────────────────
def save_candidate(
    name:             str,
    category:         str,
    confidence:       float,
    similarity_score: float,
    rank:             int,
    resume_text:      str  = "",
    filename:         str  = None,
    rejected:         bool = False
) -> int:
    """
    Save a classified candidate to the database.
    Returns the new candidate's ID.
    """
    conn   = _get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO candidates
            (name, filename, category, confidence, similarity_score,
             rank, resume_text, rejected, uploaded_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        name,
        filename,
        category,
        round(confidence, 4),
        round(similarity_score, 4),
        rank,
        resume_text[:5000],     # cap text length to 5000 chars
        1 if rejected else 0,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))

    new_id = cursor.lastrowid
    conn.commit()
    conn.close()

    print(f"[Database] Saved candidate '{name}' → ID {new_id} | Category: {category} | Rank: #{rank}")
    return new_id


# ── Save uploaded PDF ─────────────────────────────────────────────────────────
def save_pdf(uploaded_file, original_filename: str) -> str:
    """
    Save an uploaded PDF file to the uploads/ directory.
    Returns the saved filename.

    Args:
        uploaded_file:     file-like object (from Streamlit uploader)
        original_filename: original name of the uploaded file
    """
    # Sanitize filename and make unique with timestamp
    timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name    = original_filename.replace(" ", "_")
    saved_name   = f"{timestamp}_{safe_name}"
    saved_path   = os.path.join(UPLOADS_DIR, saved_name)

    with open(saved_path, "wb") as f:
        f.write(uploaded_file.read())

    print(f"[Database] PDF saved: {saved_path}")
    return saved_name


# ── Fetch all accepted candidates ─────────────────────────────────────────────
def get_all_candidates() -> list:
    """Return all accepted (non-rejected) candidates ordered by category and rank."""
    conn    = _get_connection()
    cursor  = conn.cursor()

    cursor.execute("""
        SELECT * FROM candidates
        WHERE rejected = 0
        ORDER BY category ASC, rank ASC
    """)

    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


# ── Fetch candidates by category ──────────────────────────────────────────────
def get_candidates_by_category(category: str) -> list:
    """Return all accepted candidates for a specific category, ranked."""
    conn   = _get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT * FROM candidates
        WHERE category = ? AND rejected = 0
        ORDER BY rank ASC
    """, (category.upper(),))

    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


# ── Fetch rejected candidates ─────────────────────────────────────────────────
def get_rejected_candidates() -> list:
    """Return all rejected candidates."""
    conn   = _get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT * FROM candidates
        WHERE rejected = 1
        ORDER BY uploaded_at DESC
    """)

    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


# ── Fetch all distinct categories ─────────────────────────────────────────────
def get_all_categories() -> list:
    """Return list of categories that have at least one accepted candidate."""
    conn   = _get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT DISTINCT category, COUNT(*) as count
        FROM candidates
        WHERE rejected = 0
        GROUP BY category
        ORDER BY category ASC
    """)

    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


# ── Fetch summary stats ───────────────────────────────────────────────────────
def get_stats() -> dict:
    """Return summary statistics for the dashboard."""
    conn   = _get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM candidates WHERE rejected = 0")
    total_accepted = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM candidates WHERE rejected = 1")
    total_rejected = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(DISTINCT category) FROM candidates WHERE rejected = 0")
    total_categories = cursor.fetchone()[0]

    cursor.execute("SELECT AVG(confidence) FROM candidates WHERE rejected = 0")
    avg_confidence = cursor.fetchone()[0] or 0.0

    conn.close()

    return {
        "total_accepted":   total_accepted,
        "total_rejected":   total_rejected,
        "total_processed":  total_accepted + total_rejected,
        "total_categories": total_categories,
        "avg_confidence":   round(avg_confidence * 100, 1)
    }


# ── Update ranks for a category ───────────────────────────────────────────────
def update_ranks_for_category(category: str, ranked_candidates: list):
    """
    After a new candidate is added to a category, update all ranks.

    Args:
        category:          the job category
        ranked_candidates: list of dicts with 'id' and 'rank' fields
    """
    conn   = _get_connection()
    cursor = conn.cursor()

    for candidate in ranked_candidates:
        cursor.execute("""
            UPDATE candidates SET rank = ? WHERE id = ?
        """, (candidate['rank'], candidate['id']))

    conn.commit()
    conn.close()


# ── Delete a candidate ────────────────────────────────────────────────────────
def delete_candidate(candidate_id: int):
    """Delete a candidate by ID."""
    conn   = _get_connection()
    cursor = conn.cursor()

    # Get filename first to delete PDF too
    cursor.execute("SELECT filename FROM candidates WHERE id = ?", (candidate_id,))
    row = cursor.fetchone()

    if row and row['filename']:
        pdf_path = os.path.join(UPLOADS_DIR, row['filename'])
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
            print(f"[Database] Deleted PDF: {pdf_path}")

    cursor.execute("DELETE FROM candidates WHERE id = ?", (candidate_id,))
    conn.commit()
    conn.close()
    print(f"[Database] Deleted candidate ID: {candidate_id}")


# ── Clear all data ────────────────────────────────────────────────────────────
def clear_all():
    """Delete all candidates and uploaded PDFs. Use with caution."""
    conn   = _get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM candidates")
    conn.commit()
    conn.close()

    # Clear uploads folder
    for f in os.listdir(UPLOADS_DIR):
        os.remove(os.path.join(UPLOADS_DIR, f))

    print("[Database] All data cleared.")


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  DATABASE TEST")
    print("=" * 55)

    # Initialize
    init_db()

    # Insert dummy candidates
    save_candidate("John Doe",    "FINANCE",  0.92, 0.84, 1, "experienced finance manager...")
    save_candidate("Sara Smith",  "FINANCE",  0.88, 0.79, 2, "financial analyst with CFA...")
    save_candidate("Ali Hassan",  "HR",       0.95, 0.91, 1, "hr manager recruitment...")
    save_candidate("Jane Doe",    "FINANCE",  0.21, 0.15, 0, "random text...", rejected=True)

    # Test fetchers
    print("\n── All accepted candidates ──")
    for c in get_all_candidates():
        print(f"  [{c['category']}] #{c['rank']} {c['name']} — confidence: {c['confidence']*100:.1f}%")

    print("\n── Finance only ──")
    for c in get_candidates_by_category("FINANCE"):
        print(f"  #{c['rank']} {c['name']} — similarity: {c['similarity_score']}")

    print("\n── Rejected ──")
    for c in get_rejected_candidates():
        print(f"  {c['name']} — confidence: {c['confidence']*100:.1f}%")

    print("\n── Stats ──")
    stats = get_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # Cleanup test data
    clear_all()
    print("\n  ✅ Database working correctly.")
    print("=" * 55)