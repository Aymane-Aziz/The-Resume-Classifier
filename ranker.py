import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import joblib
import os

# ── Load BERT encoder (reuse same model as classifier) ────────────────────────
BERT_NAME_PATH = os.path.join("model", "bert_model_name.txt")

with open(BERT_NAME_PATH, "r") as f:
    _bert_name = f.read().strip()

print(f"[Ranker] Loading BERT encoder: {_bert_name}")
_bert_encoder = SentenceTransformer(_bert_name)
print("[Ranker] Ready.")


# ── Job description references ────────────────────────────────────────────────
# One reference description per category.
# These are used as the "ideal candidate" vector for ranking.
JOB_DESCRIPTIONS = {
    "ACCOUNTANT": """
        Certified accountant with expertise in financial reporting, tax preparation,
        auditing, and compliance. Proficient in QuickBooks, SAP, and Excel.
        Experience with accounts payable, receivable, and general ledger management.
        Strong knowledge of GAAP and IFRS standards.
    """,
    "ADVOCATE": """
        Legal professional with experience in litigation, client representation,
        contract drafting, and legal research. Expertise in civil and criminal law,
        court proceedings, and case management. Strong negotiation and communication skills.
    """,
    "AGRICULTURE": """
        Agriculture specialist with knowledge of crop management, soil science,
        irrigation systems, and sustainable farming practices. Experience with
        agricultural machinery, pesticide application, and farm operations management.
    """,
    "APPAREL": """
        Fashion and apparel professional with experience in garment design,
        textile sourcing, production management, and quality control.
        Knowledge of fashion trends, pattern making, and retail merchandising.
    """,
    "ARTS": """
        Creative professional with expertise in visual arts, graphic design,
        illustration, and multimedia production. Proficient in Adobe Creative Suite.
        Experience in content creation, branding, and artistic direction.
    """,
    "AUTOMOBILE": """
        Automotive engineer with expertise in vehicle design, mechanical systems,
        engine diagnostics, and repair. Experience with CAD software, quality control,
        and automotive manufacturing processes. Knowledge of hybrid and electric vehicles.
    """,
    "AVIATION": """
        Aviation professional with experience in aircraft operations, flight planning,
        navigation, and safety compliance. Knowledge of FAA regulations, aircraft
        maintenance, and air traffic control procedures.
    """,
    "BANKING": """
        Banking professional with expertise in financial products, credit analysis,
        risk management, and customer relationship management. Experience with
        loan processing, investment advisory, and regulatory compliance in banking.
    """,
    "BPO": """
        Business process outsourcing specialist with experience in customer service,
        call center operations, data processing, and back-office support.
        Strong communication skills and experience with CRM tools and KPI management.
    """,
    "BUSINESS-DEVELOPMENT": """
        Business development manager with expertise in lead generation, sales strategy,
        partnership development, and market expansion. Experience in B2B sales,
        contract negotiation, revenue growth, and stakeholder management.
    """,
    "CHEF": """
        Professional chef with expertise in culinary arts, menu planning, kitchen
        management, and food safety. Experience in fine dining, food preparation,
        inventory management, and team leadership in restaurant environments.
    """,
    "CONSTRUCTION": """
        Construction project manager with expertise in site management, structural
        engineering, budget control, and safety compliance. Experience with AutoCAD,
        project scheduling, contractor coordination, and building codes.
    """,
    "CONSULTANT": """
        Management consultant with expertise in business strategy, process optimization,
        data analysis, and organizational change management. Experience delivering
        client solutions across multiple industries with strong presentation skills.
    """,
    "DESIGNER": """
        UX/UI designer with expertise in user research, wireframing, prototyping,
        and visual design. Proficient in Figma, Adobe XD, and Sketch.
        Experience in web design, mobile app design, and design systems.
    """,
    "DIGITAL-MEDIA": """
        Digital media specialist with expertise in social media management, content
        creation, SEO, and digital marketing campaigns. Experience with Google Analytics,
        Facebook Ads, video production, and influencer marketing strategies.
    """,
    "ENGINEERING": """
        Mechanical or civil engineer with expertise in technical design, project
        management, CAD modeling, and quality assurance. Experience in manufacturing,
        infrastructure projects, and cross-functional team collaboration.
    """,
    "FINANCE": """
        Finance professional with expertise in financial analysis, budgeting,
        forecasting, investment management, and risk assessment. Proficient in
        Excel, Python, and financial modeling. CFA or MBA preferred.
    """,
    "FITNESS": """
        Fitness trainer and wellness coach with expertise in personal training,
        nutrition planning, group fitness instruction, and athletic performance.
        Certified in CPR and personal training with experience in gym management.
    """,
    "HEALTHCARE": """
        Healthcare professional with expertise in patient care, clinical assessment,
        medical documentation, and treatment planning. Experience in hospital settings,
        electronic health records, and interdisciplinary team collaboration.
    """,
    "HR": """
        Human resources manager with expertise in talent acquisition, employee
        relations, performance management, and HR policy development. Experience
        with HRIS systems, onboarding, training programs, and labor law compliance.
    """,
    "INFORMATION-TECHNOLOGY": """
        Software engineer and IT professional with expertise in software development,
        system architecture, cloud computing, and cybersecurity. Proficient in
        Python, Java, SQL, and agile methodologies. Experience in DevOps and CI/CD.
    """,
    "PUBLIC-RELATIONS": """
        Public relations specialist with expertise in media relations, press release
        writing, crisis communication, and brand reputation management. Experience
        with event planning, stakeholder engagement, and corporate communications.
    """,
    "SALES": """
        Sales executive with expertise in B2B and B2C sales, account management,
        pipeline development, and CRM tools like Salesforce. Strong track record
        of meeting revenue targets, cold outreach, and client retention strategies.
    """,
    "TEACHER": """
        Educator with expertise in curriculum development, classroom management,
        student assessment, and differentiated instruction. Experience teaching
        at secondary or university level with strong communication and mentoring skills.
    """,
}


def _get_job_embedding(category: str) -> np.ndarray:
    """
    Generate BERT embedding for a job description.
    """
    description = JOB_DESCRIPTIONS.get(category.upper(), category)
    embedding   = _bert_encoder.encode(
        [description],
        batch_size=1,
        show_progress_bar=False
    )
    return embedding


def rank_candidates(candidates: list, category: str) -> list:
    """
    Rank a list of candidates within a category by cosine similarity
    to the job description.

    Args:
        candidates: list of dicts, each with at minimum:
                    { 'id', 'name', 'cleaned_text', 'confidence', ... }
        category:   job category string (e.g. "FINANCE")

    Returns:
        Same list sorted by 'similarity_score' descending,
        with 'rank' and 'similarity_score' fields added.
    """
    if not candidates:
        return []

    # Get job description embedding
    job_embedding = _get_job_embedding(category)

    # Generate embeddings for all candidates in one batch
    texts      = [c['cleaned_text'] for c in candidates]
    embeddings = _bert_encoder.encode(
        texts,
        batch_size=32,
        show_progress_bar=False
    )

    # Compute cosine similarity between each resume and the job description
    similarities = cosine_similarity(embeddings, job_embedding).flatten()

    # Attach scores to candidates
    for i, candidate in enumerate(candidates):
        candidate['similarity_score'] = float(similarities[i])

    # Sort by similarity descending
    ranked = sorted(candidates, key=lambda x: x['similarity_score'], reverse=True)

    # Assign rank
    for rank, candidate in enumerate(ranked, start=1):
        candidate['rank'] = rank

    return ranked


def rank_single_candidate(cleaned_text: str, category: str, all_candidates: list) -> dict:
    """
    Given a new candidate's cleaned text, compute their similarity score
    and determine their rank among existing candidates in the same category.

    Args:
        cleaned_text:   preprocessed resume text of the new candidate
        category:       predicted job category
        all_candidates: list of existing candidates in this category
                        (each must have 'similarity_score')

    Returns:
        dict with 'similarity_score' and 'rank'
    """
    job_embedding      = _get_job_embedding(category)
    candidate_embedding = _bert_encoder.encode(
        [cleaned_text],
        batch_size=1,
        show_progress_bar=False
    )

    score = float(cosine_similarity(candidate_embedding, job_embedding)[0][0])

    # Determine rank among existing candidates
    existing_scores = [c.get('similarity_score', 0) for c in all_candidates]
    rank = sum(1 for s in existing_scores if s > score) + 1

    return {
        'similarity_score': score,
        'rank': rank,
        'total_in_category': len(all_candidates) + 1
    }


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import pandas as pd
    from preprocessor import preprocess_resume_text

    print("\n" + "=" * 55)
    print("  RANKER TEST — Finance candidates")
    print("=" * 55)

    df       = pd.read_csv('data/Resume.csv')
    finance  = df[df['Category'] == 'FINANCE'].head(5)

    candidates = []
    for _, row in finance.iterrows():
        candidates.append({
            'id':           row['ID'],
            'name':         f"Candidate_{row['ID']}",
            'cleaned_text': preprocess_resume_text(row['Resume_str']),
            'confidence':   0.95,
        })

    ranked = rank_candidates(candidates, "FINANCE")

    print(f"\n  {'Rank':<6} {'Candidate':<20} {'Similarity Score':>16}")
    print("  " + "-" * 44)
    for c in ranked:
        print(f"  #{c['rank']:<5} {c['name']:<20} {c['similarity_score']:>16.4f}")

    print("\n  ✅ Ranker working correctly.")
    print("=" * 55)