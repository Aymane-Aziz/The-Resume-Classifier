import streamlit as st
import os
import tempfile
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from preprocessor import preprocess_resume_pdf, extract_name_from_pdf
from classifier  import classify_from_pdf, classify_from_text, REJECTION_THRESHOLD
from ranker      import rank_candidates, rank_single_candidate
from database    import (
    init_db, save_candidate, save_pdf,
    get_all_candidates, get_candidates_by_category,
    get_rejected_candidates, get_all_categories,
    get_stats, update_ranks_for_category, delete_candidate
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Resume Classifier",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Initialize DB ─────────────────────────────────────────────────────────────
init_db()

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1F4E79;
        margin-bottom: 0;
    }
    .sub-title {
        font-size: 1rem;
        color: #888;
        margin-top: 0;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #F0F4FA;
        border-radius: 10px;
        padding: 1rem 1.5rem;
        text-align: center;
        border-left: 4px solid #2E75B6;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1F4E79;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #666;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .result-accepted {
        background: #E8F5E9;
        border-left: 5px solid #2E7D32;
    }
    .result-rejected {
        background: #FFEBEE;
        border-left: 5px solid #C62828;
    }
    .rank-badge {
        background: #2E75B6;
        color: white;
        border-radius: 50%;
        width: 32px;
        height: 32px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        font-size: 0.9rem;
    }
    .category-header {
        background: #1F4E79;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 6px;
        font-weight: 600;
        margin: 1rem 0 0.5rem 0;
    }
    div[data-testid="stSidebar"] {
        background: #1F4E79;
    }
    div[data-testid="stSidebar"] * {
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)


# ── Sidebar navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📄 Resume Classifier")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["📤 Upload Resume", "📊 Dashboard"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    stats = get_stats()
    st.markdown(f"**📥 Total Processed:** {stats['total_processed']}")
    st.markdown(f"**✅ Accepted:** {stats['total_accepted']}")
    st.markdown(f"**❌ Rejected:** {stats['total_rejected']}")
    st.markdown(f"**📂 Categories:** {stats['total_categories']}")
    if stats['total_accepted'] > 0:
        st.markdown(f"**🎯 Avg Confidence:** {stats['avg_confidence']}%")
    st.markdown("---")
    st.markdown(
        "<small style='color:#aaa'>Powered by BERT + SVM<br>Resume Classifier v1.0</small>",
        unsafe_allow_html=True
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — UPLOAD RESUME
# ══════════════════════════════════════════════════════════════════════════════
if page == "📤 Upload Resume":
    st.markdown('<p class="main-title">📤 Upload Resume</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Upload a PDF resume to classify and rank the candidate automatically.</p>', unsafe_allow_html=True)

    # ── Upload form ───────────────────────────────────────────────────────────
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Upload a resume in PDF format. The system will automatically extract the candidate's name and classify the resume."
    )

    # Extract name when file is uploaded
    candidate_name = ""
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        
        candidate_name = extract_name_from_pdf(tmp_path)
        os.unlink(tmp_path)
        uploaded_file.seek(0)
    
    # Show name input (pre-filled with extracted name, editable)
    if uploaded_file:
        candidate_name = st.text_input(
            "Candidate Name",
            value=candidate_name,
            placeholder="e.g. John Doe",
            help="The name is automatically extracted from your resume. You can edit it if needed."
        )

    if uploaded_file and candidate_name:
        if st.button("🚀 Classify Resume", type="primary", use_container_width=True):
            with st.spinner("Analyzing resume... this may take a few seconds."):

                # 1. Save PDF temporarily for processing
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                # Reset file pointer and save permanently
                uploaded_file.seek(0)
                saved_filename = save_pdf(uploaded_file, uploaded_file.name)

                # 2. Classify
                result = classify_from_pdf(tmp_path)
                os.unlink(tmp_path)

                # 3. Get cleaned text for ranking
                uploaded_file.seek(0)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp2:
                    tmp2.write(uploaded_file.read())
                    tmp2_path = tmp2.name

                from preprocessor import preprocess_resume_pdf as prep_pdf
                cleaned_text = prep_pdf(tmp2_path)
                os.unlink(tmp2_path)

                # 4. Rank within category
                if not result['rejected']:
                    existing = get_candidates_by_category(result['category'])
                    ranking  = rank_single_candidate(
                        cleaned_text, result['category'], existing
                    )
                    similarity_score = ranking['similarity_score']
                    rank             = ranking['rank']

                    # Re-rank all existing candidates in this category
                    all_in_cat = existing + [{
                        'id':             -1,
                        'cleaned_text':   cleaned_text,
                        'similarity_score': similarity_score
                    }]
                    re_ranked = rank_candidates(
                        [{'id': c['id'], 'cleaned_text': c.get('resume_text', ''),
                          'name': c['name'], 'confidence': c['confidence']}
                         for c in existing],
                        result['category']
                    )
                    if re_ranked:
                        update_ranks_for_category(result['category'], re_ranked)
                else:
                    similarity_score = 0.0
                    rank             = 0

                # 5. Save to database
                candidate_id = save_candidate(
                    name             = candidate_name,
                    category         = result['category'],
                    confidence       = result['confidence'],
                    similarity_score = similarity_score,
                    rank             = rank,
                    resume_text      = cleaned_text,
                    filename         = saved_filename,
                    rejected         = result['rejected']
                )

                # 6. Store result in session state for display
                st.session_state['last_result'] = {
                    'name':             candidate_name,
                    'category':         result['category'],
                    'confidence':       result['confidence'],
                    'similarity_score': similarity_score,
                    'rank':             rank,
                    'rejected':         result['rejected'],
                    'all_scores':       result['all_scores'],
                    'candidate_id':     candidate_id
                }

    elif uploaded_file and not candidate_name:
        st.warning("⚠️ Could not automatically extract the candidate's name from the resume. Please enter it manually before classifying.")
    elif candidate_name and not uploaded_file:
        st.info("📎 Please upload a PDF resume to proceed.")

    # ── Results display ───────────────────────────────────────────────────────
    if 'last_result' in st.session_state:
        r = st.session_state['last_result']
        st.markdown("---")
        st.markdown("### 🎯 Classification Result")

        if r['rejected']:
            st.markdown(f"""
            <div class="result-box result-rejected">
                <h3>❌ Rejected</h3>
                <p><strong>Candidate:</strong> {r['name']}</p>
                <p><strong>Reason:</strong> Confidence score ({r['confidence']*100:.1f}%)
                is below the minimum threshold of {REJECTION_THRESHOLD*100:.0f}%.</p>
                <p>The resume does not clearly match any of the available job categories.</p>
            </div>
            """, unsafe_allow_html=True)

        else:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">🏷️</div>
                    <div class="metric-value">{r['category']}</div>
                    <div class="metric-label">Predicted Category</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{r['confidence']*100:.1f}%</div>
                    <div class="metric-label">Confidence Score</div>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">#{r['rank']}</div>
                    <div class="metric-label">Rank in Category</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="result-box result-accepted" style="margin-top:1rem">
                <p>✅ <strong>{r['name']}</strong> has been successfully classified as
                <strong>{r['category']}</strong> with a confidence of
                <strong>{r['confidence']*100:.1f}%</strong> and ranked
                <strong>#{r['rank']}</strong> in their category based on job fit score
                ({r['similarity_score']:.4f}).</p>
            </div>
            """, unsafe_allow_html=True)

            # Top 5 category scores
            st.markdown("#### 📊 Confidence Scores (Top 5 Categories)")
            top5     = list(r['all_scores'].items())[:5]
            labels   = [item[0] for item in top5]
            scores   = [item[1] * 100 for item in top5]
            colors   = ['#2E75B6' if l == r['category'] else '#A0C4E8' for l in labels]

            fig, ax = plt.subplots(figsize=(8, 3))
            bars = ax.barh(labels[::-1], scores[::-1], color=colors[::-1], edgecolor='white')
            ax.set_xlabel("Confidence (%)")
            ax.set_xlim(0, 100)
            for bar, score in zip(bars, scores[::-1]):
                ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                        f"{score:.1f}%", va='center', fontsize=9)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Dashboard":
    st.markdown('<p class="main-title">📊 Recruiter Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Overview of all processed candidates, ranked by job fit within each category.</p>', unsafe_allow_html=True)

    stats = get_stats()

    # ── Summary metrics ───────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{stats['total_processed']}</div>
            <div class="metric-label">Total Processed</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card" style="border-color:#2E7D32">
            <div class="metric-value" style="color:#2E7D32">{stats['total_accepted']}</div>
            <div class="metric-label">Accepted</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card" style="border-color:#C62828">
            <div class="metric-value" style="color:#C62828">{stats['total_rejected']}</div>
            <div class="metric-label">Rejected</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{stats['avg_confidence']}%</div>
            <div class="metric-label">Avg Confidence</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["✅ Accepted Candidates", "❌ Rejected", "📈 Analytics"])

    # ── Tab 1: Accepted candidates by category ────────────────────────────────
    with tab1:
        categories = get_all_categories()

        if not categories:
            st.info("📭 No candidates processed yet. Upload a resume to get started.")
        else:
            # Category filter
            cat_names = ["All Categories"] + [c['category'] for c in categories]
            selected  = st.selectbox("Filter by Category", cat_names)

            if selected == "All Categories":
                candidates = get_all_candidates()
            else:
                candidates = get_candidates_by_category(selected)

            if not candidates:
                st.info(f"No candidates in {selected} yet.")
            else:
                # Group by category
                from collections import defaultdict
                grouped = defaultdict(list)
                for c in candidates:
                    grouped[c['category']].append(c)

                for category, members in grouped.items():
                    st.markdown(f'<div class="category-header">📂 {category} — {len(members)} candidate(s)</div>', unsafe_allow_html=True)

                    # Build table data
                    table_data = []
                    for m in members:
                        table_data.append({
                            "Rank":       f"#{m['rank']}",
                            "Name":       m['name'],
                            "Confidence": f"{m['confidence']*100:.1f}%",
                            "Job Fit":    f"{m['similarity_score']:.4f}",
                            "Uploaded":   m['uploaded_at']
                        })

                    df_display = pd.DataFrame(table_data)
                    st.dataframe(
                        df_display,
                        use_container_width=True,
                        hide_index=True
                    )

                    # Delete option
                    with st.expander(f"🗑️ Remove a candidate from {category}"):
                        member_names = {m['name']: m['id'] for m in members}
                        to_delete    = st.selectbox(
                            "Select candidate to remove",
                            list(member_names.keys()),
                            key=f"del_{category}"
                        )
                        if st.button(f"Delete {to_delete}", key=f"btn_{category}", type="secondary"):
                            delete_candidate(member_names[to_delete])
                            st.success(f"✅ {to_delete} removed.")
                            st.rerun()

    # ── Tab 2: Rejected candidates ────────────────────────────────────────────
    with tab2:
        rejected = get_rejected_candidates()

        if not rejected:
            st.info("✅ No rejected candidates yet.")
        else:
            st.markdown(f"**{len(rejected)} rejected candidate(s)**")
            table_data = []
            for r in rejected:
                table_data.append({
                    "Name":       r['name'],
                    "Confidence": f"{r['confidence']*100:.1f}%",
                    "Reason":     f"Below {REJECTION_THRESHOLD*100:.0f}% threshold",
                    "Uploaded":   r['uploaded_at']
                })

            st.dataframe(
                pd.DataFrame(table_data),
                use_container_width=True,
                hide_index=True
            )

    # ── Tab 3: Analytics ──────────────────────────────────────────────────────
    with tab3:
        all_candidates = get_all_candidates()

        if len(all_candidates) < 2:
            st.info("📊 Upload at least 2 resumes to see analytics.")
        else:
            df_all = pd.DataFrame(all_candidates)

            col1, col2 = st.columns(2)

            # Chart 1: Candidates per category
            with col1:
                cat_counts = df_all['category'].value_counts()
                fig, ax    = plt.subplots(figsize=(6, 4))
                ax.bar(cat_counts.index, cat_counts.values,
                       color='#2E75B6', edgecolor='white')
                ax.set_title("Candidates per Category", fontweight='bold')
                ax.set_ylabel("Count")
                plt.xticks(rotation=45, ha='right', fontsize=8)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            # Chart 2: Confidence distribution
            with col2:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.hist(df_all['confidence'] * 100, bins=15,
                        color='#1F4E79', edgecolor='white', alpha=0.85)
                ax.set_title("Confidence Score Distribution", fontweight='bold')
                ax.set_xlabel("Confidence (%)")
                ax.set_ylabel("Number of Candidates")
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            # Model comparison charts from training
            st.markdown("---")
            st.markdown("#### 🧠 Model Training Results")
            results_dir = "results"
            charts = {
                "model_comparison.png": "Accuracy & F1 Comparison",
                "all_metrics.png":      "All Metrics Comparison",
                "confusion_matrix.png": "Confusion Matrix (Best Model)"
            }
            for filename, title in charts.items():
                path = os.path.join(results_dir, filename)
                if os.path.exists(path):
                    st.markdown(f"**{title}**")
                    st.image(path, use_column_width=True)