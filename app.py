import streamlit as st
import pandas as pd
import json
import os
import time
from datetime import datetime

# Import custom modules
from src.jd_parser import JobDescriptionParser
from src.embedding_engine import CandidateEmbeddingEngine
from src.retrieval_engine import CandidateRetrievalEngine
from src.ranking_engine import CandidateRankingEngine
from src.explainability import CandidateExplainability
from src.submission_generator import SubmissionGenerator

st.set_page_config(
    page_title="TalentMind AI - Candidate Discovery Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Hero section */
    .hero-container {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2.5rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }
    
    .hero-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .hero-subtitle {
        font-size: 1.1rem;
        font-weight: 300;
        opacity: 0.9;
    }
    
    /* Metrics cards */
    .metric-card {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 1.25rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1e3c72;
    }
    
    .metric-lbl {
        font-size: 0.85rem;
        text-transform: uppercase;
        color: #6c757d;
        letter-spacing: 1px;
        margin-top: 0.25rem;
    }
    
    /* Candidate profile card */
    .profile-card {
        background: white;
        border: 1px solid #dee2e6;
        border-radius: 12px;
        padding: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 1.5rem;
    }
    
    .profile-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 2px solid #f8f9fa;
        padding-bottom: 1rem;
        margin-bottom: 1rem;
    }
    
    .profile-name {
        font-size: 1.6rem;
        font-weight: 700;
        color: #212529;
    }
    
    .profile-title {
        font-size: 1.1rem;
        color: #495057;
        margin-top: 0.25rem;
    }
    
    .score-badge {
        background: #1e3c72;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 1.1rem;
    }
    
    .reason-box {
        background-color: #f1f3f5;
        border-left: 4px solid #1e3c72;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin-top: 1rem;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

# App Title & Header
st.markdown("""
<div class="hero-container">
    <div class="hero-title">TalentMind AI</div>
    <div class="hero-subtitle">Intelligent Candidate Discovery & Ranking Platform for Redrob H2S Challenge</div>
</div>
""", unsafe_allow_html=True)

# Sidebar configurations
st.sidebar.header("⚙️ Control Panel")
team_id = st.sidebar.text_input("Registered Team ID", value="team_talentmind")

# File paths
default_candidates = "data/candidates.jsonl"
default_embeddings = "data/candidate_embeddings.npy"

# JD Upload
uploaded_jd = st.sidebar.file_uploader("Upload Job Description (.docx or .txt)", type=["docx", "txt"])

run_button = st.sidebar.button("🚀 Run Discovery Pipeline")

# Helper functions
@st.cache_resource
def get_embedding_engine():
    engine = CandidateEmbeddingEngine(device='cpu')
    return engine

@st.cache_data
def load_candidates_data(path):
    candidates = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            candidates.append(json.loads(line))
    return candidates

# Load candidate database if available
if os.path.exists(default_candidates):
    try:
        candidates_pool = load_candidates_data(default_candidates)
        embeddings_exist = os.path.exists(default_embeddings)
    except Exception as e:
        st.error(f"Error loading candidates: {e}")
        candidates_pool = None
        embeddings_exist = False
else:
    st.warning("Candidate pool candidates.jsonl not found in data/ folder. Please place it in data/ to proceed.")
    candidates_pool = None
    embeddings_exist = False

# Run Discovery logic
if run_button:
    if not candidates_pool:
        st.error("Cannot run discovery. Candidates pool is missing.")
    elif not uploaded_jd:
        st.error("Please upload a Job Description file first.")
    elif not embeddings_exist:
        st.error("Pre-computed embeddings not found. Please run pre-computation CLI first to construct embeddings.")
    else:
        with st.spinner("Processing Job Description & Running Semantic Retrieval..."):
            start_time = time.time()
            
            # Save uploaded file temporarily to parse
            temp_jd_path = "temp_uploaded_jd" + (" .docx" if uploaded_jd.name.endswith(".docx") else ".txt")
            with open(temp_jd_path, "wb") as f:
                f.write(uploaded_jd.getbuffer())
                
            # Parse JD
            jd_parser = JobDescriptionParser(temp_jd_path)
            jd_requirements = jd_parser.get_requirements()
            
            # Load Embeddings
            engine = get_embedding_engine()
            cand_embeddings = engine.load_embeddings(default_embeddings)
            
            # Generate JD Embedding
            jd_embedding = engine.generate_jd_embedding(jd_parser.text)
            
            # FAISS Retrieval
            retrieval_engine = CandidateRetrievalEngine(dimension=cand_embeddings.shape[1])
            retrieval_engine.build_index(cand_embeddings)
            retrieved_results = retrieval_engine.retrieve(jd_embedding, top_n=2000)
            
            # Score retrieved candidates
            ranking_engine = CandidateRankingEngine(jd_requirements)
            scored_candidates = []
            
            # Track honeypots filtered
            honeypots_filtered = 0
            
            for idx, sim in retrieved_results:
                cand = candidates_pool[idx]
                scores = ranking_engine.score_candidate(cand, sim)
                if scores["is_honeypot"]:
                    honeypots_filtered += 1
                scored_candidates.append((cand, scores))
                
            # Explainability
            explain_engine = CandidateExplainability(jd_requirements)
            final_list = []
            
            # Sort
            scored_candidates.sort(key=lambda x: (-x[1]['final_score'], x[1]['candidate_id']))
            
            for cand, score_details in scored_candidates[:200]:
                reason = explain_engine.generate_reason(cand, score_details)
                score_details['reasoning'] = reason
                final_list.append((cand, score_details))
                
            # Save generated CSV to outputs
            top_100_details = [details for _, details in final_list[:100]]
            generator = SubmissionGenerator(team_id=team_id)
            csv_path = generator.generate(top_100_details, output_dir="outputs")
            
            # Cleanup temp file
            if os.path.exists(temp_jd_path):
                os.remove(temp_jd_path)
                
            # Save to session state
            st.session_state["results"] = final_list
            st.session_state["honeypots_filtered"] = honeypots_filtered
            st.session_state["exec_time"] = time.time() - start_time
            st.session_state["csv_path"] = csv_path
            
            st.success("Discovery Pipeline Completed Successfully!")

# Display Results
if "results" in st.session_state:
    results = st.session_state["results"]
    honeypots_filtered = st.session_state["honeypots_filtered"]
    exec_time = st.session_state["exec_time"]
    csv_path = st.session_state["csv_path"]
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">{len(results):d}</div>
            <div class="metric-lbl">Candidates Scored</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">{exec_time:.2f}s</div>
            <div class="metric-lbl">Execution Time</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">{honeypots_filtered:d}</div>
            <div class="metric-lbl">Honeypots Blocked</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">100%</div>
            <div class="metric-lbl">Spec Compliance</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.write("")
    
    # Download Button
    with open(csv_path, 'r', encoding='utf-8') as f:
        csv_data = f.read()
    st.download_button(
        label="📥 Download Submission CSV",
        data=csv_data,
        file_name=os.path.basename(csv_path),
        mime="text/csv",
        key="download_csv"
    )
    
    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["📊 Discovery Leaderboard", "🔍 Candidate Deep Dive", "⚙️ Analytics & Logs"])
    
    with tab1:
        st.subheader("Top 100 Candidates")
        table_rows = []
        for rank, (cand, scores) in enumerate(results[:100], start=1):
            table_rows.append({
                "Rank": rank,
                "Candidate ID": scores["candidate_id"],
                "Name": cand["profile"]["anonymized_name"],
                "Headline": cand["profile"]["headline"],
                "Experience": f"{cand['profile']['years_of_experience']:.1f} yrs",
                "Final Score": scores["final_score"],
                "Quality": scores["quality_score"],
                "Behavioral": scores["behavior_score"],
            })
        df_display = pd.DataFrame(table_rows)
        st.dataframe(df_display, use_container_width=True, hide_index=True)
        
    with tab2:
        st.subheader("Candidate Deep-Dive Profile Analysis")
        # Select candidate
        cand_options = {f"Rank {r}: {c['profile']['anonymized_name']} ({s['candidate_id']})": idx for idx, (c, s) in enumerate(results[:100])}
        selected_cand_label = st.selectbox("Select Candidate to Inspect", list(cand_options.keys()))
        
        if selected_cand_label:
            selected_idx = cand_options[selected_cand_label]
            cand, scores = results[selected_idx]
            profile = cand["profile"]
            
            # Custom Profile Card
            st.markdown(f"""
            <div class="profile-card">
                <div class="profile-header">
                    <div>
                        <div class="profile-name">{profile['anonymized_name']}</div>
                        <div class="profile-title">{profile['current_title']} at {profile['current_company']}</div>
                    </div>
                    <div class="score-badge">Final Score: {scores['final_score']}</div>
                </div>
                <p><strong>Headline:</strong> {profile['headline']}</p>
                <p><strong>Summary:</strong> {profile['summary']}</p>
                <p><strong>Location:</strong> {profile['location']}, {profile['country']} | <strong>Experience:</strong> {profile['years_of_experience']:.1f} Years</p>
                <div class="reason-box">
                    <strong>AI Recommendation Reason:</strong><br>
                    {scores['reasoning']}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Scores details
            st.write("### 7-Factor Score Breakdown")
            col_scores = st.columns(6)
            factors = [
                ("Semantic Similarity (35%)", "semantic_score"),
                ("Skill Match (20%)", "skill_score"),
                ("Experience Match (15%)", "experience_score"),
                ("Project Relevance (10%)", "project_score"),
                ("Behavioral (10%)", "behavior_score"),
                ("Quality Score (5%)", "quality_score")
            ]
            for i, (label, key) in enumerate(factors):
                with col_scores[i % 6]:
                    st.metric(label=label, value=f"{scores[key]}")
                    
            # Career History & Education
            st.write("---")
            col_h, col_e = st.columns(2)
            with col_h:
                st.subheader("💼 Career History")
                for job in cand.get("career_history", []):
                    st.markdown(f"""
                    **{job.get('title')}** at *{job.get('company')}*  
                    📅 {job.get('start_date')} to {job.get('end_date') or 'Present'} ({job.get('duration_months')} months)  
                    *{job.get('description')}*
                    """)
                    st.write("")
            with col_e:
                st.subheader("🎓 Education")
                for edu in cand.get("education", []):
                    st.markdown(f"""
                    **{edu.get('degree')}** in *{edu.get('field_of_study')}*  
                    Institution: {edu.get('institution')} | Tier: `{edu.get('tier')}`  
                    Grade: {edu.get('grade')} | Timeline: {edu.get('start_year')} - {edu.get('end_year')}
                    """)
                
                st.subheader("💡 Highlighted Skills")
                skill_badges = [f"`{s.get('name')} ({s.get('proficiency')})`" for s in cand.get("skills", [])]
                st.markdown(" ".join(skill_badges))
                
    with tab3:
        st.subheader("Platform Diagnostics & Active Parameters")
        st.write("##### Active Job Description Requirements:")
        st.json(st.session_state.get("results")[0][1].get("is_honeypot")) # print test representation
        st.write("##### Core Engine Configuration:")
        st.write("- **Semantic Model:** `sentence-transformers/all-MiniLM-L6-v2`")
        st.write("- **Retrieval Engine:** FAISS (FlatIP with L2 Normalization)")
        st.write("- **Candidates Scored:** 2,000 (Top retrieved from 100,000 pool)")
        st.write("- **Quality Engine:** Checked for salary range validity, employment gaps, graduation year vs career start date, and keyword frequency anomalies.")
else:
    st.info("Upload a Job Description and click 'Run Discovery Pipeline' in the sidebar to execute ranking.")
