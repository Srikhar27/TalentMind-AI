# TalentMind-AI
# TalentMind AI - Intelligent Candidate Discovery Platform

Intelligent Candidate Discovery & Ranking Platform built for the Redrob H2S Challenge. The system retrieves and ranks candidates from a pool of 100,000 profiles against a job description, delivering the Top 100 recommended candidates in under 30 seconds on CPU.

---

## 🚀 Key Features

* **Two-Stage Hybrid Pipeline**: Combines FAISS-based semantic vector retrieval with a robust 7-factor scoring and ranking model.
* **Honeypot & Trap Immunity**: Proactively filters out keyword-stuffed profiles, empty applications, and subtly impossible "honeypots" (such as timeline or skill duration mismatches).
* **Deterministic Tie-Breaking**: Ranks candidates with identical scores based on ascending alphabetical order of their `candidate_id`.
* **Explainable AI Layer**: Automatically generates a 1-2 sentence, non-templated reasoning block for each candidate highlighting strengths, experience, location, and potential fit gaps.
* **Interactive Streamlit Dashboard**: Provides recruiters with an elegant UI to upload JDs, run discovery, inspect scores, read reasoning cards, and download submission CSVs.

---

## 🛠️ System Architecture

```mermaid
graph TD
    JD[Job Description .docx] --> Parser[JD Parser & Keyphrase Extractor]
    Parser --> EmbeddingGen[sentence-transformers/all-MiniLM-L6-v2]
    
    Candidates[100,000 Candidates jsonl] --> Precompute[Precomputed Embeddings Engine]
    Precompute --> FAISS_Index[FAISS IndexFlatIP]
    
    EmbeddingGen --> Search[Cosine Search / Inner Product Query]
    FAISS_Index --> Search
    Search --> Retrieval[Retrieved Top 2000 Candidates]
    
    Retrieval --> ReRank[7-Factor Scoring & Ranking Engine]
    
    ReRank --> Quality[Profile Quality Engine & Honeypot Filter]
    ReRank --> Explanations[Explainability Layer / Reasoning Generator]
    
    Quality --> Top100[Top 100 Recommendations]
    Explanations --> Top100
    
    Top100 --> CSV[Submission CSV Output]
```

---

## 📈 Ranking Methodology (Scoring Framework)

The final score is normalized to a `[0, 100]` range and computed as follows:

| Factor | Weight | Scoring Logic |
| :--- | :--- | :--- |
| **Semantic Similarity** | 35% | Cosine similarity between JD and candidate text representation via FAISS. |
| **Skill Match** | 20% | Substring match against required/preferred skills, weighted by candidate proficiency. |
| **Experience Match** | 15% | Score based on target years (5-9), career tenure, and penalties for gaps & consulting firms. |
| **Project Relevance** | 10% | Semantic keyword frequency check across candidate's past work descriptions. |
| **Behavioral Signals** | 10% | Response rate, response time, login activity, and interview completion rates. |
| **Education Match** | 5% | Degree field relevance combined with university tiering (Tier 1/2/3/4). |
| **Profile Quality** | 5% | Checks for empty profiles, keyword stuffers, and invalid salary ranges. |

* **Honeypot Penalty**: Profiles with impossible timelines (e.g. job duration exceeding date ranges) or expert skills with 0 duration have their Quality Score forced to `0.0` and are completely filtered out of the top rankings.

---

## 📂 Project Structure

```text
TalentMind-AI/
├── data/                       # Dataset directory (place candidates.jsonl here)
├── src/                        # Core source package
│   ├── jd_parser.py            # Word document parser and extractor
│   ├── embedding_engine.py     # sentence-transformers embedding helper
│   ├── retrieval_engine.py     # FAISS indexing and vector search
│   ├── ranking_engine.py       # 7-factor scoring engine and anomaly detector
│   ├── explainability.py       # Candidate reasoning generator
│   └── submission_generator.py # CSV formatter and validation pipeline
├── outputs/                    # Output directory for generated CSVs
├── main.py                     # Primary CLI entry point
├── rank.py                     # Simplified execution wrapper
├── app.py                      # Streamlit interactive dashboard UI
├── requirements.txt            # Python library dependencies
├── submission_metadata.yaml    # Participant registration info
└── README.md                   # System documentation
```

---

## 💻 Setup & Installation

Ensure you have [uv](https://github.com/astral-sh/uv) or `pip` installed.

1. **Clone/create project directory** and navigate inside it.
2. **Place challenge files** in the `data/` folder:
   * `candidates.jsonl`
   * `job_description.docx`
   * `validate_submission.py`
3. **Set up virtual environment** and install dependencies:
   ```bash
   uv venv --python 3.12
   .venv\Scripts\activate
   uv pip install -r requirements.txt
   ```

---

## 🏃 How to Run

### Step 1: Precompute Candidate Embeddings (Offline)
This generates embeddings for the 100,000 candidate profiles. Note that this step exceeds the 5-minute budget but is done offline once.
```bash
python main.py --precompute --candidates ./data/candidates.jsonl --embeddings ./data/candidate_embeddings.npy
```

### Step 2: Run Online Ranking (CPU-Only)
Loads precomputed embeddings, runs FAISS search, scores the candidates, generates reasoning, and saves the CSV. This completes in **under 10 seconds**:
```bash
python rank.py --candidates ./data/candidates.jsonl --jd ./data/job_description.docx --out ./outputs/team_talentmind.csv
```

### Step 3: Run Interactive Streamlit UI
To launch the recruiter dashboard:
```bash
streamlit run app.py
```

---

## 📊 Sample Output Format

```csv
candidate_id,rank,score,reasoning
CAND_0068134,1,92.4,"Aarav Sen is an excellent fit as a Senior Applied ML Engineer with 7.5 years of experience, showing strong hands-on experience in embeddings, FAISS, and python. Excellent recruiter response rate (95%) indicates high availability."
CAND_0012945,2,91.8,"Meera Nair is a strong fit as a Machine Learning Engineer with 6.2 years of experience, demonstrating core expertise in vector databases and python. Active GitHub contributions (score: 84) confirm solid code quality."
```
