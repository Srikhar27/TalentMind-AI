import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer

class CandidateEmbeddingEngine:
    def __init__(self, model_name='all-MiniLM-L6-v2', device='cpu'):
        self.model_name = model_name
        self.device = device
        self.model = None

    def load_model(self):
        if self.model is None:
            print(f"Loading sentence-transformer model: {self.model_name} on {self.device}...")
            self.model = SentenceTransformer(self.model_name, device=self.device)
            print("Model loaded successfully.")

    def construct_candidate_text(self, cand):
        """
        Creates a search-optimized text block for embedding representation.
        Packs headline, summary, skills, and work history.
        """
        profile = cand.get('profile', {})
        skills = cand.get('skills', [])
        history = cand.get('career_history', [])
        
        # 1. Headline, Summary, Title
        headline = profile.get('headline', '')
        summary = profile.get('summary', '')
        current_title = profile.get('current_title', '')
        
        # 2. Skills
        skill_names = [s.get('name', '') for s in skills if s.get('name')]
        skills_str = ", ".join(skill_names)
        
        # 3. Work history summaries (up to 3 recent jobs)
        work_entries = []
        for i, job in enumerate(history[:3]):
            title = job.get('title', '')
            company = job.get('company', '')
            desc = job.get('description', '')
            # Truncate description to save token space
            desc_short = desc[:150] + "..." if len(desc) > 150 else desc
            work_entries.append(f"Role: {title} at {company}. Description: {desc_short}")
        work_str = " | ".join(work_entries)
        
        # Construct final combined representation
        text = f"Title: {current_title} | Headline: {headline} | Summary: {summary} | Skills: {skills_str} | Work History: {work_str}"
        return text

    def generate_embeddings(self, candidates, batch_size=128):
        """
        Generates embeddings in batches.
        """
        self.load_model()
        texts = [self.construct_candidate_text(cand) for cand in candidates]
        print(f"Encoding {len(texts)} candidate text representations...")
        embeddings = self.model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
        return embeddings

    def generate_jd_embedding(self, jd_text):
        """
        Encodes the job description text.
        """
        self.load_model()
        return self.model.encode(jd_text, convert_to_numpy=True)

    def save_embeddings(self, embeddings, file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        np.save(file_path, embeddings)
        print(f"Saved embeddings array of shape {embeddings.shape} to {file_path}")

    def load_embeddings(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Embeddings file not found at {file_path}")
        embeddings = np.load(file_path)
        print(f"Loaded embeddings array of shape {embeddings.shape} from {file_path}")
        return embeddings
