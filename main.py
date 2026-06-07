import os
import argparse
import json
import time
import numpy as np
from datetime import datetime

from src.jd_parser import JobDescriptionParser
from src.embedding_engine import CandidateEmbeddingEngine
from src.retrieval_engine import CandidateRetrievalEngine
from src.ranking_engine import CandidateRankingEngine
from src.explainability import CandidateExplainability
from src.submission_generator import SubmissionGenerator

def load_candidates(candidates_path):
    print(f"Loading candidates from {candidates_path}...")
    candidates = []
    start_time = time.time()
    with open(candidates_path, 'r', encoding='utf-8') as f:
        for line in f:
            candidates.append(json.loads(line))
    print(f"Loaded {len(candidates)} candidates in {time.time() - start_time:.2f} seconds.")
    return candidates

def handle_precompute(args):
    print("=== STARTING EMBEDDING PRE-COMPUTATION ===")
    candidates = load_candidates(args.candidates)
    
    engine = CandidateEmbeddingEngine(device=args.device)
    embeddings = engine.generate_embeddings(candidates, batch_size=args.batch_size)
    
    engine.save_embeddings(embeddings, args.embeddings)
    print("Pre-computation completed successfully!")

def handle_rank(args):
    print("=== STARTING CANDIDATE RETRIEVAL & RANKING ===")
    overall_start = time.time()
    
    # 1. Parse JD
    print(f"Parsing Job Description: {args.jd}...")
    jd_parser = JobDescriptionParser(args.jd)
    jd_requirements = jd_parser.get_requirements()
    
    # 2. Load candidate records
    candidates = load_candidates(args.candidates)
    
    # 3. Load pre-computed embeddings
    engine = CandidateEmbeddingEngine(device=args.device)
    if not os.path.exists(args.embeddings):
        print(f"Warning: Pre-computed embeddings not found at {args.embeddings}. Generating on-the-fly (this may take a few minutes)...")
        cand_embeddings = engine.generate_embeddings(candidates, batch_size=args.batch_size)
        engine.save_embeddings(cand_embeddings, args.embeddings)
    else:
        cand_embeddings = engine.load_embeddings(args.embeddings)
        
    # Verify lengths
    if len(candidates) != len(cand_embeddings):
        raise ValueError(f"Mismatch between number of candidates ({len(candidates)}) and embeddings ({len(cand_embeddings)})")

    # 4. Generate JD embedding
    print("Generating embedding for Job Description text...")
    jd_text = jd_parser.text
    jd_embedding = engine.generate_jd_embedding(jd_text)

    # 5. Build FAISS Index and retrieve Top N
    retrieval_start = time.time()
    retrieval_engine = CandidateRetrievalEngine(dimension=cand_embeddings.shape[1])
    retrieval_engine.build_index(cand_embeddings)
    retrieved_results = retrieval_engine.retrieve(jd_embedding, top_n=args.top_n)
    print(f"Retrieved Top {args.top_n} candidates in {time.time() - retrieval_start:.4f} seconds.")

    # 6. Re-rank retrieved candidates using scoring framework
    ranking_start = time.time()
    ranking_engine = CandidateRankingEngine(jd_requirements)
    
    scored_candidates = []
    print(f"Scoring retrieved candidates...")
    for idx, sim in retrieved_results:
        cand = candidates[idx]
        scores = ranking_engine.score_candidate(cand, sim)
        scored_candidates.append((cand, scores))
        
    print(f"Scored candidates in {time.time() - ranking_start:.4f} seconds.")

    # 7. Generate Explainability and format recommendations
    explain_engine = CandidateExplainability(jd_requirements)
    final_candidates_list = []
    
    # Sort by score descending to find the top candidates for reasoning generation
    scored_candidates.sort(key=lambda x: (-x[1]['final_score'], x[1]['candidate_id']))
    
    # Generate explanations for the top candidates (generate more than 100 just in case)
    print("Generating explainable reasoning for top candidates...")
    for cand, score_details in scored_candidates[:200]:
        reason = explain_engine.generate_reason(cand, score_details)
        score_details['reasoning'] = reason
        final_candidates_list.append(score_details)

    # 8. Write CSV and validate
    generator = SubmissionGenerator(team_id=args.team_id)
    csv_path = generator.generate(final_candidates_list, output_dir=os.path.dirname(args.out))
    
    # Run validator
    validator_path = os.path.join(os.path.dirname(args.candidates), "validate_submission.py")
    generator.run_validator(csv_path, validator_path)
    
    print(f"=== RANKING COMPLETED IN {time.time() - overall_start:.2f} SECONDS ===")

def main():
    parser = argparse.ArgumentParser(description="TalentMind AI candidate discovery and ranking system.")
    parser.add_argument("--precompute", action="store_true", help="Precompute candidate embeddings and exit.")
    parser.add_argument("--rank", action="store_true", help="Run the candidate ranking and generate CSV.")
    parser.add_argument("--candidates", default="data/candidates.jsonl", help="Path to candidates jsonl file.")
    parser.add_argument("--jd", default="data/job_description.docx", help="Path to job description docx file.")
    parser.add_argument("--embeddings", default="data/candidate_embeddings.npy", help="Path to save/load candidate embeddings.")
    parser.add_argument("--out", default="outputs/team_talentmind.csv", help="Path to write the final CSV.")
    parser.add_argument("--team-id", default="team_talentmind", help="Participant team ID.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for embedding generation.")
    parser.add_argument("--top-n", type=int, default=2000, help="Number of candidates to retrieve via FAISS.")
    parser.add_argument("--device", default="cpu", help="Device to run sentence-transformers on (cpu/cuda).")
    
    args = parser.parse_args()
    
    if args.precompute:
        handle_precompute(args)
    elif args.rank:
        handle_rank(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
