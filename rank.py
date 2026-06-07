import argparse
import sys
import os

# Ensure project root is in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import handle_rank

def main():
    parser = argparse.ArgumentParser(description="TalentMind AI wrapper for ranking.")
    parser.add_argument("--candidates", required=True, help="Path to candidates jsonl file.")
    parser.add_argument("--jd", default="data/job_description.docx", help="Path to job description docx file.")
    parser.add_argument("--embeddings", default="data/candidate_embeddings.npy", help="Path to candidate embeddings npy file.")
    parser.add_argument("--out", required=True, help="Path to write final CSV.")
    parser.add_argument("--team-id", default="team_talentmind", help="Participant team ID.")
    parser.add_argument("--device", default="cpu", help="Device to run inference on (cpu/cuda).")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for embedding generation.")
    parser.add_argument("--top-n", type=int, default=2000, help="Number of candidates to retrieve.")
    
    args = parser.parse_args()
    
    # Add rank flag to match main.py args
    args.rank = True
    
    handle_rank(args)

if __name__ == "__main__":
    main()
