import os
import csv
import subprocess

class SubmissionGenerator:
    def __init__(self, team_id="team_talentmind"):
        self.team_id = team_id

    def generate(self, ranked_candidates, output_dir="outputs"):
        """
        Generates the submission CSV file.
        ranked_candidates: list of dicts containing 'candidate_id', 'final_score', 'reasoning'
        """
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, f"{self.team_id}.csv")
        
        # Sort candidate list to ensure score non-increasing, tie-break by candidate_id ascending
        sorted_candidates = sorted(
            ranked_candidates,
            key=lambda x: (-x['final_score'], x['candidate_id'])
        )
        
        # Take top 100
        top_100 = sorted_candidates[:100]
        
        # Write CSV
        print(f"Writing Top 100 candidates to {csv_path}...")
        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            # Header
            writer.writerow(["candidate_id", "rank", "score", "reasoning"])
            
            # Rows
            for rank, cand in enumerate(top_100, start=1):
                writer.writerow([
                    cand['candidate_id'],
                    rank,
                    cand['final_score'],
                    cand['reasoning']
                ])
                
        print("CSV generated successfully.")
        return csv_path

    def run_validator(self, csv_path, validator_script_path):
        """
        Runs the official validate_submission.py script on the generated CSV.
        """
        if not os.path.exists(validator_script_path):
            print(f"Validator script not found at {validator_script_path}. Skipping validation.")
            return False
            
        print(f"Running validator on {csv_path}...")
        try:
            result = subprocess.run(
                ["python", validator_script_path, csv_path],
                capture_output=True,
                text=True,
                check=False
            )
            print("Validator Output:")
            print(result.stdout)
            if result.stderr:
                print("Validator Error:")
                print(result.stderr)
            return result.returncode == 0
        except Exception as e:
            print(f"Failed to run validator: {e}")
            return False
