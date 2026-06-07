import json
import os
import subprocess
import time

data_dir = r"C:\Users\ASUS\.gemini\antigravity\scratch\TalentMind-AI\data"
candidates_file = os.path.join(data_dir, 'candidates.jsonl')
test_candidates_file = os.path.join(data_dir, 'candidates_test.jsonl')

def run_test():
    print("=== STARTING PIPELINE INTEGRATION TEST ===")
    
    # 1. Extract first 1000 candidates to a test file
    print("Extracting first 1000 candidates for testing...")
    count = 0
    with open(candidates_file, 'r', encoding='utf-8') as infile, \
         open(test_candidates_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            outfile.write(line)
            count += 1
            if count >= 1000:
                break
    print(f"Created test candidate pool: {test_candidates_file}")
    
    # 2. Run precompute on the test candidates
    test_embeddings_file = os.path.join(data_dir, 'candidate_embeddings_test.npy')
    print("Running pre-computation on 1000 test candidates...")
    precompute_cmd = [
        r".venv\Scripts\python.exe", "main.py", "--precompute",
        "--candidates", test_candidates_file,
        "--embeddings", test_embeddings_file,
        "--batch-size", "128"
    ]
    start = time.time()
    subprocess.run(precompute_cmd, check=True)
    print(f"Test pre-computation finished in {time.time() - start:.2f} seconds.")
    
    # 3. Run ranking on the test candidates
    test_out_csv = r"outputs\team_talentmind_test.csv"
    print("Running ranking on test candidates...")
    rank_cmd = [
        r".venv\Scripts\python.exe", "rank.py",
        "--candidates", test_candidates_file,
        "--embeddings", test_embeddings_file,
        "--out", test_out_csv,
        "--top-n", "100" # FAISS retrieve top 100 from the 1000 pool
    ]
    start = time.time()
    subprocess.run(rank_cmd, check=True)
    print(f"Test ranking finished in {time.time() - start:.2f} seconds.")
    
    # Check if output file exists
    if os.path.exists(test_out_csv):
        print(f"\nSUCCESS! Test submission CSV generated at: {test_out_csv}")
        print("\n--- SAMPLE OUTPUT ROWS (First 5) ---")
        with open(test_out_csv, 'r', encoding='utf-8') as f:
            for i in range(6):
                print(f.readline().strip())
    else:
        print("\nError: Output CSV was not generated.")

if __name__ == '__main__':
    run_test()
