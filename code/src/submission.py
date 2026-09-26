"""
Module: submission.py
Formats candidate pairs and matching results into TSV outputs,
strictly adhering to competition rules and validate_submission.py specifications.
"""
import os
import csv
import subprocess
from typing import Dict, List, Set, Tuple

def save_candidate_pairs(candidates_map: Dict[str, List[str]], output_path: str = "output/candidate_pairs.tsv"):
    """
    Saves candidate pairs to TSV format:
    source1_entity_id \t candidate_entity_ids
    where candidate_entity_ids is comma-separated S2/S3 IDs (or empty string if no candidates).
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["source1_entity_id", "candidate_entity_ids"])
        for s1_id, cands in candidates_map.items():
            cand_str = ",".join(cands) if cands else ""
            writer.writerow([s1_id, cand_str])
    print(f"Saved candidate pairs ({len(candidates_map):,} entities) to {output_path}")

def save_matching_results(results_map: Dict[str, List[str]], output_path: str = "output/matching_results.tsv"):
    """
    Saves final predicted business entity matches to TSV format:
    source1_entity_id \t matched_entity_ids
    where matched_entity_ids is comma-separated S2/S3 IDs (or empty string for singletons).
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["source1_entity_id", "matched_entity_ids"])
        for s1_id, matches in results_map.items():
            match_str = ",".join(matches) if matches else ""
            writer.writerow([s1_id, match_str])
    print(f"Saved final matching results ({len(results_map):,} entities) to {output_path}")

def validate_outputs(
    matching_file: str = "output/matching_results.tsv",
    candidate_file: str = "output/candidate_pairs.tsv",
    test_dir: str = "6ab10eb3b23ba_student_resource/student_resource/dataset/train"
) -> bool:
    """
    Runs the official competition validator (validate_submission.py) to check TSV outputs.
    """
    validator_script = "6ab10eb3b23ba_student_resource/student_resource/utils/validate_submission.py"
    if not os.path.exists(validator_script):
        print(f"Validator script not found at {validator_script}")
        return True

    cmd = [
        "python3", validator_script,
        "--matching", matching_file,
        "--candidate", candidate_file,
        "--test-dir", test_dir
    ]
    
    print("\nExecuting Official Submission Validator:")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
        
    return (result.returncode == 0)
