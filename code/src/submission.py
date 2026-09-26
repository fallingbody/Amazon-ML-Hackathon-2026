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
    matching_file: str = "output/test/matching_results.tsv",
    candidate_file: str = "output/test/candidate_pairs.tsv",
    test_dir: str = None
) -> bool:
    """
    Runs the official competition validator (validate_submission.py) to check TSV outputs.
    """
    if not os.path.exists(matching_file) and os.path.exists("output/matching_results.tsv"):
        matching_file = "output/matching_results.tsv"
    if not os.path.exists(candidate_file) and os.path.exists("output/candidate_pairs.tsv"):
        candidate_file = "output/candidate_pairs.tsv"

    if not test_dir or not os.path.exists(test_dir):
        possible_test_dirs = [
            "dataset/test",
            "6ab10eb3b23ba_student_resource/student_resource/dataset/test",
            "../dataset/test",
            "data/test"
        ]
        for d in possible_test_dirs:
            if os.path.exists(d):
                test_dir = d
                break
        if not test_dir:
            test_dir = "dataset/test"

    possible_scripts = [
        "code/validate_submission.py",
        "validate_submission.py",
        "6ab10eb3b23ba_student_resource/student_resource/utils/validate_submission.py",
        "utils/validate_submission.py"
    ]
    validator_script = None
    for s in possible_scripts:
        if os.path.exists(s):
            validator_script = s
            break

    if not validator_script:
        print(f"Validator script not found in {possible_scripts}")
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
