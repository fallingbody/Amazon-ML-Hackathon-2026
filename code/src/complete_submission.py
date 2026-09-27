#!/usr/bin/env python3
"""
Emergency Submission Aligner & Formatter for Amazon ML Challenge 2026.
Ensures all 1,732,544 Source 1 entities exist in matching_results.tsv and candidate_pairs.tsv.
Fills any missing entities with valid empty strings (singletons), guaranteeing 100% validator pass.
"""
import os
import sys
import argparse
import pandas as pd

def find_file(paths):
    return next((p for p in paths if os.path.exists(p)), None)

def main():
    parser = argparse.ArgumentParser(description="Ensure 100% complete submission files")
    parser.add_argument("--test-s1", default=None, help="Path to test_source1.tsv")
    parser.add_argument("--matching", default="output/matching_results.tsv")
    parser.add_argument("--candidate", default="output/candidate_pairs.tsv")
    args = parser.parse_args()

    # Locate test_source1.tsv
    s1_path = args.test_s1 or find_file([
        "dataset/test/test_source1.tsv",
        "../dataset/test/test_source1.tsv",
        "../../dataset/test/test_source1.tsv",
        "6ab10eb3b23ba_student_resource/student_resource/dataset/test/test_source1.tsv",
        "student_resource/dataset/test/test_source1.tsv"
    ])
    if not s1_path:
        print("ERROR: test_source1.tsv not found! Cannot verify entity list.")
        sys.exit(1)

    print(f"Loading required Source 1 entities from {s1_path}...")
    required_ids = []
    with open(s1_path, "r", encoding="utf-8") as f:
        _ = f.readline()  # skip header
        for line in f:
            line_str = line.strip()
            if line_str:
                required_ids.append(line_str.split("\t", 1)[0].strip())
    total_required = len(required_ids)
    print(f"Total required entities: {total_required:,}")

    # Process matching_results.tsv
    match_file = find_file([args.matching, "output/test/matching_results.tsv", "output/matching_results.tsv"])
    existing_matches = {}
    if match_file and os.path.exists(match_file):
        print(f"Reading existing matches from {match_file}...")
        with open(match_file, "r", encoding="utf-8") as f:
            header = f.readline()
            for line in f:
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) >= 2:
                    existing_matches[parts[0]] = parts[1]
                elif len(parts) == 1:
                    existing_matches[parts[0]] = ""
        print(f"Loaded {len(existing_matches):,} existing matches.")

    # Write completed matching_results.tsv
    os.makedirs("output", exist_ok=True)
    os.makedirs("unwanted_submission/output", exist_ok=True)
    out_match = "output/matching_results.tsv"
    out_match_sub = "unwanted_submission/output/matching_results.tsv"
    print(f"Writing complete matching results to {out_match}...")
    with open(out_match, "w", encoding="utf-8") as f, open(out_match_sub, "w", encoding="utf-8") as f_sub:
        header_line = "source1_entity_id\tmatched_entity_ids\n"
        f.write(header_line)
        f_sub.write(header_line)
        for s1_id in required_ids:
            m = existing_matches.get(s1_id, "")
            row_line = f"{s1_id}\t{m}\n"
            f.write(row_line)
            f_sub.write(row_line)
    print(f"Successfully generated 100% complete {out_match} ({total_required:,} entities).")

    # Process candidate_pairs.tsv
    cand_file = find_file([args.candidate, "output/test/candidate_pairs.tsv", "output/candidate_pairs.tsv"])
    existing_cands = {}
    if cand_file and os.path.exists(cand_file):
        print(f"Reading existing candidates from {cand_file}...")
        with open(cand_file, "r", encoding="utf-8") as f:
            header = f.readline()
            for line in f:
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) >= 2:
                    existing_cands[parts[0]] = parts[1]
                elif len(parts) == 1:
                    existing_cands[parts[0]] = ""
        print(f"Loaded {len(existing_cands):,} existing candidates.")

    out_cand = "output/candidate_pairs.tsv"
    out_cand_sub = "unwanted_submission/output/candidate_pairs.tsv"
    print(f"Writing complete candidate pairs to {out_cand}...")
    with open(out_cand, "w", encoding="utf-8") as f, open(out_cand_sub, "w", encoding="utf-8") as f_sub:
        header_line = "source1_entity_id\tcandidate_entity_ids\n"
        f.write(header_line)
        f_sub.write(header_line)
        for s1_id in required_ids:
            # Matches must be subset of candidates
            m = existing_matches.get(s1_id, "")
            c = existing_cands.get(s1_id, m)
            if m and m not in c:
                c = f"{c},{m}" if c else m
            row_line = f"{s1_id}\t{c}\n"
            f.write(row_line)
            f_sub.write(row_line)
    print(f"Successfully generated 100% complete {out_cand} ({total_required:,} entities).")

    # Run official validator
    val_script = find_file(["code/src/validate_submission.py", "src/validate_submission.py", "validate_submission.py"])
    if val_script:
        test_dir = os.path.dirname(s1_path)
        print("\nRunning official submission validator...")
        os.system(f"python3 {val_script} --matching {out_match} --candidate {out_cand} --test-dir {test_dir}")

if __name__ == "__main__":
    main()
