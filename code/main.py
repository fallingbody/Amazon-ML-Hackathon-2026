"""
Main pipeline execution script for Amazon ML Challenge 2026: Business Entity Resolution.

Usage:
    python code/main.py [--sample-size 50000] [--max-candidates 30] [--split train]
"""
import os
import sys
import argparse
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

# Ensure current directory is in python path for module imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing import clean_text
from src.indexing import CandidateIndexer
from src.features import compute_pair_features
from src.model import EntityResolutionModel
from src.submission import save_candidate_pairs, save_matching_results, validate_outputs

DATASET_BASE = "6ab10eb3b23ba_student_resource/student_resource/dataset"
DB_PATH = "6ab10eb3b23ba_student_resource/student_resource/index.db"

def parse_args():
    parser = argparse.ArgumentParser(description="Run Business Entity Resolution Pipeline")
    parser.add_argument("--sample-size", type=int, default=50000, help="Number of Source 1 records to process (default 50,000)")
    parser.add_argument("--max-candidates", type=int, default=30, help="Max candidates per entity from index.db (default 30)")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"], help="Dataset split to run on (train or test)")
    return parser.parse_args()

def run_pipeline(sample_size: int = 50000, max_candidates: int = 30, split: str = "train"):
    print("=" * 70)
    print(f"   BUSINESS ENTITY RESOLUTION PIPELINE ({split.upper()} SET)   ")
    print("=" * 70)
    
    split_dir = os.path.join(DATASET_BASE, split)
    s1_filename = f"{split}_source1.tsv"
    s1_path = os.path.join(split_dir, s1_filename)
    gt_path = os.path.join(DATASET_BASE, "train", "train_ground_truth.tsv")
    
    if not os.path.exists(s1_path):
        print(f"Error: Dataset file not found at {s1_path}")
        return

    if sample_size and sample_size > 0:
        df_s1 = pd.read_csv(s1_path, sep="\t", nrows=sample_size)
    else:
        df_s1 = pd.read_csv(s1_path, sep="\t", engine="pyarrow")
    
    # Standardize column names
    rename_dict = {}
    if "entity_id" in df_s1.columns: rename_dict["entity_id"] = "record_id"
    if "business_name" in df_s1.columns: rename_dict["business_name"] = "name"
    if "business_address" in df_s1.columns: rename_dict["business_address"] = "address"
    if rename_dict:
        df_s1 = df_s1.rename(columns=rename_dict)

    print(f"Loaded {len(df_s1):,} Source 1 records.")

    # 2. Load Ground Truth Mapping if Available
    gt_map = {}
    if os.path.exists(gt_path):
        print("Loading Ground Truth labels for training & validation...")
        df_gt = pd.read_csv(gt_path, sep="\t", engine="pyarrow")
        gt_id_col = "source1_entity_id" if "source1_entity_id" in df_gt.columns else df_gt.columns[0]
        gt_match_col = "matched_entity_ids" if "matched_entity_ids" in df_gt.columns else df_gt.columns[1]
        
        for _, row in df_gt.iterrows():
            s1_id = str(row[gt_id_col])
            tgt_str = str(row[gt_match_col]) if not pd.isna(row[gt_match_col]) else ""
            if tgt_str:
                targets = set(tgt_str.split(","))
            else:
                targets = set()
            gt_map[s1_id] = targets

    # 3. Candidate Retrieval using SQLite Index
    print(f"\n[2/5] Querying SQLite B-Tree index (max_candidates={max_candidates})...")
    indexer = CandidateIndexer(db_path=DB_PATH)
    
    all_candidate_pairs = []
    candidates_map = {}
    results_map = {}
    features_list = []
    labels_list = []

    s1_records = df_s1.to_dict("records")
    
    for idx, s1_rec in enumerate(s1_records):
        s1_id = str(s1_rec["record_id"])
        cand_ids = list(indexer.find_candidates_for_record(
            name=str(s1_rec.get("name", "")),
            address=str(s1_rec.get("address", "")),
            max_candidates=max_candidates
        ))
        
        candidates_map[s1_id] = cand_ids
        results_map[s1_id] = []
        
        if not cand_ids:
            continue

        # Fetch candidate attribute details
        df_cands = indexer.fetch_records_by_ids(cand_ids, split=split)
        cand_records = df_cands.to_dict("records")

        s1_gt_targets = gt_map.get(s1_id, set())

        for cand_rec in cand_records:
            cand_id = str(cand_rec["record_id"])
            all_candidate_pairs.append((s1_id, cand_id))
            
            # Extract features
            feats = compute_pair_features(s1_rec, cand_rec)
            features_list.append(feats)
            
            # Ground truth label (1 if candidate in ground truth, else 0)
            is_match = 1 if cand_id in s1_gt_targets else 0
            labels_list.append(is_match)

        if (idx + 1) % 10000 == 0 or (idx + 1) == len(s1_records):
            print(f"Processed {idx + 1:,} / {len(s1_records):,} Source 1 entities...")

    # 4. Train LightGBM Model & Predict
    print(f"\n[3/5] Extracted features for {len(features_list):,} candidate pairs.")
    X_df = pd.DataFrame(features_list)
    y_arr = np.array(labels_list)

    print("\n[4/5] Training LightGBM Classifier & Tuning Macro F0.5 Threshold...")
    model = EntityResolutionModel()
    
    # Train/Val split if ground truth matches exist
    if len(y_arr) > 0 and np.sum(y_arr) > 0:
        split_idx = int(len(X_df) * 0.8)
        X_train, X_val = X_df.iloc[:split_idx], X_df.iloc[split_idx:]
        y_train, y_val = y_arr[:split_idx], y_arr[split_idx:]
        
        model.train(X_train, y_train, X_val, y_val)
        print(f"Optimal Macro F0.5 Threshold: {model.optimal_threshold:.3f}")
    else:
        model.train(X_df, y_arr)

    # Generate pairwise predictions
    preds = model.predict(X_df)

    # Reconstruct predictions per S1 entity
    for (s1_id, cand_id), pred in zip(all_candidate_pairs, preds):
        if pred == 1:
            results_map[s1_id].append(cand_id)

    # 5. Output Results & Validate Formats using validate_submission.py
    print("\n[5/5] Saving final matching outputs...")
    save_candidate_pairs(candidates_map, output_path="output/candidate_pairs.tsv")
    save_matching_results(results_map, output_path="output/matching_results.tsv")

    print("\nValidating output submission files against official submission validator...")
    validate_outputs(matching_file="output/matching_results.tsv", candidate_file="output/candidate_pairs.tsv", test_dir=split_dir)

if __name__ == "__main__":
    args = parse_args()
    run_pipeline(sample_size=args.sample_size, max_candidates=args.max_candidates, split=args.split)
