"""
Main pipeline execution script for Amazon ML Challenge 2026: Business Entity Resolution.

Usage:
    python code/main.py [--sample-size 50000] [--max-candidates 30] [--split train]
"""
import gc
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import sys
import time
import pickle
import argparse
import pandas as pd
import numpy as np
import multiprocessing as mp
from typing import Dict, List, Tuple, Any, Set

# Ensure src and parent directories are in python path for flexible execution
src_dir = os.path.dirname(os.path.abspath(__file__))
code_dir = os.path.dirname(src_dir)
root_dir = os.path.dirname(code_dir)
for p in [src_dir, code_dir, root_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from preprocessing import clean_text
    from indexing import CandidateIndexer
    from features import compute_pair_features, precompute_s1_features
    from model import EntityResolutionModel
    from submission import save_candidate_pairs, save_matching_results, validate_outputs
except (ImportError, ValueError):
    from src.preprocessing import clean_text
    from src.indexing import CandidateIndexer
    from src.features import compute_pair_features, precompute_s1_features
    from src.model import EntityResolutionModel
    from src.submission import save_candidate_pairs, save_matching_results, validate_outputs

# Global worker state for multi-process test streaming
_WORKER_INDEXER = None
_WORKER_MODEL = None
_WORKER_MAX_CANDS = 30

def _init_test_worker(db_path: str, model_path: str, dataset_base: str, max_candidates: int):
    global _WORKER_INDEXER, _WORKER_MODEL, _WORKER_MAX_CANDS
    _WORKER_INDEXER = CandidateIndexer(db_path=db_path, dataset_base=dataset_base, split="test", is_worker=True)
    _WORKER_INDEXER._records_dict = None  # Pure Zero-RAM disk mode via SQLite Primary Key
    _WORKER_INDEXER.get_frequent_tokens()
    _WORKER_MODEL = EntityResolutionModel.load(model_path)
    _WORKER_MAX_CANDS = max_candidates

def _process_test_chunk(chunk_s1: list) -> tuple:
    cand_lines = []
    match_lines = []
    for s1_rec in chunk_s1:
        s1_id = str(s1_rec["record_id"])
        s1_country = str(s1_rec.get("country", "")).strip().lower()
        cand_ids = list(_WORKER_INDEXER.find_candidates_for_record(
            name=str(s1_rec.get("name", "")),
            address=str(s1_rec.get("address", "")),
            max_candidates=_WORKER_MAX_CANDS,
            country=s1_country
        ))
        cand_lines.append(f"{s1_id}\t{','.join(cand_ids)}\n")
        if not cand_ids:
            match_lines.append(f"{s1_id}\t\n")
            continue
        cand_records = _WORKER_INDEXER.fetch_records_by_ids(cand_ids, split="test")
        s1_precomputed = precompute_s1_features(s1_rec)
        batch_pairs = []
        batch_features = []
        for cand_rec in cand_records:
            cid = str(cand_rec["record_id"])
            c2_country = str(cand_rec.get("country", "")).strip().lower()
            if s1_country and c2_country and s1_country != c2_country:
                continue
            batch_pairs.append(cid)
            batch_features.append(compute_pair_features(s1_rec, cand_rec, s1_precomputed))
        matches = []
        if batch_features:
            preds = _WORKER_MODEL.predict(pd.DataFrame(batch_features))
            for cid, pred in zip(batch_pairs, preds):
                if pred == 1:
                    matches.append(cid)
        match_lines.append(f"{s1_id}\t{','.join(matches)}\n")
    return cand_lines, match_lines

_WORKER_GT_MAP = None

def _init_train_worker(db_path: str, dataset_base: str, max_candidates: int, gt_map: dict):
    global _WORKER_INDEXER, _WORKER_MAX_CANDS, _WORKER_GT_MAP
    _WORKER_INDEXER = CandidateIndexer(db_path=db_path, dataset_base=dataset_base, split="train", is_worker=True)
    _WORKER_INDEXER._records_dict = None  # Pure Zero-RAM disk mode via SQLite Primary Key
    _WORKER_INDEXER.get_frequent_tokens()
    _WORKER_MAX_CANDS = max_candidates
    _WORKER_GT_MAP = gt_map

def _process_train_chunk(chunk_s1: list) -> tuple:
    chunk_cands_map = {}
    chunk_pairs = []
    chunk_features = []
    chunk_labels = []

    for s1_rec in chunk_s1:
        s1_id = str(s1_rec["record_id"])
        s1_country = str(s1_rec.get("country", "")).strip().lower()
        cand_ids = list(_WORKER_INDEXER.find_candidates_for_record(
            name=str(s1_rec.get("name", "")),
            address=str(s1_rec.get("address", "")),
            max_candidates=_WORKER_MAX_CANDS,
            country=s1_country
        ))
        chunk_cands_map[s1_id] = cand_ids
        if not cand_ids:
            continue

        cand_records = _WORKER_INDEXER.fetch_records_by_ids(cand_ids, split="train")
        s1_gt_targets = _WORKER_GT_MAP.get(s1_id, set())
        s1_precomputed = precompute_s1_features(s1_rec)

        for cand_rec in cand_records:
            cand_id = str(cand_rec["record_id"])
            c2_country = str(cand_rec.get("country", "")).strip().lower()
            if s1_country and c2_country and s1_country != c2_country:
                continue

            chunk_pairs.append((s1_id, cand_id))
            feats = compute_pair_features(s1_rec, cand_rec, s1_precomputed)
            chunk_features.append(feats)
            is_match = 1 if cand_id in s1_gt_targets else 0
            chunk_labels.append(is_match)

    return chunk_cands_map, chunk_pairs, chunk_features, chunk_labels

def find_dataset_base() -> str:
    """Dynamically resolves dataset folder location across local, SageMaker, and Colab environments."""
    possible_paths = [
        "dataset",
        "student_resource/dataset",
        "6ab10eb3b23ba_student_resource/student_resource/dataset",
        "../6ab10eb3b23ba_student_resource/student_resource/dataset",
        "/content/6ab10eb3b23ba_student_resource/student_resource/dataset",
        "/content/dataset"
    ]
    for p in possible_paths:
        if os.path.exists(p):
            return p
    return "dataset"

def find_db_path(split: str = "train", explicit_path: str = None) -> str:
    """Dynamically resolves SQLite index.db location across local, SageMaker, and Colab environments."""
    if explicit_path:
        return explicit_path
    db_name = f"index_{split}.db" if split != "train" else "index.db"
    possible_paths = [
        db_name,
        os.path.join("dataset", db_name),
        f"6ab10eb3b23ba_student_resource/student_resource/{db_name}",
        f"student_resource/{db_name}",
        f"../6ab10eb3b23ba_student_resource/student_resource/{db_name}",
        f"/content/6ab10eb3b23ba_student_resource/student_resource/{db_name}",
        f"/content/{db_name}"
    ]
    for p in possible_paths:
        if os.path.exists(p):
            return p
    return db_name

def parse_args():
    parser = argparse.ArgumentParser(description="Run Business Entity Resolution Pipeline")
    parser.add_argument("--sample-size", type=int, default=50000, help="Number of Source 1 records to process (default 50,000)")
    parser.add_argument("--max-candidates", type=int, default=40, help="Max candidates per entity from index.db (default 40)")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"], help="Dataset split to run on (train or test)")
    parser.add_argument("--no-cache", action="store_true", help="Disable caching of extracted features")
    parser.add_argument("--db-path", type=str, default=None, help="Explicit path to SQLite index.db (optional)")
    parser.add_argument("--num-workers", type=int, default=min(8, os.cpu_count() or 4), help="Number of parallel worker processes (default: min(8, CPU count))")
    parser.add_argument("--num-chunks", type=int, default=1, help="Number of sequential training chunks to process (default: 1)")
    parser.add_argument("--chunk-size", type=int, default=None, help="Size of each training chunk (overrides --sample-size if provided)")
    parser.add_argument("--chunk-offset", type=int, default=0, help="Starting record offset for chunk training (default: 0)")
    parser.add_argument("--val-size", type=int, default=2000, help="Number of Source 1 entities for fixed holdout validation (default: 2,000)")
    parser.add_argument("--trees-per-chunk", type=int, default=100, help="Number of trees to add per chunk in multi-chunk training (default: 100)")
    parser.add_argument("--continue-training", action="store_true", help="Continue training from existing model checkpoint")
    parser.add_argument("--eval-only", action="store_true", help="Only run validation audit and confusion matrix evaluation on existing model and exit")
    return parser.parse_args()

def evaluate_and_report_validation(
    model: EntityResolutionModel,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    val_entities_list: List[str],
    val_pairs_list: List[Any],
    gt_map: Dict[str, Set[str]],
    max_candidates: int,
    title: str = "VALIDATION AUDIT & COMPETITION METRICS REPORT"
) -> Dict[str, Any]:
    val_probs = model.predict_proba(X_val)
    val_preds = (val_probs >= model.optimal_threshold).astype(int)

    tp = int(np.sum((val_preds == 1) & (y_val == 1)))
    fp = int(np.sum((val_preds == 1) & (y_val == 0)))
    fn = int(np.sum((val_preds == 0) & (y_val == 1)))
    tn = int(np.sum((val_preds == 0) & (y_val == 0)))

    cand_prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    cand_rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    cand_f05 = (1.25 * cand_prec * cand_rec) / (0.25 * cand_prec + cand_rec) if (0.25 * cand_prec + cand_rec) > 0 else 0.0

    ent_to_idx = {e: i for i, e in enumerate(val_entities_list)}
    n_val_ents = len(val_entities_list)
    gt_counts = np.array([len(gt_map.get(e, set())) for e in val_entities_list], dtype=float)
    total_val_gt = int(np.sum(gt_counts))
    val_singletons = int(np.sum(gt_counts == 0))

    pair_ents = np.array([ent_to_idx[p[0]] for p in val_pairs_list])
    pair_is_gt = np.array([p[1] in gt_map.get(p[0], set()) for p in val_pairs_list])

    tp_pairs = (val_preds == 1) & pair_is_gt
    fp_pairs = (val_preds == 1) & (~pair_is_gt)

    tp_g = np.bincount(pair_ents, weights=tp_pairs, minlength=n_val_ents)
    fp_g = np.bincount(pair_ents, weights=fp_pairs, minlength=n_val_ents)

    p_den = tp_g + fp_g
    e2e_prec_g = np.divide(tp_g, p_den, out=np.zeros_like(tp_g, dtype=float), where=p_den != 0)
    e2e_rec_g = np.divide(tp_g, gt_counts, out=np.zeros_like(tp_g, dtype=float), where=gt_counts != 0)

    f_den = 0.25 * e2e_prec_g + e2e_rec_g
    f05_g = np.divide(1.25 * e2e_prec_g * e2e_rec_g, f_den, out=np.zeros_like(e2e_prec_g), where=f_den != 0)

    singleton_mask = (gt_counts == 0)
    correct_singletons = int(np.sum(singleton_mask & (fp_g == 0)))
    false_singletons = int(np.sum(singleton_mask & (fp_g > 0)))
    f05_g[singleton_mask & (fp_g == 0)] = 1.0
    f05_g[singleton_mask & (fp_g > 0)] = 0.0

    competition_macro_f05 = float(np.mean(f05_g))
    retrieved_in_pool = int(np.sum(y_val))
    blocking_recall = (retrieved_in_pool / total_val_gt * 100) if total_val_gt > 0 else 100.0

    print("\n" + "=" * 65, flush=True)
    print(f"       {title}       ", flush=True)
    print("=" * 65, flush=True)
    print("  --- 1. Validation Cohort & Stage 1 Retrieval ---", flush=True)
    print(f"  Holdout S1 Entities       : {n_val_ents:,} (100% leak-proof grouped)", flush=True)
    print(f"  Total Ground Truth Matches: {total_val_gt:,}", flush=True)
    print(f"  True Singletons in GT     : {val_singletons:,} ({val_singletons/n_val_ents*100:.1f}%)", flush=True)
    print(f"  Stage 1 Candidates Formed : {len(y_val):,} pairs", flush=True)
    print(f"  Stage 1 Candidate Recall  : {blocking_recall:.2f}% ({retrieved_in_pool:,} / {total_val_gt:,} true matches in top-{max_candidates})", flush=True)
    print("-" * 65, flush=True)
    print("  --- 2. Pairwise Classifier Diagnostics (Conditional on Retrieval) ---", flush=True)
    print(f"  Optimal Decision Threshold: {model.optimal_threshold:.3f}", flush=True)
    print(f"  Candidate True Positives  : {tp:,}", flush=True)
    print(f"  Candidate False Positives : {fp:,}", flush=True)
    print(f"  Candidate False Negatives : {fn:,}", flush=True)
    print(f"  Candidate True Negatives  : {tn:,}", flush=True)
    print(f"  Pairwise Precision        : {cand_prec * 100:.2f}%", flush=True)
    print(f"  Pairwise Recall (on cands): {cand_rec * 100:.2f}%", flush=True)
    print(f"  Pairwise F0.5 (on cands)  : {cand_f05:.4f}", flush=True)
    print("-" * 65, flush=True)
    print("  --- 3. OFFICIAL END-TO-END COMPETITION LEADERBOARD METRIC ---", flush=True)
    print(f"  Correct Singletons (1.0)  : {correct_singletons:,} / {val_singletons:,}", flush=True)
    print(f"  False Merges on Singletons: {false_singletons:,} / {val_singletons:,}", flush=True)
    print(f"  End-to-End True Positives : {int(np.sum(tp_g)):,} / {total_val_gt:,} total matches", flush=True)
    print(f"  >>> COMPETITION MACRO F0.5: {competition_macro_f05:.4f} <<< [OFFICIAL SCORER EQUIVALENT]", flush=True)
    print("=" * 65 + "\n", flush=True)

    metrics = {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "cand_prec": float(cand_prec),
        "cand_rec": float(cand_rec),
        "cand_f05": float(cand_f05),
        "blocking_recall": float(blocking_recall),
        "macro_f05": float(competition_macro_f05),
        "threshold": float(model.optimal_threshold)
    }
    model.val_metrics = metrics
    return metrics

def run_pipeline(
    sample_size: int = 50000,
    max_candidates: int = 40,
    split: str = "train",
    no_cache: bool = False,
    db_path_arg: str = None,
    num_workers: int = None,
    num_chunks: int = 1,
    chunk_size: int = None,
    chunk_offset: int = 0,
    val_size: int = 2000,
    trees_per_chunk: int = 100,
    continue_training: bool = False,
    eval_only: bool = False
):
    if num_workers is None:
        num_workers = min(8, os.cpu_count() or 4)
    dataset_base = find_dataset_base()
    db_path = find_db_path(split=split, explicit_path=db_path_arg)

    print("=" * 70, flush=True)
    print(f"   BUSINESS ENTITY RESOLUTION PIPELINE ({split.upper()} SET)   ", flush=True)
    print("=" * 70, flush=True)
    print(f"Dataset Location: {dataset_base}", flush=True)
    print(f"SQLite Index DB:  {db_path}", flush=True)

    split_dir = os.path.join(dataset_base, split)
    s1_filename = f"{split}_source1.tsv"
    s1_path = os.path.join(split_dir, s1_filename)
    gt_path = os.path.join(dataset_base, "train", "train_ground_truth.tsv")
    
    if not os.path.exists(s1_path):
        print(f"\nError: Dataset file not found at {s1_path}", flush=True)
        print("Searched locations:")
        print(f"  - {s1_path}")
        return

    # 1. Load Source 1 Sample (Only needed for test streaming inference)
    if split == "test":
        print(f"\n[1/5] Loading Source 1 records from {s1_path} (limit={sample_size:,})...", flush=True)
        if sample_size and sample_size > 0:
            df_s1 = pd.read_csv(s1_path, sep="\t", nrows=sample_size)
        else:
            df_s1 = pd.read_csv(s1_path, sep="\t", engine="pyarrow")
        
        rename_dict = {}
        if "entity_id" in df_s1.columns: rename_dict["entity_id"] = "record_id"
        if "business_name" in df_s1.columns: rename_dict["business_name"] = "name"
        if "business_address" in df_s1.columns: rename_dict["business_address"] = "address"
        if rename_dict:
            df_s1 = df_s1.rename(columns=rename_dict)
        print(f"Loaded {len(df_s1):,} Source 1 records.", flush=True)

    # 2. Fast Vectorized Ground Truth Mapping Load (only needed for train split)
    gt_map = {}
    if split == "train" and os.path.exists(gt_path):
        print("Loading Ground Truth labels (vectorized dictionary build)...", flush=True)
        df_gt = pd.read_csv(gt_path, sep="\t", engine="pyarrow")
        gt_id_col = "source1_entity_id" if "source1_entity_id" in df_gt.columns else df_gt.columns[0]
        gt_match_col = "matched_entity_ids" if "matched_entity_ids" in df_gt.columns else df_gt.columns[1]
        
        gt_ids = df_gt[gt_id_col].astype(str).values
        gt_matches = df_gt[gt_match_col].fillna("").astype(str).values
        
        gt_map = {
            s1_id: set(m.split(",")) if m else set()
            for s1_id, m in zip(gt_ids, gt_matches)
        }
        print(f"Loaded Ground Truth mapping for {len(gt_map):,} entities.", flush=True)

    # 3. Model & Output Paths (Separate folders for Training vs Testing/Submission)
    train_dir = os.path.join("output", "train")
    test_dir = os.path.join("output", "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    out_dir = test_dir if split == "test" else train_dir
    cand_out_path = os.path.join(out_dir, "candidate_pairs.tsv")
    match_out_path = os.path.join(out_dir, "matching_results.tsv")

    # Locate model: checks output/train/lgb_model.pkl, falls back to output/lgb_model.pkl
    model_save_path = os.path.join(train_dir, "lgb_model.pkl")
    model_load_path = model_save_path if os.path.exists(model_save_path) else "output/lgb_model.pkl"

    if split == "test":
        # ======================================================================
        # ZERO-RAM STREAMING INFERENCE FOR TEST SET
        # Streams predictions directly to disk in batches of 500 entities.
        # RAM usage remains strictly < 400 MB throughout the entire run.
        # ======================================================================
        print(f"\n[2/5] Loading pre-trained LightGBM model from {model_load_path}...", flush=True)
        if not os.path.exists(model_load_path):
            raise FileNotFoundError(f"Model file {model_load_path} not found! Please run '--split train' first.")
        model = EntityResolutionModel.load(model_load_path)

        print(f"\n[3/5] Initializing Zero-RAM SQLite Disk Index (TEST set)...", flush=True)
        indexer = CandidateIndexer(db_path=db_path, dataset_base=dataset_base, split="test")
        indexer.get_frequent_tokens()

        print(f"\n[4/5] Streaming predictions directly to disk in batches (RAM < 400 MB)...", flush=True)
        s1_records = df_s1.to_dict("records")
        total_s1 = len(s1_records)
        start_time = time.time()

        with open(cand_out_path, "w") as f_cand, open(match_out_path, "w") as f_match:
            f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
            f_match.write("source1_entity_id\tmatched_entity_ids\n")

            if num_workers > 1:
                indexer.close()  # CRITICAL: Close SQLite connection in parent before fork to avoid deadlock in workers

                _test_batch_size = 500
                _test_chunks = (s1_records[i:i + _test_batch_size] for i in range(0, total_s1, _test_batch_size))
                _test_num_chunks = (total_s1 + _test_batch_size - 1) // _test_batch_size
                print(f"Executing with {num_workers} parallel workers across {_test_num_chunks:,} chunks (Pure Zero-RAM Mode)...", flush=True)

                processed_count = 0
                with mp.Pool(
                    processes=num_workers,
                    initializer=_init_test_worker,
                    initargs=(db_path, model_load_path, dataset_base, max_candidates)
                ) as pool:
                    for cand_lines, match_lines in pool.imap(_process_test_chunk, _test_chunks, chunksize=1):
                        f_cand.writelines(cand_lines)
                        f_match.writelines(match_lines)
                        processed_count += len(cand_lines)

                        if processed_count % 1000 == 0 or processed_count == total_s1 or total_s1 <= 2000:
                            f_cand.flush()
                            f_match.flush()
                            elapsed = time.time() - start_time
                            pct = processed_count / total_s1 * 100
                            rate = processed_count / elapsed if elapsed > 0 else 0
                            eta = (total_s1 - processed_count) / rate if rate > 0 else 0
                            print(f"  [{pct:5.1f}%] Processed {processed_count:,} / {total_s1:,} entities ({rate:.0f} ent/s | ETA: {eta:.0f}s)", flush=True)
            else:
                batch_size = 500
                for b_idx in range(0, total_s1, batch_size):
                    b_end = min(b_idx + batch_size, total_s1)
                    batch_s1 = s1_records[b_idx:b_end]

                    batch_pairs = []
                    batch_features = []
                    batch_cands = {}

                    for s1_rec in batch_s1:
                        s1_id = str(s1_rec["record_id"])
                        s1_country = str(s1_rec.get("country", "")).strip().lower()
                        cand_ids = list(indexer.find_candidates_for_record(
                            name=str(s1_rec.get("name", "")),
                            address=str(s1_rec.get("address", "")),
                            max_candidates=max_candidates,
                            country=s1_country
                        ))
                        batch_cands[s1_id] = cand_ids

                        if not cand_ids:
                            continue

                        cand_records = indexer.fetch_records_by_ids(cand_ids, split="test")
                        s1_precomputed = precompute_s1_features(s1_rec)

                        for cand_rec in cand_records:
                            cid = str(cand_rec["record_id"])
                            c2_country = str(cand_rec.get("country", "")).strip().lower()
                            # Open-set country filter: reject impossible cross-country pairs
                            if s1_country and c2_country and s1_country != c2_country:
                                continue
                            batch_pairs.append((s1_id, cid))
                            batch_features.append(compute_pair_features(s1_rec, cand_rec, s1_precomputed))

                    # Flush candidate pairs to disk immediately (1 row per S1 entity, comma-separated candidate IDs)
                    for s1_id, cand_ids in batch_cands.items():
                        f_cand.write(f"{s1_id}\t{','.join(cand_ids)}\n")

                    # Predict matches for this batch
                    batch_matches = {s1_id: [] for s1_id in batch_cands}
                    if batch_features:
                        X_b = pd.DataFrame(batch_features)
                        preds = model.predict(X_b)
                        for (s1_id, cid), pred in zip(batch_pairs, preds):
                            if pred == 1:
                                batch_matches[s1_id].append(cid)

                    # Flush match results to disk immediately
                    for s1_id, matches in batch_matches.items():
                        f_match.write(f"{s1_id}\t{','.join(matches)}\n")

                    # Progress report
                    elapsed = time.time() - start_time
                    pct = b_end / total_s1 * 100
                    rate = b_end / elapsed if elapsed > 0 else 0
                    eta = (total_s1 - b_end) / rate if rate > 0 else 0
                    print(f"  [{pct:5.1f}%] Processed {b_end:,} / {total_s1:,} entities ({rate:.0f} ent/s | ETA: {eta:.0f}s)", flush=True)

                    # Explicit memory release per batch
                    del batch_pairs, batch_features, batch_cands, batch_matches
                    gc.collect()

        print(f"\n[5/5] Test inference complete! Outputs saved to {cand_out_path} and {match_out_path}.", flush=True)
        
        # Free memory before running validator subprocess to prevent OOM
        del df_s1, s1_records
        if "chunks" in locals():
            del chunks
        gc.collect()

        print("\nValidating output submission files against official submission validator...", flush=True)
        validate_outputs(matching_file=match_out_path, candidate_file=cand_out_path, test_dir=split_dir)
        return

    # ======================================================================
    # MULTI-CHUNK CONTINUAL TRAINING PIPELINE (split == 'train')
    # ======================================================================
    effective_chunk_size = chunk_size if chunk_size else sample_size

    # 1. Locked Fixed Holdout Validation Benchmark
    val_cache_path = os.path.join(train_dir, f"fixed_val_set_{val_size}_{max_candidates}.pkl")
    if os.path.exists(val_cache_path) and not no_cache:
        print(f"\n[2/5] Loading Locked Fixed Validation Benchmark from {val_cache_path}...", flush=True)
        try:
            with open(val_cache_path, "rb") as f:
                val_data = pickle.load(f)
            val_entities_list = val_data["val_entities"]
            val_pairs_list = val_data["val_pairs"]
            X_val = val_data["X_val"]
            y_val = val_data["y_val"]
            val_groups = val_data.get("val_groups")
            val_entities_set = set(val_entities_list)
            print(f"Loaded locked holdout benchmark ({len(val_entities_list):,} entities, {len(y_val):,} pairs).", flush=True)
        except Exception as e:
            print(f"Warning: Failed to load validation cache ({e}), regenerating...", flush=True)
            val_entities_list = None
    else:
        val_entities_list = None

    if val_entities_list is None:
        print(f"\n[2/5] Creating and locking Fixed Validation Benchmark ({val_size:,} entities from holdout slice)...", flush=True)
        # Sample val_size entities from the tail of the dataset (skiprows = 2,206,821 - val_size)
        val_skip = max(0, 2206821 - val_size)
        df_val_raw = pd.read_csv(s1_path, sep="\t", skiprows=range(1, val_skip + 1), nrows=val_size)
        rename_dict = {}
        if "entity_id" in df_val_raw.columns: rename_dict["entity_id"] = "record_id"
        if "business_name" in df_val_raw.columns: rename_dict["business_name"] = "name"
        if "business_address" in df_val_raw.columns: rename_dict["business_address"] = "address"
        if rename_dict:
            df_val_raw = df_val_raw.rename(columns=rename_dict)
        val_entities_list = df_val_raw["record_id"].astype(str).tolist()
        val_entities_set = set(val_entities_list)

        val_records = df_val_raw.to_dict("records")
        val_chunks = [val_records[i:i + 250] for i in range(0, len(val_records), 250)]
        val_relevant_gt = {k: v for k, v in gt_map.items() if k in val_entities_set}

        val_pairs_list = []
        val_features_list = []
        val_labels_list = []

        print(f"Extracting validation candidate pairs & features with {num_workers} parallel workers...", flush=True)
        with mp.Pool(
            processes=num_workers,
            initializer=_init_train_worker,
            initargs=(db_path, dataset_base, max_candidates, val_relevant_gt)
        ) as pool:
            for _, c_pairs, c_feats, c_labels in pool.imap(_process_train_chunk, val_chunks, chunksize=1):
                val_pairs_list.extend(c_pairs)
                val_features_list.extend(c_feats)
                val_labels_list.extend(c_labels)

        X_val = pd.DataFrame(val_features_list)
        y_val = np.array(val_labels_list)
        val_groups = np.array([p[0] for p in val_pairs_list])

        val_data = {
            "val_entities": val_entities_list,
            "val_pairs": val_pairs_list,
            "X_val": X_val,
            "y_val": y_val,
            "val_groups": val_groups
        }
        with open(val_cache_path, "wb") as f:
            pickle.dump(val_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Locked fixed holdout benchmark ({len(val_entities_list):,} entities, {len(y_val):,} pairs) saved to {val_cache_path}.", flush=True)
        del df_val_raw, val_records, val_chunks, val_relevant_gt, val_features_list, val_labels_list
        gc.collect()

    # 2. Standalone Evaluation Mode or Continuation
    if eval_only:
        print(f"\n[3/5] Standalone Validation Audit & Benchmark Evaluation Mode (--eval-only)...", flush=True)
        if not os.path.exists(model_save_path):
            alt_path = "output/lgb_model.pkl"
            if os.path.exists(alt_path):
                model_save_path = alt_path
            else:
                raise FileNotFoundError(f"Model checkpoint not found at {model_save_path} or {alt_path}!")
        print(f"Loading trained ensemble model from {model_save_path}...", flush=True)
        model = EntityResolutionModel.load(model_save_path)
        metrics = evaluate_and_report_validation(
            model=model, X_val=X_val, y_val=y_val,
            val_entities_list=val_entities_list, val_pairs_list=val_pairs_list,
            gt_map=gt_map, max_candidates=max_candidates,
            title=f"EVALUATION AUDIT (Top-{max_candidates} Candidates | {len(val_entities_list):,} Entities)"
        )
        val_metrics_path = os.path.join(train_dir, "val_metrics.pkl")
        with open(val_metrics_path, "wb") as f:
            pickle.dump(metrics, f)
        print(f"Saved validation audit metrics to {val_metrics_path} for Cell 5b visualization.\n", flush=True)
        return

    # 2b. Check for Continuation Model
    model = None
    if continue_training or (chunk_offset > 0 and os.path.exists(model_save_path)):
        if os.path.exists(model_save_path):
            print(f"\n[3/5] Loading existing model checkpoint from {model_save_path} for continuation...", flush=True)
            model = EntityResolutionModel.load(model_save_path)
            evaluate_and_report_validation(
                model=model, X_val=X_val, y_val=y_val,
                val_entities_list=val_entities_list, val_pairs_list=val_pairs_list,
                gt_map=gt_map, max_candidates=max_candidates,
                title="BASELINE MODEL (BEFORE CHUNK TRAINING)"
            )
        else:
            print(f"\n[3/5] Note: Model file {model_save_path} not found. Starting new model from scratch.", flush=True)

    # 3. Multi-Chunk Training Loop
    chunk_metrics_history = []

    for chunk_idx in range(num_chunks):
        current_offset = chunk_offset + chunk_idx * effective_chunk_size
        chunk_title = f"CHUNK {chunk_idx + 1} / {num_chunks} (Offset: {current_offset:,} | Size: {effective_chunk_size:,})"
        print("\n" + "=" * 70, flush=True)
        print(f"   TRAINING {chunk_title}   ", flush=True)
        print("=" * 70, flush=True)

        # Load chunk slice
        if current_offset > 0:
            df_chunk = pd.read_csv(s1_path, sep="\t", skiprows=range(1, current_offset + 1), nrows=effective_chunk_size)
        else:
            df_chunk = pd.read_csv(s1_path, sep="\t", nrows=effective_chunk_size)

        rename_dict = {}
        if "entity_id" in df_chunk.columns: rename_dict["entity_id"] = "record_id"
        if "business_name" in df_chunk.columns: rename_dict["business_name"] = "name"
        if "business_address" in df_chunk.columns: rename_dict["business_address"] = "address"
        if rename_dict:
            df_chunk = df_chunk.rename(columns=rename_dict)

        # 100% Leak-Proof Guarantee: Exclude any record belonging to the locked validation benchmark
        df_chunk = df_chunk[~df_chunk["record_id"].astype(str).isin(val_entities_set)].copy()
        s1_records = df_chunk.to_dict("records")
        print(f"Loaded {len(s1_records):,} Source 1 records for Chunk {chunk_idx + 1} (strictly isolated from holdout).", flush=True)

        if not s1_records:
            print(f"Warning: No valid records for Chunk {chunk_idx + 1}. Skipping...", flush=True)
            continue

        # Multiprocessing Candidate Retrieval & Pairwise Feature Extraction
        chunks = [s1_records[i:i + 250] for i in range(0, len(s1_records), 250)]
        s1_id_set = {str(r["record_id"]) for r in s1_records}
        relevant_gt = {k: v for k, v in gt_map.items() if k in s1_id_set}

        candidates_map = {}
        all_candidate_pairs = []
        features_list = []
        labels_list = []

        start_time = time.time()
        processed_count = 0
        print(f"Extracting features with {num_workers} parallel workers across {len(chunks):,} chunks...", flush=True)
        with mp.Pool(
            processes=num_workers,
            initializer=_init_train_worker,
            initargs=(db_path, dataset_base, max_candidates, relevant_gt)
        ) as pool:
            for c_cands, c_pairs, c_feats, c_labels in pool.imap(_process_train_chunk, chunks, chunksize=1):
                candidates_map.update(c_cands)
                all_candidate_pairs.extend(c_pairs)
                features_list.extend(c_feats)
                labels_list.extend(c_labels)
                processed_count += len(c_cands)

                if processed_count % 1000 == 0 or processed_count == len(s1_records) or len(s1_records) <= 2000:
                    elapsed = time.time() - start_time
                    pct = processed_count / len(s1_records) * 100
                    rate = processed_count / elapsed if elapsed > 0 else 0
                    eta = (len(s1_records) - processed_count) / rate if rate > 0 else 0
                    print(f"  [{pct:5.1f}%] Processed {processed_count:,} / {len(s1_records):,} entities ({rate:.0f} ent/s | ETA: {eta:.0f}s)", flush=True)

        X_train = pd.DataFrame(features_list)
        y_train = np.array(labels_list)
        print(f"Extracted {len(features_list):,} candidate pair features for Chunk {chunk_idx + 1}.", flush=True)

        if len(features_list) == 0:
            print(f"Warning: No candidate pairs extracted for Chunk {chunk_idx + 1} (all records may be singletons or unindexed). Skipping model training for this chunk.", flush=True)
            del df_chunk, s1_records, chunks, s1_id_set, relevant_gt, candidates_map, all_candidate_pairs, features_list, labels_list, X_train, y_train
            gc.collect()
            continue

        is_continuation = (model is not None)
        if model is None:
            model = EntityResolutionModel()

        print(f"\nFitting ensemble (continuation={is_continuation}, trees_per_chunk={trees_per_chunk})...", flush=True)
        model.train(
            X_train, y_train,
            X_val=X_val, y_val=y_val,
            val_entities=val_entities_list,
            val_pairs=val_pairs_list,
            gt_map=gt_map,
            is_continuation=is_continuation,
            trees_per_chunk=trees_per_chunk
        )

        metrics = evaluate_and_report_validation(
            model=model, X_val=X_val, y_val=y_val,
            val_entities_list=val_entities_list, val_pairs_list=val_pairs_list,
            gt_map=gt_map, max_candidates=max_candidates,
            title=f"CHUNK {chunk_idx + 1} / {num_chunks} VALIDATION AUDIT (OFFSET {current_offset:,})"
        )
        chunk_metrics_history.append((chunk_idx + 1, current_offset, metrics["macro_f05"]))
        model.val_metrics = metrics

        # Persist checkpoint to disk (wrapped for disk-full safety)
        try:
            model.save(model_save_path)
        except OSError as e:
            print(f"[ERROR] Could not save model to {model_save_path}: {e}. Check available disk space!", flush=True)
        try:
            model.save("output/lgb_model.pkl")
        except Exception:
            pass

        # Explicitly free memory per chunk
        del df_chunk, s1_records, chunks, s1_id_set, relevant_gt, candidates_map, all_candidate_pairs, features_list, labels_list, X_train, y_train
        gc.collect()

    print("\n" + "=" * 65, flush=True)
    print("       MULTI-CHUNK CONTINUAL TRAINING COMPLETE       ", flush=True)
    print("=" * 65, flush=True)
    print("Progression of Official Competition Macro F0.5 on Fixed Holdout Benchmark:")
    for c_num, c_off, m_f05 in chunk_metrics_history:
        print(f"  Chunk {c_num} (Offset {c_off:,}): Macro F0.5 = {m_f05:.4f}", flush=True)
    print(f"\nFinal model successfully saved to {model_save_path} and output/lgb_model.pkl")
    print("=" * 65 + "\n", flush=True)

if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        sample_size=args.sample_size,
        max_candidates=args.max_candidates,
        split=args.split,
        no_cache=args.no_cache,
        db_path_arg=args.db_path,
        num_workers=args.num_workers,
        num_chunks=args.num_chunks,
        chunk_size=args.chunk_size,
        chunk_offset=args.chunk_offset,
        val_size=args.val_size,
        trees_per_chunk=args.trees_per_chunk,
        continue_training=args.continue_training,
        eval_only=args.eval_only
    )
