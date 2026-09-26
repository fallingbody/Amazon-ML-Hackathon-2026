import time
import os
import sys
sys.path.insert(0, "code")
from src.indexing import CandidateIndexer
from src.features import compute_pair_features
import pandas as pd

def profile():
    print("Loading indexer...")
    indexer = CandidateIndexer()
    indexer.load_records_dict()
    print("Pre-caching tokens...")
    indexer.get_frequent_tokens()
    
    print("Loading S1 data...")
    dataset_base = "6ab10eb3b23ba_student_resource/student_resource/dataset"
    if not os.path.exists(dataset_base):
        dataset_base = "dataset"
    
    # We'll just mock 200 S1 records to see the speed
    s1_records = [
        {"record_id": f"S1_{i}", "name": f"Business Name {i}", "address": f"123 Street {i}"}
        for i in range(200)
    ]
    
    print("Starting loop profiling...")
    start_total = time.time()
    query_times = []
    fetch_times = []
    feat_times = []
    
    for i, s1_rec in enumerate(s1_records):
        t0 = time.time()
        cand_ids = list(indexer.find_candidates_for_record(
            name=s1_rec["name"],
            address=s1_rec["address"],
            max_candidates=30
        ))
        t1 = time.time()
        
        if cand_ids:
            cand_records = indexer.fetch_records_by_ids(cand_ids)
        else:
            cand_records = []
        t2 = time.time()
        
        for cand_rec in cand_records:
            compute_pair_features(s1_rec, cand_rec)
        t3 = time.time()
        
        query_times.append(t1 - t0)
        fetch_times.append(t2 - t1)
        feat_times.append(t3 - t2)
    
    total = time.time() - start_total
    print(f"Total time for 200 entities: {total:.4f}s ({200/total:.1f} ent/s)")
    print(f"Avg Query time: {sum(query_times)/len(query_times):.4f}s")
    print(f"Avg Fetch time: {sum(fetch_times)/len(fetch_times):.4f}s")
    print(f"Avg Feat time:  {sum(feat_times)/len(feat_times):.4f}s")

if __name__ == "__main__":
    profile()
