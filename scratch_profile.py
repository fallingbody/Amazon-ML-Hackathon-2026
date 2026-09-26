import time
import os
import sys
sys.path.insert(0, "code")
from src.indexing import CandidateIndexer
from src.features import compute_pair_features

indexer = CandidateIndexer()
indexer.load_records_dict()

start = time.time()
cand_ids = list(indexer.find_candidates_for_record("AMAZON SERVICES", "123 MAIN ST"))
end_query = time.time()

cand_records = indexer.fetch_records_by_ids(cand_ids)
s1_rec = {"name": "AMAZON SERVICES", "address": "123 MAIN ST"}

for cand_rec in cand_records[:30]:
    compute_pair_features(s1_rec, cand_rec)
end_feats = time.time()

print(f"Query time: {end_query - start:.4f}s")
print(f"Features time: {end_feats - end_query:.4f}s for {len(cand_records[:30])} cands")
