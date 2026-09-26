import time
import sys
import cProfile
import pstats
sys.path.insert(0, "code")
from src.features import compute_pair_features, precompute_s1_features
from src.indexing import CandidateIndexer

s1_rec = {"name": "Amazon Web Services Inc", "address": "410 Terry Ave N, Seattle, WA 98109"}
cand_rec = {"name": "Amazon.com Services LLC", "address": "410 Terry Avenue North Seattle Washington 98109"}

s1_pre = precompute_s1_features(s1_rec)

def run():
    for _ in range(5000):
        compute_pair_features(s1_rec, cand_rec, s1_pre)

cProfile.run('run()', 'stats')
p = pstats.Stats('stats')
p.sort_stats('time').print_stats(15)
