import sys
import psutil
import os
sys.path.insert(0, "code")
from src.indexing import CandidateIndexer

process = psutil.Process(os.getpid())
print(f"Memory before: {process.memory_info().rss / 1024 / 1024:.2f} MB")

indexer = CandidateIndexer()
indexer.load_records_dict()

print(f"Memory after records_dict: {process.memory_info().rss / 1024 / 1024:.2f} MB")
