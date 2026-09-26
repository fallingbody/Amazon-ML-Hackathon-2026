"""
Module: indexing.py
Handles zero-RAM candidate blocking and attribute retrieval using the SQLite index.db database.
"""
import os
import sqlite3
import pandas as pd
from typing import List, Set, Dict, Any
from .preprocessing import extract_tokens

DEFAULT_DB_PATH = "6ab10eb3b23ba_student_resource/student_resource/index.db"
DEFAULT_DATASET_BASE = "6ab10eb3b23ba_student_resource/student_resource/dataset"

class CandidateIndexer:
    """Class to interact with SQLite index.db for zero-RAM candidate lookup."""
    
    def __init__(self, db_path: str = DEFAULT_DB_PATH, dataset_base: str = DEFAULT_DATASET_BASE):
        self.db_path = db_path
        self.dataset_base = dataset_base
        self._ensure_records_table_indexed()

    def get_connection(self):
        return sqlite3.connect(self.db_path)

    def _ensure_records_table_indexed(self, split: str = "train"):
        """Ensures 'records' table exists in index.db for zero-RAM SQL lookups."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='records'")
        exists = cursor.fetchone()
        
        if exists:
            conn.close()
            return

        print("\nCreating 'records' table in SQLite index.db for Zero-RAM candidate lookups...", flush=True)
        cursor.execute("PRAGMA synchronous = OFF")
        cursor.execute("PRAGMA journal_mode = MEMORY")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS records (
                record_id TEXT PRIMARY KEY,
                name TEXT,
                address TEXT,
                country TEXT,
                dataset TEXT
            )
        """)
        conn.commit()

        split_dir = os.path.join(self.dataset_base, split)
        for src_num in [2, 3]:
            tsv_path = os.path.join(split_dir, f"{split}_source{src_num}.tsv")
            if not os.path.exists(tsv_path):
                continue
                
            print(f"Fast-indexing Source {src_num} attributes into SQLite disk database...", flush=True)
            # High-speed chunked vector insertion
            chunk_count = 0
            for chunk in pd.read_csv(tsv_path, sep="\t", chunksize=250000):
                id_col = "entity_id" if "entity_id" in chunk.columns else chunk.columns[0]
                name_col = "business_name" if "business_name" in chunk.columns else chunk.columns[1]
                addr_col = "business_address" if "business_address" in chunk.columns else chunk.columns[2]
                ctry_col = "country" if "country" in chunk.columns else chunk.columns[3]
                
                chunk["dataset"] = f"source{src_num}"
                records_to_insert = list(
                    chunk[[id_col, name_col, addr_col, ctry_col, "dataset"]]
                    .fillna("")
                    .astype(str)
                    .itertuples(index=False, name=None)
                )
                
                cursor.executemany(
                    "INSERT OR IGNORE INTO records (record_id, name, address, country, dataset) VALUES (?, ?, ?, ?, ?)",
                    records_to_insert
                )
                chunk_count += len(records_to_insert)
                print(f"  Indexed {chunk_count:,} Source {src_num} records into SQLite...", flush=True)

            conn.commit()

        conn.close()
        print("SQLite 'records' table indexing complete!\n", flush=True)

    def find_candidates_for_record(self, name: str, address: str, max_candidates: int = 50) -> Set[str]:
        """
        Extracts search tokens from name and address and queries index.db
        to retrieve matching candidate record IDs from S2 and S3.
        """
        tokens = extract_tokens(name, address)
        if not tokens:
            return set()

        conn = self.get_connection()
        cursor = conn.cursor()

        # Inspect column name in token_index
        cursor.execute("PRAGMA table_info(token_index)")
        cols = [row[1] for row in cursor.fetchall()]
        id_col = "record_id" if "record_id" in cols else "entity_id"

        candidate_counts = {}
        for token in tokens:
            cursor.execute(f"SELECT {id_col} FROM token_index WHERE token = ?", (token,))
            rows = cursor.fetchall()
            for (rec_id,) in rows:
                candidate_counts[rec_id] = candidate_counts.get(rec_id, 0) + 1

        conn.close()

        # Sort candidates by number of matching tokens in descending order
        sorted_candidates = sorted(candidate_counts.items(), key=lambda x: x[1], reverse=True)
        return {rec_id for rec_id, _ in sorted_candidates[:max_candidates]}

    def fetch_records_by_ids(self, record_ids: List[str], split: str = "train") -> pd.DataFrame:
        """
        Fetches full record attributes directly from SQLite disk database in < 1ms.
        Zero RAM overhead.
        """
        if not record_ids:
            return pd.DataFrame()

        conn = self.get_connection()
        placeholders = ",".join(["?"] * len(record_ids))
        query = f"SELECT record_id, name, address, country, dataset FROM records WHERE record_id IN ({placeholders})"
        df = pd.read_sql_query(query, conn, params=record_ids)
        conn.close()
        return df
