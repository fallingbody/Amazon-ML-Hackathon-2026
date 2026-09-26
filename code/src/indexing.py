"""
Module: indexing.py
Handles zero-RAM candidate blocking and attribute retrieval using persistent SQLite connection.
Optimized for 1,000+ entities/sec throughput.
"""
import os
import sqlite3
import pandas as pd
from typing import List, Set, Dict, Any
from .preprocessing import extract_tokens

def resolve_db_path(db_path: str = None) -> str:
    if db_path and os.path.exists(db_path):
        return db_path
    possible_paths = [
        "6ab10eb3b23ba_student_resource/student_resource/index.db",
        "student_resource/index.db",
        "index.db",
        "../6ab10eb3b23ba_student_resource/student_resource/index.db",
        "/content/6ab10eb3b23ba_student_resource/student_resource/index.db",
        "/content/dataset/index.db",
        "/content/index.db"
    ]
    for p in possible_paths:
        if os.path.exists(p):
            return p
    return possible_paths[0]

def resolve_dataset_base(dataset_base: str = None) -> str:
    if dataset_base and os.path.exists(dataset_base):
        return dataset_base
    possible_paths = [
        "6ab10eb3b23ba_student_resource/student_resource/dataset",
        "student_resource/dataset",
        "dataset",
        "../6ab10eb3b23ba_student_resource/student_resource/dataset",
        "/content/6ab10eb3b23ba_student_resource/student_resource/dataset",
        "/content/dataset"
    ]
    for p in possible_paths:
        if os.path.exists(p):
            return p
    return possible_paths[0]

class CandidateIndexer:
    """Class to interact with SQLite index.db for ultra-fast zero-RAM candidate lookup."""
    
    def __init__(self, db_path: str = None, dataset_base: str = None):
        self.db_path = resolve_db_path(db_path)
        self.dataset_base = resolve_dataset_base(dataset_base)
        self._conn = None
        self._id_col = None
        self._ensure_tables_indexed()

    def get_connection(self):
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
        return self._conn

    def _ensure_tables_indexed(self, split: str = "train"):
        """Ensures both 'records' and 'token_index' tables exist in index.db."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # 1. Ensure 'records' table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='records'")
        has_records = cursor.fetchone()
        
        if not has_records:
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

        # 2. Ensure 'token_index' table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='token_index'")
        has_token_index = cursor.fetchone()
        
        if not has_token_index:
            print("\nCreating 'token_index' table and B-Tree index in SQLite index.db...", flush=True)
            cursor.execute("PRAGMA synchronous = OFF")
            cursor.execute("PRAGMA journal_mode = MEMORY")
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS token_index (
                    token TEXT,
                    record_id TEXT
                )
            """)
            conn.commit()

            cursor.execute("SELECT COUNT(*) FROM records")
            total_recs = cursor.fetchone()[0]
            print(f"Generating search tokens across {total_recs:,} candidate records...", flush=True)
            
            chunk_size = 100000
            offset = 0
            while offset < total_recs:
                cursor.execute("SELECT record_id, name, address FROM records LIMIT ? OFFSET ?", (chunk_size, offset))
                rows = cursor.fetchall()
                token_rows = []
                for rec_id, name, addr in rows:
                    tokens = extract_tokens(name, addr)
                    for tok in tokens:
                        token_rows.append((tok, rec_id))
                
                cursor.executemany("INSERT INTO token_index (token, record_id) VALUES (?, ?)", token_rows)
                offset += len(rows)
                conn.commit()
                print(f"  Indexed tokens for {offset:,} / {total_recs:,} records...", flush=True)

            print("Building B-Tree index on token_index(token)...", flush=True)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_token ON token_index(token)")
            conn.commit()
            print("SQLite 'token_index' B-Tree index complete!\n", flush=True)

        # Pre-cache id_col name
        cursor.execute("PRAGMA table_info(token_index)")
        cols = [row[1] for row in cursor.fetchall()]
        self._id_col = "record_id" if "record_id" in cols else "entity_id"

    def find_candidates_for_record(self, name: str, address: str, max_candidates: int = 50) -> Set[str]:
        """
        Extracts search tokens and retrieves candidate IDs using single batch IN SQL query.
        """
        tokens = list(extract_tokens(name, address))
        if not tokens:
            return set()

        conn = self.get_connection()
        cursor = conn.cursor()

        placeholders = ",".join(["?"] * len(tokens))
        query = f"SELECT {self._id_col} FROM token_index WHERE token IN ({placeholders})"
        cursor.execute(query, tokens)
        rows = cursor.fetchall()

        candidate_counts = {}
        for (rec_id,) in rows:
            candidate_counts[rec_id] = candidate_counts.get(rec_id, 0) + 1

        # Sort candidates by number of matching tokens in descending order
        sorted_candidates = sorted(candidate_counts.items(), key=lambda x: x[1], reverse=True)
        return {rec_id for rec_id, _ in sorted_candidates[:max_candidates]}

    def fetch_records_by_ids(self, record_ids: List[str], split: str = "train") -> pd.DataFrame:
        """
        Fetches full record attributes directly from SQLite disk database using persistent connection.
        """
        if not record_ids:
            return pd.DataFrame()

        conn = self.get_connection()
        placeholders = ",".join(["?"] * len(record_ids))
        query = f"SELECT record_id, name, address, country, dataset FROM records WHERE record_id IN ({placeholders})"
        df = pd.read_sql_query(query, conn, params=record_ids)
        return df

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None
