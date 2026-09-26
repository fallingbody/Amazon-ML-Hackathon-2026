"""
Module: indexing.py
Handles candidate blocking and retrieval using the SQLite index.db database
and candidate attribute lookup from TSV source files.
"""
import os
import sqlite3
import pandas as pd
from typing import List, Set, Dict, Any
from .preprocessing import extract_tokens

DEFAULT_DB_PATH = "6ab10eb3b23ba_student_resource/student_resource/index.db"
DEFAULT_DATASET_BASE = "6ab10eb3b23ba_student_resource/student_resource/dataset"

class CandidateIndexer:
    """Class to interact with SQLite index.db for fast candidate lookup."""
    
    def __init__(self, db_path: str = DEFAULT_DB_PATH, dataset_base: str = DEFAULT_DATASET_BASE):
        self.db_path = db_path
        self.dataset_base = dataset_base
        self._candidates_df = None

    def get_connection(self):
        return sqlite3.connect(self.db_path)

    def load_candidate_records(self, split: str = "train"):
        """Loads Source 2 and Source 3 TSVs into an indexed PyArrow DataFrame."""
        if self._candidates_df is not None:
            return self._candidates_df

        split_dir = os.path.join(self.dataset_base, split)
        frames = []
        for src_num in [2, 3]:
            tsv_path = os.path.join(split_dir, f"{split}_source{src_num}.tsv")
            if os.path.exists(tsv_path):
                df = pd.read_csv(tsv_path, sep="\t", engine="pyarrow")
                # Standardize column names
                rename_map = {}
                if "entity_id" in df.columns: rename_map["entity_id"] = "record_id"
                if "business_name" in df.columns: rename_map["business_name"] = "name"
                if "business_address" in df.columns: rename_map["business_address"] = "address"
                if rename_map:
                    df = df.rename(columns=rename_map)
                df["dataset"] = f"source{src_num}"
                frames.append(df)

        if frames:
            full_df = pd.concat(frames, ignore_index=True)
            full_df = full_df.drop_duplicates(subset=["record_id"])
            self._candidates_df = full_df.set_index("record_id", drop=False)
        else:
            self._candidates_df = pd.DataFrame(columns=["record_id", "name", "address", "country", "dataset"]).set_index("record_id", drop=False)
            
        return self._candidates_df

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

        # Dynamically inspect column name in token_index
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
        Fetches full record attributes for given candidate record IDs.
        """
        if not record_ids:
            return pd.DataFrame()

        cand_df = self.load_candidate_records(split=split)
        valid_ids = [rid for rid in record_ids if rid in cand_df.index]
        if not valid_ids:
            return pd.DataFrame()

        return cand_df.loc[valid_ids].copy()
