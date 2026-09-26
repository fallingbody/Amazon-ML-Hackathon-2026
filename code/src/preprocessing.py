"""
Module: preprocessing.py
Handles text cleaning, unicode normalization, legal suffix removal, consecutive duplicate word removal,
and token extraction for Business Entity Resolution across US, India, France, and international sources.
"""
import re
import unicodedata
import pandas as pd
from typing import List, Set

# Expanded regex covering US, Indian, German, French, and international legal suffixes
LEGAL_SUFFIXES_REGEX = re.compile(
    r"\b(corp|corporation|inc|incorporated|ltd|limited|pvt|private|co|company|llc|plc|gmbh|sa|sasu|sarl|eurl|snc|gie|bv|srl|pc|llp|lp)\b",
    flags=re.IGNORECASE
)

# Regex to strip website URLs and domain paths inserted into business names
URL_REGEX = re.compile(r"\b(https?://|www\.)?\S+\.(com|org|net|in|fr|co|io|edu|gov|biz)\b", flags=re.IGNORECASE)

STOP_WORDS = {
    "street", "road", "avenue", "drive", "lane", "boulevard", "court", "place",
    "suite", "floor", "building", "unit", "near", "opposite", "behind", "block",
    "north", "south", "east", "west", "central", "main", "state", "city", "town",
    "center", "centre", "group", "services", "solutions", "international", "global",
    "traders", "trading", "enterprise", "enterprises", "india", "pvt", "ltd", "france", "us"
}

def remove_accents(text: str) -> str:
    """Normalizes unicode characters and strips diacritics (e.g., é -> e, ä -> a)."""
    if not isinstance(text, str):
        return ""
    nfkd_form = unicodedata.normalize("NFKD", text)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])

def remove_consecutive_duplicates(text: str) -> str:
    """Removes synthetic consecutive repeated words (e.g., 'LLC LLC' -> 'LLC', 'Odyssey Odyssey' -> 'Odyssey')."""
    words = text.split()
    if not words:
        return ""
    deduped = [words[0]]
    for w in words[1:]:
        if w.lower() != deduped[-1].lower():
            deduped.append(w)
    return " ".join(deduped)

def clean_text(text: str) -> str:
    """
    Cleans string by removing accents, URLs, legal suffixes, special noise prefixes,
    consecutive duplicate words, and standardizing whitespace.
    """
    if not isinstance(text, str) or pd.isna(text):
        return ""
    
    # 1. Unicode accent normalization
    text = remove_accents(text).lower()
    
    # 2. Strip URLs and domains
    text = URL_REGEX.sub("", text)
    
    # 3. Strip legal suffixes
    text = LEGAL_SUFFIXES_REGEX.sub("", text)
    
    # 4. Remove non-alphanumeric noise characters
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    
    # 5. Deduplicate consecutive identical words
    text = remove_consecutive_duplicates(text)
    
    return text

def clean_text_series(series: pd.Series) -> pd.Series:
    """Vectorized C-speed text cleaning for Pandas Series."""
    return series.fillna("").astype(str).apply(clean_text)

def extract_tokens(name: str, address: str) -> Set[str]:
    """Extracts informative search tokens (>= 3 characters, excluding stop words) from name and address."""
    full_text = f"{clean_text(name)} {clean_text(address)}"
    tokens = [t for t in full_text.split() if len(t) >= 3 and t not in STOP_WORDS]
    return set(tokens)

RE_DIGITS = re.compile(r"\b\d+\b")

def extract_house_numbers(text: str) -> Set[str]:
    """Extracts numeric building/house numbers from an address string."""
    if not isinstance(text, str) or pd.isna(text):
        return set()
    return set(RE_DIGITS.findall(text))

