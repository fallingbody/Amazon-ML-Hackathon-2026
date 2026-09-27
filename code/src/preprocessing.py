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
RE_US_ZIP = re.compile(r"\b(\d{5})(?:-\d{4})?\b")
RE_IN_PIN = re.compile(r"\b([1-9]\d{5})\b")
RE_WORD = re.compile(r"\b[a-z]{3,}\b")

DOMAIN_KEYWORDS = {
    "health": {"hospital", "clinic", "dental", "pharmacy", "chemist", "medical", "diagnostics", "healthcare", "optical", "doctor", "physicians", "pediatric", "cardiology"},
    "hospitality": {"hotel", "resort", "inn", "suites", "lodge", "motel", "hostel", "residency", "banquet", "hospitality"},
    "education": {"school", "college", "academy", "institute", "university", "vidyalaya", "convent", "classes", "tuition"},
    "automotive": {"auto", "motors", "garage", "tyre", "tires", "automobile", "hero", "honda", "maruti", "bajaj", "workshop", "wheels"},
    "food": {"restaurant", "cafe", "bakery", "sweets", "dhaba", "pizza", "kitchen", "snacks", "caterers", "dining", "foods"},
    "finance": {"bank", "finance", "investments", "insurance", "loan", "wealth", "securities"},
    "legal": {"legal", "advocate", "attorney", "law"}
}

def extract_house_numbers(text: str) -> Set[str]:
    """Extracts numeric building/house numbers from an address string."""
    if not isinstance(text, str) or pd.isna(text):
        return set()
    return set(RE_DIGITS.findall(text))

def extract_postal_code(text: str, country: str = "") -> str:
    """Extracts normalized 5-digit US zip or 6-digit Indian PIN code."""
    if not text or not isinstance(text, str):
        return ""
    c = country.strip().lower()
    if c == "in":
        m = RE_IN_PIN.findall(text)
        return m[-1] if m else ""
    elif c == "us":
        m = RE_US_ZIP.findall(text)
        return m[-1] if m else ""
    else:
        m6 = RE_IN_PIN.findall(text)
        if m6:
            return m6[-1]
        m5 = RE_US_ZIP.findall(text)
        if m5:
            return m5[-1]
    return ""

def extract_domain_category(name: str, address: str = "") -> str:
    """Classifies entity into a high-level domain (health, hospitality, education, etc.) to detect domain clashes."""
    text = f"{name} {address}".lower()
    words = set(RE_WORD.findall(text))
    for cat, kws in DOMAIN_KEYWORDS.items():
        if words & kws:
            return cat
    return ""

def check_acronym_match(name1: str, name2: str) -> float:
    """Detects whether one name is an acronym or abbreviation of the other."""
    if not name1 or not name2:
        return 0.0
    w1 = [w for w in name1.split() if w]
    w2 = [w for w in name2.split() if w]
    if len(name1) <= 6 and name1.isalpha() and len(w2) >= 2:
        initials2 = "".join(w[0] for w in w2).lower()
        if name1.lower() == initials2 or initials2.startswith(name1.lower()):
            return 1.0
    if len(name2) <= 6 and name2.isalpha() and len(w1) >= 2:
        initials1 = "".join(w[0] for w in w1).lower()
        if name2.lower() == initials1 or initials1.startswith(name2.lower()):
            return 1.0
    return 0.0

def jaro_winkler(s1: str, s2: str) -> float:
    """Vector-friendly Jaro-Winkler string similarity with prefix scaling."""
    if not s1 or not s2:
        return 0.0
    if s1 == s2:
        return 1.0
    len1, len2 = len(s1), len(s2)
    max_dist = max(len1, len2) // 2 - 1
    if max_dist < 0:
        max_dist = 0
    match1 = [False] * len1
    match2 = [False] * len2
    matches = 0
    for i in range(len1):
        start = max(0, i - max_dist)
        end = min(i + max_dist + 1, len2)
        for j in range(start, end):
            if match2[j] or s1[i] != s2[j]:
                continue
            match1[i] = match2[j] = True
            matches += 1
            break
    if matches == 0:
        return 0.0
    k = transpositions = 0
    for i in range(len1):
        if not match1[i]:
            continue
        while not match2[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1
    t = transpositions / 2.0
    sim = (matches / len1 + matches / len2 + (matches - t) / matches) / 3.0
    prefix = 0
    for i in range(min(4, min(len1, len2))):
        if s1[i] == s2[i]:
            prefix += 1
        else:
            break
    return min(1.0, sim + 0.1 * prefix * (1.0 - sim))


