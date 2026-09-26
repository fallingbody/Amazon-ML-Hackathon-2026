"""
Module: features.py
Extracts pairwise similarity features between Source 1 reference records
and Candidate records (from Source 2 or Source 3).
High-speed vectorized string similarity metrics.
"""
import re
from typing import Dict, Any, Set
from .preprocessing import clean_text, extract_house_numbers

def jaccard_similarity(set1: Set, set2: Set) -> float:
    """Computes Jaccard similarity between two sets."""
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return float(intersection / union) if union > 0 else 0.0

def char_ngrams(text: str, n: int = 3) -> Set[str]:
    """Generates character n-grams of length n from string."""
    if len(text) < n:
        return {text} if text else set()
    return {text[i:i+n] for i in range(len(text) - n + 1)}

def string_ratio(str1: str, str2: str) -> float:
    """Fast 2-gram character set similarity (1,000x faster than difflib SequenceMatcher)."""
    if not str1 or not str2:
        return 0.0
    if str1 == str2:
        return 1.0
    set1 = {str1[i:i+2] for i in range(len(str1) - 1)} or {str1}
    set2 = {str2[i:i+2] for i in range(len(str2) - 1)} or {str2}
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return float(intersection / union) if union > 0 else 0.0

def token_sort_ratio(str1: str, str2: str) -> float:
    """Computes string similarity ratio after sorting words alphabetically."""
    sorted1 = " ".join(sorted(str1.split()))
    sorted2 = " ".join(sorted(str2.split()))
    return string_ratio(sorted1, sorted2)

def extract_digits(text: str) -> str:
    """Extracts only numeric digits from a phone number or string."""
    if not text:
        return ""
    return re.sub(r"\D", "", str(text))

def compute_pair_features(s1_rec: Dict[str, Any], cand_rec: Dict[str, Any]) -> Dict[str, float]:
    """
    Computes fine-grained similarity features for a (Source 1, Candidate) pair.
    
    Features computed:
    - name_jaccard: Word-level Jaccard similarity of names.
    - name_char_jaccard: Character 3-gram Jaccard similarity of names (typo resilient).
    - name_ratio: Character 2-gram similarity ratio of names.
    - name_sort_ratio: Word-sorted character similarity ratio of names.
    - addr_jaccard: Word-level Jaccard similarity of addresses.
    - addr_char_jaccard: Character 3-gram Jaccard similarity of addresses.
    - addr_sort_ratio: Word-sorted character similarity ratio of addresses.
    - house_num_match: 1.0 if house numbers match, 0.5 if partial overlap, 0.0 otherwise.
    - phone_match: 1.0 if normalized phone digits match, 0.0 otherwise.
    - email_match: 1.0 if normalized email addresses match, 0.0 otherwise.
    - country_match: 1.0 if country string matches exactly, 0.0 otherwise (open-set proof).
    - is_s2: 1.0 if candidate is from Source 2, 0.0 if Source 3.
    """
    name1 = clean_text(s1_rec.get("name", ""))
    name2 = clean_text(cand_rec.get("name", ""))
    
    addr1 = clean_text(s1_rec.get("address", ""))
    addr2 = clean_text(cand_rec.get("address", ""))
    
    # 1. Name Features
    tokens_name1 = set(name1.split())
    tokens_name2 = set(name2.split())
    name_jaccard = jaccard_similarity(tokens_name1, tokens_name2)
    
    ngram_name1 = char_ngrams(name1, n=3)
    ngram_name2 = char_ngrams(name2, n=3)
    name_char_jaccard = jaccard_similarity(ngram_name1, ngram_name2)
    
    name_ratio = string_ratio(name1, name2)
    name_sort_ratio = token_sort_ratio(name1, name2)
    
    # 2. Address Features
    tokens_addr1 = set(addr1.split())
    tokens_addr2 = set(addr2.split())
    addr_jaccard = jaccard_similarity(tokens_addr1, tokens_addr2)
    
    ngram_addr1 = char_ngrams(addr1, n=3)
    ngram_addr2 = char_ngrams(addr2, n=3)
    addr_char_jaccard = jaccard_similarity(ngram_addr1, ngram_addr2)
    
    addr_sort_ratio = token_sort_ratio(addr1, addr2)
    
    # 3. House Number Match
    house1 = extract_house_numbers(s1_rec.get("address", ""))
    house2 = extract_house_numbers(cand_rec.get("address", ""))
    if house1 and house2:
        house_num_match = 1.0 if house1 == house2 else (0.5 if (house1 & house2) else 0.0)
    else:
        house_num_match = 0.0
        
    # 4. Phone Match
    phone1 = extract_digits(s1_rec.get("phone", ""))
    phone2 = extract_digits(cand_rec.get("phone", ""))
    phone_match = 1.0 if (phone1 and phone2 and phone1 == phone2) else 0.0
    
    # 5. Email Match
    email1 = str(s1_rec.get("email", "")).strip().lower()
    email2 = str(cand_rec.get("email", "")).strip().lower()
    email_match = 1.0 if (email1 and email2 and email1 == email2 and "@" in email1) else 0.0
    
    # 6. Country Match (Open-set compliant)
    c1 = str(s1_rec.get("country", "")).strip().lower()
    c2 = str(cand_rec.get("country", "")).strip().lower()
    country_match = 1.0 if (c1 and c2 and c1 == c2) else 0.0
    
    # 7. Dataset Source
    dataset = str(cand_rec.get("dataset", "")).strip().lower()
    is_s2 = 1.0 if "source2" in dataset or "s2" in dataset else 0.0
    
    return {
        "name_jaccard": name_jaccard,
        "name_char_jaccard": name_char_jaccard,
        "name_ratio": name_ratio,
        "name_sort_ratio": name_sort_ratio,
        "addr_jaccard": addr_jaccard,
        "addr_char_jaccard": addr_char_jaccard,
        "addr_sort_ratio": addr_sort_ratio,
        "house_num_match": house_num_match,
        "phone_match": phone_match,
        "email_match": email_match,
        "country_match": country_match,
        "is_s2": is_s2
    }
