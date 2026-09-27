"""
Module: features.py
Extracts pairwise similarity features between Source 1 reference records
and Candidate records (from Source 2 or Source 3).
High-speed vectorized string similarity metrics.
"""
import re
from typing import Dict, Any, Set
from .preprocessing import (
    clean_text, extract_house_numbers, extract_postal_code,
    extract_domain_category, check_acronym_match, jaro_winkler
)

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

def precompute_s1_features(s1_rec: Dict[str, Any]) -> Dict[str, Any]:
    """Precomputes string tokens and n-grams for S1 record to avoid recalculating it per candidate."""
    name1 = clean_text(s1_rec.get("name", ""))
    addr1 = clean_text(s1_rec.get("address", ""))
    words_name1 = name1.split()
    words_addr1 = addr1.split()
    
    return {
        "name1": name1,
        "addr1": addr1,
        "tokens_name1": set(words_name1),
        "first_word1": words_name1[0] if words_name1 else "",
        "last_word1": words_name1[-1] if words_name1 else "",
        "ngram_name1": char_ngrams(name1, n=3),
        "tokens_addr1": set(words_addr1),
        "ngram_addr1": char_ngrams(addr1, n=3),
        "house1": extract_house_numbers(s1_rec.get("address", "")),
        "c1": str(s1_rec.get("country", "")).strip().lower(),
        "postal1": extract_postal_code(s1_rec.get("address", ""), str(s1_rec.get("country", ""))),
        "domain1": extract_domain_category(name1, addr1)
    }

def compute_pair_features(s1_rec: Dict[str, Any], cand_rec: Dict[str, Any], s1_precomputed: Dict[str, Any] = None) -> Dict[str, float]:
    """
    Computes fine-grained similarity and conflict features for a (Source 1, Candidate) pair.
    Engineered for maximum Precision and Macro F0.5 optimization.
    """
    if s1_precomputed:
        name1 = s1_precomputed["name1"]
        addr1 = s1_precomputed["addr1"]
        tokens_name1 = s1_precomputed["tokens_name1"]
        first_word1 = s1_precomputed.get("first_word1", "")
        last_word1 = s1_precomputed.get("last_word1", "")
        ngram_name1 = s1_precomputed["ngram_name1"]
        tokens_addr1 = s1_precomputed["tokens_addr1"]
        ngram_addr1 = s1_precomputed["ngram_addr1"]
        house1 = s1_precomputed["house1"]
        c1 = s1_precomputed["c1"]
    else:
        name1 = clean_text(s1_rec.get("name", ""))
        addr1 = clean_text(s1_rec.get("address", ""))
        words1 = name1.split()
        tokens_name1 = set(words1)
        first_word1 = words1[0] if words1 else ""
        last_word1 = words1[-1] if words1 else ""
        ngram_name1 = char_ngrams(name1, n=3)
        tokens_addr1 = set(addr1.split())
        ngram_addr1 = char_ngrams(addr1, n=3)
        house1 = extract_house_numbers(s1_rec.get("address", ""))
        c1 = str(s1_rec.get("country", "")).strip().lower()

    name2 = clean_text(cand_rec.get("name", ""))
    addr2 = clean_text(cand_rec.get("address", ""))
    words2 = name2.split()
    first_word2 = words2[0] if words2 else ""
    last_word2 = words2[-1] if words2 else ""
    tokens_name2 = set(words2)
    tokens_addr2 = set(addr2.split())
    
    # 1. Name Features
    name_jaccard = jaccard_similarity(tokens_name1, tokens_name2)
    ngram_name2 = char_ngrams(name2, n=3)
    name_char_jaccard = jaccard_similarity(ngram_name1, ngram_name2)
    name_ratio = string_ratio(name1, name2)
    name_sort_ratio = token_sort_ratio(name1, name2)
    name_exact = 1.0 if (name1 and name1 == name2) else 0.0
    
    # Asymmetric word containment and length ratios
    len_name1 = len(tokens_name1)
    len_name2 = len(tokens_name2)
    overlap_name = len(tokens_name1 & tokens_name2)
    min_name_len = min(len_name1, len_name2)
    max_name_len = max(len_name1, len_name2)
    name_containment = overlap_name / min_name_len if min_name_len > 0 else 0.0
    name_s1_in_cand = overlap_name / len_name1 if len_name1 > 0 else 0.0
    name_cand_in_s1 = overlap_name / len_name2 if len_name2 > 0 else 0.0
    name_len_diff = float(abs(len_name1 - len_name2))
    name_len_ratio = float(min_name_len / max_name_len) if max_name_len > 0 else 0.0

    first_word_match = 1.0 if (first_word1 and first_word2 and first_word1 == first_word2) else 0.0
    last_word_match = 1.0 if (last_word1 and last_word2 and last_word1 == last_word2) else 0.0
    
    # 2. Address Features
    addr_jaccard = jaccard_similarity(tokens_addr1, tokens_addr2)
    ngram_addr2 = char_ngrams(addr2, n=3)
    addr_char_jaccard = jaccard_similarity(ngram_addr1, ngram_addr2)
    addr_sort_ratio = token_sort_ratio(addr1, addr2)
    addr_exact = 1.0 if (addr1 and addr1 == addr2) else 0.0
    
    len_addr1 = len(tokens_addr1)
    len_addr2 = len(tokens_addr2)
    overlap_addr = len(tokens_addr1 & tokens_addr2)
    min_addr_len = min(len_addr1, len_addr2)
    addr_containment = overlap_addr / min_addr_len if min_addr_len > 0 else 0.0
    addr_s1_in_cand = overlap_addr / len_addr1 if len_addr1 > 0 else 0.0
    addr_cand_in_s1 = overlap_addr / len_addr2 if len_addr2 > 0 else 0.0

    # Joint Interaction
    both_match_score = name_jaccard * addr_jaccard
    both_exact = 1.0 if (name_exact == 1.0 and addr_exact == 1.0) else 0.0
    
    # 3. House Number Signals (Match & Conflict False-Positive Killer)
    house2 = extract_house_numbers(cand_rec.get("address", ""))
    if house1 and house2:
        house_num_match = 1.0 if house1 == house2 else (0.5 if (house1 & house2) else 0.0)
        house_num_conflict = 1.0 if not (house1 & house2) else 0.0
    else:
        house_num_match = 0.0
        house_num_conflict = 0.0
        
    # 4. Country Match (Open-set compliant)
    c2 = str(cand_rec.get("country", "")).strip().lower()
    country_match = 1.0 if (c1 and c2 and c1 == c2) else 0.0
    
    # 5. Dataset Source
    dataset = str(cand_rec.get("dataset", "")).strip().lower()
    is_s2 = 1.0 if "source2" in dataset or "s2" in dataset else 0.0

    # 6. Sprint 1 High-Signal Features (Precision & Recall Boosters)
    name_jaro = jaro_winkler(name1, name2)
    addr_jaro = jaro_winkler(addr1, addr2)
    acronym_match = check_acronym_match(name1, name2)

    # Postal / PIN Code Match & Conflict
    postal1 = s1_precomputed.get("postal1", "") if s1_precomputed else extract_postal_code(s1_rec.get("address", ""), c1)
    postal2 = extract_postal_code(cand_rec.get("address", ""), c2)
    if postal1 and postal2:
        postal_match = 1.0 if postal1 == postal2 else 0.0
        postal_conflict = 1.0 if postal1 != postal2 and (not c1 or not c2 or c1 == c2) else 0.0
    else:
        postal_match = 0.0
        postal_conflict = 0.0

    # Business Domain Category Match & Conflict (e.g. Hospital vs Hotel)
    domain1 = s1_precomputed.get("domain1", "") if s1_precomputed else extract_domain_category(name1, addr1)
    domain2 = extract_domain_category(name2, addr2)
    if domain1 and domain2:
        domain_match = 1.0 if domain1 == domain2 else 0.0
        domain_conflict = 1.0 if domain1 != domain2 else 0.0
    else:
        domain_match = 0.0
        domain_conflict = 0.0
    
    return {
        "name_jaccard": name_jaccard,
        "name_char_jaccard": name_char_jaccard,
        "name_ratio": name_ratio,
        "name_sort_ratio": name_sort_ratio,
        "name_exact": name_exact,
        "name_containment": name_containment,
        "name_s1_in_cand": name_s1_in_cand,
        "name_cand_in_s1": name_cand_in_s1,
        "name_len_diff": name_len_diff,
        "name_len_ratio": name_len_ratio,
        "first_word_match": first_word_match,
        "last_word_match": last_word_match,
        "name_jaro": name_jaro,
        "acronym_match": acronym_match,
        "addr_jaccard": addr_jaccard,
        "addr_char_jaccard": addr_char_jaccard,
        "addr_sort_ratio": addr_sort_ratio,
        "addr_containment": addr_containment,
        "addr_s1_in_cand": addr_s1_in_cand,
        "addr_cand_in_s1": addr_cand_in_s1,
        "addr_exact": addr_exact,
        "addr_jaro": addr_jaro,
        "both_match_score": both_match_score,
        "both_exact": both_exact,
        "house_num_match": house_num_match,
        "house_num_conflict": house_num_conflict,
        "postal_match": postal_match,
        "postal_conflict": postal_conflict,
        "domain_match": domain_match,
        "domain_conflict": domain_conflict,
        "country_match": country_match,
        "is_s2": is_s2
    }

