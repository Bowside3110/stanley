# src/horse_matcher.py

from difflib import SequenceMatcher
import re
import numpy as np

def normalize_horse_name(name) -> str:
    """
    Normalize horse name for matching.
    - Lowercase
    - Remove country suffix like (AUS), (NZ), (IRE), (GB), etc.
    - Remove special characters
    - Remove extra whitespace
    
    Examples:
        "Heavenly Thought (AUS)" -> "heavenly thought"
        "Lucky Star" -> "lucky star"
        "Ka Ying Star (IRE)" -> "ka ying star"
    """
    # Handle None, NaN, or float values
    if name is None:
        return ""
    if isinstance(name, float) and np.isnan(name):
        return ""
        
    # Convert to string if it's not already
    if not isinstance(name, str):
        name = str(name).strip()
        
    # Return empty string if the result is empty or equals 'nan'
    if not name or name.lower() == 'nan':
        return ""
    
    # Remove country suffix pattern BEFORE lowercasing: (XXX) at the end
    # Common suffixes: (AUS), (NZ), (IRE), (GB), (FR), (USA), (SAF), (JPN)
    name = re.sub(r'\s*\([A-Z]{2,4}\)\s*$', '', name)
    
    # Convert to lowercase
    name = name.lower()
    
    # Remove special characters but keep spaces
    name = re.sub(r'[^\w\s]', '', name)
    
    # Remove extra whitespace
    name = ' '.join(name.split())
    
    return name.strip()


def normalize_jockey_name(name) -> str:
    """
    Normalize jockey name for matching.
    - Lowercase
    - Remove all punctuation (periods, commas)
    - Normalize whitespace
    
    Examples:
        "A S Cruz" -> "a s cruz"
        "A. S. Cruz" -> "a s cruz"
        "M L Yeung" -> "m l yeung"
    """
    # Handle None, NaN, or float values
    if name is None:
        return ""
    if isinstance(name, float) and np.isnan(name):
        return ""
        
    # Convert to string if it's not already
    if not isinstance(name, str):
        name = str(name).strip()
        
    # Return empty string if the result is empty or equals 'nan'
    if not name or name.lower() == 'nan':
        return ""
    
    # Convert to lowercase
    name = name.lower()
    
    # Remove all punctuation
    name = re.sub(r'[^\w\s]', '', name)
    
    # Normalize whitespace
    name = ' '.join(name.split())
    
    return name.strip()


def normalize_trainer_name(name) -> str:
    """
    Normalize trainer name for matching.
    - Lowercase
    - Remove all punctuation (periods, commas, ampersands)
    - Normalize whitespace
    - Handle partnership names
    
    Examples:
        "A S Cruz" -> "a s cruz"
        "Ben, Will & JD Hayes" -> "ben will jd hayes"
        "M Price & M Kent (Jnr)" -> "m price m kent jnr"
        "P O'Sullivan" -> "p osullivan"
    """
    # Handle None, NaN, or float values
    if name is None:
        return ""
    if isinstance(name, float) and np.isnan(name):
        return ""
        
    # Convert to string if it's not already
    if not isinstance(name, str):
        name = str(name).strip()
        
    # Return empty string if the result is empty or equals 'nan'
    if not name or name.lower() == 'nan':
        return ""
    
    # Convert to lowercase
    name = name.lower()
    
    # Remove all punctuation (including ampersands, commas, periods, parentheses)
    name = re.sub(r'[^\w\s]', '', name)
    
    # Normalize whitespace
    name = ' '.join(name.split())
    
    return name.strip()


def fuzzy_match_horse(target_name: str, name_to_ids_index: dict, threshold: float = 0.85) -> tuple:
    """
    Find the best matching horse from candidates using fuzzy matching.
    
    Args:
        target_name: The horse name to match (e.g., "Heavenly Thought" or "Heavenly Thought (AUS)")
        name_to_ids_index: Dict of {normalized_name: [list of horse_ids]} to search through
        threshold: Minimum similarity score (0-1) to consider a match
        
    Returns:
        Tuple of (normalized_name, confidence_score) or (None, 0.0)
    """
    target_norm = normalize_horse_name(target_name)
    
    if not target_norm:
        return None, 0.0
    
    # Try exact match first (fast path)
    if target_norm in name_to_ids_index:
        return target_norm, 1.0
    
    # Fuzzy match using SequenceMatcher
    best_match_name = None
    best_score = 0.0
    
    for candidate_norm in name_to_ids_index.keys():
        score = SequenceMatcher(None, target_norm, candidate_norm).ratio()
        
        if score > best_score and score >= threshold:
            best_score = score
            best_match_name = candidate_norm
    
    return best_match_name, best_score


def build_horse_name_index(conn) -> dict:
    """
    Build an index mapping normalized horse names to lists of horse IDs.
    Returns dict of {normalized_name: [horse_id1, horse_id2, ...]}
    
    This handles cases where the same horse appears with different name variations
    (e.g., "Heavenly Thought" vs "Heavenly Thought (AUS)").
    """
    import pandas as pd
    
    # Query all unique horses from runners table
    query = """
    SELECT DISTINCT horse_id, horse
    FROM runners
    WHERE horse_id IS NOT NULL 
      AND horse IS NOT NULL
      AND horse != ''
    """
    
    df = pd.read_sql_query(query, conn)
    
    # Build mapping from normalized name to list of horse IDs
    name_to_ids = {}
    for _, row in df.iterrows():
        normalized = normalize_horse_name(row['horse'])
        if normalized:
            if normalized not in name_to_ids:
                name_to_ids[normalized] = []
            name_to_ids[normalized].append(row['horse_id'])
    
    total_ids = sum(len(ids) for ids in name_to_ids.values())
    print(f"Built horse name index with {len(name_to_ids)} unique normalized names mapping to {total_ids} horse IDs")
    
    return name_to_ids


def build_jockey_name_index(conn) -> dict:
    """
    Build an index mapping normalized jockey names to lists of jockey IDs.
    Returns dict of {normalized_name: [jockey_id1, jockey_id2, ...]}
    
    This handles cases where the same jockey appears with different IDs.
    Note: IDs are unreliable (same ID can map to different people), so we use names as primary key.
    """
    import pandas as pd
    
    # Query all unique jockeys from runners table
    query = """
    SELECT DISTINCT jockey, jockey_id
    FROM runners
    WHERE jockey IS NOT NULL 
      AND jockey != ''
    """
    
    df = pd.read_sql_query(query, conn)
    
    # Build mapping from normalized name to list of jockey IDs
    name_to_ids = {}
    for _, row in df.iterrows():
        normalized = normalize_jockey_name(row['jockey'])
        if normalized:
            if normalized not in name_to_ids:
                name_to_ids[normalized] = []
            # Only add non-null IDs
            if row['jockey_id'] is not None:
                name_to_ids[normalized].append(row['jockey_id'])
    
    total_ids = sum(len(ids) for ids in name_to_ids.values())
    print(f"Built jockey name index with {len(name_to_ids)} unique normalized names mapping to {total_ids} jockey IDs")
    
    return name_to_ids


def build_trainer_name_index(conn) -> dict:
    """
    Build an index mapping normalized trainer names to lists of trainer IDs.
    Returns dict of {normalized_name: [trainer_id1, trainer_id2, ...]}
    
    This handles cases where the same trainer appears with different IDs.
    Note: IDs are unreliable (same ID can map to different people), so we use names as primary key.
    """
    import pandas as pd
    
    # Query all unique trainers from runners table
    query = """
    SELECT DISTINCT trainer, trainer_id
    FROM runners
    WHERE trainer IS NOT NULL 
      AND trainer != ''
    """
    
    df = pd.read_sql_query(query, conn)
    
    # Build mapping from normalized name to list of trainer IDs
    name_to_ids = {}
    for _, row in df.iterrows():
        normalized = normalize_trainer_name(row['trainer'])
        if normalized:
            if normalized not in name_to_ids:
                name_to_ids[normalized] = []
            # Only add non-null IDs
            if row['trainer_id'] is not None:
                name_to_ids[normalized].append(row['trainer_id'])
    
    total_ids = sum(len(ids) for ids in name_to_ids.values())
    print(f"Built trainer name index with {len(name_to_ids)} unique normalized names mapping to {total_ids} trainer IDs")
    
    return name_to_ids


def test_normalization():
    """Test cases for horse name normalization"""
    test_cases = [
        ("Heavenly Thought (AUS)", "heavenly thought"),
        ("Heavenly Thought", "heavenly thought"),
        ("Ka Ying Star (IRE)", "ka ying star"),
        ("Lucky Star", "lucky star"),
        ("Golden Sixty (AUS)", "golden sixty"),
        ("California Spangle (NZ)", "california spangle"),
        ("Beauty Joy", "beauty joy"),
        ("Fantastic Treasure (GB)", "fantastic treasure"),
    ]
    
    print("Testing horse name normalization:\n")
    all_passed = True
    
    for input_name, expected in test_cases:
        result = normalize_horse_name(input_name)
        passed = result == expected
        status = "✓" if passed else "✗"
        print(f"{status} '{input_name}' -> '{result}' (expected: '{expected}')")
        if not passed:
            all_passed = False
    
    print(f"\n{'All tests passed!' if all_passed else 'Some tests failed!'}")
    return all_passed


def test_fuzzy_matching():
    """Test fuzzy matching logic"""
    # Simulate a small database
    candidate_horses = {
        "H001": "Heavenly Thought (AUS)",
        "H002": "Lucky Star",
        "H003": "Ka Ying Star (IRE)",
        "H004": "Golden Sixty (AUS)",
        "H005": "Beauty Joy",
    }
    
    test_cases = [
        ("Heavenly Thought", "H001", 1.0),  # Exact match (ignoring suffix)
        ("Heavenly Thought (AUS)", "H001", 1.0),  # Exact with suffix
        ("Lucky Star", "H002", 1.0),
        ("Ka Ying Star", "H003", 1.0),
        ("Golden Sixty", "H004", 1.0),
        ("Heavenly", None, 0.0),  # Too short, should not match
        ("Beauty", None, 0.0),  # Partial name, should not match at default threshold
    ]
    
    print("\nTesting fuzzy matching:\n")
    all_passed = True
    
    for input_name, expected_id, expected_min_conf in test_cases:
        matched_id, confidence = fuzzy_match_horse(input_name, candidate_horses, threshold=0.85)
        
        if expected_id is None:
            passed = matched_id is None
            status = "✓" if passed else "✗"
            print(f"{status} '{input_name}' -> No match (confidence: {confidence:.2f})")
        else:
            passed = matched_id == expected_id and confidence >= expected_min_conf
            status = "✓" if passed else "✗"
            expected_name = candidate_horses.get(expected_id, "?")
            print(f"{status} '{input_name}' -> '{expected_name}' (confidence: {confidence:.2f})")
        
        if not passed:
            all_passed = False
    
    print(f"\n{'All matching tests passed!' if all_passed else 'Some matching tests failed!'}")
    return all_passed


if __name__ == "__main__":
    print("=" * 60)
    print("Horse Matcher Test Suite")
    print("=" * 60)
    
    norm_passed = test_normalization()
    match_passed = test_fuzzy_matching()
    
    if norm_passed and match_passed:
        print("\n✅ All tests passed! The matcher is ready to use.")
    else:
        print("\n⚠️ Some tests failed. Please review the implementation.")