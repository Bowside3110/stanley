# src/horse_matcher.py

from difflib import SequenceMatcher
import re

def normalize_horse_name(name: str) -> str:
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
    if not name:
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


def fuzzy_match_horse(target_name: str, candidate_names: dict, threshold: float = 0.85) -> tuple:
    """
    Find the best matching horse from candidates using fuzzy matching.
    
    Args:
        target_name: The horse name to match (e.g., "Heavenly Thought" or "Heavenly Thought (AUS)")
        candidate_names: Dict of {horse_id: horse_name} to search through
        threshold: Minimum similarity score (0-1) to consider a match
        
    Returns:
        Tuple of (matched_horse_id, confidence_score) or (None, 0.0)
    """
    target_norm = normalize_horse_name(target_name)
    
    if not target_norm:
        return None, 0.0
    
    best_match_id = None
    best_score = 0.0
    
    for horse_id, horse_name in candidate_names.items():
        candidate_norm = normalize_horse_name(horse_name)
        
        if not candidate_norm:
            continue
            
        # Try exact match first (fast path)
        if target_norm == candidate_norm:
            return horse_id, 1.0
        
        # Fuzzy match using SequenceMatcher
        score = SequenceMatcher(None, target_norm, candidate_norm).ratio()
        
        if score > best_score and score >= threshold:
            best_score = score
            best_match_id = horse_id
    
    return best_match_id, best_score


def build_horse_name_index(conn) -> dict:
    """
    Build an index of all historical horses from the database.
    Returns dict of {horse_id: horse_name}
    """
    import pandas as pd
    
    # Query all unique horses from historical results
    # We want horses that have actually raced (have results)
    query = """
    SELECT DISTINCT r.horse_id, r.horse
    FROM runners r
    WHERE r.horse_id IS NOT NULL 
      AND r.horse IS NOT NULL
      AND r.horse != ''
    """
    
    df = pd.read_sql_query(query, conn)
    
    # Create dictionary
    horse_index = dict(zip(df['horse_id'], df['horse']))
    
    print(f"Built horse name index with {len(horse_index)} unique horses")
    
    return horse_index


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