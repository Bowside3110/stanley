"""
HKJC Results Scraper - HTML Backup Method

Scrapes race results from HKJC's ResultsAll page as a backup to the API-based fetcher.
Uses HTML parsing to extract finishing positions and odds for all races on a given date.

Adapted for Stanley's patterns:
- Print statements with emoji for logging
- Raw sqlite3 database operations
- Follows existing scraper conventions
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
from typing import Optional, Tuple, List
import time
import re
from enum import Enum

HKJC_RESULTS_URL = (
    "https://racing.hkjc.com/racing/information/English/Racing/ResultsAll.aspx?RaceDate={date}"
)


class ResultStatus(Enum):
    SUCCESS = "success"
    NO_MEETING = "no_meeting"
    MEETING_ABANDONED = "meeting_abandoned"
    PARSE_ERROR = "parse_error"
    NETWORK_ERROR = "network_error"
    HTML_STRUCTURE_CHANGED = "html_structure_changed"


class HKJCResultsScraper:
    """
    HTML-based scraper for HKJC race results with:
    - Race and venue identification
    - Robust error handling
    - Retry logic with exponential backoff
    """
    
    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })
    
    def _extract_race_number(self, text: str) -> Optional[int]:
        """Extract race number from text like 'Race 1', 'RACE 5', etc."""
        match = re.search(r'Race\s+(\d+)', text, re.IGNORECASE)
        if match:
            return int(match.group(1))
        return None
    
    def _extract_venue(self, text: str) -> Optional[str]:
        """Extract venue from text and normalize to Stanley's format."""
        text_upper = text.upper()
        if 'HAPPY VALLEY' in text_upper or 'HV' in text_upper:
            return 'Happy Valley'
        elif 'SHA TIN' in text_upper or 'ST' in text_upper:
            return 'Sha Tin'
        return None
    
    def _check_meeting_status(self, soup: BeautifulSoup) -> Tuple[bool, str]:
        """
        Check if meeting occurred or was abandoned.
        Returns: (meeting_occurred, message)
        """
        page_text = soup.get_text()
        
        # Check for common "no results" indicators
        no_meeting_indicators = [
            "No race meeting",
            "No races scheduled",
            "No results available",
            "There is no race"
        ]
        
        abandoned_indicators = [
            "Meeting abandoned",
            "Race meeting abandoned",
            "Abandoned"
        ]
        
        for indicator in abandoned_indicators:
            if indicator.lower() in page_text.lower():
                return False, "Meeting abandoned"
        
        for indicator in no_meeting_indicators:
            if indicator.lower() in page_text.lower():
                return False, "No meeting scheduled"
        
        return True, "Meeting occurred"
    
    def _is_valid_result_table(self, header_cells: List[str]) -> bool:
        """
        Validate that a table is actually a race result table.
        Requires multiple key markers to be present.
        """
        # Check for key columns in runner results table
        has_position = any('Pla' in cell for cell in header_cells)
        has_horse = any('Horse' in cell for cell in header_cells)
        has_jockey = any('Jockey' in cell for cell in header_cells)
        
        # Must have position, horse, and jockey to be a valid result table
        return has_position and has_horse and has_jockey
    
    def _fetch_with_retry(self, url: str) -> requests.Response:
        """Fetch URL with exponential backoff retry logic."""
        for attempt in range(self.max_retries):
            try:
                resp = self.session.get(url, timeout=15)
                resp.raise_for_status()
                return resp
            except requests.RequestException as e:
                wait_time = 2 ** attempt
                if attempt < self.max_retries - 1:
                    print(f"   ⚠️  Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"   ❌ All {self.max_retries} attempts failed")
                    raise
    
    def get_results_for_date(self, date_str: str) -> Tuple[pd.DataFrame, ResultStatus, str]:
        """
        Fetch and parse HKJC race results for a given date.
        
        Args:
            date_str: 'YYYY-MM-DD' or 'YYYY/MM/DD'
        
        Returns:
            Tuple of (DataFrame, ResultStatus, message)
        """
        # Normalize date to HKJC format YYYY/MM/DD
        date_str_clean = date_str.replace("-", "/")
        
        try:
            parsed_date = datetime.strptime(date_str_clean, "%Y/%m/%d")
        except ValueError:
            return pd.DataFrame(), ResultStatus.PARSE_ERROR, "Invalid date format"
        
        url = HKJC_RESULTS_URL.format(date=date_str_clean)
        print(f"📥 Fetching results from {url}")
        
        try:
            resp = self._fetch_with_retry(url)
        except requests.RequestException as e:
            return pd.DataFrame(), ResultStatus.NETWORK_ERROR, f"Network error: {str(e)}"
        
        soup = BeautifulSoup(resp.text, "html.parser")
        
        # Check meeting status
        meeting_occurred, status_msg = self._check_meeting_status(soup)
        if not meeting_occurred:
            status = ResultStatus.MEETING_ABANDONED if "abandoned" in status_msg.lower() else ResultStatus.NO_MEETING
            print(f"   ℹ️  {status_msg}")
            return pd.DataFrame(), status, status_msg
        
        # Parse results
        all_rows = []
        current_race = None
        current_venue = None
        tables_found = 0
        
        # Iterate through all elements to capture context before tables
        for element in soup.find_all(['h2', 'h3', 'h4', 'div', 'table', 'span']):
            text = element.get_text(strip=True)
            
            # Try to extract race number
            race_num = self._extract_race_number(text)
            if race_num:
                current_race = race_num
            
            # Try to extract venue
            venue = self._extract_venue(text)
            if venue:
                current_venue = venue
            
            # Process tables
            if element.name == 'table':
                # HKJC uses 'td' for headers in result tables, not 'th'
                rows = element.find_all("tr")
                if not rows:
                    continue
                
                # First row is typically the header
                first_row = rows[0]
                header_cells = [cell.get_text(strip=True) for cell in first_row.find_all(['td', 'th'])]
                
                if not header_cells or not self._is_valid_result_table(header_cells):
                    continue
                
                tables_found += 1
                
                # Parse data rows (skip first row which is header)
                for tr in rows[1:]:
                    cells = tr.find_all(['td', 'th'])
                    if not cells or len(cells) != len(header_cells):
                        continue
                    
                    row = {}
                    for col_name, cell in zip(header_cells, cells):
                        row[col_name] = cell.get_text(strip=True)
                    
                    # Add metadata
                    row["RaceDate"] = date_str_clean
                    row["RaceNumber"] = current_race
                    row["Venue"] = current_venue
                    
                    all_rows.append(row)
        
        # Validate results
        if not all_rows:
            if tables_found == 0:
                print(f"   ⚠️  No valid result tables found - HTML structure may have changed")
                return pd.DataFrame(), ResultStatus.HTML_STRUCTURE_CHANGED, "No valid result tables detected"
            else:
                print(f"   ⚠️  Found {tables_found} tables but extracted no rows")
                return pd.DataFrame(), ResultStatus.PARSE_ERROR, "Tables found but no data extracted"
        
        df = pd.DataFrame(all_rows)
        
        # Data quality checks
        races_missing_number = df['RaceNumber'].isna().sum()
        races_missing_venue = df['Venue'].isna().sum()
        
        if races_missing_number > 0:
            print(f"   ⚠️  {races_missing_number} runners missing race number")
        if races_missing_venue > 0:
            print(f"   ⚠️  {races_missing_venue} runners missing venue")
        
        print(f"   ✅ Parsed {len(df)} runners from {df['RaceNumber'].nunique()} races")
        
        return df, ResultStatus.SUCCESS, f"Parsed {len(df)} runners"


def scrape_results_for_date(date_str: str) -> Tuple[pd.DataFrame, ResultStatus, str]:
    """
    Convenience function to scrape results for a single date.
    
    Args:
        date_str: Date in 'YYYY-MM-DD' format
    
    Returns:
        Tuple of (DataFrame, ResultStatus, message)
    """
    scraper = HKJCResultsScraper(max_retries=3)
    return scraper.get_results_for_date(date_str)


if __name__ == "__main__":
    # Test scraper
    import sys
    
    if len(sys.argv) > 1:
        date = sys.argv[1]
    else:
        date = "2024-11-09"
    
    print("=" * 80)
    print(f"TESTING HKJC RESULTS SCRAPER - {date}")
    print("=" * 80)
    
    df, status, message = scrape_results_for_date(date)
    
    print(f"\n📊 Status: {status.value}")
    print(f"📊 Message: {message}")
    
    if status == ResultStatus.SUCCESS:
        print(f"\n✅ Found {len(df)} runners across {df['RaceNumber'].nunique()} races")
        print(f"✅ Venues: {df['Venue'].unique()}")
        print(f"\nColumns: {list(df.columns)}")
        print(f"\nFirst few rows:")
        print(df.head(10))
    else:
        print(f"\n❌ Scraping failed: {message}")

