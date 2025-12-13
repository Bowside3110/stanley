"""
Database query functions for API endpoints
"""
from src.db_config import get_connection, get_placeholder
from typing import List, Dict, Any, Optional
from datetime import datetime


def get_upcoming_races() -> List[Dict[str, Any]]:
    """
    Fetch races with post_time > now
    
    Returns:
        List of race dictionaries with race_id, race_name, course, post_time
    """
    conn = get_connection()
    cur = conn.cursor()
    placeholder = get_placeholder()
    
    try:
        # Query races table for future races
        query = f"""
            SELECT 
                race_id, 
                race_name, 
                course, 
                post_time,
                date
            FROM races
            WHERE post_time > {placeholder}
            ORDER BY post_time ASC
        """
        
        cur.execute(query, (datetime.now().isoformat(),))
        rows = cur.fetchall()
        
        # Convert rows to dictionaries
        races = []
        for row in rows:
            race = dict(row)
            races.append(race)
        
        return races
    
    finally:
        conn.close()


def get_race_predictions(race_id: str) -> Optional[Dict[str, Any]]:
    """
    Fetch predictions for specific race
    
    Args:
        race_id: The race identifier
        
    Returns:
        Dictionary with race info and list of predictions, or None if not found
    """
    conn = get_connection()
    cur = conn.cursor()
    placeholder = get_placeholder()
    
    try:
        # First, get race information
        race_query = f"""
            SELECT 
                race_id, 
                race_name, 
                course, 
                post_time
            FROM races
            WHERE race_id = {placeholder}
        """
        
        cur.execute(race_query, (race_id,))
        race_row = cur.fetchone()
        
        if not race_row:
            return None
        
        race_info = dict(race_row)
        
        # Now get predictions for runners in this race
        runners_query = f"""
            SELECT 
                horse,
                draw,
                predicted_rank,
                predicted_score,
                win_odds
            FROM runners
            WHERE race_id = {placeholder}
                AND predicted_rank IS NOT NULL
            ORDER BY predicted_rank ASC
        """
        
        cur.execute(runners_query, (race_id,))
        runner_rows = cur.fetchall()
        
        # Convert runner rows to list of dictionaries
        predictions = []
        for row in runner_rows:
            runner = dict(row)
            predictions.append(runner)
        
        # Combine race info with predictions
        result = {
            **race_info,
            'predictions': predictions
        }
        
        return result
    
    finally:
        conn.close()


def get_all_current_predictions() -> List[Dict[str, Any]]:
    """
    Fetch predictions for all upcoming races
    
    Returns:
        List of race dictionaries, each containing race info and predictions
    """
    conn = get_connection()
    cur = conn.cursor()
    placeholder = get_placeholder()
    
    try:
        # Get all upcoming races with predictions
        races_query = f"""
            SELECT DISTINCT
                r.race_id,
                r.race_name,
                r.course,
                r.post_time
            FROM races r
            INNER JOIN runners ru ON r.race_id = ru.race_id
            WHERE r.post_time > {placeholder}
                AND ru.predicted_rank IS NOT NULL
            ORDER BY r.post_time ASC
        """
        
        cur.execute(races_query, (datetime.now().isoformat(),))
        race_rows = cur.fetchall()
        
        results = []
        
        # For each race, fetch its predictions
        for race_row in race_rows:
            race_dict = dict(race_row)
            race_id = race_dict['race_id']
            
            # Get predictions for this race
            runners_query = f"""
                SELECT 
                    horse,
                    draw,
                    predicted_rank,
                    predicted_score,
                    win_odds
                FROM runners
                WHERE race_id = {placeholder}
                    AND predicted_rank IS NOT NULL
                ORDER BY predicted_rank ASC
            """
            
            cur.execute(runners_query, (race_id,))
            runner_rows = cur.fetchall()
            
            predictions = [dict(row) for row in runner_rows]
            
            # Combine race info with predictions
            race_with_predictions = {
                **race_dict,
                'predictions': predictions
            }
            
            results.append(race_with_predictions)
        
        return results
    
    finally:
        conn.close()


def get_past_predictions(limit: int = 10) -> List[Dict[str, Any]]:
    """
    Fetch recent past races with predictions and results
    
    Args:
        limit: Maximum number of races to return
        
    Returns:
        List of race dictionaries with predictions and actual results
    """
    conn = get_connection()
    cur = conn.cursor()
    placeholder = get_placeholder()
    
    try:
        # Get recent past races with predictions
        races_query = f"""
            SELECT DISTINCT
                r.race_id,
                r.race_name,
                r.course,
                r.post_time,
                r.date
            FROM races r
            INNER JOIN runners ru ON r.race_id = ru.race_id
            WHERE r.post_time < {placeholder}
                AND ru.predicted_rank IS NOT NULL
            ORDER BY r.post_time DESC
            LIMIT {placeholder}
        """
        
        cur.execute(races_query, (datetime.now().isoformat(), limit))
        race_rows = cur.fetchall()
        
        results = []
        
        # For each race, fetch its predictions and results
        for race_row in race_rows:
            race_dict = dict(race_row)
            race_id = race_dict['race_id']
            
            # Get predictions and actual results for this race
            runners_query = f"""
                SELECT 
                    horse,
                    draw,
                    predicted_rank,
                    predicted_score,
                    win_odds,
                    position as actual_position
                FROM runners
                WHERE race_id = {placeholder}
                    AND predicted_rank IS NOT NULL
                ORDER BY predicted_rank ASC
            """
            
            cur.execute(runners_query, (race_id,))
            runner_rows = cur.fetchall()
            
            predictions = [dict(row) for row in runner_rows]
            
            # Combine race info with predictions
            race_with_predictions = {
                **race_dict,
                'predictions': predictions
            }
            
            results.append(race_with_predictions)
        
        return results
    
    finally:
        conn.close()

