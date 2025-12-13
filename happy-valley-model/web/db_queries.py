"""
Database query functions for API endpoints
"""
from src.db_config import get_connection, get_placeholder
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
import traceback

# Configure logging
logger = logging.getLogger(__name__)


def get_upcoming_races() -> List[Dict[str, Any]]:
    """
    Fetch races with post_time > now
    
    Returns:
        List of race dictionaries with race_id, race_name, course, post_time
    """
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        placeholder = get_placeholder()
        
        # Query races table for future races
        # Cast post_time to TIMESTAMP for proper datetime comparison in PostgreSQL
        # For SQLite, CAST is a no-op since it's dynamically typed
        query = f"""
            SELECT 
                race_id, 
                race_name, 
                course, 
                post_time,
                date
            FROM races
            WHERE CAST(post_time AS TIMESTAMP) > CAST({placeholder} AS TIMESTAMP)
            ORDER BY post_time ASC
        """
        
        current_time = datetime.now().isoformat()
        logger.info(f"Querying upcoming races with current_time={current_time}")
        
        cur.execute(query, (current_time,))
        rows = cur.fetchall()
        
        logger.info(f"Found {len(rows)} upcoming races")
        
        # Convert rows to dictionaries
        races = []
        for row in rows:
            race = dict(row)
            races.append(race)
        
        return races
    except Exception as e:
        logger.error(f"Error in get_upcoming_races: {str(e)}")
        logger.error(traceback.format_exc())
        raise
    finally:
        if conn:
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
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        placeholder = get_placeholder()
        
        # Get all upcoming races with predictions
        # Cast post_time to TIMESTAMP for proper datetime comparison in PostgreSQL
        races_query = f"""
            SELECT DISTINCT
                r.race_id,
                r.race_name,
                r.course,
                r.post_time
            FROM races r
            INNER JOIN runners ru ON r.race_id = ru.race_id
            WHERE CAST(r.post_time AS TIMESTAMP) > CAST({placeholder} AS TIMESTAMP)
                AND ru.predicted_rank IS NOT NULL
            ORDER BY r.post_time ASC
        """
        
        current_time = datetime.now().isoformat()
        logger.info(f"Querying predictions with current_time={current_time}")
        
        cur.execute(races_query, (current_time,))
        race_rows = cur.fetchall()
        
        logger.info(f"Found {len(race_rows)} races with predictions")
        
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
            
            logger.debug(f"Race {race_id}: {len(predictions)} predictions")
            
            # Combine race info with predictions
            race_with_predictions = {
                **race_dict,
                'predictions': predictions
            }
            
            results.append(race_with_predictions)
        
        return results
    except Exception as e:
        logger.error(f"Error in get_all_current_predictions: {str(e)}")
        logger.error(traceback.format_exc())
        raise
    finally:
        if conn:
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
        # Cast post_time to TIMESTAMP for proper datetime comparison
        races_query = f"""
            SELECT DISTINCT
                r.race_id,
                r.race_name,
                r.course,
                r.post_time,
                r.date
            FROM races r
            INNER JOIN runners ru ON r.race_id = ru.race_id
            WHERE CAST(r.post_time AS TIMESTAMP) < CAST({placeholder} AS TIMESTAMP)
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


def get_prediction_accuracy() -> Dict[str, Any]:
    """
    Calculate overall prediction accuracy metrics
    
    Returns:
        Dictionary containing various accuracy metrics:
        - total_races: Total number of races with predictions and results
        - top1_correct: Number of races where predicted #1 finished in top 3
        - top1_wins: Number of races where predicted #1 actually won
        - top3_in_top3: Number of races where all top 3 predictions finished in top 3
        - accuracy_pct: Percentage of top1 correct predictions
        - win_rate: Percentage of top1 wins
    """
    conn = get_connection()
    cur = conn.cursor()
    placeholder = get_placeholder()
    
    try:
        # Query for races with predictions and actual results
        # Cast post_time to TIMESTAMP for proper datetime comparison
        query = f"""
            SELECT 
                r.race_id,
                r.race_name,
                r.course,
                r.post_time,
                r.date
            FROM races r
            INNER JOIN runners ru ON r.race_id = ru.race_id
            WHERE CAST(r.post_time AS TIMESTAMP) < CAST({placeholder} AS TIMESTAMP)
                AND ru.predicted_rank IS NOT NULL
                AND ru.position IS NOT NULL
                AND ru.position != ''
            GROUP BY r.race_id, r.race_name, r.course, r.post_time, r.date
            ORDER BY r.post_time DESC
        """
        
        cur.execute(query, (datetime.now().isoformat(),))
        race_rows = cur.fetchall()
        
        total_races = len(race_rows)
        top1_correct = 0
        top1_wins = 0
        top3_in_top3 = 0
        
        # Analyze each race
        for race_row in race_rows:
            race_id = race_row['race_id']
            
            # Get runners with predictions and actual positions
            runners_query = f"""
                SELECT 
                    horse,
                    predicted_rank,
                    position
                FROM runners
                WHERE race_id = {placeholder}
                    AND predicted_rank IS NOT NULL
                    AND position IS NOT NULL
                    AND position != ''
                ORDER BY predicted_rank ASC
            """
            
            cur.execute(runners_query, (race_id,))
            runner_rows = cur.fetchall()
            
            if not runner_rows:
                total_races -= 1  # Don't count this race
                continue
            
            runners = [dict(row) for row in runner_rows]
            
            # Find predicted #1
            predicted_first = next((r for r in runners if r['predicted_rank'] == 1), None)
            
            if predicted_first:
                try:
                    actual_pos = int(predicted_first['position'])
                    # Check if predicted #1 finished in top 3
                    if actual_pos <= 3:
                        top1_correct += 1
                    # Check if predicted #1 won
                    if actual_pos == 1:
                        top1_wins += 1
                except (ValueError, TypeError):
                    pass
            
            # Check if all top 3 predictions finished in top 3
            top3_predicted = [r for r in runners if r['predicted_rank'] <= 3]
            if len(top3_predicted) == 3:
                try:
                    top3_actual_positions = [int(r['position']) for r in top3_predicted]
                    if all(pos <= 3 for pos in top3_actual_positions):
                        top3_in_top3 += 1
                except (ValueError, TypeError):
                    pass
        
        # Calculate percentages
        accuracy_pct = (top1_correct / total_races * 100) if total_races > 0 else 0
        win_rate = (top1_wins / total_races * 100) if total_races > 0 else 0
        top3_accuracy = (top3_in_top3 / total_races * 100) if total_races > 0 else 0
        
        return {
            'total_races': total_races,
            'top1_correct': top1_correct,
            'top1_wins': top1_wins,
            'top3_in_top3': top3_in_top3,
            'accuracy_pct': round(accuracy_pct, 1),
            'win_rate': round(win_rate, 1),
            'top3_accuracy': round(top3_accuracy, 1)
        }
    
    finally:
        conn.close()


def get_recent_performance(limit: int = 10) -> List[Dict[str, Any]]:
    """
    Get last N races performance with detailed analysis
    
    Args:
        limit: Maximum number of races to return
        
    Returns:
        List of race dictionaries with predictions, results, and accuracy
    """
    conn = get_connection()
    cur = conn.cursor()
    placeholder = get_placeholder()
    
    try:
        # Get recent past races with predictions and results
        # Cast post_time to TIMESTAMP for proper datetime comparison
        races_query = f"""
            SELECT DISTINCT
                r.race_id,
                r.race_name,
                r.course,
                r.post_time,
                r.date
            FROM races r
            INNER JOIN runners ru ON r.race_id = ru.race_id
            WHERE CAST(r.post_time AS TIMESTAMP) < CAST({placeholder} AS TIMESTAMP)
                AND ru.predicted_rank IS NOT NULL
                AND ru.position IS NOT NULL
                AND ru.position != ''
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
            
            runners = [dict(row) for row in runner_rows]
            
            # Calculate accuracy for this race
            predicted_first = next((r for r in runners if r['predicted_rank'] == 1), None)
            
            top1_correct = False
            top1_win = False
            actual_winner = None
            
            if predicted_first and predicted_first.get('actual_position'):
                try:
                    actual_pos = int(predicted_first['actual_position'])
                    top1_correct = actual_pos <= 3
                    top1_win = actual_pos == 1
                except (ValueError, TypeError):
                    pass
            
            # Find actual winner
            for runner in runners:
                if runner.get('actual_position'):
                    try:
                        if int(runner['actual_position']) == 1:
                            actual_winner = runner['horse']
                            break
                    except (ValueError, TypeError):
                        pass
            
            # Combine race info with predictions and analysis
            race_with_analysis = {
                **race_dict,
                'predictions': runners[:5],  # Only show top 5
                'predicted_winner': predicted_first['horse'] if predicted_first else None,
                'actual_winner': actual_winner,
                'top1_correct': top1_correct,
                'top1_win': top1_win
            }
            
            results.append(race_with_analysis)
        
        return results
    
    finally:
        conn.close()

