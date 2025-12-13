"""
Pydantic models for API responses
"""
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime


class Horse(BaseModel):
    """Individual horse prediction"""
    horse: str
    draw: int
    predicted_rank: int
    predicted_score: float
    win_odds: Optional[float] = None


class Race(BaseModel):
    """Race with predictions"""
    race_id: str
    race_name: str
    course: str
    post_time: str
    predictions: List[Horse]


class RaceSummary(BaseModel):
    """Race summary for dropdown/listing"""
    race_id: str
    race_name: str
    course: str
    post_time: str
    race_number: Optional[int] = None


class SchedulerStatus(BaseModel):
    """Scheduler status information"""
    active: bool
    next_job: Optional[str] = None

