from fastapi import FastAPI, Request, Form, Depends, HTTPException, status
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from datetime import datetime, timedelta
from web.auth import create_access_token, verify_password, get_current_user, USERS
from web.db_queries import get_upcoming_races, get_race_predictions, get_all_current_predictions, get_past_predictions, get_prediction_accuracy, get_recent_performance
from web.models import Race, RaceSummary, SchedulerStatus
from typing import List
import os
import threading
import json
from pathlib import Path

app = FastAPI(title="Stanley Racing Predictions")
templates = Jinja2Templates(directory="web/templates")
app.mount("/static", StaticFiles(directory="web/static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request, user: str = Depends(get_current_user)):
    """Main dashboard page"""
    try:
        predictions = get_all_current_predictions()
        races = get_upcoming_races()
        
        return templates.TemplateResponse("dashboard.html", {
            "request": request,
            "user": user,
            "predictions": predictions,
            "races": races,
            "error": None,
            "now": datetime.now().strftime("%I:%M:%S %p")
        })
    except Exception as e:
        return templates.TemplateResponse("dashboard.html", {
            "request": request,
            "user": user,
            "predictions": [],
            "races": [],
            "error": str(e),
            "now": datetime.now().strftime("%I:%M:%S %p")
        })

@app.get("/health")
async def health():
    """
    Health check endpoint for DigitalOcean App Platform
    Tests database connectivity and returns system status
    """
    try:
        # Import here to avoid circular imports
        from src.db_config import get_connection
        
        # Test database connection
        conn = get_connection()
        conn.close()
        
        return {
            "status": "healthy",
            "database": "connected",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "database": "disconnected",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, error: str = None):
    return templates.TemplateResponse("login.html", {
        "request": request,
        "error": error
    })

@app.post("/login")
async def login(username: str = Form(...), password: str = Form(...)):
    # Verify credentials
    if username not in USERS:
        return RedirectResponse(url="/login?error=Invalid+credentials", status_code=303)
    
    hashed_password = USERS[username]
    if not verify_password(password, hashed_password):
        return RedirectResponse(url="/login?error=Invalid+credentials", status_code=303)
    
    # Create JWT token
    access_token = create_access_token(data={"sub": username})
    
    # Create redirect response with cookie
    response = RedirectResponse(url="/", status_code=303)
    response.set_cookie(
        key="access_token",
        value=f"Bearer {access_token}",
        httponly=True,
        max_age=1440 * 60,  # 24 hours in seconds
        samesite="lax"
    )
    return response

@app.get("/logout")
async def logout():
    response = RedirectResponse(url="/login", status_code=303)
    response.delete_cookie(key="access_token")
    return response


@app.get("/analytics", response_class=HTMLResponse)
async def analytics(request: Request, user: str = Depends(get_current_user)):
    """Analytics page showing prediction performance"""
    try:
        accuracy = get_prediction_accuracy()
        recent = get_recent_performance(limit=10)
        
        return templates.TemplateResponse("analytics.html", {
            "request": request,
            "user": user,
            "accuracy": accuracy,
            "recent": recent,
            "error": None
        })
    except Exception as e:
        return templates.TemplateResponse("analytics.html", {
            "request": request,
            "user": user,
            "accuracy": None,
            "recent": [],
            "error": str(e)
        })


# ========== API Endpoints ==========

@app.get("/api/races", response_model=List[RaceSummary])
async def list_races(user: str = Depends(get_current_user)):
    """
    Get list of upcoming races for dropdown/listing
    
    Returns:
        List of race summaries with basic info
    """
    try:
        races = get_upcoming_races()
        return races
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@app.get("/api/predictions", response_model=List[Race])
async def get_predictions(user: str = Depends(get_current_user)):
    """
    Get all current predictions for upcoming races
    
    Returns:
        List of races with their predictions
    """
    try:
        predictions = get_all_current_predictions()
        return predictions
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@app.get("/api/predictions/{race_id}", response_model=Race)
async def get_race_prediction(race_id: str, user: str = Depends(get_current_user)):
    """
    Get predictions for specific race
    
    Args:
        race_id: The race identifier (e.g., RACE_20251214_0010)
        
    Returns:
        Race with predictions
    """
    try:
        prediction = get_race_predictions(race_id)
        if not prediction:
            raise HTTPException(status_code=404, detail=f"Race {race_id} not found or has no predictions")
        return prediction
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@app.get("/api/past-predictions")
async def get_past_predictions_endpoint(
    limit: int = 10,
    user: str = Depends(get_current_user)
):
    """
    Get recent past races with predictions and actual results
    
    Args:
        limit: Maximum number of races to return (default: 10)
        
    Returns:
        List of races with predictions and actual results
    """
    try:
        predictions = get_past_predictions(limit=limit)
        return predictions
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@app.get("/api/scheduler/status", response_model=SchedulerStatus)
async def get_scheduler_status(user: str = Depends(get_current_user)):
    """
    Get scheduler status
    
    Returns:
        Scheduler status information
    """
    control_file = Path("data/scheduler_control.json")
    try:
        if control_file.exists():
            with open(control_file, 'r') as f:
                data = json.load(f)
                return {
                    "active": data.get('enabled', True),
                    "next_job": None  # TODO: Query scheduler DB for next job
                }
        else:
            # Default to enabled if file doesn't exist
            return {
                "active": True,
                "next_job": None
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading scheduler status: {str(e)}")


@app.post("/api/scheduler/toggle")
async def toggle_scheduler(user: str = Depends(get_current_user)):
    """
    Enable/disable scheduler
    
    Returns:
        New scheduler state
    """
    control_file = Path("data/scheduler_control.json")
    
    try:
        # Read current state
        if control_file.exists():
            with open(control_file, 'r') as f:
                data = json.load(f)
        else:
            data = {"enabled": True}
        
        # Toggle state
        new_state = not data.get('enabled', True)
        data['enabled'] = new_state
        data['last_updated'] = datetime.now().isoformat()
        data['updated_by'] = user
        
        # Write back
        control_file.parent.mkdir(parents=True, exist_ok=True)
        with open(control_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        return {
            "status": "success",
            "enabled": new_state,
            "message": f"Scheduler {'enabled' if new_state else 'disabled'}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error toggling scheduler: {str(e)}")


@app.post("/api/predict-meeting")
async def trigger_meeting_prediction(user: str = Depends(get_current_user)):
    """
    Trigger prediction for entire meeting
    
    Runs make_predictions in background thread
    """
    try:
        from src.make_predictions import main as make_predictions
        
        thread = threading.Thread(target=make_predictions)
        thread.daemon = True
        thread.start()
        
        return {
            "status": "started",
            "message": "Meeting prediction running in background"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting meeting prediction: {str(e)}")


@app.post("/api/predict-race")
async def trigger_race_prediction(
    user: str = Depends(get_current_user)
):
    """
    Trigger prediction for next upcoming race
    
    Note: Currently predicts the next race only. For specific race prediction,
    use the meeting prediction instead.
    """
    try:
        from src.predict_next_race import main as predict_next_race
        
        thread = threading.Thread(target=predict_next_race)
        thread.daemon = True
        thread.start()
        
        return {
            "status": "started",
            "message": "Prediction for next race running in background"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting race prediction: {str(e)}")


@app.post("/api/refresh-odds")
async def refresh_odds(user: str = Depends(get_current_user)):
    """
    Refresh odds and regenerate predictions
    
    Runs update_odds in background thread
    """
    try:
        from src.update_odds import main as update_odds
        
        thread = threading.Thread(target=update_odds)
        thread.daemon = True
        thread.start()
        
        return {
            "status": "started",
            "message": "Odds refresh running in background"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting odds refresh: {str(e)}")

