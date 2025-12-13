from fastapi import FastAPI, Request, Form, Depends, HTTPException, status
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from datetime import datetime, timedelta
from web.auth import create_access_token, verify_password, get_current_user, USERS
from web.db_queries import get_upcoming_races, get_race_predictions, get_all_current_predictions, get_past_predictions
from web.models import Race, RaceSummary, SchedulerStatus
from typing import List
import os

app = FastAPI(title="Stanley Racing Predictions")
templates = Jinja2Templates(directory="web/templates")
app.mount("/static", StaticFiles(directory="web/static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request, user: str = Depends(get_current_user)):
    return templates.TemplateResponse("base.html", {
        "request": request,
        "user": user
    })

@app.get("/health")
async def health():
    return {"status": "healthy"}

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
    Get scheduler status (placeholder for future implementation)
    
    Returns:
        Scheduler status information
    """
    # TODO: Implement actual scheduler status checking
    # For now, return a placeholder response
    return {
        "active": False,
        "next_job": None
    }

