from fastapi import FastAPI, Request, Form, Depends, HTTPException, status
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, RedirectResponse
from datetime import datetime, timedelta
from web.auth import create_access_token, verify_password, get_current_user, USERS
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

