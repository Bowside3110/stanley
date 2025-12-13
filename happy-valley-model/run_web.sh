#!/bin/bash
# Run the Stanley Racing Predictions web application

# Navigate to the project directory
cd "$(dirname "$0")"

# Activate virtual environment
source venv/bin/activate

# Run the FastAPI server with uvicorn
uvicorn web.main:app --host 0.0.0.0 --port 8000 --reload

