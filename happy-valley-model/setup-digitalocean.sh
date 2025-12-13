#!/bin/bash
# DigitalOcean Droplet Setup Script for Stanley Racing System
# Run this on a fresh Ubuntu 22.04 droplet

set -e  # Exit on error

echo "════════════════════════════════════════════════════════════════"
echo "  Stanley Racing System - DigitalOcean Setup"
echo "════════════════════════════════════════════════════════════════"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# ============================================================================
# 1. Update System
# ============================================================================
echo -e "\n${BLUE}[1/7] Updating system packages...${NC}"
apt update && apt upgrade -y

# ============================================================================
# 2. Install Dependencies
# ============================================================================
echo -e "\n${BLUE}[2/7] Installing Python 3.11, Node.js, SQLite...${NC}"
apt install -y python3.11 python3.11-venv python3-pip nodejs npm sqlite3 git

# ============================================================================
# 3. Set up Project Directory
# ============================================================================
echo -e "\n${BLUE}[3/7] Setting up project directory...${NC}"
PROJECT_DIR="/opt/stanley/happy-valley-model"
mkdir -p $PROJECT_DIR
cd $PROJECT_DIR

# If you're deploying from git:
# git clone <your-repo-url> $PROJECT_DIR
# For now, assume project is already uploaded

# ============================================================================
# 4. Set up Python Virtual Environment
# ============================================================================
echo -e "\n${BLUE}[4/7] Creating Python virtual environment...${NC}"
python3.11 -m venv venv
source venv/bin/activate

echo -e "${GREEN}Installing Python packages from requirements.txt...${NC}"
pip install --upgrade pip
pip install -r requirements.txt

# ============================================================================
# 5. Install Node.js Dependencies
# ============================================================================
echo -e "\n${BLUE}[5/7] Installing Node.js dependencies...${NC}"
cd scripts
npm install
cd ..

# ============================================================================
# 6. Create Required Directories
# ============================================================================
echo -e "\n${BLUE}[6/7] Creating data and log directories...${NC}"
mkdir -p data/historical
mkdir -p data/predictions
mkdir -p data/processed
mkdir -p data/raw
mkdir -p data/models
mkdir -p logs

# ============================================================================
# 7. Set up Environment Variables
# ============================================================================
echo -e "\n${BLUE}[7/7] Setting up environment variables...${NC}"

if [ ! -f .env ]; then
    echo -e "${RED}WARNING: .env file not found!${NC}"
    echo "Please create .env file with your credentials:"
    echo "  1. Copy env.template to .env"
    echo "  2. Fill in SMTP and Twilio credentials"
    echo ""
    echo "Example:"
    echo "  cp env.template .env"
    echo "  nano .env"
    echo ""
else
    echo -e "${GREEN}.env file found ✓${NC}"
fi

# ============================================================================
# 8. Create systemd Service
# ============================================================================
echo -e "\n${BLUE}Creating systemd service...${NC}"

cat > /etc/systemd/system/stanley-scheduler.service << EOF
[Unit]
Description=Stanley Racing Scheduler
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=$PROJECT_DIR
Environment="PATH=$PROJECT_DIR/venv/bin:/usr/local/bin:/usr/bin:/bin"
ExecStart=$PROJECT_DIR/venv/bin/python scripts/scheduler.py
Restart=always
RestartSec=10
StandardOutput=append:/var/log/stanley-scheduler.log
StandardError=append:/var/log/stanley-scheduler-error.log

[Install]
WantedBy=multi-user.target
EOF

# ============================================================================
# 9. Enable but Don't Start (manual step)
# ============================================================================
echo -e "\n${BLUE}Enabling systemd service...${NC}"
systemctl daemon-reload
systemctl enable stanley-scheduler

echo ""
echo "════════════════════════════════════════════════════════════════"
echo -e "${GREEN}✓ Setup Complete!${NC}"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Next Steps:"
echo ""
echo "1. Create .env file with credentials:"
echo "   cd $PROJECT_DIR"
echo "   cp env.template .env"
echo "   nano .env"
echo ""
echo "2. Test the scheduler manually:"
echo "   cd $PROJECT_DIR"
echo "   source venv/bin/activate"
echo "   python scripts/scheduler.py"
echo "   (Press Ctrl+C to stop)"
echo ""
echo "3. Start the scheduler service:"
echo "   systemctl start stanley-scheduler"
echo ""
echo "4. Check service status:"
echo "   systemctl status stanley-scheduler"
echo ""
echo "5. View logs:"
echo "   journalctl -u stanley-scheduler -f"
echo "   tail -f $PROJECT_DIR/logs/scheduler.log"
echo ""
echo "════════════════════════════════════════════════════════════════"

