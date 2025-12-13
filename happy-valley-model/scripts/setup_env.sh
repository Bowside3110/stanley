#!/bin/bash
# Add PostgreSQL configuration to .env file

ENV_FILE="/Users/bendunn/Stanley/happy-valley-model/.env"

# Check if USE_POSTGRES already exists
if grep -q "USE_POSTGRES" "$ENV_FILE" 2>/dev/null; then
    echo "✅ PostgreSQL settings already in .env"
else
    echo "" >> "$ENV_FILE"
    echo "# ============================================================================" >> "$ENV_FILE"
    echo "# DATABASE CONFIGURATION (PostgreSQL Migration)" >> "$ENV_FILE"
    echo "# ============================================================================" >> "$ENV_FILE"
    echo "USE_POSTGRES=true" >> "$ENV_FILE"
    echo 'DATABASE_URL=postgresql://username:password@host:port/database?sslmode=require' >> "$ENV_FILE"
    echo "" >> "$ENV_FILE"
    echo "✅ Added PostgreSQL configuration to .env"
fi

echo ""
echo "Current database settings:"
grep -A 2 "USE_POSTGRES" "$ENV_FILE" 2>/dev/null || echo "USE_POSTGRES not found"

