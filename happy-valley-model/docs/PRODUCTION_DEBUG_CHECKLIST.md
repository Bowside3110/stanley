# Production Debugging Checklist

After deploying the fixes, use this checklist to diagnose any remaining issues.

---

## ✅ Immediate Checks (After Deploy)

### 1. Check Application Logs
```bash
# If using DigitalOcean App Platform
doctl apps logs <app-id> --type run --follow

# Look for:
# - "INFO - Querying upcoming races..." messages
# - "ERROR - Error in ..." messages with full stack traces
# - Any PostgreSQL connection errors
```

### 2. Test Endpoints Manually
```bash
# Get your production URL
PROD_URL="https://your-app.ondigitalocean.app"

# Test health endpoint (should work without auth)
curl $PROD_URL/health

# Test after logging in (grab cookie from browser)
curl -H "Cookie: access_token=YOUR_TOKEN" $PROD_URL/api/races
curl -H "Cookie: access_token=YOUR_TOKEN" $PROD_URL/api/predictions
```

### 3. Check Database Connection
```bash
# Connect to production database
doctl databases connection <db-id> --output json

# Or through the web portal:
# - Go to DigitalOcean > Databases
# - Connect to stanley-dbase
# - Run these queries:
```

**SQL Queries to Run:**
```sql
-- Check if races table exists and has data
SELECT COUNT(*) as race_count FROM races;

-- Check upcoming races
SELECT race_id, race_name, course, post_time 
FROM races 
WHERE post_time > NOW() 
ORDER BY post_time ASC 
LIMIT 10;

-- Check if any races have predictions
SELECT 
    r.race_id, 
    r.race_name, 
    COUNT(ru.horse) as runner_count,
    COUNT(CASE WHEN ru.predicted_rank IS NOT NULL THEN 1 END) as predictions_count
FROM races r
LEFT JOIN runners ru ON r.race_id = ru.race_id
WHERE r.post_time > NOW()
GROUP BY r.race_id, r.race_name
ORDER BY r.post_time ASC;

-- Check runners table structure
SELECT column_name, data_type 
FROM information_schema.columns 
WHERE table_name = 'runners';
```

---

## 🔍 Common Issues & Solutions

### Issue 1: "No upcoming races with predictions"

**Symptoms:**
- `/api/predictions` returns empty array `[]`
- Dashboard shows "No predictions available"

**Diagnosis:**
```sql
-- Check if races exist but predictions are missing
SELECT race_id, race_name, post_time 
FROM races 
WHERE post_time > NOW();

-- Check if runners exist but predicted_rank is NULL
SELECT race_id, horse, predicted_rank 
FROM runners 
WHERE race_id IN (
    SELECT race_id FROM races WHERE post_time > NOW()
)
LIMIT 20;
```

**Solution:**
Run the prediction pipeline:
```bash
# Locally or in a worker container
python -m src.make_predictions
```

---

### Issue 2: "Timezone mismatch"

**Symptoms:**
- Races exist but query returns empty
- `post_time > NOW()` comparison failing

**Diagnosis:**
```sql
-- Check current database time
SELECT NOW() as db_time;

-- Check race times
SELECT race_id, post_time, 
       post_time > NOW() as is_future,
       post_time - NOW() as time_until_race
FROM races 
ORDER BY post_time DESC 
LIMIT 10;
```

**Solution:**
If times are in different timezones, update `web/db_queries.py`:
```python
# For PostgreSQL with timezone awareness
from datetime import datetime, timezone

# Instead of:
datetime.now().isoformat()

# Use:
datetime.now(timezone.utc).isoformat()
```

---

### Issue 3: "Column not found errors"

**Symptoms:**
- Error: `column "predicted_rank" does not exist`
- Error: `column "post_time" does not exist`

**Diagnosis:**
```sql
-- List all columns in runners table
SELECT column_name FROM information_schema.columns 
WHERE table_name = 'runners';

-- List all columns in races table
SELECT column_name FROM information_schema.columns 
WHERE table_name = 'races';
```

**Solution:**
Column names might differ between SQLite and PostgreSQL. Check migration:
- Review `POSTGRESQL_MIGRATION_REPORT.md`
- Verify column names match schema in `src/db_config.py`

---

### Issue 4: "Database connection timeout"

**Symptoms:**
- Slow queries
- Connection errors
- "connection already closed"

**Diagnosis:**
- Check connection pool settings
- Look for leaked connections

**Solution:**
Ensure all queries properly close connections (already fixed in db_queries.py):
```python
conn = None
try:
    conn = get_connection()
    # ... queries ...
finally:
    if conn:
        conn.close()
```

---

### Issue 5: "Permission denied"

**Symptoms:**
- Error: `permission denied for table races`

**Diagnosis:**
```sql
-- Check current user permissions
SELECT current_user;

-- Check table permissions
SELECT grantee, privilege_type 
FROM information_schema.table_privileges 
WHERE table_name IN ('races', 'runners');
```

**Solution:**
```sql
-- Grant necessary permissions
GRANT SELECT, INSERT, UPDATE ON races TO your_db_user;
GRANT SELECT, INSERT, UPDATE ON runners TO your_db_user;
```

---

## 🐛 Debug Logging Commands

### Enable Debug Mode Temporarily
Add to `.do/app.yaml` or environment variables:
```yaml
envs:
  - key: LOG_LEVEL
    value: "DEBUG"
```

### View Specific Module Logs
```python
# In web/main.py or web/db_queries.py, change:
logging.basicConfig(level=logging.INFO, ...)

# To:
logging.basicConfig(level=logging.DEBUG, ...)
```

### Tail Production Logs with Filtering
```bash
# Watch for errors only
doctl apps logs <app-id> --type run --follow | grep ERROR

# Watch for specific endpoint
doctl apps logs <app-id> --type run --follow | grep "/api/predictions"

# Watch for database queries
doctl apps logs <app-id> --type run --follow | grep "Querying"
```

---

## 📊 Key Metrics to Monitor

### 1. Response Times
- `/api/races` should be < 500ms
- `/api/predictions` should be < 1s (depends on data volume)

### 2. Error Rates
- Should be 0% after fixes
- Any 500 errors should have detailed logs

### 3. Database Queries
- Check for slow queries (> 2s)
- Look for N+1 query patterns

### 4. Memory Usage
- Watch for memory leaks
- Ensure connections are properly closed

---

## 🎯 Quick Test Script

Save as `test_production.sh`:

```bash
#!/bin/bash

PROD_URL="https://your-app.ondigitalocean.app"

echo "Testing Health Endpoint..."
curl -s $PROD_URL/health | jq .

echo -e "\n\nTesting Login Page..."
curl -s -I $PROD_URL/login | grep "HTTP"

echo -e "\n\nChecking if app is responsive..."
response_time=$(curl -o /dev/null -s -w '%{time_total}' $PROD_URL/health)
echo "Health check response time: ${response_time}s"

if (( $(echo "$response_time > 2" | bc -l) )); then
    echo "⚠️  WARNING: Slow response time!"
else
    echo "✅ Response time OK"
fi
```

---

## 📞 If All Else Fails

### 1. Check DigitalOcean Status
- https://status.digitalocean.com/
- Check if there are any ongoing incidents

### 2. Review Environment Variables
```bash
# List all environment variables in production
doctl apps spec get <app-id> --format yaml | grep -A 20 "envs:"
```

### 3. Compare Local vs Production
```bash
# Local environment
echo $DATABASE_URL
echo $USE_POSTGRES

# Production environment (from DigitalOcean dashboard)
# Settings > App-Level Environment Variables
```

### 4. Restart Application
```bash
# Force redeploy (sometimes fixes stuck state)
doctl apps create-deployment <app-id>
```

---

## ✅ Success Indicators

You'll know the fixes worked when:

1. **Logs show detailed information:**
   ```
   INFO - Querying upcoming races with current_time=...
   INFO - Found X upcoming races
   ```

2. **No more "uvicorn: error: unrecognized arguments" messages**

3. **API endpoints return data or meaningful errors:**
   ```json
   {"detail": "Database error: relation 'races' does not exist"}
   ```
   Instead of generic 500 errors

4. **Refresh Odds button works** without crashing

---

**Last Updated:** December 13, 2025

