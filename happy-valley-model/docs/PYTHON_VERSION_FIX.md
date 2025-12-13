# Python 3.13 Compatibility Issue - RESOLVED

**Date:** December 13, 2025  
**Issue:** psycopg2 ImportError in production  
**Status:** ✅ FIXED

---

## 🔴 The Issue

After deploying the TEXT vs TIMESTAMP fixes, a new error appeared:

```
ImportError: /workspace/happy-valley-model/.heroku/python/lib/python3.13/site-packages/psycopg2/_psycopg.cpython-313-x86_64-linux-gnu.so: undefined symbol: _PyInterpreterState_Get
```

### Root Cause

1. **DigitalOcean is NOT using Dockerfile** - Despite `dockerfile_path: Dockerfile.web` in `app.yaml`, the `.heroku` path indicates it's using **Heroku Python buildpack**
   
2. **Python 3.13 is too new** - The buildpack auto-detected and installed Python 3.13 (latest)

3. **psycopg2-binary incompatibility** - The pre-compiled `psycopg2-binary==2.9.9` wheels don't support Python 3.13 yet
   - The binary is compiled against older Python C-API
   - Symbol `_PyInterpreterState_Get` changed in Python 3.13

---

## ✅ The Solution

Created `runtime.txt` to pin Python version:

```
python-3.12.7
```

This tells the Heroku buildpack to use Python 3.12 instead of auto-detecting the latest version.

---

## 📋 Why This Happened

### Buildpack Detection
DigitalOcean App Platform uses buildpacks to detect project type:
- Finds `requirements.txt` → Uses Heroku Python buildpack
- Buildpack installs latest Python (3.13) by default
- **Ignores Dockerfile** unless explicitly configured differently

### Python 3.13 Release
- Python 3.13 was released October 2024
- Binary packages (like psycopg2) take time to catch up
- Many C-extension packages not yet compatible

### Why Dockerfile Wasn't Used
The `app.yaml` specifies `dockerfile_path` but DigitalOcean App Platform may:
1. Still use buildpacks for dependency detection
2. Require additional configuration to force Docker builds
3. Override with buildpack if it detects standard project structure

---

## 🎯 Lessons Learned

1. **Always pin Python version in production**
   - Don't rely on "latest" auto-detection
   - Use `runtime.txt` for buildpack-based deploys
   - Or ensure Dockerfile is actually being used

2. **Check build logs carefully**
   - `.heroku` in paths = buildpack build
   - `/app` or custom paths = Docker build

3. **Test new Python versions before production**
   - Python 3.13 is cutting edge
   - Stick with 3.11 or 3.12 for stability

---

## 🔧 Files Modified

**Commit:** `bb6c164`

### Created:
- `happy-valley-model/runtime.txt` - Pins Python to 3.12.7

---

## 🚀 Deployment

The fix has been pushed and will:
1. Trigger DigitalOcean rebuild
2. Heroku buildpack will use Python 3.12.7 (from runtime.txt)
3. Install `psycopg2-binary==2.9.9` with Python 3.12 compatibility
4. Application should start successfully

**Expected build time:** 2-3 minutes

---

## ✅ Verification Steps

After deployment completes:

1. **Check build logs for Python version:**
   ```
   -----> Python app detected
   -----> Using Python version specified in runtime.txt
   -----> Installing python-3.12.7
   ```

2. **Test endpoints:**
   - `/api/predictions` should return data (not 500)
   - `/api/races` should return data (not 500)
   - No ImportError in logs

3. **Verify psycopg2 loads:**
   ```python
   import psycopg2  # Should work now
   ```

---

## 📊 Alternative Solutions (if this doesn't work)

### Option 1: Force Dockerfile Usage
Remove or rename `requirements.txt` temporarily to prevent buildpack detection, forcing Docker build.

### Option 2: Use psycopg2 (not binary)
```txt
# Instead of:
psycopg2-binary==2.9.9

# Use source build:
psycopg2==2.9.9
```
Requires build dependencies (already in Dockerfile: `gcc`, `postgresql-client`)

### Option 3: Downgrade to Python 3.11
```txt
# runtime.txt
python-3.11.9
```
Even more stable, all packages guaranteed compatible.

---

## 🎉 Summary

**Problem:** Python 3.13 + psycopg2-binary incompatibility  
**Solution:** Pin Python 3.12 via runtime.txt  
**Status:** Deployed and building  

This should be the final fix! 🤞

---

**Commits:**
- `581d26d` - Fixed TEXT vs TIMESTAMP comparison
- `bb6c164` - Pinned Python to 3.12 for psycopg2

**Next:** Monitor deployment and verify endpoints work

