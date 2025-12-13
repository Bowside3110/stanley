# DigitalOcean Deployment Guide

## Overview

This guide covers deploying the Stanley Racing Predictions application to DigitalOcean App Platform with:
- **Web Service**: FastAPI application for dashboard and API
- **Worker Service**: Background scheduler for predictions
- **PostgreSQL Database**: Managed database for race data

## Prerequisites

- GitHub repository with code
- DigitalOcean account
- PostgreSQL database already created and migrated
- Domain name (optional)

## Architecture

```
┌─────────────────────────────────────────────────────┐
│ DigitalOcean App Platform                           │
│                                                      │
│  ┌──────────────┐         ┌─────────────────┐      │
│  │ Web Service  │         │ Worker Service  │      │
│  │ (Uvicorn)    │         │ (Scheduler)     │      │
│  │ Port 8000    │         │                 │      │
│  └──────┬───────┘         └────────┬────────┘      │
│         │                          │               │
│         └──────────┬───────────────┘               │
│                    │                               │
│         ┌──────────▼──────────────┐                │
│         │ PostgreSQL Database     │                │
│         │ (Managed)               │                │
│         └─────────────────────────┘                │
└─────────────────────────────────────────────────────┘
```

## Deployment Files

The following files configure the deployment:

- `Dockerfile.web` - Container for web service
- `Dockerfile.worker` - Container for worker service
- `.do/app.yaml` - App Platform configuration
- `.dockerignore` - Files to exclude from Docker builds
- `scripts/wait_for_db.py` - Database startup check

## Step-by-Step Deployment

### 1. Prepare Your GitHub Repository

Push all code to GitHub:

```bash
git add .
git commit -m "Add DigitalOcean deployment configuration"
git push origin main
```

### 2. Generate Secrets

Generate a secure JWT secret key:

```bash
openssl rand -hex 32
```

Save this value for environment configuration.

### 3. Create App in DigitalOcean

1. Log into DigitalOcean console
2. Navigate to Apps → Create App
3. Choose "GitHub" as source
4. Select your repository
5. Select `main` branch
6. Import configuration from `.do/app.yaml`

### 4. Configure Environment Variables

Update the following SECRET variables in the DigitalOcean App Platform console:

#### Web Service Secrets:
- `JWT_SECRET_KEY` - Output from `openssl rand -hex 32`
- `SMTP_USERNAME` - Your Gmail address
- `SMTP_PASSWORD` - Gmail app password (not your regular password)

#### Worker Service Secrets:
- `TWILIO_ACCOUNT_SID` - From Twilio dashboard
- `TWILIO_AUTH_TOKEN` - From Twilio dashboard
- `TWILIO_FROM_PHONE` - Your Twilio phone number
- `SMTP_USERNAME` - Your Gmail address
- `SMTP_PASSWORD` - Gmail app password

#### Database Connection:
The `DATABASE_URL` is automatically set from the managed PostgreSQL database.

### 5. Verify Configuration

Review the app configuration:
- **Region**: Singapore (sgp) - closest to Hong Kong
- **Web service**: basic-xxs ($5/month)
- **Worker service**: basic-xxs ($5/month)
- **Database**: Already provisioned ($15/month)

### 6. Deploy

Click "Create Resources" to start deployment.

The deployment process will:
1. Build Docker images for web and worker
2. Run database connectivity checks
3. Start services
4. Configure health checks

### 7. Post-Deployment Verification

After deployment completes:

1. **Check Web Service**:
   - Visit your app URL (e.g., `https://stanley-racing-xxxxx.ondigitalocean.app`)
   - Verify `/health` endpoint returns healthy status
   - Login with credentials

2. **Check Worker Service**:
   - View logs in DigitalOcean console
   - Verify scheduler started successfully
   - Check for scheduled jobs

3. **Test Functionality**:
   - Verify predictions are displayed
   - Test manual prediction triggers
   - Confirm email/SMS alerts work

## Environment Variables Reference

### Web Service

| Variable | Type | Description | Example |
|----------|------|-------------|---------|
| USE_POSTGRES | value | Use PostgreSQL | `true` |
| DATABASE_URL | auto | Database connection string | Auto-set |
| JWT_SECRET_KEY | secret | JWT token signing key | (generated) |
| SMTP_SERVER | value | Email server | `smtp.gmail.com` |
| SMTP_PORT | value | Email port | `587` |
| SMTP_USERNAME | secret | Email username | `your-email@gmail.com` |
| SMTP_PASSWORD | secret | Email app password | (from Gmail) |
| ALERT_EMAIL | value | Alert recipient | `adamsalistair1978@gmail.com` |

### Worker Service

| Variable | Type | Description | Example |
|----------|------|-------------|---------|
| USE_POSTGRES | value | Use PostgreSQL | `true` |
| DATABASE_URL | auto | Database connection string | Auto-set |
| TWILIO_ACCOUNT_SID | secret | Twilio account SID | (from Twilio) |
| TWILIO_AUTH_TOKEN | secret | Twilio auth token | (from Twilio) |
| TWILIO_FROM_PHONE | value | Twilio phone number | `+1234567890` |
| ALERT_PHONE | value | SMS recipient | `+61417676973` |
| SMTP_SERVER | value | Email server | `smtp.gmail.com` |
| SMTP_PORT | value | Email port | `587` |
| SMTP_USERNAME | secret | Email username | `your-email@gmail.com` |
| SMTP_PASSWORD | secret | Email app password | (from Gmail) |
| ALERT_EMAIL | value | Alert recipient | `adamsalistair1978@gmail.com` |

## Cost Breakdown

Monthly costs:

| Service | Size | Cost |
|---------|------|------|
| Web Service | basic-xxs | $5/month |
| Worker Service | basic-xxs | $5/month |
| PostgreSQL | Dev Database | $15/month |
| **Total** | | **$25/month** |

## Scaling Considerations

### When to Scale Up

**Web Service** - Upgrade if:
- Response times > 2 seconds
- CPU usage consistently > 80%
- Multiple concurrent users

**Worker Service** - Upgrade if:
- Predictions timeout
- Scheduler jobs miss deadlines
- Memory usage high during model training

### Recommended Upgrades

| Scenario | Web | Worker | Cost |
|----------|-----|--------|------|
| Current | basic-xxs | basic-xxs | $10/mo |
| Moderate traffic | basic-xs | basic-xs | $20/mo |
| High traffic | basic-s | basic-s | $40/mo |
| Production | professional-xs | professional-xs | $100/mo |

## Monitoring & Logs

### View Logs

1. Navigate to your app in DigitalOcean console
2. Click on service (web or worker)
3. View "Runtime Logs" tab

### Health Checks

The web service has an automatic health check at `/health`:

```bash
curl https://your-app.ondigitalocean.app/health
```

Expected response:
```json
{
  "status": "healthy",
  "database": "connected",
  "timestamp": "2025-12-13T10:30:00"
}
```

## Troubleshooting

### Database Connection Issues

If services can't connect to database:

1. Check `DATABASE_URL` is set correctly
2. Verify database is in same region (sgp)
3. Check database firewall allows app services
4. Review logs for connection errors

### Scheduler Not Running

If predictions aren't being generated:

1. Check worker service logs
2. Verify `scripts/fetch_race_times.py` ran successfully
3. Check database has upcoming races
4. Verify Twilio/SMTP credentials are correct

### Build Failures

If Docker build fails:

1. Check `requirements.txt` for invalid packages
2. Verify `package.json` exists in scripts/
3. Review build logs for specific errors
4. Test Docker build locally first

## Local Testing (Optional)

Before deploying, test Docker containers locally:

```bash
# Build images
docker build -f Dockerfile.web -t stanley-web .
docker build -f Dockerfile.worker -t stanley-worker .

# Test web container
docker run -p 8000:8000 --env-file .env stanley-web

# Test worker container
docker run --env-file .env stanley-worker
```

## Updating the Application

To deploy changes:

```bash
git add .
git commit -m "Your changes"
git push origin main
```

DigitalOcean will automatically rebuild and redeploy if `deploy_on_push: true` is enabled.

## Rollback

If a deployment fails:

1. Go to App Platform console
2. Navigate to "Deployments" tab
3. Click on previous successful deployment
4. Click "Rollback to this deployment"

## Custom Domain (Optional)

To use a custom domain:

1. Go to App Platform console
2. Navigate to "Settings" → "Domains"
3. Click "Add Domain"
4. Follow DNS configuration instructions

## Security Notes

1. **Never commit secrets** to Git
2. Use DigitalOcean's SECRET type for sensitive values
3. Rotate JWT_SECRET_KEY periodically
4. Use Gmail app passwords, not regular passwords
5. Enable 2FA on DigitalOcean account

## Support

For issues:
- Check DigitalOcean documentation: https://docs.digitalocean.com/products/app-platform/
- Review application logs in console
- Test locally with Docker first
- Verify all environment variables are set

## Next Steps

After successful deployment:

1. Set up monitoring alerts in DigitalOcean
2. Configure backup schedule for PostgreSQL
3. Test full prediction workflow
4. Monitor costs and resource usage
5. Consider upgrading to production database tier for reliability

