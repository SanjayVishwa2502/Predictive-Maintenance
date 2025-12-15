# Service Startup Summary

## Answer to Your Question

**Out of the 4 services (Frontend, Backend, Celery, Flower), you need to run ALL 4 manually.**

However, I've created a **single startup script** that launches all 4 automatically! 🎉

## Easy Way: Use the Startup Script

### Just run this:
```powershell
.\start_dashboard.ps1
```

This will automatically open **4 PowerShell windows**, one for each service:

1. ✅ **Backend (FastAPI)** - http://localhost:8000
2. ✅ **Celery Worker** - Background task processing
3. ✅ **Flower** - http://localhost:5555 (monitoring)
4. ✅ **Frontend (React)** - http://localhost:5173 (auto-opens in browser)

### To stop everything:
```powershell
.\stop_dashboard.ps1
```

## What Runs Automatically (No Action Needed)

These are already configured as Windows services:

- ✅ **PostgreSQL** - Database (starts with Windows)
- ✅ **Redis** - Message broker (starts with Windows)

## Summary Table

| Service | Manual? | Method | Port |
|---------|---------|--------|------|
| PostgreSQL | ❌ Auto | Windows Service | 5433 |
| Redis | ❌ Auto | Windows Service | 6379 |
| Backend | ✅ Manual | `start_dashboard.ps1` | 8000 |
| Celery | ✅ Manual | `start_dashboard.ps1` | N/A |
| Flower | ✅ Manual | `start_dashboard.ps1` | 5555 |
| Frontend | ✅ Manual | `start_dashboard.ps1` | 5173 |

## Recommendation

**Use the startup script!** Instead of opening 4 terminals manually, just run:

```powershell
.\start_dashboard.ps1
```

It will:
- ✅ Check PostgreSQL and Redis are running
- ✅ Start all 4 services in separate windows
- ✅ Show you all the URLs
- ✅ Open the frontend in your browser

Much easier than managing 4 terminals! 😊

## During Development

Keep the **4 PowerShell windows open** while you work. They show logs for:
- Backend: API requests, database queries
- Celery: Task execution, errors
- Flower: Web UI (visit http://localhost:5555)
- Frontend: Build logs, hot reload

When done, close them all with `.\stop_dashboard.ps1`
