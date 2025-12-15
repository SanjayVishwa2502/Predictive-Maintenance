# Dashboard Startup Guide

## Quick Start

### Option 1: Batch File (Double-Click) ⚡
Simply **double-click** `start_dashboard.bat` to start all services.

### Option 2: PowerShell Script 🚀
Right-click `start_dashboard.ps1` → **Run with PowerShell**

Or from PowerShell terminal:
```powershell
.\start_dashboard.ps1
```

## What Gets Started

The startup script launches **4 PowerShell windows**, each running a different service:

### 1. Backend Server (FastAPI) 🔧
- **URL:** http://localhost:8000
- **API Docs:** http://localhost:8000/docs
- **Purpose:** REST API for frontend, database operations, authentication
- **Port:** 8000

### 2. Celery Worker ⚙️
- **Purpose:** Processes background tasks (GAN training, ML predictions, LLM explanations)
- **Pool:** Solo (Windows compatible)
- **Queues:** celery (default), gan, ml, llm (future)

### 3. Flower Monitoring 📊
- **URL:** http://localhost:5555
- **Purpose:** Real-time monitoring of Celery tasks and workers
- **Features:** Task history, worker stats, performance metrics
- **Port:** 5555

### 4. Frontend Server (React + Vite) 🎨
- **URL:** http://localhost:5173
- **Purpose:** Dashboard UI (auto-opens in browser)
- **Features:** Hot module reload, fast refresh
- **Port:** 5173

## Services That Run Automatically

These Windows services start automatically with your computer:

✅ **PostgreSQL 18** - Database (port 5433)  
✅ **Redis** - Message broker & result backend (port 6379)

You don't need to manually start/stop these!

## Stopping Services

### Option 1: Batch File 🛑
Double-click `stop_dashboard.bat`

### Option 2: PowerShell Script
```powershell
.\stop_dashboard.ps1
```

### Option 3: Manual
Close all PowerShell windows that were opened by the startup script.

**Note:** PostgreSQL and Redis will continue running (they're Windows services).

## Manual Startup (Alternative)

If you prefer to start services individually in separate terminals:

### Terminal 1 - Backend
```powershell
cd "C:\Projects\Predictive Maintenance\frontend\server"
& "C:/Projects/Predictive Maintenance/venv/Scripts/Activate.ps1"
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Terminal 2 - Celery Worker
```powershell
cd "C:\Projects\Predictive Maintenance\frontend\server"
& "C:/Projects/Predictive Maintenance/venv/Scripts/Activate.ps1"
celery -A celery_app worker --loglevel=info --pool=solo
```

### Terminal 3 - Flower (Optional)
```powershell
cd "C:\Projects\Predictive Maintenance\frontend\server"
& "C:/Projects/Predictive Maintenance/venv/Scripts/Activate.ps1"
celery -A celery_app flower --port=5555
```

### Terminal 4 - Frontend
```powershell
cd "C:\Projects\Predictive Maintenance\frontend\client"
npm run dev
```

## Service URLs

| Service | URL | Purpose |
|---------|-----|---------|
| **Frontend** | http://localhost:5173 | Main dashboard UI |
| **Backend API** | http://localhost:8000 | REST API |
| **API Docs** | http://localhost:8000/docs | Interactive API documentation |
| **Flower** | http://localhost:5555 | Celery task monitoring |

## Files

- `start_dashboard.bat` - Windows batch file for easy startup
- `start_dashboard.ps1` - PowerShell startup script (colored output, service checks)
- `stop_dashboard.bat` - Windows batch file to stop all services
- `stop_dashboard.ps1` - PowerShell shutdown script

## Troubleshooting

### Services won't start
Check if PostgreSQL and Redis are running:
```powershell
Get-Service postgresql-x64-18, Redis
```

If not running, start them:
```powershell
Start-Service postgresql-x64-18
Start-Service Redis
```

### Port already in use
If you see "port already in use" errors:
1. Run `stop_dashboard.ps1` to clean up
2. Check for orphaned processes:
   ```powershell
   Get-Process | Where-Object {$_.ProcessName -match "node|python"}
   ```
3. Kill any remaining processes

### Flower not accessible
Flower is optional. If it doesn't start, the rest of the system will still work.
You can skip it and monitor tasks through the backend API instead.

## Requirements

- Python virtual environment at `C:\Projects\Predictive Maintenance\venv`
- Node.js installed (for frontend)
- All dependencies installed:
  - Backend: `pip install -r frontend/server/requirements.txt`
  - Frontend: `cd frontend/client && npm install`

## Troubleshooting

**Port already in use:**
- Backend (8000): Another FastAPI instance is running
- Frontend (5173): Another Vite dev server is running
- Solution: Run `stop_dashboard.bat` to kill all processes

**Services not starting:**
- Check if virtual environment exists
- Ensure all dependencies are installed
- Check PowerShell execution policy: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

## Manual Start (Alternative)

If scripts don't work, start manually:

### Backend:
```powershell
cd "C:\Projects\Predictive Maintenance\frontend\server"
& "C:\Projects\Predictive Maintenance\venv\Scripts\Activate.ps1"
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Celery:
```powershell
cd "C:\Projects\Predictive Maintenance\frontend\server"
& "C:\Projects\Predictive Maintenance\venv\Scripts\Activate.ps1"
celery -A celery_app worker --loglevel=info --pool=solo
```

### Frontend:
```powershell
cd "C:\Projects\Predictive Maintenance\frontend\client"
npm run dev
```
