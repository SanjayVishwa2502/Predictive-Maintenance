# FRONTEND REBUILD REFERENCE - December 3, 2025

## 🎯 YOU ARE REBUILDING FRONTEND - READ THIS FIRST

### ✅ What's Working (DO NOT TOUCH)
- **Backend**: `frontend/server/` - ALL ENDPOINTS WORKING PERFECTLY
- **GAN Pipeline**: `GAN/` - 28/29 machines with complete data
- **Scripts**: All Python scripts tested and working
- **Database**: PostgreSQL connected
- **AI Parser**: LLaMA 3.1-8B working

### ❌ What's Broken (Why You're Rebuilding)
- **Frontend**: React components show wrong status (backend returns TRUE, UI shows FALSE)

---

## 📦 KEEP THESE FILES - ALREADY WORKING

```
✅ KEEP: frontend/server/          (Entire backend folder)
✅ KEEP: GAN/                       (All data and scripts)
✅ KEEP: LLM/                       (AI model)
✅ KEEP: ml_models/                 (ML models if any)
✅ KEEP: *.md                       (All documentation)
✅ KEEP: verify_all_machines.py     (Verification script)
✅ KEEP: requirements.txt           (Python deps)

❌ DELETE: frontend/client/         (Rebuild this only)
```

---

## 🔌 WORKING API ENDPOINTS (Test Before Rebuild)

### Verification Endpoint (SOURCE OF TRUTH)
```powershell
# This shows REAL file system status
Invoke-RestMethod -Uri "http://localhost:8000/api/gan/machines/verification"
```

### Status Endpoint (Also Working)
```powershell
# Returns TRUE values for existing machines
Invoke-RestMethod -Uri "http://localhost:8000/api/gan/machines/cnc_brother_speedio_001/status"

# Output:
# has_metadata: True
# has_seed_data: True
# has_tvae_model: True
# has_synthetic_data: True
```

### All Working Endpoints:
```
GET  /api/gan/machines                      - List 29 machines ✓
GET  /api/gan/machines/{id}/status          - Get status ✓
GET  /api/gan/machines/verification         - File system check ✓
POST /api/gan/machines                      - Create machine ✓
POST /api/gan/machines/{id}/seed            - Generate seed (10 min) ✓
POST /api/gan/machines/{id}/train           - Train TVAE (30 min) ✓
POST /api/gan/machines/{id}/generate        - Generate data (10 min) ✓
GET  /api/gan/machines/{id}/validate        - Validate ✓
POST /api/gan/machines/parse-ai             - AI parse ✓
POST /api/gan/machines/upload               - Upload JSON ✓
GET  /api/gan/machines/template             - Download template ✓
GET  /api/gan/machines/types                - List machine types ✓
```

---

## 🚀 REBUILD OPTION 1: React + Vite (Current Stack)

### Step 1: Delete and Recreate
```powershell
cd "C:\Projects\Predictive Maintenance\frontend"
Remove-Item -Recurse -Force client
npm create vite@latest client -- --template react-ts
cd client
npm install
```

### Step 2: Install Dependencies
```powershell
npm install @mui/material @emotion/react @emotion/styled
npm install @mui/icons-material
npm install axios
npm install zustand
npm install react-router-dom
```

### Step 3: Key Files to Create

#### `src/services/api.ts`
```typescript
import axios from 'axios';

export const api = axios.create({
  baseURL: 'http://localhost:8000',
  timeout: 30000, // 30s default
});
```

#### `src/modules/gan/ganApi.ts`
```typescript
import { api } from '../../services/api';

export async function listMachines() {
  const response = await api.get('/api/gan/machines');
  return response.data;
}

export async function getMachineStatus(machineId: string) {
  const response = await api.get(`/api/gan/machines/${machineId}/status`);
  return response.data;
}

export async function getVerification() {
  const response = await api.get('/api/gan/machines/verification');
  return response.data;
}

export async function generateSeedData(machineId: string, samples: number = 50000) {
  const response = await api.post(
    `/api/gan/machines/${machineId}/seed`,
    { samples },
    { timeout: 600000 } // 10 min
  );
  return response.data;
}

export async function startTraining(machineId: string, epochs: number = 500) {
  const response = await api.post(
    `/api/gan/machines/${machineId}/train`,
    { epochs },
    { timeout: 3600000 } // 60 min
  );
  return response.data;
}

export async function generateData(machineId: string, samples: number = 50000) {
  const response = await api.post(
    `/api/gan/machines/${machineId}/generate`,
    { samples },
    { timeout: 600000 } // 10 min
  );
  return response.data;
}
```

#### `src/modules/gan/components/MachineList.tsx`
```typescript
import { useEffect, useState } from 'react';
import { getVerification } from '../ganApi';

export function MachineList() {
  const [machines, setMachines] = useState([]);
  
  useEffect(() => {
    loadMachines();
  }, []);
  
  async function loadMachines() {
    const data = await getVerification();
    setMachines(data.machines);
  }
  
  return (
    <div>
      {machines.map(m => (
        <div key={m.machine_id}>
          <h3>{m.machine_id}</h3>
          <p>Seed: {m.has_seed ? '✓' : '✗'}</p>
          <p>Model: {m.has_model ? '✓' : '✗'}</p>
          <p>Synthetic: {m.has_synthetic ? '✓' : '✗'}</p>
        </div>
      ))}
    </div>
  );
}
```

---

## 🚀 REBUILD OPTION 2: Simple HTML (Fastest - No Build)

### Single File Solution

Create `simple-dashboard.html`:
```html
<!DOCTYPE html>
<html>
<head>
  <title>GAN Dashboard</title>
  <style>
    body { font-family: Arial; padding: 20px; }
    .machine { border: 1px solid #ddd; padding: 15px; margin: 10px 0; }
    .complete { background: #e8f5e9; }
    .incomplete { background: #fff3e0; }
    .status { display: inline-block; margin: 5px; }
    .check { color: green; }
    .cross { color: red; }
    button { padding: 10px 20px; margin: 5px; cursor: pointer; }
  </style>
</head>
<body>
  <h1>GAN Machine Dashboard</h1>
  <div id="summary"></div>
  <div id="machines"></div>
  
  <script>
    const API_BASE = 'http://localhost:8000/api/gan';
    
    async function loadDashboard() {
      const response = await fetch(`${API_BASE}/machines/verification`);
      const data = await response.json();
      
      // Show summary
      document.getElementById('summary').innerHTML = `
        <h2>Summary</h2>
        <p>Total: ${data.summary.total} | Complete: ${data.summary.complete}</p>
      `;
      
      // Show machines
      const machinesHTML = data.machines.map(m => {
        const complete = m.has_metadata && m.has_seed && m.has_model && m.has_synthetic;
        return `
          <div class="machine ${complete ? 'complete' : 'incomplete'}">
            <h3>${m.machine_id}</h3>
            <div class="status">Metadata: <span class="${m.has_metadata ? 'check' : 'cross'}">${m.has_metadata ? '✓' : '✗'}</span></div>
            <div class="status">Seed: <span class="${m.has_seed ? 'check' : 'cross'}">${m.has_seed ? '✓' : '✗'}</span></div>
            <div class="status">Model: <span class="${m.has_model ? 'check' : 'cross'}">${m.has_model ? '✓' : '✗'}</span></div>
            <div class="status">Synthetic: <span class="${m.has_synthetic ? 'check' : 'cross'}">${m.has_synthetic ? '✓' : '✗'}</span></div>
            ${!complete ? `<div><button onclick="completeMachine('${m.machine_id}')">Complete This Machine</button></div>` : ''}
          </div>
        `;
      }).join('');
      
      document.getElementById('machines').innerHTML = machinesHTML;
    }
    
    async function completeMachine(machineId) {
      alert(`For ${machineId}, run these commands in order:\n\n1. Generate Seed: POST ${API_BASE}/machines/${machineId}/seed\n2. Train Model: POST ${API_BASE}/machines/${machineId}/train\n3. Generate Data: POST ${API_BASE}/machines/${machineId}/generate`);
    }
    
    loadDashboard();
    setInterval(loadDashboard, 30000); // Refresh every 30s
  </script>
</body>
</html>
```

**To use**: Just open this file in browser - no npm, no build, works instantly!

---

## ⚙️ CRITICAL TIMEOUTS

```typescript
// These are REQUIRED or operations will fail:

Training:        60 minutes (3600000ms)
Seed Generation: 10 minutes (600000ms)
Data Generation: 10 minutes (600000ms)
AI Parsing:       5 minutes (300000ms)
Default:         30 seconds (30000ms)
```

---

## 📝 WORKFLOW LOGIC

```typescript
// When user selects existing machine:
const status = await getMachineStatus(machineId);

// Set wizard state:
{
  machineId: machineId,
  profileCreated: true,  // Always true if machine exists
  seedGenerated: status.has_seed_data,
  trainingComplete: status.has_tvae_model,
  generationComplete: status.has_synthetic_data,
}

// Determine next step:
if (!status.has_seed_data) → goto "Seed Generation" step
else if (!status.has_tvae_model) → goto "Training" step
else if (!status.has_synthetic_data) → goto "Generation" step
else → goto "Validation" step
```

---

## 🎯 TESTING AFTER REBUILD

1. Start backend: `uvicorn main:app --reload` (in frontend/server/)
2. Test API: http://localhost:8000/docs
3. Start frontend: `npm run dev` (in frontend/client/)
4. Open browser: http://localhost:3000/
5. Select existing machine → Should show ✓✓✓✓
6. Complete cnc_doosan_dnm_001 → Should take ~45 min total

---

## 💡 MY RECOMMENDATION

**Use Simple HTML first** to verify everything works, then rebuild React if needed.

**Simple HTML advantages**:
- No build process
- No dependencies
- Works in 2 minutes
- Uses verification endpoint (source of truth)
- Easy to debug

**React advantages**:
- Better UX
- Reusable components
- Type safety
- Scalable

---

**I'm ready to help with either approach. Which would you like?**

1. React rebuild (I'll provide all component code)
2. Simple HTML (I'll provide complete working file)
3. Something else?
