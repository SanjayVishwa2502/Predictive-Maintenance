# GAN Dashboard Integration Plan
**Phase 3.7.6: Integrate GAN Workflow into ML Dashboard**

**Date:** December 16, 2025  
**Status:** 📋 READY TO IMPLEMENT

---

## Executive Summary

**Problem Identified:** The GAN system (Phase 3.7.2) for synthetic data generation is **fully implemented in the backend** but has **NO frontend UI integration**. Users cannot:
- Upload new machine profiles
- Generate seed data
- Train TVAE models
- Generate synthetic datasets
- Monitor GAN workflows

**Current State:**
- ✅ Backend: 17 GAN API endpoints implemented (`/api/gan/*`)
- ✅ Backend: GANManager service operational with Celery tasks
- ✅ Backend: Database schema for GAN jobs and profiles
- ❌ Frontend: **NO GAN UI components exist**
- ❌ Frontend: ML Dashboard only shows predictions, not data generation

**Architecture Decision:**
- **Add a collapsible side navigation panel** to the ML Dashboard (hamburger menu or side drawer)
- GAN "New Machine Wizard" will be one option in this navigation panel
- Panel will include multiple application features: GAN, Settings, History, Reports, etc.
- Single-machine workflow (one machine at a time, no batch operations)
- Main dashboard view switches based on selected option

**Goal:** Implement a scalable navigation structure in the ML Dashboard with GAN workflow as the first major feature, enabling future expansion with settings, history tracking, and other tools.

---

## Architecture Analysis

### Backend (Already Complete ✅)

**GAN API Endpoints (17 total):**

| Category | Endpoint | Method | Purpose |
|----------|----------|--------|---------|
| **Templates** | `/api/gan/templates` | GET | List all machine templates |
| | `/api/gan/templates/{type}` | GET | Get specific template |
| | `/api/gan/examples/{type}` | GET | Get example profile |
| **Profiles** | `/api/gan/profiles/upload` | POST | Upload machine profile |
| | `/api/gan/profiles/validate` | POST | Validate profile |
| | `/api/gan/profiles/{id}/edit` | PUT | Edit profile |
| **Workflow** | `/api/gan/seed/generate` | POST | Generate seed data (Celery) |
| | `/api/gan/seed/{machine_id}/status` | GET | Check seed status |
| | `/api/gan/train` | POST | Train TVAE model (Celery) |
| | `/api/gan/train/{job_id}/status` | GET | Check training status |
| | `/api/gan/generate` | POST | Generate synthetic data (Celery) |
| | `/api/gan/generate/{job_id}/status` | GET | Check generation status |
| **Management** | `/api/gan/machines` | GET | List all machines |
| | `/api/gan/machines/{id}` | GET | Get machine details |
| | `/api/gan/machines/{id}` | DELETE | Delete machine |
| **Monitoring** | `/api/gan/tasks/{task_id}` | GET | Get task status |
| | `/api/gan/health` | GET | Health check |

**Services:**
- `GANManager` (Singleton): Model caching, seed generation, TVAE training
- `Celery Tasks`: Async background processing for long-running operations
- `Redis`: Task queue + caching
- `PostgreSQL`: Metadata storage (tables: `gan_profiles`, `gan_training_jobs`, `gan_generation_jobs`)

### Frontend (Missing ❌)

**What Needs to Be Built:**

1. **GAN Dashboard Page** - New top-level route
2. **Machine Profile Manager** - Upload, validate, edit profiles
3. **Workflow Stepper** - 4-step wizard (Profile → Seed → Train → Generate)
4. **Task Monitor** - Real-time Celery task progress
5. **Data Visualization** - Charts for seed data, training loss, synthetic quality
6. **Machine List** - Browse all machines with status badges

---

## Implementation Plan

### Phase 3.7.6.1: Core UI Structure - Side Navigation Panel (3-4 hours)

**Goal:** Implement collapsible side navigation panel in ML Dashboard

#### 1.1 Add Side Navigation Drawer

**File:** `client/src/modules/ml/pages/MLDashboardPage.tsx`

Add MUI Drawer component with navigation options:
```tsx
const [drawerOpen, setDrawerOpen] = useState(false);
const [selectedView, setSelectedView] = useState('predictions'); // predictions, gan, settings, history

<Box sx={{ display: 'flex' }}>
  {/* Top Bar with Menu Button */}
  <AppBar position="fixed">
    <Toolbar>
      <IconButton onClick={() => setDrawerOpen(!drawerOpen)}>
        <MenuIcon />
      </IconButton>
      <Typography>ML Dashboard - {getViewTitle(selectedView)}</Typography>
    </Toolbar>
  </AppBar>

  {/* Side Navigation Drawer */}
  <Drawer
    anchor="left"
    open={drawerOpen}
    onClose={() => setDrawerOpen(false)}
    variant="temporary" // Can be changed to "persistent" for always-visible
  >
    <NavigationPanel 
      selectedView={selectedView}
      onSelectView={(view) => {
        setSelectedView(view);
        setDrawerOpen(false);
      }}
    />
  </Drawer>

  {/* Main Content Area */}
  <Box component="main" sx={{ flexGrow: 1, p: 3 }}>
    {selectedView === 'predictions' && <PredictionsView />}
    {selectedView === 'gan' && <GANWizardView />}
    {selectedView === 'settings' && <SettingsView />}
    {selectedView === 'history' && <HistoryView />}
  </Box>
</Box>
```

#### 1.2 Create Navigation Panel Component

**File:** `client/src/modules/ml/components/NavigationPanel.tsx`

```tsx
interface NavOption {
  id: string;
  label: string;
  icon: ReactNode;
  description: string;
}

const navOptions: NavOption[] = [
  {
    id: 'predictions',
    label: 'Predictions',
    icon: <TimelineIcon />,
    description: 'Run ML predictions on machines'
  },
  {
    id: 'gan',
    label: 'New Machine Wizard',
    icon: <AutoFixHighIcon />,
    description: 'Generate synthetic training data'
  },
  {
    id: 'history',
    label: 'Prediction History',
    icon: <HistoryIcon />,
    description: 'View past predictions and trends'
  },
  {
    id: 'reports',
    label: 'Reports',
    icon: <AssessmentIcon />,
    description: 'Generate analysis reports'
  },
  {
    id: 'datasets',
    label: 'Dataset Manager',
    icon: <StorageIcon />,
    description: 'Manage training datasets'
  },
  {
    id: 'settings',
    label: 'Settings',
    icon: <SettingsIcon />,
    description: 'Configure dashboard preferences'
  }
];

<List>
  {navOptions.map((option) => (
    <ListItem 
      button 
      key={option.id}
      selected={selectedView === option.id}
      onClick={() => onSelectView(option.id)}
    >
      <ListItemIcon>{option.icon}</ListItemIcon>
      <ListItemText 
        primary={option.label}
        secondary={option.description}
      />
    </ListItem>
  ))}
</List>
```

#### 1.3 Create Module Structure

```
client/src/modules/ml/
├── pages/
│   └── MLDashboardPage.tsx           # Updated with drawer
├── components/
│   ├── NavigationPanel.tsx           # NEW - Side nav menu
│   ├── predictions/
│   │   └── PredictionsView.tsx       # Existing predictions (refactored)
│   ├── gan/
│   │   ├── GANWizardView.tsx         # NEW - Main GAN view
│   │   ├── MachineProfileUpload.tsx  # Upload wizard
│   │   ├── WorkflowStepper.tsx       # 4-step workflow
│   │   ├── TaskMonitor.tsx           # Celery task tracker
│   │   ├── SeedDataChart.tsx         # Visualize seed data
│   │   ├── TrainingProgressChart.tsx # TVAE loss curves
│   │   └── SyntheticDataPreview.tsx  # Sample generated data
│   ├── history/
│   │   └── HistoryView.tsx           # NEW - Prediction history
│   ├── reports/
│   │   └── ReportsView.tsx           # NEW - Report generator
│   ├── datasets/
│   │   └── DatasetManagerView.tsx    # NEW - Dataset manager
│   └── settings/
│       └── SettingsView.tsx          # NEW - Settings panel
├── api/
│   └── ganApi.ts                     # GAN API client
└── types/
    └── gan.types.ts                  # TypeScript interfaces
```

**Deliverables:**
- ✅ Collapsible side drawer implemented
- ✅ Navigation panel with 6 menu options
- ✅ View switching logic functional
- ✅ Folder structure created for all views
- ✅ Responsive design (mobile-friendly)

---

### Phase 3.7.6.2: Machine Profile Management (3-4 hours)

**Goal:** Enable users to upload and manage machine profiles

#### 2.1 Profile Upload Component

**File:** `MachineProfileUpload.tsx`

**Features:**
- Drag-and-drop file upload (JSON/YAML/Excel)
- Template download buttons (11 machine types)
- Real-time validation with error highlighting
- Example profile browser

**API Integration:**
```typescript
// Download template
GET /api/gan/templates/{machine_type}

// Upload profile
POST /api/gan/profiles/upload
FormData: { file: File }

// Validate profile
POST /api/gan/profiles/validate
Body: { profile_data: object }
```

**UI Flow:**
```
Step 1: Download Template
  → Dropdown: Select machine type (motor, pump, cnc, etc.)
  → Button: "Download Template" → JSON file

Step 2: Fill Profile
  → User edits JSON file externally

Step 3: Upload File
  → Drag-drop zone OR file browser
  → Immediate validation

Step 4: Review & Fix
  → Show validation errors in red
  → Inline editor to fix issues
  → Re-validate button

Step 5: Confirm & Save
  → Button: "Create Machine Profile"
  → Success: Navigate to workflow stepper
```

**Components:**
```tsx
<MachineProfileUpload>
  <TemplateSelector />
  <FileUploadZone />
  <ValidationResults />
  <ProfileEditor />
  <ActionButtons />
</MachineProfileUpload>
```

#### 2.2 Machine List Component

**File:** `MachineList.tsx`

**Features:**
- Table with columns: ID, Type, Manufacturer, Status, Actions
- Status badges: Draft, Seed Generated, Model Trained, Ready
- Search/filter by type or manufacturer
- Actions: View details, Edit, Delete, Start workflow

**API Integration:**
```typescript
GET /api/gan/machines
Response: { machines: Machine[], total: number }

GET /api/gan/machines/{machine_id}
Response: MachineDetails

DELETE /api/gan/machines/{machine_id}
```

**Deliverables:**
- ✅ Upload wizard with validation
- ✅ Template download functionality
- ✅ Machine list with CRUD operations
- ✅ Status badges and filtering

---

### Phase 3.7.6.3: Workflow Stepper (4-5 hours)

**Goal:** Guide users through the 4-step GAN workflow

#### 3.1 Workflow Steps

**Step 1: Generate Seed Data**
- Input: Number of samples (default 10,000)
- Button: "Generate Physics-Based Seed"
- Progress: Celery task status (0-100%)
- Output: CSV file path + row count
- Visualization: Time-series plot of RUL decay

**Step 2: Train TVAE Model**
- Input: Epochs (default 300), batch size (default 500)
- Button: "Train TVAE Model"
- Progress: Real-time loss curve chart
- Output: Model file path + final loss
- Estimated time: ~5-10 minutes

**Step 3: Generate Synthetic Data**
- Input: Total samples (default 50,000), split ratio (70/15/15)
- Button: "Generate Synthetic Dataset"
- Progress: Celery task with samples/second rate
- Output: 3 Parquet files (train/val/test) + row counts
- Preview: First 10 rows of generated data

**Step 4: Validation & Download**
- Quality metrics: Feature distributions, correlation matrices
- Comparison charts: Original seed vs. synthetic
- Download buttons for train/val/test datasets

#### 3.2 Implementation

**File:** `WorkflowStepper.tsx`

```tsx
const steps = [
  { label: 'Generate Seed', component: <SeedGenerationStep /> },
  { label: 'Train TVAE', component: <TVAETrainingStep /> },
  { label: 'Generate Synthetic', component: <SyntheticGenerationStep /> },
  { label: 'Validate & Download', component: <ValidationStep /> }
];

<Stepper activeStep={currentStep}>
  {steps.map((step) => (
    <Step key={step.label}>
      <StepLabel>{step.label}</StepLabel>
    </Step>
  ))}
</Stepper>

<Box sx={{ mt: 3 }}>
  {steps[currentStep].component}
</Box>
```

**API Integration:**
```typescript
// Step 1: Seed Generation
POST /api/gan/seed/generate
Body: { machine_id, num_samples }
Response: { task_id, status: "pending" }

GET /api/gan/seed/{machine_id}/status
Response: { status: "completed", file_path, row_count }

// Step 2: Training
POST /api/gan/train
Body: { machine_id, epochs, batch_size }
Response: { job_id, status: "running" }

GET /api/gan/train/{job_id}/status
Response: { status: "running", progress: 65, loss: 0.045 }

// Step 3: Generation
POST /api/gan/generate
Body: { machine_id, num_samples, split_ratio }
Response: { job_id }

GET /api/gan/generate/{job_id}/status
Response: { 
  status: "completed", 
  files: { train, val, test },
  row_counts: { train: 35000, val: 7500, test: 7500 }
}
```

**Deliverables:**
- ✅ 4-step workflow stepper
- ✅ Real-time task progress tracking
- ✅ Visualizations for each step
- ✅ Download functionality

---

### Phase 3.7.6.4: Task Monitoring (2-3 hours)

**Goal:** Real-time visibility into Celery background tasks

#### 4.1 Task Monitor Component

**File:** `TaskMonitor.tsx`

**Features:**
- Live task list (running, pending, completed, failed)
- Progress bars with percentage
- ETA (estimated time remaining)
- Cancel button for running tasks
- Logs viewer (stdout from Celery)
- Auto-refresh every 2 seconds

**API Integration:**
```typescript
GET /api/gan/tasks/{task_id}
Response: {
  task_id,
  status: "running" | "success" | "failure",
  progress: 45,
  current: 135,
  total: 300,
  result: {...},
  started_at,
  completed_at,
  logs: "Epoch 135/300..."
}
```

**UI Components:**
```tsx
<TaskMonitor>
  <TaskList>
    {tasks.map(task => (
      <TaskCard key={task.id}>
        <TaskHeader>{task.name}</TaskHeader>
        <LinearProgress value={task.progress} />
        <Typography>{task.progress}% - ETA: {task.eta}</Typography>
        <Button onClick={() => cancelTask(task.id)}>Cancel</Button>
      </TaskCard>
    ))}
  </TaskList>
</TaskMonitor>
```

**Deliverables:**
- ✅ Task list with real-time updates
- ✅ Progress tracking for long-running operations
- ✅ Error handling with retry options

---

### Phase 3.7.6.5: Data Visualization (2-3 hours)

**Goal:** Visual insights into seed data, training, and synthetic quality

#### 5.1 Charts to Implement

**1. Seed Data Time Series**
- X-axis: Timestamp
- Y-axis: RUL (Remaining Useful Life)
- Multiple series: Each sensor reading
- Highlights: Degradation patterns

**2. Training Loss Curve**
- X-axis: Epoch
- Y-axis: Loss value
- Line chart with smoothing
- Markers for best epoch

**3. Feature Distribution Comparison**
- Original seed data vs. synthetic data
- Histograms side-by-side
- Statistical metrics (mean, std, min, max)

**4. Correlation Matrix Heatmap**
- Sensor correlations in synthetic data
- Color-coded (red = high correlation)
- Validates temporal dependencies

**Libraries:**
- Recharts (already in project)
- MUI DataGrid (already in project)
- react-chartjs-2 (if needed for heatmaps)

**Files:**
- `SeedDataChart.tsx`
- `TrainingProgressChart.tsx`
- `FeatureDistributionChart.tsx`
- `CorrelationHeatmap.tsx`

**Deliverables:**
- ✅ 4 visualization components
- ✅ Interactive charts with tooltips
- ✅ Export chart as PNG

---

### Phase 3.7.6.6: View Integration & Additional Features (2-3 hours)

**Goal:** Seamless experience across all navigation views

#### 6.1 Shared State Management

**Global Dashboard State:**
- Selected machine persists across all views
- Connection status shared globally
- Task notifications visible in all views
- Navigation breadcrumbs show current location

**Implementation:**
```tsx
// Use Zustand or Context for global state
const useDashboardStore = create((set) => ({
  selectedMachine: null,
  connectionStatus: 'connected',
  activeTasks: [],
  selectedView: 'predictions',
  setSelectedMachine: (machine) => set({ selectedMachine: machine }),
  setSelectedView: (view) => set({ selectedView: view })
}));
```

#### 6.2 Additional View Implementations (Stub for now)

**1. History View** (`HistoryView.tsx`)
- Table of past predictions with timestamps
- Filter by machine, date range, prediction type
- Export to CSV functionality
- Charts showing prediction trends over time

**2. Reports View** (`ReportsView.tsx`)
- Generate PDF/Excel reports for machines
- Custom report builder (select metrics, date range)
- Scheduled reports (email delivery)
- Report templates (daily summary, weekly analysis)

**3. Dataset Manager View** (`DatasetManagerView.tsx`)
- List all training datasets (original + synthetic)
- Upload new datasets
- Dataset statistics and quality metrics
- Delete/archive old datasets

**4. Settings View** (`SettingsView.tsx`)
- API endpoint configuration
- Polling interval settings
- Theme customization (dark/light mode)
- Notification preferences
- User profile management

#### 6.3 Cross-View Navigation

**Quick Actions:**
- Predictions View → "Need Training Data?" → Opens GAN Wizard
- GAN Wizard → "Test Predictions" → Opens Predictions View
- Any View → "View History" → Opens History View
- Task Monitor (global) → Shows across all views in a persistent banner

**Files to Update:**
- `MLDashboardPage.tsx` - Global state management
- `NavigationPanel.tsx` - Badge indicators for active tasks
- All view components - Import shared state hook

**Deliverables:**
- ✅ Shared state across all views
- ✅ 4 additional view stubs (History, Reports, Datasets, Settings)
- ✅ Cross-view navigation with context preservation
- ✅ Persistent task monitor banner

---

## Technical Specifications

### TypeScript Interfaces

**File:** `client/src/modules/gan/types/gan.types.ts`

```typescript
export interface MachineProfile {
  machine_id: string;
  machine_type: string;
  manufacturer: string;
  model: string;
  sensors: Sensor[];
  operational_parameters: OperationalParams;
  rul_configuration?: RULConfig;
}

export interface Sensor {
  name: string;
  unit: string;
  type: 'numerical' | 'categorical';
  description?: string;
}

export interface OperationalParams {
  rated_power_kW: number;
  rated_speed_rpm: number;
  rated_voltage_V: number;
}

export interface RULConfig {
  initial_rul: number;
  failure_modes: string[];
  degradation_rate: 'slow' | 'medium' | 'fast';
}

export interface GANTask {
  task_id: string;
  task_type: 'seed_generation' | 'training' | 'generation';
  machine_id: string;
  status: 'pending' | 'running' | 'success' | 'failure';
  progress: number; // 0-100
  started_at: string;
  completed_at?: string;
  result?: any;
  error?: string;
}

export interface MachineWorkflowStatus {
  machine_id: string;
  status: 'draft' | 'seed_generated' | 'model_trained' | 'data_generated';
  seed_data?: {
    file_path: string;
    row_count: number;
    generated_at: string;
  };
  model?: {
    file_path: string;
    epochs: number;
    loss: number;
    trained_at: string;
  };
  synthetic_data?: {
    train_path: string;
    val_path: string;
    test_path: string;
    row_counts: { train: number; val: number; test: number };
    generated_at: string;
  };
}
```

### API Client

**File:** `client/src/modules/gan/api/ganApi.ts`

```typescript
import axios from 'axios';

const API_BASE = process.env.VITE_API_BASE_URL || 'http://localhost:8000';

export const ganApi = {
  // Templates
  getTemplates: () => 
    axios.get(`${API_BASE}/api/gan/templates`),
  
  getTemplate: (machineType: string) => 
    axios.get(`${API_BASE}/api/gan/templates/${machineType}`),
  
  // Profiles
  uploadProfile: (file: File) => {
    const formData = new FormData();
    formData.append('file', file);
    return axios.post(`${API_BASE}/api/gan/profiles/upload`, formData);
  },
  
  validateProfile: (profileData: object) => 
    axios.post(`${API_BASE}/api/gan/profiles/validate`, profileData),
  
  // Workflow
  generateSeed: (machineId: string, numSamples: number = 10000) => 
    axios.post(`${API_BASE}/api/gan/seed/generate`, { 
      machine_id: machineId, 
      num_samples: numSamples 
    }),
  
  getSeedStatus: (machineId: string) => 
    axios.get(`${API_BASE}/api/gan/seed/${machineId}/status`),
  
  trainTVAE: (machineId: string, epochs: number = 300) => 
    axios.post(`${API_BASE}/api/gan/train`, { 
      machine_id: machineId, 
      epochs 
    }),
  
  getTrainingStatus: (jobId: string) => 
    axios.get(`${API_BASE}/api/gan/train/${jobId}/status`),
  
  generateSynthetic: (machineId: string, numSamples: number = 50000) => 
    axios.post(`${API_BASE}/api/gan/generate`, { 
      machine_id: machineId, 
      num_samples: numSamples 
    }),
  
  getGenerationStatus: (jobId: string) => 
    axios.get(`${API_BASE}/api/gan/generate/${jobId}/status`),
  
  // Management
  getMachines: () => 
    axios.get(`${API_BASE}/api/gan/machines`),
  
  getMachineDetails: (machineId: string) => 
    axios.get(`${API_BASE}/api/gan/machines/${machineId}`),
  
  deleteMachine: (machineId: string) => 
    axios.delete(`${API_BASE}/api/gan/machines/${machineId}`),
  
  // Tasks
  getTaskStatus: (taskId: string) => 
    axios.get(`${API_BASE}/api/gan/tasks/${taskId}`),
};
```

---

## UI/UX Design Specifications

### Color Scheme (Match ML Dashboard)

```css
/* Primary Colors */
--primary-blue: #2563eb
--primary-purple: #7c3aed
--success-green: #10b981
--warning-orange: #f59e0b
--error-red: #ef4444

/* Status Badges */
--status-draft: #64748b (Gray)
--status-processing: #3b82f6 (Blue)
--status-completed: #10b981 (Green)
--status-failed: #ef4444 (Red)

/* Background */
--bg-primary: #0a0e27
--bg-secondary: #131829
--surface: rgba(255, 255, 255, 0.05)
```

### Typography

- **Headers:** Poppins (600 weight)
- **Body:** Inter (400 weight)
- **Monospace (logs):** Fira Code

### Layout (Side Navigation Panel Design)

**Default View (Drawer Closed):**
```
┌──────────────────────────────────────────────────────────────┐
│  [≡] ML Dashboard - Predictions          🔔 Tasks (2) 👤     │ ← Top Bar
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Machine Selector: [Motor 001 ▼]      Status: ● Online     │
│                                                              │
│  ┌────────────────────────────────────────────────────┐     │
│  │  Sensor Readings                                   │     │
│  │  • Temperature: 65°C  • Vibration: 2.3 mm/s       │     │
│  └────────────────────────────────────────────────────┘     │
│                                                              │
│  [Run Prediction]  [View History]                          │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

**Side Drawer Open:**
```
┌───────────────┬──────────────────────────────────────────────┐
│               │  [≡] ML Dashboard - New Machine Wizard      │
│ Navigation    │                                              │
│               ├──────────────────────────────────────────────┤
│ ◉ Predictions │  Machine Selector: [Motor 001 ▼]            │
│               │                                              │
│ ○ New Machine │  ┌──────────────────────────────────────┐   │
│   Wizard      │  │  GAN Workflow Stepper                 │   │
│   ⚡ Generate │  │  ① Profile ── ② Seed ── ③ Train ── ④  │   │
│   Data        │  └──────────────────────────────────────┘   │
│               │                                              │
│ ○ Prediction  │  Step 1: Upload Machine Profile            │
│   History     │  ┌────────────────┐                         │
│   📊 View     │  │ Drag & Drop    │                         │
│   Trends      │  │ JSON/YAML/     │  [Download Template]    │
│               │  │ Excel File     │                         │
│ ○ Reports     │  └────────────────┘                         │
│   📄 Generate │                                              │
│   Analytics   │  [Validate Profile]  [Continue]            │
│               │                                              │
│ ○ Dataset     │  ⚙️ Active Tasks:                           │
│   Manager     │  • Training Model: 45% ━━━━━━━━░░░░        │
│   💾 Manage   │                                              │
│   Data        │                                              │
│               │                                              │
│ ○ Settings    │                                              │
│   ⚙️ Config   │                                              │
│               │                                              │
└───────────────┴──────────────────────────────────────────────┘
```

**Key Features:**
- 🍔 **Hamburger Menu** (≡) to toggle drawer
- 📱 **Responsive**: Drawer overlays on mobile, can be persistent on desktop
- 🔔 **Global Task Banner**: Shows active background tasks across all views
- 🎯 **Context Preservation**: Selected machine carries across views
- 🎨 **Visual Hierarchy**: Icons + labels + descriptions in navigation

---

## Testing Plan

### Unit Tests (Vitest)
- Profile validation logic
- API client error handling
- Status badge rendering
- Chart data transformations

### Integration Tests (Playwright)
- Upload profile flow end-to-end
- Workflow stepper progression
- Task cancellation
- File downloads

### Manual Testing Checklist
- [ ] Upload valid JSON profile → Success
- [ ] Upload invalid profile → Error messages
- [ ] Generate seed data → Task completes
- [ ] Train TVAE → Loss curve displays
- [ ] Generate synthetic → Files downloadable
- [ ] Cancel running task → Task stops
- [ ] Delete machine → Confirmation dialog
- [ ] Navigate between ML & GAN → State persists

---

## Deployment Checklist

### Backend Verification
- [ ] GAN API routes accessible (`/api/gan/*`)
- [ ] Celery workers running
- [ ] Redis connected
- [ ] Database tables exist (`gan_profiles`, `gan_training_jobs`)
- [ ] GANManager service healthy

### Frontend Deployment
- [ ] Build passes without errors
- [ ] Environment variables set (`VITE_API_BASE_URL`)
- [ ] Bundle size acceptable (<2MB for GAN module)
- [ ] All routes accessible
- [ ] CORS configured correctly

---

## Success Metrics

### Performance
- Page load time: <2 seconds
- API response time: <500ms (95th percentile)
- Task status polling: 2-second intervals
- Chart rendering: <100ms

### User Experience
- Profile upload success rate: >95%
- Workflow completion rate: >80%
- Task cancellation success: 100%
- User satisfaction score: >4.5/5

### Technical
- Code coverage: >70%
- Zero console errors
- Accessibility score (Lighthouse): >90
- Mobile responsiveness: 100%

---

## Timeline Estimate

| Phase | Hours | Tasks |
|-------|-------|-------|
| 3.7.6.1 - Side Navigation Panel | 3-4 | Drawer, NavigationPanel, 6 menu items, view routing |
| 3.7.6.2 - Profile Management | 3-4 | Upload wizard, validation, template downloads |
| 3.7.6.3 - Workflow Stepper | 4-5 | 4-step wizard, API integration, progress tracking |
| 3.7.6.4 - Task Monitoring | 2-3 | Real-time updates, progress bars, global banner |
| 3.7.6.5 - Visualizations | 2-3 | Charts for seed/training/quality metrics |
| 3.7.6.6 - View Integration | 2-3 | Shared state, stub views, cross-navigation |
| **Testing & Refinement** | 2-3 | E2E tests, responsive design, bug fixes |
| **Documentation** | 1-2 | User guide, navigation docs |
| **TOTAL** | **19-27 hours** | **2-3 days** |

---

## Next Steps

### Immediate Actions (Start Here)
1. **Review this plan** - Confirm scope and approach
2. **Create Phase 3.7.6.1** - Scaffold the core structure
3. **Build MachineProfileUpload** - Most critical component
4. **Implement WorkflowStepper** - Core user journey
5. **Add TaskMonitor** - Real-time feedback
6. **Testing & Refinement** - Ensure quality

### Architecture Decisions (Confirmed by User)
1. ✅ **Navigation Structure:** Collapsible side navigation panel
   - Hamburger menu (≡) to toggle drawer
   - 6 navigation options: Predictions, GAN Wizard, History, Reports, Datasets, Settings
   - Scalable design for future features
   - Can be persistent (always visible) or temporary (overlay)
   
2. ✅ **GAN Integration:** "New Machine Wizard" as a navigation option
   - Accessible from side panel alongside other tools
   - Full-screen view when selected (not cramped in a tab)
   - Dedicated workspace for the 4-step workflow
   
3. ✅ **Workflow Type:** Single-machine workflow only (no batch operations)
   - User works with one machine at a time
   - Simpler UX, clearer focus
   - Machine selection persists across all views
   
4. ✅ **Development Approach:** Implement first, refine based on feedback
   - Build core GAN workflow first
   - Stub out other views (History, Reports, etc.) for future development
   - User will provide feature updates after GAN completion

---

## Appendix: Backend API Reference

### GET /api/gan/templates
**Response:**
```json
[
  {
    "machine_type": "motor",
    "display_name": "Electric Motor",
    "sensor_count": 8,
    "example_available": true
  },
  ...
]
```

### POST /api/gan/profiles/upload
**Request:**
```
FormData: { file: motor_profile.json }
```

**Response:**
```json
{
  "profile_id": "uuid-1234",
  "machine_id": "motor_001",
  "status": "validated",
  "validation_errors": [],
  "next_step": "generate_seed"
}
```

### POST /api/gan/seed/generate
**Request:**
```json
{
  "machine_id": "motor_001",
  "num_samples": 10000
}
```

**Response:**
```json
{
  "task_id": "celery-task-xyz",
  "status": "pending",
  "estimated_duration_seconds": 30
}
```

### GET /api/gan/seed/{machine_id}/status
**Response:**
```json
{
  "machine_id": "motor_001",
  "status": "completed",
  "file_path": "/seed_data/motor_001_temporal_seed.csv",
  "row_count": 10000,
  "file_size_mb": 2.3,
  "generated_at": "2025-12-16T10:30:00Z"
}
```

---

---

## Design Comparison: Before vs. After

### ❌ Old Approach (Separate Route - Rejected)
```
App Routes:
  /dashboard → Home
  /ml → ML Dashboard (Predictions only)
  /gan → GAN Dashboard (Separate page)
  /settings → Settings

Problems:
- Users navigate away from ML context
- Duplicate machine selectors
- State not shared between pages
- Fragmented experience
```

### ❌ Tab Approach (Initial Idea - Rejected)
```
ML Dashboard:
  [Predictions] [Data Generation]
  
Problems:
- Limited space for multiple features
- Tabs can get crowded (6+ tabs?)
- Not scalable for many features
- Mobile UX issues
```

### ✅ Final Approach (Side Navigation Panel - Approved)
```
ML Dashboard with Side Drawer:
  [≡] Menu
    ├─ Predictions
    ├─ New Machine Wizard (GAN)
    ├─ Prediction History
    ├─ Reports
    ├─ Dataset Manager
    └─ Settings

Benefits:
✅ Scalable (add unlimited features)
✅ Professional UX (common in enterprise apps)
✅ Mobile-friendly (drawer collapses)
✅ Shared context (machine selection, tasks)
✅ Clear visual hierarchy
✅ Full-screen workspace for each feature
```

---

## Navigation Options Summary

| Option | Label | Icon | Description | Status |
|--------|-------|------|-------------|--------|
| 1 | Predictions | 📈 | Run ML predictions on machines | ✅ Existing |
| 2 | New Machine Wizard | ⚡ | Generate synthetic training data (GAN) | 🚧 To Build |
| 3 | Prediction History | 📊 | View past predictions and trends | 📝 Stub |
| 4 | Reports | 📄 | Generate analysis reports | 📝 Stub |
| 5 | Dataset Manager | 💾 | Manage training datasets | 📝 Stub |
| 6 | Settings | ⚙️ | Configure dashboard preferences | 📝 Stub |

**Implementation Priority:**
1. Phase 1: Build side navigation structure + GAN Wizard (full implementation)
2. Phase 2: User feedback and refinements
3. Phase 3: Implement other views (History, Reports, etc.) based on user needs

---

**END OF IMPLEMENTATION PLAN**

Ready to proceed with Phase 3.7.6.1 (Side Navigation Panel)? 🚀
