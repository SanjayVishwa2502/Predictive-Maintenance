# Predictive Maintenance System

A comprehensive, production-grade predictive maintenance platform that leverages synthetic data generation, machine learning models, and real-time monitoring to predict equipment failures and optimize maintenance schedules for industrial machinery.

---

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Core Components](#core-components)
- [Technology Stack](#technology-stack)
- [Installation and Setup](#installation-and-setup)
- [Usage](#usage)
- [Project Status](#project-status)
- [Documentation](#documentation)

---

## Overview

The Predictive Maintenance System is designed to monitor industrial equipment health, predict failures, and provide actionable insights through an integrated web-based dashboard. The system combines four core AI workflows:

1. **Generative Adversarial Networks (GAN)** - Synthetic sensor data generation
2. **Machine Learning (ML)** - Real-time failure prediction and classification
3. **Large Language Models (LLM)** - Natural language explanations and reporting
4. **Visual Language Models (VLM)** - Visual diagnostics and anomaly detection

The system currently supports 26+ industrial machine profiles across multiple categories including motors, pumps, compressors, CNC machines, hydraulic systems, and cooling towers.

---

## System Architecture

### High-Level Architecture

The system is structured as a multi-tier application with the following layers:

```
┌─────────────────────────────────────────────────────────────┐
│                     Frontend Dashboard                       │
│              (React + Material-UI + Vite)                   │
└────────────────────┬────────────────────────────────────────┘
                     │ REST API
┌────────────────────┴────────────────────────────────────────┐
│                    Backend Services                          │
│              (FastAPI + Celery + Redis)                     │
├──────────────────────────────────────────────────────────────┤
│  • ML Manager      • GAN Manager      • Task Orchestration  │
│  • Authentication  • Rate Limiting    • WebSocket Support   │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────┴────────────────────────────────────────┐
│                  Data & Model Layers                         │
├──────────────────────────────────────────────────────────────┤
│  • PostgreSQL DB   • Redis Cache      • MLflow Tracking     │
│  • Model Artifacts • Synthetic Data   • Training Logs       │
└──────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Machine Profile Creation**: Users define machine specifications through the New Machine Wizard
2. **Synthetic Data Generation**: TVAE models generate physics-based synthetic sensor data
3. **Model Training**: AutoGluon trains classification, regression, anomaly detection, and time-series models
4. **Real-Time Monitoring**: Dashboard displays live predictions and equipment health status
5. **Task Management**: Background workers process training, prediction, and generation tasks asynchronously

---

## Core Components

### 1. GAN Module (`GAN/`)

The GAN module handles synthetic sensor data generation using Table Variational Autoencoder (TVAE) models.

**Key Features:**
- Machine profile management with comprehensive validation
- Physics-based seed data generation
- TVAE model training with quality assessment
- Synthetic dataset generation (train/validation/test splits)
- Quality metrics and distribution comparison

**Key Files:**
- `services/gan_training_service.py` - Core TVAE training orchestration
- `services/profile_validation_service.py` - Machine profile validation
- `validate_new_machine.py` - End-to-end machine validation pipeline
- `metadata/` - Machine profile metadata storage
- `data/synthetic/` - Generated synthetic datasets

### 2. ML Models Module (`ml_models/`)

The ML module contains trained prediction models and training infrastructure.

**Model Types:**
- **Classification**: Equipment failure type prediction
- **Regression**: Remaining Useful Life (RUL) estimation
- **Anomaly Detection**: Outlier and anomaly identification
- **Time-Series**: Sequential pattern analysis and forecasting

**Key Features:**
- AutoGluon-based automated machine learning
- Per-machine model customization
- Model versioning and artifact management
- Performance metrics and validation reports
- Model inventory and lifecycle management

**Directory Structure:**
```
ml_models/
├── models/
│   ├── classification/<machine_id>/    # Classification models
│   ├── regression/<machine_id>/        # Regression models
│   ├── anomaly/<machine_id>/          # Anomaly detection models
│   └── timeseries/<machine_id>/       # Time-series models
├── reports/                           # Training and validation reports
├── scripts/                           # Training and evaluation scripts
└── notebooks/                         # Jupyter notebooks for analysis
```

### 3. Frontend Dashboard (`frontend/`)

A modern web-based dashboard built with React and Material-UI.

**Views:**
- **Predictions**: Real-time equipment health monitoring and failure prediction
- **New Machine Wizard**: Guided workflow for adding new machine profiles
- **Model Training**: Interface for training ML models with progress tracking
- **Manage Models**: Model inventory, deletion, and retraining controls
- **Tasks**: Background task monitoring with status tracking
- **Dataset Downloads**: Access to synthetic and processed datasets
- **Continue Workflow**: Resume interrupted GAN profile workflows

**Key Features:**
- Real-time task status updates via polling
- Machine filtering and selection
- Workflow state persistence across browser sessions
- Responsive design with dark theme
- Interactive charts and visualizations

**Technical Stack:**
- **Frontend Framework**: React 18 with TypeScript
- **UI Library**: Material-UI (MUI) v6
- **Build Tool**: Vite 7
- **State Management**: React Context API
- **HTTP Client**: Axios

### 4. Backend API (`frontend/server/`)

FastAPI-based REST API with asynchronous task processing.

**Endpoints:**

**ML Endpoints** (`/api/ml/`):
- `GET /machines` - List all available machines
- `GET /machines/{machine_id}/status` - Get machine status
- `POST /predict/classification` - Run failure classification
- `POST /predict/rul` - Estimate remaining useful life
- `GET /models/inventory` - Get model inventory
- `POST /models/train` - Initiate model training
- `DELETE /models/{model_id}` - Delete specific model
- `DELETE /machines/{machine_id}/models` - Delete all models for machine

**GAN Endpoints** (`/api/gan/`):
- `POST /upload-profile` - Upload machine profile
- `POST /validate-profile` - Validate machine configuration
- `POST /train` - Start TVAE model training
- `POST /generate` - Generate synthetic data
- `GET /tasks/{task_id}` - Get task status
- `DELETE /tasks/{task_id}` - Cancel running task
- `GET /workflow/continue` - Retrieve workflow state
- `PUT /workflow/continue` - Store workflow state
- `DELETE /workflow/continue` - Clear workflow state

**Task Management** (`/api/tasks/`):
- `GET /` - List all tasks
- `GET /{task_id}` - Get task details
- `DELETE /{task_id}` - Cancel task

**Key Features:**
- JWT-based authentication (ready for deployment)
- Rate limiting and request throttling
- CORS configuration for development and production
- Comprehensive error handling and logging
- Background task orchestration via Celery
- Redis-based caching and session management

### 5. LLM Module (`LLM/`)

Integration layer for large language models to provide natural language explanations.

**Features:**
- RAG (Retrieval-Augmented Generation) knowledge base
- Context-aware explanation generation
- Integration with ML prediction pipeline
- Automated report generation

**Status**: Integration framework complete; model deployment pending

### 6. Data Ingestion Module (`data_ingestion/`)

Pipeline for ingesting, validating, and refining real-world sensor data.

**Workflow:**
1. Upload dataset in supported formats (CSV, Excel, Parquet, JSON)
2. Validate and clean data
3. Map columns to machine profile sensors
4. Merge with seed data (optional)
5. Refine TVAE models with real data
6. Generate augmented datasets
7. Compare quality metrics

**Directory Structure:**
```
data_ingestion/
├── raw/                  # Original uploaded files
├── processed/            # Cleaned and transformed data
├── merged/               # Seed + real data combined
├── refined_models/       # TVAE models refined on real data
├── augmented/            # Augmented datasets
├── reports/              # Quality comparison reports
└── scripts/              # Ingestion and refinement utilities
```

---

## Technology Stack

### Backend

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **API Framework** | FastAPI 0.115+ | REST API and WebSocket support |
| **Task Queue** | Celery 5.4+ | Asynchronous background processing |
| **Message Broker** | Redis 7.4+ | Task queue and caching |
| **Database** | PostgreSQL 18 | User data and configuration storage |
| **ML Framework** | AutoGluon 1.4+ | Automated machine learning |
| **GAN Framework** | CTGAN/TVAE (SDV 1.16+) | Synthetic data generation |
| **Experiment Tracking** | MLflow 2.18+ | Model versioning and tracking |
| **Python Runtime** | Python 3.10+ | Core runtime environment |

### Frontend

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Framework** | React 18 | UI framework |
| **Language** | TypeScript 5+ | Type-safe development |
| **UI Library** | Material-UI 6 | Component library |
| **Build Tool** | Vite 7 | Fast development and builds |
| **HTTP Client** | Axios | API communication |
| **Charts** | Recharts | Data visualization |

### Infrastructure

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **OS** | Windows Server / Linux | Production hosting |
| **Virtualization** | Docker (optional) | Containerized deployment |
| **Process Management** | PM2 / Supervisor | Service management |
| **Reverse Proxy** | Nginx (optional) | Load balancing and SSL |

---

## Installation and Setup

### Prerequisites

- **Python**: Version 3.10 or higher
- **Node.js**: Version 18 or higher
- **PostgreSQL**: Version 16 or higher (service running on port 5433)
- **Redis**: Version 7.4 or higher (service running on port 6379)
- **Git**: For version control

### Installation Steps

#### 1. Clone Repository

```bash
git clone <repository-url>
cd "Predictive Maintenance"
```

#### 2. Backend Setup

```powershell
# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install Python dependencies
pip install -r frontend/server/requirements.txt
pip install -r GAN/requirements.txt
pip install -r ml_models/requirements.txt
```

#### 3. Database Configuration

```powershell
# Create database and tables
cd frontend/server
python create_database.py
```

#### 4. Environment Configuration

Create `frontend/server/.env` file:

```env
DATABASE_URL=postgresql://postgres:<your_password>@localhost:5433/predictive_maintenance
REDIS_URL=redis://localhost:6379/0
SECRET_KEY=<your-secret-key>
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0
```

#### 5. Frontend Setup

```powershell
cd frontend/client
npm install
```

### Running the Application

#### Quick Start (Recommended)

Use the provided batch scripts to start all services:

**Windows:**
```powershell
# Start all services
.\start_dashboard.ps1

# Stop all services
.\stop_dashboard.ps1
```

This will launch separate terminal windows for:
1. **Backend API** (FastAPI on port 8000)
2. **Celery Worker(s)** (Background task processor)
3. **Frontend** (React development server on port 5173)

#### Manual Start (Advanced)

**Terminal 1 - Backend API:**
```powershell
cd frontend/server
.\venv\Scripts\Activate.ps1
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 - Celery Worker:**
```powershell
cd frontend/server
.\venv\Scripts\Activate.ps1
celery -A celery_app worker --loglevel=info --pool=solo
```

**Terminal 3 - Frontend:**
```powershell
cd frontend/client
npm run dev
```

### Accessing the Application

| Service | URL | Description |
|---------|-----|-------------|
| **Dashboard** | http://localhost:5173 | Main application interface |
| **API Documentation** | http://localhost:8000/docs | Interactive API documentation (Swagger) |
| **API Health** | http://localhost:8000/health | System health check endpoint |

---

## Usage

### Adding a New Machine Profile

1. Navigate to **New Machine Wizard** in the dashboard
2. Select machine type and configuration template
3. Upload machine profile JSON or fill in the form
4. Validate profile (checks sensor configuration and physics parameters)
5. Generate seed data (physics-based initial dataset)
6. Train TVAE model (typically 5-10 minutes)
7. Generate synthetic datasets (train/validation/test splits)
8. Validate quality metrics (should achieve >0.85 quality score)
9. Deploy to production (machine appears in predictions view)

### Training ML Models

1. Navigate to **Model Training** view
2. Select machine from dropdown (shows machines without trained models)
3. Select model types to train (Classification, Regression, Anomaly, Time-Series)
4. Click **Start Training**
5. Monitor progress in real-time
6. View training logs (collapsible section)
7. Check training reports in **Manage Models** after completion

### Monitoring Tasks

1. Navigate to **Tasks** view to see all background jobs
2. Click on any running task to view its page (Training or GAN)
3. Cancel tasks if needed
4. Review completed task history

### Managing Models

1. Navigate to **Manage Models** view
2. View model inventory with details (type, size, performance metrics)
3. Retrain models individually or all at once per machine
4. Delete models to free disk space
5. Filter machines by training status

### Continuing GAN Workflows

If a GAN workflow is interrupted:
1. Click **Continue Workflow** in navigation panel
2. System automatically resumes at the last saved step
3. Complete remaining workflow steps
4. Workflow state persists across browser sessions

---

## Project Status

### Current Implementation Status

| Component | Status | Completion |
|-----------|--------|------------|
| **GAN Synthetic Data Generation** | ✅ Complete | 100% |
| **Machine Profile Management** | ✅ Complete | 100% |
| **Classification Models** | ✅ Complete | 100% |
| **Regression Models** | ✅ Complete | 100% |
| **Anomaly Detection Models** | ✅ Complete | 100% |
| **Time-Series Forecasting Models** | ✅ Complete | 100% |
| **Dashboard & Frontend** | ✅ Complete | 100% |
| **Backend API & Services** | ✅ Complete | 100% |
| **Task Management System** | ✅ Complete | 100% |
| **Model Training Pipeline** | ✅ Complete | 100% |
| **LLM Integration Framework** | ✅ Complete | 100% |
| **Data Ingestion Module** | ⏳ Planned | 0% |

### Known Limitations

1. **Data Ingestion**: Real dataset upload and refinement workflow not yet implemented
2. **Real Data Fine-Tuning**: Models currently trained on 100% synthetic data
3. **LLM Model Deployment**: Integration framework complete but model not deployed
4. **VLM Integration**: Planned for future release
5. **Authentication**: JWT infrastructure present but not enforced in development mode

### Performance Metrics

**Classification Models (10 machines trained):**
- Average F1 Score: 0.77
- Average Accuracy: 94.9%
- Average Model Size: 237 MB

**Regression Models (10 machines trained):**
- Average R² Score: 0.75
- Average MAE: 12.3 cycles

**GAN Quality (21 machines):**
- Average Quality Score: 0.91+
- Datasets Generated: 50,000 rows per machine

---

## Documentation

### Core Documentation

- **[STARTUP_GUIDE.md](STARTUP_GUIDE.md)** - Complete startup and shutdown procedures
- **[PROJECT_STATUS_SUMMARY.md](PROJECT_STATUS_SUMMARY.md)** - Detailed project status and blockers
- **[FUTURE_SCOPE_ROADMAP.md](FUTURE_SCOPE_ROADMAP.md)** - Planned enhancements and roadmap

### Component Documentation

**GAN Module:**
- `GAN/WORKFLOW_TEST_NEW_MACHINE.md` - Step-by-step new machine workflow
- `GAN/GAN_DASHBOARD_IMPLEMENTATION_PLAN.md` - GAN integration architecture

**ML Module:**
- `ML_PART_OVERVIEW.md` - ML component architecture overview
- `ML_TRAINING_WORKFLOW.md` - Training pipeline documentation
- `ml_models/PHASE_2_ML_DETAILED_APPROACH.md` - Detailed ML implementation guide

**Frontend:**
- `frontend/DASHBOARD_ARCHITECTURE_AND_WORKFLOW.md` - Dashboard architecture
- `frontend/DASHBOARD_DETAILED_SPECIFICATION.md` - Component specifications
- `frontend/FRONTEND_REBUILD_GUIDE.md` - Development and build guide

**Data Ingestion:**
- `data_ingestion/README.md` - Data ingestion pipeline overview
- `PHASE_3.7.6_EXISTING_DATASET_REFINEMENT.md` - Dataset refinement workflow

### Phase Reports

Completion reports for each development phase are available in the root directory and component folders:
- `PHASE_3.7.2.3_COMPLETION_REPORT.md` - Dashboard UI implementation
- `PHASE_3.7.6.2_COMPLETION_REPORT.md` - Model management features
- `GAN_DASHBOARD_COMPLETION_STATUS.md` - GAN dashboard integration

### API Documentation

Interactive API documentation is available at http://localhost:8000/docs when the backend is running.

---

## System Requirements

### Minimum Requirements

- **CPU**: 4 cores (8 recommended for training)
- **RAM**: 8 GB (16 GB recommended)
- **Disk**: 50 GB free space (100 GB+ for extensive model training)
- **GPU**: Optional (CPU training supported but slower)
- **Network**: Stable internet connection for package installation

### Recommended Production Configuration

- **CPU**: 8+ cores
- **RAM**: 32 GB
- **Disk**: 500 GB SSD
- **GPU**: NVIDIA GPU with 8+ GB VRAM (for LLM/VLM deployment)
- **OS**: Ubuntu 22.04 LTS or Windows Server 2022

---

## Maintenance and Operations

### Backup Procedures

**Critical Data:**
- PostgreSQL database (user data, configurations)
- Machine profiles (`GAN/metadata/`)
- Trained models (`ml_models/models/`)
- Synthetic datasets (`GAN/data/synthetic/`)

**Backup Commands:**
```powershell
# Database backup
pg_dump -U postgres -h localhost -p 5433 predictive_maintenance > backup.sql

# Model and data backup (compress)
Compress-Archive -Path "ml_models/models" -DestinationPath "backup_models.zip"
Compress-Archive -Path "GAN/data/synthetic" -DestinationPath "backup_data.zip"
```

### Log Management

**Log Locations:**
- Backend API: `frontend/server/app.log`
- Celery Worker: Terminal output or system logs
- Training Logs: `ml_models/outputs/`
- GAN Logs: `GAN/outputs/`

### Monitoring

**Health Checks:**
- Backend: `GET http://localhost:8000/health`
- Redis: `redis-cli ping`
- PostgreSQL: `pg_isready -h localhost -p 5433`
- Celery: Access Flower at http://localhost:5555

---

## Troubleshooting

### Common Issues

**Backend fails to start:**
- Check PostgreSQL service is running
- Check Redis service is running
- Verify `.env` file configuration
- Check port 8000 is not in use

**Celery worker not processing tasks:**
- Verify Redis connection
- Check Celery worker is running
- Use Flower to inspect worker status
- Check task logs for errors

**Frontend build fails:**
- Clear node_modules: `rm -rf node_modules && npm install`
- Check Node.js version (18+)
- Clear Vite cache: `rm -rf node_modules/.vite`

**Model training fails:**
- Verify synthetic data exists for machine
- Check disk space availability
- Review training logs in `ml_models/outputs/`
- Ensure Python dependencies are installed

**GAN training fails:**
- Validate machine profile format
- Check seed data generation
- Review GAN logs in `GAN/outputs/`
- Verify SDV library version compatibility

---

## Contributing

This is a production system under active development. For modifications:

1. Follow existing code structure and naming conventions
2. Add appropriate error handling and logging
3. Update documentation for new features
4. Test changes thoroughly in development environment
5. Submit detailed change descriptions

---

## License

This project is proprietary software developed for industrial predictive maintenance applications.

---

## Contact and Support

For technical support or questions regarding system operation, refer to the documentation files in the repository or contact the development team.

---

**Last Updated**: December 23, 2025  
**Version**: 1.0  
**Status**: Production Ready (Core System Complete)
