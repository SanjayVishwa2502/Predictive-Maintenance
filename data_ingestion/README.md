# Data Ingestion Module
**Purpose:** Ingest, validate, and refine existing machine datasets for TVAE model improvement

## Directory Structure

```
data_ingestion/
├── raw/                    # Original uploaded files (unmodified)
├── processed/              # Cleaned and transformed data
├── merged/                 # Seed data + Real data combined
├── refined_models/         # TVAE models refined on real data
├── augmented/              # Augmented datasets (35K/7.5K/7.5K)
├── reports/                # Quality comparison reports
└── scripts/                # Ingestion and refinement scripts
    └── utils/              # Helper modules
```

## Workflow

1. **Upload Dataset** → `raw/{machine_id}/`
2. **Validate & Clean** → `processed/{machine_id}/`
3. **Map Columns** → Match dataset columns to machine profile sensors
4. **Merge with Seed Data** (optional) → `merged/{machine_id}/`
5. **Refine TVAE Model** → `refined_models/{machine_id}/`
6. **Generate Augmented Data** → `augmented/{machine_id}/`
7. **Compare Quality** → `reports/{machine_id}_comparison.json`

## Supported Formats

- CSV (.csv)
- Excel (.xlsx, .xls)
- Parquet (.parquet)
- JSON (.json)
- SCADA exports (custom parsers)

## Usage

See `PHASE_3.7.6_EXISTING_DATASET_REFINEMENT.md` for complete implementation guide.

## Scripts (To Be Implemented)

### Phase 3.7.6.1: Data Ingestion
- `scripts/utils/format_parsers.py` - Parse CSV/Excel/Parquet/JSON
- `scripts/utils/data_cleaners.py` - Validate and clean data
- `scripts/ingest_dataset.py` - Main ingestion script

### Phase 3.7.6.2: Column Mapping
- `scripts/utils/mappers.py` - Fuzzy matching for column mapping
- `scripts/transform_data.py` - Apply transformations

### Phase 3.7.6.3: TVAE Refinement
- `scripts/merge_datasets.py` - Merge seed + real data
- `scripts/refine_tvae.py` - Transfer learning implementation
- `scripts/compare_distributions.py` - Quality comparison

### Phase 3.7.6.4: Data Augmentation
- `scripts/augment_data.py` - Generate large datasets from refined models

## Status

**Phase 3.7.6:** Not Started (Planned)  
**Estimated Completion:** 2-3 weeks after start

See project timeline in `PHASE_3.7.6_EXISTING_DATASET_REFINEMENT.md`
