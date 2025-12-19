/**
 * GAN Types - Phase 3.7.6.2
 * 
 * TypeScript interfaces for GAN workflow:
 * - Machine profiles
 * - Sensors and operational parameters
 * - Task monitoring
 * - Workflow status tracking
 */

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
  rated_power_kW?: number;
  rated_speed_rpm?: number;
  rated_voltage_V?: number;
  rated_current_A?: number;
  rated_torque_Nm?: number;
  rated_temperature_C?: number;
  [key: string]: number | undefined; // Allow additional parameters
}

export interface RULConfig {
  initial_rul: number;
  failure_modes: string[];
  degradation_rate: 'slow' | 'medium' | 'fast';
}

export type MachineStatus =
  | 'not_started'
  | 'seed_generated'
  | 'training'
  | 'trained'
  | 'synthetic_generated'
  | 'failed';

export interface MachineWorkflowStatus {
  machine_id: string;
  status: MachineStatus;
  has_metadata: boolean;
  has_seed_data: boolean;
  has_trained_model: boolean;
  has_synthetic_data: boolean;
  can_generate_seed: boolean;
  can_train_model: boolean;
  can_generate_synthetic: boolean;
  last_updated?: string;
}

export interface MachineDetails {
  machine_id: string;
  machine_type: string;
  manufacturer: string;
  model: string;
  num_sensors: number;
  degradation_states: number;
  status: MachineWorkflowStatus;
  created_at: string;
  updated_at: string;
}

export interface MachineTemplate {
  machine_type: string;
  display_name: string;
  sensor_count: number;
  example_available: boolean;
}

export interface ValidationError {
  field: string;
  message: string;
  line?: number;
  severity?: 'info' | 'warning' | 'error';
}

export interface ProfileUploadResponse {
  profile_id: string;
  machine_id: string;
  status: 'validated' | 'invalid';
  validation_errors: ValidationError[];
  next_step?: string;
  // Optional fields used by some UI flows
  warnings?: ValidationError[];
  profile?: any;
  // Backend fields (when staged via /profiles/upload)
  success?: boolean;
  message?: string;
  validation_required?: boolean;
}

export interface SeedGenerationRequest {
  samples: number;
}

export interface SeedGenerationResponse {
  machine_id: string;
  samples_generated: number;
  file_path: string;
  file_size_mb: number;
  generation_time_seconds: number;
  timestamp: string;
}

export interface TrainingRequest {
  epochs: number;
  batch_size: number;
}

export interface TrainingResponse {
  success: boolean;
  machine_id: string;
  task_id: string;
  epochs: number;
  estimated_time_minutes: number;
  websocket_url: string;
  message: string;
}

export interface GenerationRequest {
  train_samples: number;
  val_samples: number;
  test_samples: number;
}

export interface GenerationResponse {
  machine_id: string;
  train_samples: number;
  val_samples: number;
  test_samples: number;
  train_file: string;
  val_file: string;
  test_file: string;
  generation_time_seconds: number;
  timestamp: string;
}

export interface MachineListResponse {
  total: number;
  machines: string[];
  machine_details?: MachineDetails[];
}

export interface TaskProgress {
  current: number;
  total: number;
  progress_percent: number;
  epoch?: number;
  loss?: number;
  stage?: string;
  message?: string;
}

export interface TaskStatusResponse {
  task_id: string;
  status: 'PENDING' | 'STARTED' | 'PROGRESS' | 'SUCCESS' | 'FAILURE' | 'RETRY' | 'REVOKED';
  machine_id?: string | null;
  progress?: TaskProgress | null;
  result?: any;
  error?: string | null;
  logs?: string | null;
  started_at?: string | null;
  completed_at?: string | null;
}

export interface ValidateDataQualityResponse {
  machine_id: string;
  valid: boolean;
  num_samples: number;
  num_features: number;
  null_values: number;
  quality_score: number;
  message: string;
}

export interface FeatureStats {
  mean: number;
  std: number;
  min: number;
  max: number;
}

export interface FeatureDistributionSummary {
  feature: string;
  bin_edges: number[];
  seed_counts: number[];
  synthetic_counts: number[];
  seed_stats: FeatureStats;
  synthetic_stats: FeatureStats;
}

export interface SeedSeriesPoint {
  timestamp: string;
  rul: number;
  [key: string]: number | string;
}

export interface VisualizationSummaryResponse {
  machine_id: string;
  seed_series: {
    points: SeedSeriesPoint[];
    sensor_keys: string[];
  };
  distributions: FeatureDistributionSummary[];
  correlation: {
    features: string[];
    matrix: number[][];
  };
}

// Machine types enum
export const MACHINE_TYPES = [
  { value: 'motor', label: 'Electric Motor' },
  { value: 'pump', label: 'Centrifugal Pump' },
  { value: 'compressor', label: 'Air Compressor' },
  { value: 'cnc', label: 'CNC Machine' },
  { value: 'bearing', label: 'Bearing' },
  { value: 'gearbox', label: 'Gearbox' },
  { value: 'turbine', label: 'Turbine' },
  { value: 'generator', label: 'Generator' },
  { value: 'hvac', label: 'HVAC System' },
  { value: 'conveyor', label: 'Conveyor Belt' },
  { value: 'robot', label: 'Industrial Robot' },
] as const;

export type MachineType = typeof MACHINE_TYPES[number]['value'];
