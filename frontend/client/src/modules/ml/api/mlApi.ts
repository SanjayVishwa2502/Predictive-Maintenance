/**
 * ML API Client
 * 
 * API client for ML predictions and sensor simulation controls
 */

import axios from 'axios';
import type { AxiosResponse } from 'axios';

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

// ============================================================================
// Types
// ============================================================================

export interface MachineStatusResponse {
  machine_id: string;
  is_running: boolean;
  latest_sensors: Record<string, number>;
  last_update: string;
  sensor_count: number;
  data_stamp?: string;
  run_id?: string | null;
}

export interface SnapshotItem {
  machine_id: string;
  data_stamp: string;
  sensor_data: Record<string, number>;
  run_id?: string | null;
  has_run?: boolean;
}

export interface MachineSnapshotsResponse {
  machine_id: string;
  snapshots: SnapshotItem[];
}

export interface MachineRunsResponse {
  machine_id: string;
  runs: Array<{
    run_id: string;
    machine_id: string;
    data_stamp: string;
    created_at: string;
    has_details: boolean;
  }>;
}

export interface RunDetailsResponse {
  run_id: string;
  machine_id: string;
  data_stamp: string;
  created_at: string;
  sensor_data: Record<string, number>;
  predictions: Record<string, any>;
  llm: Record<string, any>;
}

export interface AutoRunStatusResponse {
  machine_id: string;
  running: boolean;
  interval_seconds?: number;
  last_run_at?: string | null;
}

export interface SimulationStatusResponse {
  machine_id: string;
  is_running: boolean;
  current_row: number;
  total_rows: number;
  progress_percent: number;
  sensor_count: number;
  started_at: string;
  last_update: string;
  cycle_count: number;
}

export interface SimulationControlResponse {
  success: boolean;
  message: string;
  simulation?: SimulationStatusResponse;
  count?: number;
}

export interface ActiveSimulationsResponse {
  active_simulations: string[];
  count: number;
}

// ============================================================================
// API Functions
// ============================================================================

/**
 * Get machine status including latest sensor readings
 */
export async function getMachineStatus(machineId: string): Promise<MachineStatusResponse> {
  const response: AxiosResponse<MachineStatusResponse> = await axios.get(
    `${API_BASE}/api/ml/machines/${encodeURIComponent(machineId)}/status`
  );
  return response.data;
}

export async function getMachineSnapshots(machineId: string, limit = 200): Promise<MachineSnapshotsResponse> {
  const response: AxiosResponse<MachineSnapshotsResponse> = await axios.get(
    `${API_BASE}/api/ml/machines/${encodeURIComponent(machineId)}/snapshots`,
    { params: { limit } }
  );
  return response.data;
}

export async function getMachineRuns(machineId: string, limit = 200): Promise<MachineRunsResponse> {
  const response: AxiosResponse<MachineRunsResponse> = await axios.get(
    `${API_BASE}/api/ml/machines/${encodeURIComponent(machineId)}/runs`,
    { params: { limit } }
  );
  return response.data;
}

export async function getRunDetails(runId: string): Promise<RunDetailsResponse> {
  const response: AxiosResponse<RunDetailsResponse> = await axios.get(
    `${API_BASE}/api/ml/runs/${encodeURIComponent(runId)}`
  );
  return response.data;
}

export async function startAutoRuns(machineId: string, intervalSeconds = 150): Promise<AutoRunStatusResponse> {
  const response: AxiosResponse<AutoRunStatusResponse> = await axios.post(
    `${API_BASE}/api/ml/machines/${encodeURIComponent(machineId)}/auto/start`,
    null,
    { params: { interval_seconds: intervalSeconds } }
  );
  return response.data;
}

export async function stopAutoRuns(machineId: string): Promise<AutoRunStatusResponse> {
  const response: AxiosResponse<AutoRunStatusResponse> = await axios.post(
    `${API_BASE}/api/ml/machines/${encodeURIComponent(machineId)}/auto/stop`
  );
  return response.data;
}

export async function getAutoRunsStatus(machineId: string): Promise<AutoRunStatusResponse> {
  const response: AxiosResponse<AutoRunStatusResponse> = await axios.get(
    `${API_BASE}/api/ml/machines/${encodeURIComponent(machineId)}/auto/status`
  );
  return response.data;
}

export async function autoRunOnce(machineId: string): Promise<{ run_id: string; machine_id: string; data_stamp: string; created_at: string }> {
  const response: AxiosResponse<{ run_id: string; machine_id: string; data_stamp: string; created_at: string }> = await axios.post(
    `${API_BASE}/api/ml/machines/${encodeURIComponent(machineId)}/auto/run_once`
  );
  return response.data;
}

/**
 * Start sensor simulation for a machine
 */
export async function startSimulation(machineId: string): Promise<SimulationControlResponse> {
  const response: AxiosResponse<SimulationControlResponse> = await axios.post(
    `${API_BASE}/api/ml/simulation/start/${encodeURIComponent(machineId)}`
  );
  return response.data;
}

/**
 * Stop sensor simulation for a machine
 */
export async function stopSimulation(machineId: string): Promise<SimulationControlResponse> {
  const response: AxiosResponse<SimulationControlResponse> = await axios.post(
    `${API_BASE}/api/ml/simulation/stop/${encodeURIComponent(machineId)}`
  );
  return response.data;
}

/**
 * Get simulation status for a machine
 */
export async function getSimulationStatus(machineId: string): Promise<SimulationStatusResponse> {
  const response: AxiosResponse<SimulationStatusResponse> = await axios.get(
    `${API_BASE}/api/ml/simulation/status/${encodeURIComponent(machineId)}`
  );
  return response.data;
}

/**
 * Get list of active simulations
 */
export async function getActiveSimulations(): Promise<ActiveSimulationsResponse> {
  const response: AxiosResponse<ActiveSimulationsResponse> = await axios.get(
    `${API_BASE}/api/ml/simulation/active`
  );
  return response.data;
}

/**
 * Stop all active simulations
 */
export async function stopAllSimulations(): Promise<SimulationControlResponse> {
  const response: AxiosResponse<SimulationControlResponse> = await axios.post(
    `${API_BASE}/api/ml/simulation/stop-all`
  );
  return response.data;
}

export default {
  getMachineStatus,
  getMachineSnapshots,
  getMachineRuns,
  getRunDetails,
  startAutoRuns,
  stopAutoRuns,
  getAutoRunsStatus,
  autoRunOnce,
  startSimulation,
  stopSimulation,
  getSimulationStatus,
  getActiveSimulations,
  stopAllSimulations,
};
