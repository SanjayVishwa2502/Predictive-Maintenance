/**
 * ML Models Management API
 *
 * Wraps backend endpoints for model inventory + deletion.
 */

import axios from 'axios';
import type { AxiosResponse } from 'axios';

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

export type InventoryStatus = 'missing' | 'available' | 'corrupted';
export type ModelType = 'classification' | 'regression' | 'anomaly' | 'timeseries';

export interface ModelArtifactStatus {
  status: InventoryStatus;
  model_dir?: string | null;
  report_path?: string | null;
  issues?: string[];
}

export interface MachineModelInventory {
  machine_id: string;
  display_name?: string | null;
  category?: string | null;
  manufacturer?: string | null;
  model?: string | null;
  models: Record<ModelType, ModelArtifactStatus>;
}

export interface ModelInventoryResponse {
  machines: MachineModelInventory[];
  total: number;
}

export interface DeleteModelResponse {
  machine_id: string;
  model_type: ModelType;
  deleted_model_dir: boolean;
  deleted_report_file: boolean;
}

export interface DeleteAllModelsResponse {
  machine_id: string;
  results: Partial<Record<ModelType, DeleteModelResponse>>;
  errors: Record<string, string>;
}

export const mlModelsApi = {
  getInventory: async (): Promise<ModelInventoryResponse> => {
    const resp: AxiosResponse<ModelInventoryResponse> = await axios.get(`${API_BASE}/api/ml/models/inventory`);
    return resp.data;
  },

  deleteModel: async (machineId: string, modelType: ModelType): Promise<DeleteModelResponse> => {
    const resp: AxiosResponse<DeleteModelResponse> = await axios.delete(
      `${API_BASE}/api/ml/models/${encodeURIComponent(machineId)}/${encodeURIComponent(modelType)}`
    );
    return resp.data;
  },

  deleteAllModels: async (machineId: string): Promise<DeleteAllModelsResponse> => {
    const resp: AxiosResponse<DeleteAllModelsResponse> = await axios.delete(
      `${API_BASE}/api/ml/models/${encodeURIComponent(machineId)}`
    );
    return resp.data;
  },
};
