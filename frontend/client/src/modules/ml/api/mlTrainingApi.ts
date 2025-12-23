/**
 * ML Training API Service - Phase 3.7.8.4
 *
 * Minimal client for starting ML model training jobs.
 * (Progress polling + results UI are implemented in later phases.)
 */

import axios from 'axios';
import type { AxiosResponse } from 'axios';
import type { TaskStatusResponse } from '../types/gan.types';

// Get API base URL from environment
const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

export type ModelType = 'classification' | 'regression' | 'anomaly' | 'timeseries';

export interface StartTrainingRequest {
  machine_id: string;
  model_types: ModelType[];
  time_limit_per_model?: number;
}

export interface StartTrainingResponse {
  success: boolean;
  machine_id: string;
  task_id: string;
  message: string;
}

export type PrimaryMetricKey = 'accuracy' | 'f1' | 'r2' | 'rmse' | 'mape';

export interface TrainingModelResult {
  model_type: ModelType;
  report?: any;
  report_path?: string;
  model_dir?: string;
  primary_metric?: { key: PrimaryMetricKey; label: string; value: number; higher_is_better: boolean };
}

export interface TrainingResults {
  task_id: string;
  machine_id: string;
  complete_system?: boolean;
  models: TrainingModelResult[];
}

function pickNumber(val: any): number | null {
  return typeof val === 'number' && Number.isFinite(val) ? val : null;
}

function tryMetric(report: any, keys: string[]): number | null {
  if (!report || typeof report !== 'object') return null;

  for (const k of keys) {
    const direct = pickNumber((report as any)[k]);
    if (direct !== null) return direct;

    const nestedPaths: string[][] = [
      ['metrics', k],
      ['summary', k],
      ['best_model', k],
    ];
    for (const path of nestedPaths) {
      let cur: any = report;
      for (const seg of path) cur = cur && typeof cur === 'object' ? cur[seg] : undefined;
      const nested = pickNumber(cur);
      if (nested !== null) return nested;
    }
  }
  return null;
}

function computePrimaryMetric(modelType: ModelType, report: any): TrainingModelResult['primary_metric'] {
  if (modelType === 'classification') {
    const f1 = tryMetric(report, ['f1', 'f1_score']);
    if (f1 !== null) return { key: 'f1', label: 'F1', value: f1, higher_is_better: true };
    const acc = tryMetric(report, ['accuracy']);
    if (acc !== null) return { key: 'accuracy', label: 'Accuracy', value: acc, higher_is_better: true };
    return undefined;
  }
  if (modelType === 'regression') {
    const r2 = tryMetric(report, ['r2', 'r2_score']);
    if (r2 !== null) return { key: 'r2', label: 'R²', value: r2, higher_is_better: true };
    const rmse = tryMetric(report, ['rmse']);
    if (rmse !== null) return { key: 'rmse', label: 'RMSE', value: rmse, higher_is_better: false };
    return undefined;
  }
  if (modelType === 'anomaly') {
    const f1 = tryMetric(report, ['f1', 'f1_score']);
    if (f1 !== null) return { key: 'f1', label: 'F1', value: f1, higher_is_better: true };
    return undefined;
  }
  if (modelType === 'timeseries') {
    const mape = tryMetric(report, ['mape']);
    if (mape !== null) return { key: 'mape', label: 'MAPE', value: mape, higher_is_better: false };
    return undefined;
  }
  return undefined;
}

function toModelType(s: string): ModelType | null {
  const v = String(s || '').toLowerCase();
  if (v === 'classification' || v === 'regression' || v === 'anomaly' || v === 'timeseries') return v;
  return null;
}

export const mlTrainingApi = {
  /**
   * Start ML training.
   *
   * Implementation detail: This uses the backend batch endpoint so the
   * recommended "train all 4" workflow is a single task.
   */
  startTraining: async (request: StartTrainingRequest): Promise<StartTrainingResponse> => {
    const response: AxiosResponse<StartTrainingResponse> = await axios.post(
      `${API_BASE}/api/ml/train/batch`,
      {
        machine_id: request.machine_id,
        model_types: request.model_types,
        time_limit_per_model: request.time_limit_per_model,
      }
    );
    return response.data;
  },

  /**
   * Poll ML training task status (same shape as GAN TaskStatusResponse).
   */
  getTaskStatus: async (taskId: string): Promise<TaskStatusResponse> => {
    const response: AxiosResponse<TaskStatusResponse> = await axios.get(`${API_BASE}/api/ml/tasks/${taskId}`);
    return response.data;
  },

  /**
   * Best-effort task cancellation.
   */
  cancelTask: async (taskId: string): Promise<{ success: boolean; task_id: string; message: string }> => {
    const response: AxiosResponse<{ success: boolean; task_id: string; message: string }> = await axios.post(
      `${API_BASE}/api/ml/tasks/${taskId}/cancel`
    );
    return response.data;
  },

  /**
   * Derive a results object from the existing task status endpoint.
   *
   * Note: There is currently no dedicated backend "results" endpoint, so this
   * function normalizes the Celery task result payload into a UI-friendly shape.
   */
  getTrainingResults: async (taskId: string): Promise<TrainingResults> => {
    const status = await mlTrainingApi.getTaskStatus(taskId);
    if (status.status !== 'SUCCESS') {
      throw new Error('Training results are only available after task completion');
    }

    const result: any = status.result;
    const machineId =
      (typeof result?.machine_id === 'string' && result.machine_id) ||
      (typeof status.machine_id === 'string' && status.machine_id) ||
      '';

    const models: TrainingModelResult[] = [];

    // Batch shape: { results: { classification: { report, model_dir, report_path, ... }, ... } }
    if (result && typeof result === 'object' && result.results && typeof result.results === 'object') {
      for (const [k, entry] of Object.entries(result.results)) {
        const mt = toModelType(k);
        if (!mt) continue;
        const report = (entry as any)?.report;
        models.push({
          model_type: mt,
          report,
          report_path: (entry as any)?.report_path,
          model_dir: (entry as any)?.model_dir,
          primary_metric: computePrimaryMetric(mt, report),
        });
      }
    }

    // Single-model shape: { model_type, report, model_dir, report_path }
    if (models.length === 0 && typeof result?.model_type === 'string') {
      const mt = toModelType(result.model_type);
      if (mt) {
        models.push({
          model_type: mt,
          report: result.report,
          report_path: result.report_path,
          model_dir: result.model_dir,
          primary_metric: computePrimaryMetric(mt, result.report),
        });
      }
    }

    return {
      task_id: taskId,
      machine_id: machineId,
      complete_system: Boolean(result?.complete_system),
      models,
    };
  },
};
