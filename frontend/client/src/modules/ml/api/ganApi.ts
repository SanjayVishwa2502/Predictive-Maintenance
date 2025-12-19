/**
 * GAN API Service - Phase 3.7.6.2
 * 
 * API client for all 17 GAN endpoints:
 * - Templates (2 endpoints)
 * - Profiles (3 endpoints)
 * - Workflow (6 endpoints)
 * - Management (3 endpoints)
 * - Monitoring (2 endpoints)
 * - Health (1 endpoint)
 */

import axios from 'axios';
import type { AxiosResponse } from 'axios';
import type {
  MachineTemplate,
  MachineProfile,
  ProfileUploadResponse,
  ValidationError,
  SeedGenerationRequest,
  SeedGenerationResponse,
  TrainingRequest,
  TrainingResponse,
  GenerationRequest,
  GenerationResponse,
  MachineListResponse,
  MachineDetails,
  TaskStatusResponse,
  ValidateDataQualityResponse,
  VisualizationSummaryResponse,
} from '../types/gan.types';

// Get API base URL from environment
const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

/**
 * GAN API Client
 */
export const ganApi = {
  // ========== TEMPLATES ==========

  /**
   * Get list of all machine templates
   */
  getTemplates: async (): Promise<MachineTemplate[]> => {
    const response: AxiosResponse<MachineTemplate[]> = await axios.get(
      `${API_BASE}/api/gan/templates`
    );
    return response.data;
  },

  /**
   * Get specific machine template by type
   */
  getTemplate: async (machineType: string): Promise<MachineProfile> => {
    const response: AxiosResponse<MachineProfile> = await axios.get(
      `${API_BASE}/api/gan/templates/${machineType}`
    );
    return response.data;
  },

  /**
   * Get example profile for a machine type
   */
  getExampleProfile: async (machineType: string): Promise<MachineProfile> => {
    const response: AxiosResponse<MachineProfile> = await axios.get(
      `${API_BASE}/api/gan/examples/${machineType}`
    );
    return response.data;
  },

  // ========== PROFILES ==========

  /**
   * Upload machine profile file (JSON/YAML/Excel)
   * Uses backend authoritative validation (duplicates, TVAE readiness)
   */
  uploadProfile: async (file: File, machineType?: string): Promise<ProfileUploadResponse> => {
    try {
      // Parse file content
      const content = await parseFileContent(file);
      const profileData = JSON.parse(content);

      const selectedType = String(machineType || '').trim();
      const existingType = String((profileData as any)?.machine_type || '').trim();
      const existingCategory = String((profileData as any)?.category || '').trim();

      // Prefer an explicit machine type declared in the profile itself.
      // This avoids false mismatches when the UI has a stale/previous selection.
      const effectiveType = existingType || existingCategory || selectedType;

      if (effectiveType) {
        (profileData as any).machine_type = effectiveType;
        (profileData as any).category = effectiveType;
      }

      // Stage the profile on disk (backend performs immediate duplicate check)
      const uploadResp: AxiosResponse<any> = await axios.post(
        `${API_BASE}/api/gan/profiles/upload`,
        profileData
      );

      const profile_id = String(uploadResp.data?.profile_id || '');
      const machine_id = String(uploadResp.data?.machine_id || (profileData as any)?.machine_id || 'unknown');

      if (!profile_id) {
        throw new Error('Profile upload failed: missing profile_id');
      }

      // Validate the staged profile (full validator + 14 fallback rules)
      const validateResp: AxiosResponse<any> = await axios.post(
        `${API_BASE}/api/gan/profiles/${profile_id}/validate`,
        { strict: true }
      );

      const issues = (validateResp.data?.issues || []) as Array<{
        severity: 'info' | 'warning' | 'error' | string;
        field: string;
        message: string;
      }>;

      const validation_errors: ValidationError[] = issues.map((i) => ({
        field: i.field || 'profile',
        message: i.message,
        severity: (i.severity as any) || 'error',
      }));

      const valid = Boolean(validateResp.data?.valid);
      const canProceed = Boolean(validateResp.data?.can_proceed);
      const hasBlocking = validation_errors.some((e) => e.severity === 'error');
      const status: 'validated' | 'invalid' = valid && canProceed && !hasBlocking ? 'validated' : 'invalid';

      return {
        profile_id,
        machine_id,
        status,
        validation_errors,
        next_step: status === 'validated' ? 'Proceed to Create Machine' : 'Fix validation errors',
        profile: profileData,
      };
    } catch (error: any) {
      // Preserve backend error messages (e.g., duplicate machine_id) instead of masking them
      // as JSON parse issues.
      if (axios.isAxiosError(error)) {
        const data: any = error.response?.data;
        const detail = data?.detail ?? data?.message;
        if (typeof detail === 'string' && detail.trim()) {
          throw new Error(detail);
        }
        if (detail && typeof detail === 'object') {
          const msg = String((detail as any)?.message || '').trim();
          if (msg) throw new Error(msg);
          try {
            throw new Error(JSON.stringify(detail));
          } catch {
            throw new Error('Request failed');
          }
        }
        throw new Error(error.message || 'Request failed');
      }

      // Local parsing or unexpected error
      throw new Error(`Failed to parse profile: ${error?.message || String(error)}`);
    }
  },

  /**
   * Create machine from a staged profile_id
   */
  createMachineFromProfile: async (profileId: string): Promise<MachineDetails> => {
    const response: AxiosResponse<MachineDetails> = await axios.post(
      `${API_BASE}/api/gan/machines`,
      null,
      { params: { profile_id: profileId } }
    );
    return response.data;
  },

  /**
   * Validate profile data via backend authoritative validator
   */
  validateProfile: async (profileData: object): Promise<{
    valid: boolean;
    errors: ValidationError[];
  }> => {
    const response: AxiosResponse<any> = await axios.post(
      `${API_BASE}/api/gan/profiles/validate-inline`,
      {
        profile_data: profileData,
        strict: true,
      }
    );

    const issues = (response.data?.issues || []) as Array<{
      severity: 'info' | 'warning' | 'error' | string;
      field: string;
      message: string;
    }>;

    // Keep API shape (`errors`) for compatibility, but include severity so the UI can classify correctly.
    const errors: ValidationError[] = issues.map((i) => ({
      field: i.field || 'profile',
      message: i.message,
      severity: (i.severity as any) || 'error',
    }));

    return {
      valid: Boolean(response.data?.valid),
      errors,
    };
  },

  /**
   * Replace/update a staged profile (by profile_id) with edited JSON.
   * This is required so the subsequent Create Machine uses the edited machine_id.
   */
  updateStagedProfile: async (profileId: string, profileData: object): Promise<void> => {
    await axios.put(`${API_BASE}/api/gan/profiles/${profileId}/edit`, profileData);
  },

  /**
   * Validate a staged profile (by profile_id) using the authoritative backend validator.
   */
  validateStagedProfile: async (profileId: string): Promise<{
    valid: boolean;
    can_proceed: boolean;
    machine_id: string;
    issues: Array<{ severity: string; field: string; message: string }>;
  }> => {
    const resp: AxiosResponse<any> = await axios.post(
      `${API_BASE}/api/gan/profiles/${profileId}/validate`,
      { strict: true }
    );
    return {
      valid: Boolean(resp.data?.valid),
      can_proceed: Boolean(resp.data?.can_proceed),
      machine_id: String(resp.data?.machine_id || ''),
      issues: (resp.data?.issues || []) as any,
    };
  },

  /**
   * Edit existing machine profile
   */
  editProfile: async (
    profileId: string,
    profileData: Partial<MachineProfile>
  ): Promise<MachineProfile> => {
    const response: AxiosResponse<MachineProfile> = await axios.put(
      `${API_BASE}/api/gan/profiles/${profileId}/edit`,
      profileData
    );
    return response.data;
  },

  // ========== WORKFLOW: SEED GENERATION ==========

  /**
   * Generate physics-based seed data
   */
  generateSeed: async (
    machineId: string,
    request: SeedGenerationRequest
  ): Promise<SeedGenerationResponse> => {
    const response: AxiosResponse<SeedGenerationResponse> = await axios.post(
      `${API_BASE}/api/gan/machines/${machineId}/seed`,
      request
    );
    return response.data;
  },

  // ========== WORKFLOW: TVAE TRAINING ==========

  /**
   * Train TVAE model on seed data
   */
  trainModel: async (
    machineId: string,
    request: TrainingRequest
  ): Promise<TrainingResponse> => {
    const response: AxiosResponse<TrainingResponse> = await axios.post(
      `${API_BASE}/api/gan/machines/${machineId}/train`,
      request
    );
    return response.data;
  },

  /**
   * Get Celery task status (polling)
   */
  getTaskStatus: async (taskId: string): Promise<TaskStatusResponse> => {
    const response: AxiosResponse<TaskStatusResponse> = await axios.get(
      `${API_BASE}/api/gan/tasks/${taskId}`
    );
    return response.data;
  },

  // ========== VISUALIZATION ============

  getVisualizationSummary: async (
    machineId: string,
    opts?: { points?: number; bins?: number; max_features?: number }
  ): Promise<VisualizationSummaryResponse> => {
    const response: AxiosResponse<VisualizationSummaryResponse> = await axios.get(
      `${API_BASE}/api/gan/machines/${machineId}/visualizations/summary`,
      { params: opts }
    );
    return response.data;
  },

  /**
   * Cancel a running Celery task (best-effort).
   */
  cancelTask: async (taskId: string): Promise<{ success: boolean; task_id: string; message: string }> => {
    const response: AxiosResponse<{ success: boolean; task_id: string; message: string }> = await axios.post(
      `${API_BASE}/api/gan/tasks/${taskId}/cancel`
    );
    return response.data;
  },

  // ========== WORKFLOW: SYNTHETIC GENERATION ==========

  /**
   * Generate synthetic data using trained TVAE
   */
  generateSynthetic: async (
    machineId: string,
    request: GenerationRequest
  ): Promise<GenerationResponse> => {
    const response: AxiosResponse<GenerationResponse> = await axios.post(
      `${API_BASE}/api/gan/machines/${machineId}/generate`,
      request
    );
    return response.data;
  },

  /**
   * Validate generated data quality
   */
  validateDataQuality: async (machineId: string): Promise<any> => {
    const response: AxiosResponse<ValidateDataQualityResponse> = await axios.get(
      `${API_BASE}/api/gan/machines/${machineId}/validate`
    );
    return response.data;
  },

  /**
   * Get download URL for generated parquet split
   */
  getDatasetDownloadUrl: (machineId: string, split: 'train' | 'val' | 'test'): string => {
    return `${API_BASE}/api/gan/machines/${machineId}/data/${split}/download`;
  },

  /**
   * Get download URL for combined CSV (train/val/test merged)
   */
  getDatasetDownloadCsvUrl: (machineId: string): string => {
    return `${API_BASE}/api/gan/machines/${machineId}/data/download/csv`;
  },

  // ========== MANAGEMENT ==========

  /**
   * Get list of all machines
   */
  getMachines: async (): Promise<MachineListResponse> => {
    const response: AxiosResponse<MachineListResponse> = await axios.get(
      `${API_BASE}/api/gan/machines`
    );
    return response.data;
  },

  /**
   * Get detailed machine information
   */
  getMachineDetails: async (machineId: string): Promise<MachineDetails> => {
    const response: AxiosResponse<MachineDetails> = await axios.get(
      `${API_BASE}/api/gan/machines/${machineId}`
    );
    return response.data;
  },

  /**
   * Delete machine and associated data
   */
  deleteMachine: async (machineId: string): Promise<{ message: string }> => {
    const response: AxiosResponse<{ message: string }> = await axios.delete(
      `${API_BASE}/api/gan/machines/${machineId}`
    );
    return response.data;
  },

  // ========== MONITORING ==========

  /**
   * Get Celery task status
   */
  /**
   * Health check endpoint
   */
  healthCheck: async (): Promise<{ status: string; timestamp: string }> => {
    const response: AxiosResponse<{ status: string; timestamp: string }> =
      await axios.get(`${API_BASE}/api/gan/health`);
    return response.data;
  },
};

/**
 * Utility: Download file from URL
 */
export const downloadFile = (url: string, filename: string) => {
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
};

/**
 * Utility: Download template as JSON file (uses local template)
 */
export const downloadTemplateAsFile = async (machineType: string) => {
  try {
    // Fetch the local template file from public folder
    const response = await fetch('/templates/machine_profile_template.json');
    if (!response.ok) {
      throw new Error(`Failed to fetch template: ${response.statusText}`);
    }
    
    const templateText = await response.text();
    const blob = new Blob([templateText], {
      type: 'application/json',
    });
    const url = URL.createObjectURL(blob);
    downloadFile(url, `${machineType}_profile_template.json`);
    URL.revokeObjectURL(url);
  } catch (error) {
    console.error('Failed to download template:', error);
    throw error;
  }
};

/**
 * Utility: Parse uploaded file content
 */
export const parseFileContent = (file: File): Promise<string> => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      if (e.target?.result) {
        resolve(e.target.result as string);
      } else {
        reject(new Error('Failed to read file'));
      }
    };
    reader.onerror = () => reject(new Error('File read error'));
    reader.readAsText(file);
  });
};

export default ganApi;
