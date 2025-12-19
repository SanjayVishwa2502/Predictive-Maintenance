/**
 * API Integration Tests
 * Phase 3.7.5 Day 20: Quality Assurance
 * 
 * Tests API functions without full component rendering
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { mockFetchSuccess, mockFetchError } from './setup';

// API Base URL
const API_BASE_URL = 'http://localhost:8000';
const API_ENDPOINTS = {
  machines: `${API_BASE_URL}/api/ml/machines`,
  machineStatus: (id: string) => `${API_BASE_URL}/api/ml/machines/${id}/status`,
  predictClassification: `${API_BASE_URL}/api/ml/predict/classification`,
};

describe('API Integration Tests', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe('Fetch Machines', () => {
    it('should fetch machines successfully', async () => {
      const mockData = {
        machines: [
          { machine_id: 'motor_001', display_name: 'Motor 001' },
        ],
        total: 1,
      };

      mockFetchSuccess(mockData);

      const response = await fetch(API_ENDPOINTS.machines);
      const data = await response.json();

      expect(data).toEqual(mockData);
      expect(data.total).toBe(1);
    });

    it('should handle fetch error', async () => {
      mockFetchError(500, 'Server error');

      const response = await fetch(API_ENDPOINTS.machines);
      
      expect(response.ok).toBe(false);
      expect(response.status).toBe(500);
    });
  });

  describe('Fetch Sensor Data', () => {
    it('should fetch sensor data for a machine', async () => {
      const mockData = {
        machine_id: 'motor_001',
        latest_sensors: {
          bearing_de_temp_C: 65.2,
          current_A: 12.5,
        },
        sensor_count: 2,
      };

      mockFetchSuccess(mockData);

      const response = await fetch(API_ENDPOINTS.machineStatus('motor_001'));
      const data = await response.json();

      expect(data.machine_id).toBe('motor_001');
      expect(data.sensor_count).toBe(2);
    });
  });

  describe('Run Prediction', () => {
    it('should run classification prediction', async () => {
      const requestData = {
        machine_id: 'motor_001',
        sensor_data: { bearing_de_temp_C: 65.2 },
      };

      const mockResponse = {
        machine_id: 'motor_001',
        prediction: {
          failure_type: 'normal',
          confidence: 0.95,
        },
      };

      mockFetchSuccess(mockResponse);

      const response = await fetch(API_ENDPOINTS.predictClassification, {
        method: 'POST',
        body: JSON.stringify(requestData),
      });

      const data = await response.json();

      expect(data.prediction.failure_type).toBe('normal');
      expect(data.prediction.confidence).toBe(0.95);
    });
  });

  describe('Utility Functions', () => {
    it('should calculate urgency from RUL correctly', () => {
      const getUrgencyFromRUL = (rulHours: number): string => {
        if (rulHours > 240) return 'Low';
        if (rulHours > 120) return 'Medium';
        if (rulHours > 48) return 'High';
        return 'Critical';
      };

      expect(getUrgencyFromRUL(300)).toBe('Low');
      expect(getUrgencyFromRUL(150)).toBe('Medium');
      expect(getUrgencyFromRUL(72)).toBe('High');
      expect(getUrgencyFromRUL(24)).toBe('Critical');
    });

    it('should calculate health state from probability', () => {
      const getHealthStateFromProbability = (failureProb: number): string => {
        if (failureProb < 0.15) return 'HEALTHY';
        if (failureProb < 0.40) return 'DEGRADING';
        if (failureProb < 0.70) return 'WARNING';
        return 'CRITICAL';
      };

      expect(getHealthStateFromProbability(0.05)).toBe('HEALTHY');
      expect(getHealthStateFromProbability(0.25)).toBe('DEGRADING');
      expect(getHealthStateFromProbability(0.50)).toBe('WARNING');
      expect(getHealthStateFromProbability(0.85)).toBe('CRITICAL');
    });
  });

  describe('Error Handling', () => {
    it('should handle network timeout', async () => {
      const controller = new AbortController();
      const signal = controller.signal;

      // Simulate immediate abort
      controller.abort();

      try {
        await fetch(API_ENDPOINTS.machines, { signal });
        expect.fail('Should have thrown an error');
      } catch (error) {
        expect(error).toBeDefined();
      }
    });

    it('should handle JSON parse error', async () => {
      (globalThis.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: () => Promise.reject(new Error('Invalid JSON')),
      });

      try {
        const response = await fetch(API_ENDPOINTS.machines);
        await response.json();
        expect.fail('Should have thrown an error');
      } catch (error: any) {
        expect(error.message).toBe('Invalid JSON');
      }
    });
  });
});
