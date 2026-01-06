/**
 * API Configuration Service
 * Phase 3.7.9: Common endpoint for service discovery
 * 
 * Fetches server configuration from the backend including:
 * - Available services and endpoints
 * - Server port and host info
 * - Feature flags
 * - VLM integration configuration
 */

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

// ============================================================================
// Types
// ============================================================================

export interface ServiceInfo {
  name: string;
  available: boolean;
  base_path: string;
  description: string;
}

export interface ServerConfig {
  api_version: string;
  environment: string;
  host: string;
  port: number;
  base_url: string;
  debug: boolean;
  services: ServiceInfo[];
  cors_origins: string[];
  features: Record<string, boolean>;
}

export interface EndpointsResponse {
  total_categories: number;
  endpoints: Record<string, Record<string, string>>;
}

export interface VLMConfig {
  vlm_available: boolean;
  note: string;
  expected_endpoints: Record<string, string>;
  recommended_protocols: string[];
  configuration: {
    storage_key: string;
    default_port: number;
    configure_at: string;
  };
}

// ============================================================================
// API Functions
// ============================================================================

/**
 * Get server configuration from the backend.
 * Returns info about available services, port, features, etc.
 */
export async function getServerConfig(): Promise<ServerConfig> {
  const response = await fetch(`${API_BASE_URL}/api/config`, {
    method: 'GET',
    headers: { 'Content-Type': 'application/json' },
    signal: AbortSignal.timeout(10000),
  });
  
  if (!response.ok) {
    throw new Error(`Failed to fetch server config: ${response.status}`);
  }
  
  return response.json();
}

/**
 * Get list of all available API endpoints.
 */
export async function getAvailableEndpoints(): Promise<EndpointsResponse> {
  const response = await fetch(`${API_BASE_URL}/api/config/endpoints`, {
    method: 'GET',
    headers: { 'Content-Type': 'application/json' },
    signal: AbortSignal.timeout(10000),
  });
  
  if (!response.ok) {
    throw new Error(`Failed to fetch endpoints: ${response.status}`);
  }
  
  return response.json();
}

/**
 * Get VLM integration configuration.
 * Describes how to connect to external VLM devices (e.g., Jetson).
 */
export async function getVLMConfig(): Promise<VLMConfig> {
  const response = await fetch(`${API_BASE_URL}/api/config/vlm`, {
    method: 'GET',
    headers: { 'Content-Type': 'application/json' },
    signal: AbortSignal.timeout(10000),
  });
  
  if (!response.ok) {
    throw new Error(`Failed to fetch VLM config: ${response.status}`);
  }
  
  return response.json();
}

/**
 * Test connection to an external endpoint (e.g., VLM/Jetson device).
 * @param endpoint - The endpoint URL to test (e.g., http://192.168.1.50:8080)
 * @returns Object with connection status and latency
 */
export async function testExternalConnection(endpoint: string): Promise<{
  connected: boolean;
  latency_ms: number | null;
  error: string | null;
  health_data: Record<string, unknown> | null;
}> {
  const normalizedEndpoint = endpoint.trim().replace(/\/$/, '');
  if (!normalizedEndpoint) {
    return { connected: false, latency_ms: null, error: 'Empty endpoint', health_data: null };
  }

  const startTime = performance.now();
  
  try {
    // Try /health endpoint first (common convention)
    const response = await fetch(`${normalizedEndpoint}/health`, {
      method: 'GET',
      signal: AbortSignal.timeout(5000),
      mode: 'cors',
    });
    
    const latency = Math.round(performance.now() - startTime);
    
    if (response.ok) {
      let healthData = null;
      try {
        healthData = await response.json();
      } catch {
        // Response might not be JSON
      }
      return { connected: true, latency_ms: latency, error: null, health_data: healthData };
    }
    
    return { connected: false, latency_ms: latency, error: `HTTP ${response.status}`, health_data: null };
  } catch (err) {
    // Try no-cors mode as fallback (can't read response but can detect reachability)
    try {
      await fetch(`${normalizedEndpoint}/health`, {
        method: 'GET',
        signal: AbortSignal.timeout(5000),
        mode: 'no-cors',
      });
      const latency = Math.round(performance.now() - startTime);
      return { 
        connected: true, 
        latency_ms: latency, 
        error: null, 
        health_data: { note: 'CORS blocked - endpoint reachable but response not readable' } 
      };
    } catch {
      const latency = Math.round(performance.now() - startTime);
      const errorMsg = err instanceof Error ? err.message : 'Connection failed';
      return { connected: false, latency_ms: latency, error: errorMsg, health_data: null };
    }
  }
}

/**
 * Get the configured API base URL.
 */
export function getApiBaseUrl(): string {
  return API_BASE_URL;
}

/**
 * Check if the main backend server is reachable.
 */
export async function checkBackendHealth(): Promise<{
  healthy: boolean;
  version: string | null;
  error: string | null;
}> {
  try {
    const response = await fetch(`${API_BASE_URL}/health`, {
      method: 'GET',
      signal: AbortSignal.timeout(5000),
    });
    
    if (response.ok) {
      const data = await response.json();
      return { healthy: true, version: data.version || null, error: null };
    }
    
    return { healthy: false, version: null, error: `HTTP ${response.status}` };
  } catch (err) {
    const errorMsg = err instanceof Error ? err.message : 'Connection failed';
    return { healthy: false, version: null, error: errorMsg };
  }
}
