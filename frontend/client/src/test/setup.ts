/**
 * Vitest Test Setup
 * Phase 3.7.5 Day 20: Quality Assurance
 * 
 * Global test configuration and setup
 */

import { afterEach, vi } from 'vitest';
import { cleanup } from '@testing-library/react';
import '@testing-library/jest-dom/vitest';

// Cleanup after each test
afterEach(() => {
  cleanup();
});

// Mock environment variables
vi.stubEnv('VITE_API_BASE_URL', 'http://localhost:8000');
vi.stubEnv('VITE_API_TIMEOUT', '10000');
vi.stubEnv('VITE_POLLING_INTERVAL', '30000');

// Mock window.matchMedia
Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: vi.fn().mockImplementation((query) => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: vi.fn(),
    removeListener: vi.fn(),
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    dispatchEvent: vi.fn(),
  })),
});

const g = globalThis as any;

// Mock ResizeObserver
g.ResizeObserver = vi.fn().mockImplementation(() => ({
  observe: vi.fn(),
  unobserve: vi.fn(),
  disconnect: vi.fn(),
}));

// Mock IntersectionObserver
g.IntersectionObserver = vi.fn().mockImplementation(() => ({
  observe: vi.fn(),
  unobserve: vi.fn(),
  disconnect: vi.fn(),
}));

// Mock fetch globally
g.fetch = vi.fn();

// Helper to mock successful fetch responses
export const mockFetchSuccess = (data: any) => {
  (g.fetch as any).mockResolvedValueOnce({
    ok: true,
    status: 200,
    json: async () => data,
  });
};

// Helper to mock fetch errors
export const mockFetchError = (status: number, message: string) => {
  (g.fetch as any).mockResolvedValueOnce({
    ok: false,
    status,
    statusText: message,
    json: async () => ({ detail: message }),
  });
};
