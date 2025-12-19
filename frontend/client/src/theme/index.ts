/**
 * Theme Module Exports
 * Phase 3.7.3 Day 16.1: MUI Theme Setup
 * 
 * Central export point for theme-related utilities
 */

export { professionalTheme, default as theme } from './professionalTheme';
export { ThemeProvider } from './ThemeProvider';
export type { ProfessionalTheme } from './professionalTheme';

// Re-export MUI theme utilities for convenience
export { useTheme } from '@mui/material/styles';
export type { Theme } from '@mui/material/styles';
