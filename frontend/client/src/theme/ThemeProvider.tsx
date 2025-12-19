/**
 * Theme Provider Component
 * Phase 3.7.3 Day 16.1: MUI Theme Setup
 * 
 * Wraps the application with MUI ThemeProvider and CssBaseline
 */

import React from 'react';
import { ThemeProvider as MuiThemeProvider } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { professionalTheme } from './professionalTheme';

interface ThemeProviderProps {
  children: React.ReactNode;
}

/**
 * ThemeProvider Component
 * 
 * Provides the professional dark theme to all child components.
 * Includes CssBaseline for consistent baseline styles.
 * 
 * Usage:
 *   <ThemeProvider>
 *     <App />
 *   </ThemeProvider>
 */
export const ThemeProvider: React.FC<ThemeProviderProps> = ({ children }) => {
  return (
    <MuiThemeProvider theme={professionalTheme}>
      {/* CssBaseline: Normalize styles across browsers */}
      <CssBaseline />
      
      {/* Render children with theme applied */}
      {children}
    </MuiThemeProvider>
  );
};

export default ThemeProvider;
