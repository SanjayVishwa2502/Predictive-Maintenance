/**
 * SettingsContext
 * 
 * React context that provides centralized access to all application settings.
 * Handles:
 * - Theme management (dark/light mode)
 * - Auto-prediction settings
 * - Notification preferences
 * - LLM configuration
 * - Data display limits
 * - Export preferences
 */

import React, { createContext, useContext, useState, useCallback, useEffect, useMemo } from 'react';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import type { Theme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';

// ============================================================================
// Types
// ============================================================================

export interface AppSettings {
  // Theme
  theme: 'dark' | 'light';
  
  // Auto-Prediction
  autoPredictionEnabled: boolean;
  autoPredictionInterval: number; // seconds
  
  // Notifications
  notificationsEnabled: boolean;
  notificationSound: boolean;
  notificationCriticalOnly: boolean;
  
  // LLM
  llmEnabled: boolean;
  llmTemperature: number;
  llmMaxTokens: number;
  llmTimeout: number; // seconds
  
  // VLM
  vlmEndpoint: string;
  
  // Data Display
  historyItemsLimit: number;
  chartDataPoints: number;
  
  // Export
  exportFormat: 'csv' | 'json';
  exportIncludeTimestamps: boolean;
}

export interface SettingsContextType {
  settings: AppSettings;
  updateSetting: <K extends keyof AppSettings>(key: K, value: AppSettings[K]) => void;
  updateSettings: (updates: Partial<AppSettings>) => void;
  resetToDefaults: (section?: keyof typeof DEFAULTS_BY_SECTION) => void;
  refreshSettings: () => void;
  theme: Theme;
}

// ============================================================================
// Constants
// ============================================================================

const STORAGE_KEYS: Record<keyof AppSettings, string> = {
  theme: 'pm_ui_theme',
  autoPredictionEnabled: 'pm_auto_prediction_enabled',
  autoPredictionInterval: 'pm_auto_prediction_interval',
  notificationsEnabled: 'pm_notifications_enabled',
  notificationSound: 'pm_notification_sound',
  notificationCriticalOnly: 'pm_notification_critical_only',
  llmEnabled: 'pm_llm_enabled',
  llmTemperature: 'pm_llm_temperature',
  llmMaxTokens: 'pm_llm_max_tokens',
  llmTimeout: 'pm_llm_timeout',
  vlmEndpoint: 'pm_vlm_endpoint',
  historyItemsLimit: 'pm_history_items_limit',
  chartDataPoints: 'pm_chart_data_points',
  exportFormat: 'pm_export_format',
  exportIncludeTimestamps: 'pm_export_include_timestamps',
};

const DEFAULT_SETTINGS: AppSettings = {
  theme: 'dark',
  autoPredictionEnabled: true,
  autoPredictionInterval: 150, // 2.5 minutes
  notificationsEnabled: true,
  notificationSound: true,
  notificationCriticalOnly: false,
  llmEnabled: true,
  llmTemperature: 0.7,
  llmMaxTokens: 512,
  llmTimeout: 120,
  vlmEndpoint: '',
  historyItemsLimit: 100,
  chartDataPoints: 50,
  exportFormat: 'csv',
  exportIncludeTimestamps: true,
};

const DEFAULTS_BY_SECTION = {
  theme: {
    theme: 'dark' as const,
  },
  autoPrediction: {
    autoPredictionEnabled: true,
    autoPredictionInterval: 150,
  },
  notifications: {
    notificationsEnabled: true,
    notificationSound: true,
    notificationCriticalOnly: false,
  },
  llm: {
    llmEnabled: true,
    llmTemperature: 0.7,
    llmMaxTokens: 512,
    llmTimeout: 120,
  },
  vlm: {
    vlmEndpoint: '',
  },
  data: {
    historyItemsLimit: 100,
    chartDataPoints: 50,
    exportFormat: 'csv' as const,
    exportIncludeTimestamps: true,
  },
};

// ============================================================================
// Theme Definitions
// ============================================================================

const darkTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#667eea',
      light: '#8b9cf0',
      dark: '#4a5fc7',
    },
    secondary: {
      main: '#764ba2',
      light: '#9b6fc4',
      dark: '#5a3880',
    },
    background: {
      default: '#0f172a',
      paper: '#1e293b',
    },
    success: {
      main: '#10b981',
      light: '#34d399',
      dark: '#059669',
    },
    warning: {
      main: '#f59e0b',
      light: '#fbbf24',
      dark: '#d97706',
    },
    error: {
      main: '#ef4444',
      light: '#f87171',
      dark: '#dc2626',
    },
    text: {
      primary: '#e5e7eb',
      secondary: '#9ca3af',
    },
  },
  typography: {
    fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
  },
  components: {
    MuiCssBaseline: {
      styleOverrides: {
        body: {
          backgroundColor: '#0f172a',
          backgroundImage: 'linear-gradient(135deg, #0f172a 0%, #1e293b 100%)',
          minHeight: '100vh',
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundImage: 'none',
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          borderRadius: 8,
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: 12,
          border: '1px solid rgba(255, 255, 255, 0.1)',
        },
      },
    },
  },
});

const lightTheme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#5a67d8',
      light: '#7c85e0',
      dark: '#434bb0',
    },
    secondary: {
      main: '#6b46c1',
      light: '#8b6dd3',
      dark: '#553c9a',
    },
    background: {
      default: '#f8fafc',
      paper: '#ffffff',
    },
    success: {
      main: '#059669',
      light: '#10b981',
      dark: '#047857',
    },
    warning: {
      main: '#d97706',
      light: '#f59e0b',
      dark: '#b45309',
    },
    error: {
      main: '#dc2626',
      light: '#ef4444',
      dark: '#b91c1c',
    },
    text: {
      primary: '#1e293b',
      secondary: '#64748b',
    },
  },
  typography: {
    fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
  },
  components: {
    MuiCssBaseline: {
      styleOverrides: {
        body: {
          backgroundColor: '#f8fafc',
          backgroundImage: 'linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%)',
          minHeight: '100vh',
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundImage: 'none',
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          borderRadius: 8,
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: 12,
          border: '1px solid rgba(0, 0, 0, 0.1)',
        },
      },
    },
    MuiAccordion: {
      styleOverrides: {
        root: {
          '&.MuiAccordion-root': {
            background: 'linear-gradient(135deg, rgba(255, 255, 255, 0.95) 0%, rgba(248, 250, 252, 0.95) 100%)',
            border: '1px solid rgba(0, 0, 0, 0.1)',
          },
        },
      },
    },
  },
});

// ============================================================================
// Helper Functions
// ============================================================================

function loadSettings(): AppSettings {
  const settings = { ...DEFAULT_SETTINGS };
  
  try {
    for (const [key, storageKey] of Object.entries(STORAGE_KEYS)) {
      const stored = localStorage.getItem(storageKey);
      if (stored !== null) {
        const settingKey = key as keyof AppSettings;
        const defaultValue = DEFAULT_SETTINGS[settingKey];
        
        if (typeof defaultValue === 'boolean') {
          (settings as Record<string, unknown>)[settingKey] = stored === 'true';
        } else if (typeof defaultValue === 'number') {
          const parsed = parseFloat(stored);
          if (!isNaN(parsed)) {
            (settings as Record<string, unknown>)[settingKey] = parsed;
          }
        } else {
          (settings as Record<string, unknown>)[settingKey] = stored;
        }
      }
    }
  } catch {
    console.warn('Failed to load settings from localStorage');
  }
  
  return settings;
}

function saveSetting(key: keyof AppSettings, value: unknown): void {
  try {
    const storageKey = STORAGE_KEYS[key];
    localStorage.setItem(storageKey, String(value));
  } catch {
    console.warn(`Failed to save setting: ${key}`);
  }
}

// ============================================================================
// Context
// ============================================================================

const SettingsContext = createContext<SettingsContextType | undefined>(undefined);

// ============================================================================
// Provider Component
// ============================================================================

interface SettingsProviderProps {
  children: React.ReactNode;
}

export function SettingsProvider({ children }: SettingsProviderProps) {
  const [settings, setSettings] = useState<AppSettings>(() => loadSettings());

  // Update a single setting
  const updateSetting = useCallback(<K extends keyof AppSettings>(key: K, value: AppSettings[K]) => {
    setSettings(prev => {
      const newSettings = { ...prev, [key]: value };
      saveSetting(key, value);
      return newSettings;
    });
  }, []);

  // Update multiple settings at once
  const updateSettings = useCallback((updates: Partial<AppSettings>) => {
    setSettings(prev => {
      const newSettings = { ...prev, ...updates };
      for (const [key, value] of Object.entries(updates)) {
        saveSetting(key as keyof AppSettings, value);
      }
      return newSettings;
    });
  }, []);

  // Reset settings to defaults (optionally by section)
  const resetToDefaults = useCallback((section?: keyof typeof DEFAULTS_BY_SECTION) => {
    if (section) {
      const sectionDefaults = DEFAULTS_BY_SECTION[section];
      updateSettings(sectionDefaults as Partial<AppSettings>);
    } else {
      // Reset all settings
      setSettings(DEFAULT_SETTINGS);
      for (const [key, value] of Object.entries(DEFAULT_SETTINGS)) {
        saveSetting(key as keyof AppSettings, value);
      }
    }
  }, [updateSettings]);

  // Refresh settings from localStorage
  const refreshSettings = useCallback(() => {
    setSettings(loadSettings());
  }, []);

  // Apply theme to document
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', settings.theme);
    document.body.classList.remove('theme-dark', 'theme-light');
    document.body.classList.add(`theme-${settings.theme}`);
  }, [settings.theme]);

  // Get current theme object
  const theme = useMemo(() => {
    return settings.theme === 'dark' ? darkTheme : lightTheme;
  }, [settings.theme]);

  const contextValue = useMemo(() => ({
    settings,
    updateSetting,
    updateSettings,
    resetToDefaults,
    refreshSettings,
    theme,
  }), [settings, updateSetting, updateSettings, resetToDefaults, refreshSettings, theme]);

  return (
    <SettingsContext.Provider value={contextValue}>
      <ThemeProvider theme={theme}>
        <CssBaseline />
        {children}
      </ThemeProvider>
    </SettingsContext.Provider>
  );
}

// ============================================================================
// Hook
// ============================================================================

export function useSettings(): SettingsContextType {
  const context = useContext(SettingsContext);
  if (context === undefined) {
    throw new Error('useSettings must be used within a SettingsProvider');
  }
  return context;
}

// ============================================================================
// Exports
// ============================================================================

export { DEFAULT_SETTINGS, DEFAULTS_BY_SECTION };
export default SettingsContext;
