/**
 * SettingsView Component
 *
 * Centralized settings page with sections for:
 * - Theme & Display Preferences
 * - Auto-Prediction Settings
 * - VLM/Jetson Endpoint Configuration
 * - LLM Runtime Controls
 * - Notification Preferences
 * - Data & Export Settings
 * - User Management (Admin only)
 */

import { useState, useCallback, useEffect } from 'react';
import {
  Box,
  Paper,
  Typography,
  TextField,
  Button,
  Switch,
  FormControlLabel,
  Stack,
  Alert,
  CircularProgress,
  Divider,
  Chip,
  Slider,
  InputAdornment,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  RadioGroup,
  Radio,
} from '@mui/material';
import type { SelectChangeEvent } from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
  Refresh as RefreshIcon,
  Save as SaveIcon,
  DeleteSweep as ClearCacheIcon,
  Videocam as VideocamIcon,
  Psychology as LLMIcon,
  Palette as ThemeIcon,
  Notifications as NotificationIcon,
  Schedule as ScheduleIcon,
  TableChart as DataIcon,
  DarkMode as DarkModeIcon,
  LightMode as LightModeIcon,
  RestartAlt as RestoreIcon,
} from '@mui/icons-material';
import UserManagementView from './UserManagementView';
import { useSettings, DEFAULTS_BY_SECTION } from '../../../../contexts/SettingsContext';

// ============================================================================
// Constants & Types
// ============================================================================

// LocalStorage keys
const SETTINGS_KEYS = {
  VLM_ENDPOINT: 'pm_vlm_endpoint',
  LLM_ENABLED: 'pm_llm_enabled',
  LLM_TEMPERATURE: 'pm_llm_temperature',
  LLM_MAX_TOKENS: 'pm_llm_max_tokens',
  LLM_TIMEOUT: 'pm_llm_timeout',
  UI_THEME: 'pm_ui_theme',
  AUTO_PREDICTION_ENABLED: 'pm_auto_prediction_enabled',
  AUTO_PREDICTION_INTERVAL: 'pm_auto_prediction_interval',
  NOTIFICATIONS_ENABLED: 'pm_notifications_enabled',
  NOTIFICATION_SOUND: 'pm_notification_sound',
  NOTIFICATION_CRITICAL_ONLY: 'pm_notification_critical_only',
  HISTORY_ITEMS_LIMIT: 'pm_history_items_limit',
  CHART_DATA_POINTS: 'pm_chart_data_points',
  EXPORT_FORMAT: 'pm_export_format',
  EXPORT_INCLUDE_TIMESTAMPS: 'pm_export_include_timestamps',
};

interface SettingsViewProps {
  userRole?: 'admin' | 'operator' | 'viewer';
}

// ============================================================================
// Helper Functions
// ============================================================================

function getStoredValue<T>(key: string, defaultValue: T): T {
  try {
    const stored = localStorage.getItem(key);
    if (stored === null) return defaultValue;
    if (typeof defaultValue === 'boolean') {
      return (stored === 'true') as unknown as T;
    }
    if (typeof defaultValue === 'number') {
      const parsed = parseFloat(stored);
      return (isNaN(parsed) ? defaultValue : parsed) as unknown as T;
    }
    return stored as unknown as T;
  } catch {
    return defaultValue;
  }
}

function setStoredValue(key: string, value: string | number | boolean): void {
  try {
    localStorage.setItem(key, String(value));
  } catch {
    // ignore
  }
}

// ============================================================================
// Main Component
// ============================================================================

export default function SettingsView({ userRole }: SettingsViewProps) {
  // Theme Settings
  const [uiTheme, setUiTheme] = useState(() =>
    getStoredValue(SETTINGS_KEYS.UI_THEME, 'dark')
  );
  const [themeSaved, setThemeSaved] = useState(false);

  // VLM Settings
  const [vlmEndpoint, setVlmEndpoint] = useState(() =>
    getStoredValue(SETTINGS_KEYS.VLM_ENDPOINT, '')
  );
  const [vlmStatus, setVlmStatus] = useState<'unknown' | 'connected' | 'error'>('unknown');
  const [checkingVlm, setCheckingVlm] = useState(false);
  const [vlmSaved, setVlmSaved] = useState(false);

  // LLM Settings
  const [llmEnabled, setLlmEnabled] = useState(() =>
    getStoredValue(SETTINGS_KEYS.LLM_ENABLED, true)
  );
  const [llmTemperature, setLlmTemperature] = useState(() =>
    getStoredValue(SETTINGS_KEYS.LLM_TEMPERATURE, 0.7)
  );
  const [llmMaxTokens, setLlmMaxTokens] = useState(() =>
    getStoredValue(SETTINGS_KEYS.LLM_MAX_TOKENS, 512)
  );
  const [llmTimeout, setLlmTimeout] = useState(() =>
    getStoredValue(SETTINGS_KEYS.LLM_TIMEOUT, 120)
  );
  const [llmSaved, setLlmSaved] = useState(false);

  // Auto-Prediction Settings
  const [autoPredictionEnabled, setAutoPredictionEnabled] = useState(() =>
    getStoredValue(SETTINGS_KEYS.AUTO_PREDICTION_ENABLED, true)
  );
  const [autoPredictionInterval, setAutoPredictionInterval] = useState(() =>
    getStoredValue(SETTINGS_KEYS.AUTO_PREDICTION_INTERVAL, 150)
  );
  const [autoPredictionSaved, setAutoPredictionSaved] = useState(false);

  // Notification Settings
  const [notificationsEnabled, setNotificationsEnabled] = useState(() =>
    getStoredValue(SETTINGS_KEYS.NOTIFICATIONS_ENABLED, true)
  );
  const [notificationSound, setNotificationSound] = useState(() =>
    getStoredValue(SETTINGS_KEYS.NOTIFICATION_SOUND, true)
  );
  const [criticalOnly, setCriticalOnly] = useState(() =>
    getStoredValue(SETTINGS_KEYS.NOTIFICATION_CRITICAL_ONLY, false)
  );
  const [notificationSaved, setNotificationSaved] = useState(false);

  // Data & Export Settings
  const [historyLimit, setHistoryLimit] = useState(() =>
    getStoredValue(SETTINGS_KEYS.HISTORY_ITEMS_LIMIT, 100)
  );
  const [chartDataPoints, setChartDataPoints] = useState(() =>
    getStoredValue(SETTINGS_KEYS.CHART_DATA_POINTS, 50)
  );
  const [exportFormat, setExportFormat] = useState(() =>
    getStoredValue(SETTINGS_KEYS.EXPORT_FORMAT, 'csv')
  );
  const [includeTimestamps, setIncludeTimestamps] = useState(() =>
    getStoredValue(SETTINGS_KEYS.EXPORT_INCLUDE_TIMESTAMPS, true)
  );
  const [dataSaved, setDataSaved] = useState(false);
  // Cache clearing
  const [clearingCache, setClearingCache] = useState(false);
  const [cacheCleared, setCacheCleared] = useState(false);

  // Expanded accordion state
  const [expanded, setExpanded] = useState<string | false>('theme');

  // Get the settings context for syncing
  const { updateSetting: updateContextSetting, refreshSettings } = useSettings();

  // Sync local state with context on mount
  useEffect(() => {
    refreshSettings();
  }, [refreshSettings]);

  // ============================================================================
  // Restore Defaults Functions
  // ============================================================================

  const restoreThemeDefaults = () => {
    const defaults = DEFAULTS_BY_SECTION.theme;
    setUiTheme(defaults.theme);
    setStoredValue(SETTINGS_KEYS.UI_THEME, defaults.theme);
    updateContextSetting('theme', defaults.theme);
    document.documentElement.setAttribute('data-theme', defaults.theme);
    setThemeSaved(true);
    setTimeout(() => setThemeSaved(false), 2000);
  };

  const restoreAutoPredictionDefaults = () => {
    const defaults = DEFAULTS_BY_SECTION.autoPrediction;
    setAutoPredictionEnabled(defaults.autoPredictionEnabled);
    setAutoPredictionInterval(defaults.autoPredictionInterval);
    setStoredValue(SETTINGS_KEYS.AUTO_PREDICTION_ENABLED, defaults.autoPredictionEnabled);
    setStoredValue(SETTINGS_KEYS.AUTO_PREDICTION_INTERVAL, defaults.autoPredictionInterval);
    updateContextSetting('autoPredictionEnabled', defaults.autoPredictionEnabled);
    updateContextSetting('autoPredictionInterval', defaults.autoPredictionInterval);
    setAutoPredictionSaved(true);
    setTimeout(() => setAutoPredictionSaved(false), 2000);
  };

  const restoreVlmDefaults = () => {
    const defaults = DEFAULTS_BY_SECTION.vlm;
    setVlmEndpoint(defaults.vlmEndpoint);
    setStoredValue(SETTINGS_KEYS.VLM_ENDPOINT, defaults.vlmEndpoint);
    updateContextSetting('vlmEndpoint', defaults.vlmEndpoint);
    setVlmStatus('unknown');
    setVlmSaved(true);
    setTimeout(() => setVlmSaved(false), 2000);
  };

  const restoreLlmDefaults = () => {
    const defaults = DEFAULTS_BY_SECTION.llm;
    setLlmEnabled(defaults.llmEnabled);
    setLlmTemperature(defaults.llmTemperature);
    setLlmMaxTokens(defaults.llmMaxTokens);
    setLlmTimeout(defaults.llmTimeout);
    setStoredValue(SETTINGS_KEYS.LLM_ENABLED, defaults.llmEnabled);
    setStoredValue(SETTINGS_KEYS.LLM_TEMPERATURE, defaults.llmTemperature);
    setStoredValue(SETTINGS_KEYS.LLM_MAX_TOKENS, defaults.llmMaxTokens);
    setStoredValue(SETTINGS_KEYS.LLM_TIMEOUT, defaults.llmTimeout);
    updateContextSetting('llmEnabled', defaults.llmEnabled);
    updateContextSetting('llmTemperature', defaults.llmTemperature);
    updateContextSetting('llmMaxTokens', defaults.llmMaxTokens);
    updateContextSetting('llmTimeout', defaults.llmTimeout);
    setLlmSaved(true);
    setTimeout(() => setLlmSaved(false), 2000);
  };

  const restoreNotificationDefaults = () => {
    const defaults = DEFAULTS_BY_SECTION.notifications;
    setNotificationsEnabled(defaults.notificationsEnabled);
    setNotificationSound(defaults.notificationSound);
    setCriticalOnly(defaults.notificationCriticalOnly);
    setStoredValue(SETTINGS_KEYS.NOTIFICATIONS_ENABLED, defaults.notificationsEnabled);
    setStoredValue(SETTINGS_KEYS.NOTIFICATION_SOUND, defaults.notificationSound);
    setStoredValue(SETTINGS_KEYS.NOTIFICATION_CRITICAL_ONLY, defaults.notificationCriticalOnly);
    updateContextSetting('notificationsEnabled', defaults.notificationsEnabled);
    updateContextSetting('notificationSound', defaults.notificationSound);
    updateContextSetting('notificationCriticalOnly', defaults.notificationCriticalOnly);
    setNotificationSaved(true);
    setTimeout(() => setNotificationSaved(false), 2000);
  };

  const restoreDataDefaults = () => {
    const defaults = DEFAULTS_BY_SECTION.data;
    setHistoryLimit(defaults.historyItemsLimit);
    setChartDataPoints(defaults.chartDataPoints);
    setExportFormat(defaults.exportFormat);
    setIncludeTimestamps(defaults.exportIncludeTimestamps);
    setStoredValue(SETTINGS_KEYS.HISTORY_ITEMS_LIMIT, defaults.historyItemsLimit);
    setStoredValue(SETTINGS_KEYS.CHART_DATA_POINTS, defaults.chartDataPoints);
    setStoredValue(SETTINGS_KEYS.EXPORT_FORMAT, defaults.exportFormat);
    setStoredValue(SETTINGS_KEYS.EXPORT_INCLUDE_TIMESTAMPS, defaults.exportIncludeTimestamps);
    updateContextSetting('historyItemsLimit', defaults.historyItemsLimit);
    updateContextSetting('chartDataPoints', defaults.chartDataPoints);
    updateContextSetting('exportFormat', defaults.exportFormat);
    updateContextSetting('exportIncludeTimestamps', defaults.exportIncludeTimestamps);
    setDataSaved(true);
    setTimeout(() => setDataSaved(false), 2000);
  };

  // ============================================================================
  // Theme Functions
  // ============================================================================

  const handleThemeChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const newTheme = event.target.value;
    setUiTheme(newTheme);
  };

  const saveThemeSettings = () => {
    setStoredValue(SETTINGS_KEYS.UI_THEME, uiTheme);
    updateContextSetting('theme', uiTheme as 'dark' | 'light');
    // Apply theme by updating data attribute
    document.documentElement.setAttribute('data-theme', uiTheme);
    setThemeSaved(true);
    setTimeout(() => setThemeSaved(false), 2000);
  };

  // ============================================================================
  // VLM Endpoint Functions
  // ============================================================================

  const checkVlmConnection = useCallback(async () => {
    if (!vlmEndpoint.trim()) {
      setVlmStatus('unknown');
      return;
    }

    setCheckingVlm(true);
    try {
      // Try to reach the VLM endpoint with a simple request
      const url = vlmEndpoint.trim().replace(/\/$/, '');
      await fetch(`${url}/health`, {
        method: 'GET',
        signal: AbortSignal.timeout(5000),
        mode: 'no-cors', // Jetson might not have CORS headers
      });

      // With no-cors, we can't read the response, but if it doesn't throw, it's reachable
      setVlmStatus('connected');
    } catch {
      // Try without /health
      try {
        await fetch(vlmEndpoint.trim(), {
          method: 'HEAD',
          signal: AbortSignal.timeout(5000),
          mode: 'no-cors',
        });
        setVlmStatus('connected');
      } catch {
        setVlmStatus('error');
      }
    } finally {
      setCheckingVlm(false);
    }
  }, [vlmEndpoint]);

  const saveVlmSettings = () => {
    setStoredValue(SETTINGS_KEYS.VLM_ENDPOINT, vlmEndpoint.trim());
    updateContextSetting('vlmEndpoint', vlmEndpoint.trim());
    setVlmSaved(true);
    setTimeout(() => setVlmSaved(false), 2000);
  };

  // ============================================================================
  // LLM Settings Functions
  // ============================================================================

  const saveLlmSettings = () => {
    setStoredValue(SETTINGS_KEYS.LLM_ENABLED, llmEnabled);
    setStoredValue(SETTINGS_KEYS.LLM_TEMPERATURE, llmTemperature);
    setStoredValue(SETTINGS_KEYS.LLM_MAX_TOKENS, llmMaxTokens);
    setStoredValue(SETTINGS_KEYS.LLM_TIMEOUT, llmTimeout);
    updateContextSetting('llmEnabled', llmEnabled);
    updateContextSetting('llmTemperature', llmTemperature);
    updateContextSetting('llmMaxTokens', llmMaxTokens);
    updateContextSetting('llmTimeout', llmTimeout);
    setLlmSaved(true);
    setTimeout(() => setLlmSaved(false), 2000);
  };

  // ============================================================================
  // Auto-Prediction Functions
  // ============================================================================

  const handleAutoPredictionIntervalChange = (event: SelectChangeEvent<number>) => {
    setAutoPredictionInterval(Number(event.target.value));
  };

  const saveAutoPredictionSettings = () => {
    setStoredValue(SETTINGS_KEYS.AUTO_PREDICTION_ENABLED, autoPredictionEnabled);
    setStoredValue(SETTINGS_KEYS.AUTO_PREDICTION_INTERVAL, autoPredictionInterval);
    updateContextSetting('autoPredictionEnabled', autoPredictionEnabled);
    updateContextSetting('autoPredictionInterval', autoPredictionInterval);
    setAutoPredictionSaved(true);
    setTimeout(() => setAutoPredictionSaved(false), 2000);
  };

  // ============================================================================
  // Notification Functions
  // ============================================================================

  const saveNotificationSettings = () => {
    setStoredValue(SETTINGS_KEYS.NOTIFICATIONS_ENABLED, notificationsEnabled);
    setStoredValue(SETTINGS_KEYS.NOTIFICATION_SOUND, notificationSound);
    setStoredValue(SETTINGS_KEYS.NOTIFICATION_CRITICAL_ONLY, criticalOnly);
    updateContextSetting('notificationsEnabled', notificationsEnabled);
    updateContextSetting('notificationSound', notificationSound);
    updateContextSetting('notificationCriticalOnly', criticalOnly);
    setNotificationSaved(true);
    setTimeout(() => setNotificationSaved(false), 2000);
  };

  // ============================================================================
  // Data & Export Functions
  // ============================================================================

  const saveDataSettings = () => {
    setStoredValue(SETTINGS_KEYS.HISTORY_ITEMS_LIMIT, historyLimit);
    setStoredValue(SETTINGS_KEYS.CHART_DATA_POINTS, chartDataPoints);
    setStoredValue(SETTINGS_KEYS.EXPORT_FORMAT, exportFormat);
    setStoredValue(SETTINGS_KEYS.EXPORT_INCLUDE_TIMESTAMPS, includeTimestamps);
    updateContextSetting('historyItemsLimit', historyLimit);
    updateContextSetting('chartDataPoints', chartDataPoints);
    updateContextSetting('exportFormat', exportFormat as 'csv' | 'json');
    updateContextSetting('exportIncludeTimestamps', includeTimestamps);
    setDataSaved(true);
    setTimeout(() => setDataSaved(false), 2000);
  };

  // ============================================================================
  // Cache Clearing
  // ============================================================================

  const clearBrowserCache = async () => {
    setClearingCache(true);
    try {
      // Clear specific app-related localStorage items (not auth tokens)
      const keysToKeep = ['pm_access_token', 'pm_refresh_token', 'pm_user_info'];
      const allKeys = Object.keys(localStorage);

      allKeys.forEach((key) => {
        if (key.startsWith('pm_') && !keysToKeep.includes(key)) {
          localStorage.removeItem(key);
        }
      });

      // Reset all settings to defaults
      setUiTheme('dark');
      setVlmEndpoint('');
      setLlmEnabled(true);
      setLlmTemperature(0.7);
      setLlmMaxTokens(512);
      setLlmTimeout(120);
      setAutoPredictionEnabled(true);
      setAutoPredictionInterval(150);
      setNotificationsEnabled(true);
      setNotificationSound(true);
      setCriticalOnly(false);
      setHistoryLimit(100);
      setChartDataPoints(50);
      setExportFormat('csv');
      setIncludeTimestamps(true);

      // Refresh context to sync with defaults
      refreshSettings();

      setCacheCleared(true);
      setTimeout(() => setCacheCleared(false), 3000);
    } finally {
      setClearingCache(false);
    }
  };

  // ============================================================================
  // Accordion Handler
  // ============================================================================

  const handleAccordionChange =
    (panel: string) => (_event: React.SyntheticEvent, isExpanded: boolean) => {
      setExpanded(isExpanded ? panel : false);
    };

  // ============================================================================
  // Render
  // ============================================================================

  return (
    <Box>
      {/* Header */}
      <Box sx={{ mb: 4 }}>
        <Typography
          variant="h4"
          sx={{
            fontWeight: 700,
            background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            mb: 1,
          }}
        >
          Settings
        </Typography>
        <Typography variant="body1" sx={{ color: '#d1d5db' }}>
          Dashboard configuration and system preferences
        </Typography>
      </Box>

      {/* Success/Error messages */}
      {cacheCleared && (
        <Alert severity="success" sx={{ mb: 2 }}>
          Browser cache cleared successfully. Settings reset to defaults.
        </Alert>
      )}

      {/* Settings Accordions */}
      <Stack spacing={2}>
        {/* ================================================================== */}
        {/* Theme & Display */}
        {/* ================================================================== */}
        <Accordion
          expanded={expanded === 'theme'}
          onChange={handleAccordionChange('theme')}
          sx={{
            background: 'linear-gradient(135deg, rgba(31, 41, 55, 0.9) 0%, rgba(17, 24, 39, 0.9) 100%)',
            border: '1px solid rgba(255, 255, 255, 0.1)',
            '&:before': { display: 'none' },
          }}
        >
          <AccordionSummary expandIcon={<ExpandMoreIcon sx={{ color: '#9ca3af' }} />}>
            <Stack direction="row" spacing={2} alignItems="center">
              <ThemeIcon sx={{ color: '#8b5cf6' }} />
              <Box>
                <Typography variant="subtitle1" sx={{ color: '#e5e7eb', fontWeight: 600 }}>
                  Theme & Display
                </Typography>
                <Typography variant="caption" sx={{ color: '#9ca3af' }}>
                  Customize dashboard appearance
                </Typography>
              </Box>
              <Chip
                size="small"
                icon={uiTheme === 'dark' ? <DarkModeIcon sx={{ fontSize: 16 }} /> : <LightModeIcon sx={{ fontSize: 16 }} />}
                label={uiTheme === 'dark' ? 'Dark Mode' : 'Light Mode'}
                color={uiTheme === 'dark' ? 'default' : 'warning'}
                variant="outlined"
                sx={{ ml: 'auto' }}
              />
            </Stack>
          </AccordionSummary>
          <AccordionDetails>
            <Stack spacing={3}>
              <Box>
                <Typography variant="body2" sx={{ color: '#9ca3af', mb: 2 }}>
                  Select Theme
                </Typography>
                <RadioGroup row value={uiTheme} onChange={handleThemeChange}>
                  <FormControlLabel
                    value="dark"
                    control={<Radio />}
                    label={
                      <Stack direction="row" spacing={1} alignItems="center">
                        <DarkModeIcon sx={{ color: '#6b7280' }} />
                        <Box>
                          <Typography variant="body2" sx={{ color: '#e5e7eb' }}>
                            Dark Mode
                          </Typography>
                          <Typography variant="caption" sx={{ color: '#9ca3af' }}>
                            Easier on eyes, recommended for extended use
                          </Typography>
                        </Box>
                      </Stack>
                    }
                    sx={{
                      mr: 4,
                      p: 2,
                      borderRadius: 2,
                      border: uiTheme === 'dark' ? '2px solid #667eea' : '1px solid rgba(255,255,255,0.1)',
                      bgcolor: uiTheme === 'dark' ? 'rgba(102, 126, 234, 0.1)' : 'transparent',
                    }}
                  />
                  <FormControlLabel
                    value="light"
                    control={<Radio />}
                    label={
                      <Stack direction="row" spacing={1} alignItems="center">
                        <LightModeIcon sx={{ color: '#f59e0b' }} />
                        <Box>
                          <Typography variant="body2" sx={{ color: '#e5e7eb' }}>
                            Light Mode
                          </Typography>
                          <Typography variant="caption" sx={{ color: '#9ca3af' }}>
                            Brighter interface for well-lit environments
                          </Typography>
                        </Box>
                      </Stack>
                    }
                    sx={{
                      p: 2,
                      borderRadius: 2,
                      border: uiTheme === 'light' ? '2px solid #f59e0b' : '1px solid rgba(255,255,255,0.1)',
                      bgcolor: uiTheme === 'light' ? 'rgba(245, 158, 11, 0.1)' : 'transparent',
                    }}
                  />
                </RadioGroup>
              </Box>

              <Divider sx={{ borderColor: 'rgba(255, 255, 255, 0.1)' }} />

              <Stack direction="row" spacing={2}>
                <Button
                  variant="contained"
                  startIcon={<SaveIcon />}
                  onClick={saveThemeSettings}
                  color={themeSaved ? 'success' : 'primary'}
                >
                  {themeSaved ? 'Saved!' : 'Apply Theme'}
                </Button>
                <Button
                  variant="outlined"
                  startIcon={<RestoreIcon />}
                  onClick={restoreThemeDefaults}
                >
                  Restore Defaults
                </Button>
                <Button
                  variant="outlined"
                  color="warning"
                  startIcon={clearingCache ? <CircularProgress size={16} /> : <ClearCacheIcon />}
                  onClick={clearBrowserCache}
                  disabled={clearingCache}
                >
                  Reset All Settings
                </Button>
              </Stack>

              <Alert severity="info">
                Theme changes are saved to your browser. Resetting will restore all settings to defaults.
              </Alert>
            </Stack>
          </AccordionDetails>
        </Accordion>

        {/* ================================================================== */}
        {/* Auto-Prediction Settings */}
        {/* ================================================================== */}
        <Accordion
          expanded={expanded === 'autopred'}
          onChange={handleAccordionChange('autopred')}
          sx={{
            background: 'linear-gradient(135deg, rgba(31, 41, 55, 0.9) 0%, rgba(17, 24, 39, 0.9) 100%)',
            border: '1px solid rgba(255, 255, 255, 0.1)',
            '&:before': { display: 'none' },
          }}
        >
          <AccordionSummary expandIcon={<ExpandMoreIcon sx={{ color: '#9ca3af' }} />}>
            <Stack direction="row" spacing={2} alignItems="center">
              <ScheduleIcon sx={{ color: '#10b981' }} />
              <Box>
                <Typography variant="subtitle1" sx={{ color: '#e5e7eb', fontWeight: 600 }}>
                  Auto-Prediction
                </Typography>
                <Typography variant="caption" sx={{ color: '#9ca3af' }}>
                  Configure automatic prediction scheduling
                </Typography>
              </Box>
              <Chip
                size="small"
                label={autoPredictionEnabled ? 'Enabled' : 'Disabled'}
                color={autoPredictionEnabled ? 'success' : 'default'}
                variant="outlined"
                sx={{ ml: 'auto' }}
              />
            </Stack>
          </AccordionSummary>
          <AccordionDetails>
            <Stack spacing={3}>
              <FormControlLabel
                control={
                  <Switch
                    checked={autoPredictionEnabled}
                    onChange={(e) => setAutoPredictionEnabled(e.target.checked)}
                    color="primary"
                  />
                }
                label={
                  <Box>
                    <Typography variant="body1" sx={{ color: '#e5e7eb' }}>
                      Enable Auto-Prediction
                    </Typography>
                    <Typography variant="caption" sx={{ color: '#9ca3af' }}>
                      Automatically run predictions at regular intervals when monitoring a machine
                    </Typography>
                  </Box>
                }
              />

              <FormControl fullWidth disabled={!autoPredictionEnabled}>
                <InputLabel id="auto-interval-label">Prediction Interval</InputLabel>
                <Select
                  labelId="auto-interval-label"
                  value={autoPredictionInterval}
                  label="Prediction Interval"
                  onChange={handleAutoPredictionIntervalChange}
                >
                  <MenuItem value={60}>Every 1 minute (Fast)</MenuItem>
                  <MenuItem value={90}>Every 1.5 minutes</MenuItem>
                  <MenuItem value={120}>Every 2 minutes</MenuItem>
                  <MenuItem value={150}>Every 2.5 minutes (Default)</MenuItem>
                  <MenuItem value={180}>Every 3 minutes</MenuItem>
                  <MenuItem value={300}>Every 5 minutes (Slow)</MenuItem>
                </Select>
              </FormControl>

              <Alert severity="info">
                Faster intervals provide more up-to-date predictions but increase system load. 
                Recommended: 2-3 minutes for production use.
              </Alert>

              <Stack direction="row" spacing={2}>
                <Button
                  variant="contained"
                  startIcon={<SaveIcon />}
                  onClick={saveAutoPredictionSettings}
                  color={autoPredictionSaved ? 'success' : 'primary'}
                >
                  {autoPredictionSaved ? 'Saved!' : 'Save Settings'}
                </Button>
                <Button
                  variant="outlined"
                  startIcon={<RestoreIcon />}
                  onClick={restoreAutoPredictionDefaults}
                >
                  Restore Defaults
                </Button>
              </Stack>
            </Stack>
          </AccordionDetails>
        </Accordion>

        {/* ================================================================== */}
        {/* VLM / Jetson Configuration */}
        {/* ================================================================== */}
        <Accordion
          expanded={expanded === 'vlm'}
          onChange={handleAccordionChange('vlm')}
          sx={{
            background: 'linear-gradient(135deg, rgba(31, 41, 55, 0.9) 0%, rgba(17, 24, 39, 0.9) 100%)',
            border: '1px solid rgba(255, 255, 255, 0.1)',
            '&:before': { display: 'none' },
          }}
        >
          <AccordionSummary expandIcon={<ExpandMoreIcon sx={{ color: '#9ca3af' }} />}>
            <Stack direction="row" spacing={2} alignItems="center">
              <VideocamIcon sx={{ color: '#10b981' }} />
              <Box>
                <Typography variant="subtitle1" sx={{ color: '#e5e7eb', fontWeight: 600 }}>
                  VLM / Jetson Integration
                </Typography>
                <Typography variant="caption" sx={{ color: '#9ca3af' }}>
                  Configure Vision Language Model endpoint (Jetson Orin Nano)
                </Typography>
              </Box>
              {vlmEndpoint && (
                <Chip
                  size="small"
                  label={vlmStatus === 'connected' ? 'Configured' : vlmStatus === 'error' ? 'Error' : 'Not Tested'}
                  color={vlmStatus === 'connected' ? 'success' : vlmStatus === 'error' ? 'error' : 'default'}
                  variant="outlined"
                  sx={{ ml: 'auto' }}
                />
              )}
            </Stack>
          </AccordionSummary>
          <AccordionDetails>
            <Stack spacing={3}>
              <Alert severity="info" sx={{ mb: 1 }}>
                Configure the endpoint URL for your Jetson Orin Nano running the VLM inference service.
                This will be used for visual machine monitoring integration.
              </Alert>

              <TextField
                label="VLM Endpoint URL"
                value={vlmEndpoint}
                onChange={(e) => setVlmEndpoint(e.target.value)}
                fullWidth
                placeholder="http://192.168.1.100:8080"
                helperText="Enter the base URL of your Jetson VLM service"
                InputProps={{
                  startAdornment: (
                    <InputAdornment position="start">
                      <VideocamIcon sx={{ color: '#6b7280' }} />
                    </InputAdornment>
                  ),
                }}
              />

              <Stack direction="row" spacing={2}>
                <Button
                  variant="outlined"
                  startIcon={checkingVlm ? <CircularProgress size={16} /> : <RefreshIcon />}
                  onClick={checkVlmConnection}
                  disabled={checkingVlm || !vlmEndpoint.trim()}
                >
                  Test Connection
                </Button>
                <Button
                  variant="contained"
                  startIcon={<SaveIcon />}
                  onClick={saveVlmSettings}
                  color={vlmSaved ? 'success' : 'primary'}
                >
                  {vlmSaved ? 'Saved!' : 'Save'}
                </Button>
                <Button
                  variant="outlined"
                  startIcon={<RestoreIcon />}
                  onClick={restoreVlmDefaults}
                >
                  Restore Defaults
                </Button>
              </Stack>

              {vlmStatus !== 'unknown' && (
                <Alert severity={vlmStatus === 'connected' ? 'success' : 'warning'}>
                  {vlmStatus === 'connected'
                    ? 'VLM endpoint appears reachable (Note: Full validation requires VLM service response)'
                    : 'Could not reach VLM endpoint. Check the URL and ensure the Jetson service is running.'}
                </Alert>
              )}
            </Stack>
          </AccordionDetails>
        </Accordion>

        {/* ================================================================== */}
        {/* LLM Runtime Controls */}
        {/* ================================================================== */}
        <Accordion
          expanded={expanded === 'llm'}
          onChange={handleAccordionChange('llm')}
          sx={{
            background: 'linear-gradient(135deg, rgba(31, 41, 55, 0.9) 0%, rgba(17, 24, 39, 0.9) 100%)',
            border: '1px solid rgba(255, 255, 255, 0.1)',
            '&:before': { display: 'none' },
          }}
        >
          <AccordionSummary expandIcon={<ExpandMoreIcon sx={{ color: '#9ca3af' }} />}>
            <Stack direction="row" spacing={2} alignItems="center">
              <LLMIcon sx={{ color: '#f59e0b' }} />
              <Box>
                <Typography variant="subtitle1" sx={{ color: '#e5e7eb', fontWeight: 600 }}>
                  LLM Runtime Controls
                </Typography>
                <Typography variant="caption" sx={{ color: '#9ca3af' }}>
                  Configure Large Language Model inference parameters
                </Typography>
              </Box>
              <Chip
                size="small"
                label={llmEnabled ? 'Enabled' : 'Disabled'}
                color={llmEnabled ? 'success' : 'default'}
                variant="outlined"
                sx={{ ml: 'auto' }}
              />
            </Stack>
          </AccordionSummary>
          <AccordionDetails>
            <Stack spacing={3}>
              <FormControlLabel
                control={
                  <Switch
                    checked={llmEnabled}
                    onChange={(e) => setLlmEnabled(e.target.checked)}
                    color="primary"
                  />
                }
                label={
                  <Box>
                    <Typography variant="body1" sx={{ color: '#e5e7eb' }}>
                      Enable LLM Explanations
                    </Typography>
                    <Typography variant="caption" sx={{ color: '#9ca3af' }}>
                      When disabled, predictions will run without AI-generated explanations (faster)
                    </Typography>
                  </Box>
                }
              />

              <Divider sx={{ borderColor: 'rgba(255, 255, 255, 0.1)' }} />

              <Box>
                <Typography variant="body2" sx={{ color: '#9ca3af', mb: 2 }}>
                  Temperature: {llmTemperature.toFixed(2)}
                </Typography>
                <Slider
                  value={llmTemperature}
                  onChange={(_, value) => setLlmTemperature(value as number)}
                  min={0}
                  max={1}
                  step={0.05}
                  disabled={!llmEnabled}
                  marks={[
                    { value: 0, label: '0 (Precise)' },
                    { value: 0.5, label: '0.5' },
                    { value: 1, label: '1 (Creative)' },
                  ]}
                  sx={{
                    color: '#667eea',
                    '& .MuiSlider-markLabel': { color: '#6b7280', fontSize: '0.75rem' },
                  }}
                />
              </Box>

              <TextField
                label="Max Tokens"
                type="number"
                value={llmMaxTokens}
                onChange={(e) => setLlmMaxTokens(Math.max(64, Math.min(4096, parseInt(e.target.value) || 512)))}
                disabled={!llmEnabled}
                helperText="Maximum tokens for LLM response (64-4096)"
                inputProps={{ min: 64, max: 4096 }}
              />

              <TextField
                label="Timeout (seconds)"
                type="number"
                value={llmTimeout}
                onChange={(e) => setLlmTimeout(Math.max(30, Math.min(600, parseInt(e.target.value) || 120)))}
                disabled={!llmEnabled}
                helperText="Maximum wait time for LLM response (30-600 seconds)"
                inputProps={{ min: 30, max: 600 }}
              />

              <Stack direction="row" spacing={2}>
                <Button
                  variant="contained"
                  startIcon={<SaveIcon />}
                  onClick={saveLlmSettings}
                  color={llmSaved ? 'success' : 'primary'}
                >
                  {llmSaved ? 'Saved!' : 'Save LLM Settings'}
                </Button>
                <Button
                  variant="outlined"
                  startIcon={<RestoreIcon />}
                  onClick={restoreLlmDefaults}
                >
                  Restore Defaults
                </Button>
              </Stack>
            </Stack>
          </AccordionDetails>
        </Accordion>

        {/* ================================================================== */}
        {/* Notification Settings */}
        {/* ================================================================== */}
        <Accordion
          expanded={expanded === 'notifications'}
          onChange={handleAccordionChange('notifications')}
          sx={{
            background: 'linear-gradient(135deg, rgba(31, 41, 55, 0.9) 0%, rgba(17, 24, 39, 0.9) 100%)',
            border: '1px solid rgba(255, 255, 255, 0.1)',
            '&:before': { display: 'none' },
          }}
        >
          <AccordionSummary expandIcon={<ExpandMoreIcon sx={{ color: '#9ca3af' }} />}>
            <Stack direction="row" spacing={2} alignItems="center">
              <NotificationIcon sx={{ color: '#ec4899' }} />
              <Box>
                <Typography variant="subtitle1" sx={{ color: '#e5e7eb', fontWeight: 600 }}>
                  Notifications
                </Typography>
                <Typography variant="caption" sx={{ color: '#9ca3af' }}>
                  Configure alerts and notification preferences
                </Typography>
              </Box>
              <Chip
                size="small"
                label={notificationsEnabled ? 'On' : 'Off'}
                color={notificationsEnabled ? 'success' : 'default'}
                variant="outlined"
                sx={{ ml: 'auto' }}
              />
            </Stack>
          </AccordionSummary>
          <AccordionDetails>
            <Stack spacing={3}>
              <FormControlLabel
                control={
                  <Switch
                    checked={notificationsEnabled}
                    onChange={(e) => setNotificationsEnabled(e.target.checked)}
                    color="primary"
                  />
                }
                label={
                  <Box>
                    <Typography variant="body1" sx={{ color: '#e5e7eb' }}>
                      Enable Notifications
                    </Typography>
                    <Typography variant="caption" sx={{ color: '#9ca3af' }}>
                      Show toast notifications for prediction results and system events
                    </Typography>
                  </Box>
                }
              />

              <FormControlLabel
                control={
                  <Switch
                    checked={notificationSound}
                    onChange={(e) => setNotificationSound(e.target.checked)}
                    color="primary"
                    disabled={!notificationsEnabled}
                  />
                }
                label={
                  <Box>
                    <Typography variant="body1" sx={{ color: notificationsEnabled ? '#e5e7eb' : '#6b7280' }}>
                      Sound Alerts
                    </Typography>
                    <Typography variant="caption" sx={{ color: '#9ca3af' }}>
                      Play audio for critical alerts (requires browser permission)
                    </Typography>
                  </Box>
                }
              />

              <FormControlLabel
                control={
                  <Switch
                    checked={criticalOnly}
                    onChange={(e) => setCriticalOnly(e.target.checked)}
                    color="warning"
                    disabled={!notificationsEnabled}
                  />
                }
                label={
                  <Box>
                    <Typography variant="body1" sx={{ color: notificationsEnabled ? '#e5e7eb' : '#6b7280' }}>
                      Critical Alerts Only
                    </Typography>
                    <Typography variant="caption" sx={{ color: '#9ca3af' }}>
                      Only show notifications for critical/failure predictions
                    </Typography>
                  </Box>
                }
              />

              <Stack direction="row" spacing={2}>
                <Button
                  variant="contained"
                  startIcon={<SaveIcon />}
                  onClick={saveNotificationSettings}
                  color={notificationSaved ? 'success' : 'primary'}
                >
                  {notificationSaved ? 'Saved!' : 'Save Notification Settings'}
                </Button>
                <Button
                  variant="outlined"
                  startIcon={<RestoreIcon />}
                  onClick={restoreNotificationDefaults}
                >
                  Restore Defaults
                </Button>
              </Stack>
            </Stack>
          </AccordionDetails>
        </Accordion>

        {/* ================================================================== */}
        {/* Data & Export Settings */}
        {/* ================================================================== */}
        <Accordion
          expanded={expanded === 'data'}
          onChange={handleAccordionChange('data')}
          sx={{
            background: 'linear-gradient(135deg, rgba(31, 41, 55, 0.9) 0%, rgba(17, 24, 39, 0.9) 100%)',
            border: '1px solid rgba(255, 255, 255, 0.1)',
            '&:before': { display: 'none' },
          }}
        >
          <AccordionSummary expandIcon={<ExpandMoreIcon sx={{ color: '#9ca3af' }} />}>
            <Stack direction="row" spacing={2} alignItems="center">
              <DataIcon sx={{ color: '#14b8a6' }} />
              <Box>
                <Typography variant="subtitle1" sx={{ color: '#e5e7eb', fontWeight: 600 }}>
                  Data & Export
                </Typography>
                <Typography variant="caption" sx={{ color: '#9ca3af' }}>
                  Configure data display limits and export preferences
                </Typography>
              </Box>
            </Stack>
          </AccordionSummary>
          <AccordionDetails>
            <Stack spacing={3}>
              <Typography variant="body2" sx={{ color: '#9ca3af', fontWeight: 600 }}>
                Display Limits
              </Typography>

              <TextField
                label="History Items Limit"
                type="number"
                value={historyLimit}
                onChange={(e) => setHistoryLimit(Math.max(10, Math.min(1000, parseInt(e.target.value) || 100)))}
                helperText="Maximum prediction history items to display (10-1000)"
                inputProps={{ min: 10, max: 1000 }}
              />

              <TextField
                label="Chart Data Points"
                type="number"
                value={chartDataPoints}
                onChange={(e) => setChartDataPoints(Math.max(10, Math.min(200, parseInt(e.target.value) || 50)))}
                helperText="Maximum data points shown in sensor charts (10-200)"
                inputProps={{ min: 10, max: 200 }}
              />

              <Divider sx={{ borderColor: 'rgba(255, 255, 255, 0.1)' }} />

              <Typography variant="body2" sx={{ color: '#9ca3af', fontWeight: 600 }}>
                Export Preferences
              </Typography>

              <FormControl fullWidth>
                <InputLabel id="export-format-label">Default Export Format</InputLabel>
                <Select
                  labelId="export-format-label"
                  value={exportFormat}
                  label="Default Export Format"
                  onChange={(e) => setExportFormat(e.target.value)}
                >
                  <MenuItem value="csv">CSV (Comma-Separated Values)</MenuItem>
                  <MenuItem value="json">JSON (JavaScript Object Notation)</MenuItem>
                </Select>
              </FormControl>

              <FormControlLabel
                control={
                  <Switch
                    checked={includeTimestamps}
                    onChange={(e) => setIncludeTimestamps(e.target.checked)}
                    color="primary"
                  />
                }
                label={
                  <Box>
                    <Typography variant="body1" sx={{ color: '#e5e7eb' }}>
                      Include Timestamps
                    </Typography>
                    <Typography variant="caption" sx={{ color: '#9ca3af' }}>
                      Add ISO timestamps to exported data files
                    </Typography>
                  </Box>
                }
              />

              <Stack direction="row" spacing={2}>
                <Button
                  variant="contained"
                  startIcon={<SaveIcon />}
                  onClick={saveDataSettings}
                  color={dataSaved ? 'success' : 'primary'}
                >
                  {dataSaved ? 'Saved!' : 'Save Data Settings'}
                </Button>
                <Button
                  variant="outlined"
                  startIcon={<RestoreIcon />}
                  onClick={restoreDataDefaults}
                >
                  Restore Defaults
                </Button>
              </Stack>
            </Stack>
          </AccordionDetails>
        </Accordion>

        {/* ================================================================== */}
        {/* User Management (Admin Only) */}
        {/* ================================================================== */}
        {userRole === 'admin' && (
          <Paper
            elevation={3}
            sx={{
              p: 0,
              background: 'linear-gradient(135deg, rgba(31, 41, 55, 0.9) 0%, rgba(17, 24, 39, 0.9) 100%)',
              border: '1px solid rgba(255, 255, 255, 0.1)',
            }}
          >
            <UserManagementView userRole={userRole} />
          </Paper>
        )}
      </Stack>
    </Box>
  );
}
