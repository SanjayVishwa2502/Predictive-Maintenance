/**
 * ML Dashboard Page - Phase 3.7.5 Day 19.2
 * 
 * Main dashboard for Machine Learning predictions and monitoring
 * Integrates all ML components into a unified interface:
 * - Machine Selector
 * - Real-time Sensor Dashboard
 * - Prediction Card with ML results
 * - Sensor Trend Charts
 * - LLM Explanation Modal
 * - Prediction History Table
 * 
 * Architecture: Single-machine monitoring (not fleet-wide)
 * Data Flow: Select machine → Load sensors → Run prediction → View history
 * 
 * Day 19.1 Features:
 * - HTTP polling for sensor data (30-second intervals)
 * - Connection status indicator
 * - Optimistic UI updates
 * - Background sync when tab inactive
 * - Offline mode handling with retry
 * 
 * Day 19.2 Features:
 * - Backend API integration (replaced all mock data)
 * - Real ML predictions from trained models
 * - Live sensor data from backend
 * - Prediction history from database
 * - Health check monitoring
 */

// API Configuration
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';
const API_ENDPOINTS = {
  machines: `${API_BASE_URL}/api/ml/machines`,
  machineStatus: (id: string) => `${API_BASE_URL}/api/ml/machines/${id}/status`,
  predictClassification: `${API_BASE_URL}/api/ml/predict/classification`,
  predictRUL: `${API_BASE_URL}/api/ml/predict/rul`,
  predictAnomaly: `${API_BASE_URL}/api/ml/predict/anomaly`,
  predictTimeseries: `${API_BASE_URL}/api/ml/predict/timeseries`,
  predictionHistory: (id: string) => `${API_BASE_URL}/api/ml/machines/${id}/history`,
  health: `${API_BASE_URL}/api/ml/health`,
  llmInfo: `${API_BASE_URL}/api/llm/info`,
};

// Text-output mode is manual-triggered (no auto-refresh).

import { useState, useEffect, useMemo, useRef, useCallback } from 'react';
import {
  Container,
  Box,
  Paper,
  Typography,
  Breadcrumbs,
  Link,
  Alert,
  Snackbar,
  Chip,
  Drawer,
  AppBar,
  Toolbar,
  IconButton,
  CssBaseline,
  Avatar,
  Tooltip,
  Menu,
  MenuItem,
  Divider,
  ListItemIcon,
  ListItemText,
} from '@mui/material';
import { alpha } from '@mui/material/styles';
import {
  Wifi as WifiIcon,
  WifiOff as WifiOffIcon,
  CloudDone as CloudDoneIcon,
  CloudOff as CloudOffIcon,
  Sync as SyncIcon,
  Menu as MenuIcon,
  History as HistoryIcon,
  Logout as LogoutIcon,
  HourglassEmpty as HourglassEmptyIcon,
} from '@mui/icons-material';
import MachineSelector from '../modules/ml/components/MachineSelector';
import SensorDashboard from '../modules/ml/components/SensorDashboard';
import PredictionCard from '../modules/ml/components/PredictionCard';
import SensorCharts from '../modules/ml/components/SensorCharts';
import LLMExplanationModal from '../modules/ml/components/LLMExplanationModal';
import PredictionHistory from '../modules/ml/components/PredictionHistory';
import NavigationPanel from '../modules/ml/components/NavigationPanel';
import DatasetDownloadsView from '../modules/ml/components/gan/DatasetDownloadsView';
import { TaskSessionProvider } from '../modules/ml/context/TaskSessionContext';
import TaskSessionView from '../modules/ml/components/TaskSessionView';
import GANWizardView from '../modules/ml/components/gan/GANWizardView';
import GlobalTaskBanner from '../modules/ml/components/GlobalTaskBanner';
import ModelTrainingView from '../modules/ml/components/training/ModelTrainingView';
import ManageModelsView from '../modules/ml/components/models/ManageModelsView';
import TaskStatusPoller from '../modules/ml/components/TaskStatusPoller';
import SettingsView from '../modules/ml/components/settings/SettingsView';
import { DashboardProvider, useDashboard } from '../modules/ml/context/DashboardContext';
import { ganApi } from '../modules/ml/api/ganApi';
import VLMIntegrationView from '../modules/vlm/VLMIntegrationView';
import type { MachineBaselineResponse } from '../modules/ml/types/gan.types';
import type { SensorReading } from '../modules/ml/components/SensorCharts';
import type { HistoricalPrediction } from '../modules/ml/components/PredictionHistory';
import type { PredictionResult as PredictionCardResult } from '../modules/ml/components/PredictionCard';
import { RoleBadge, clearTokens, getCachedUserInfo, ROLE_CONFIG } from '../App';
import type { UserInfo, UserRole } from '../App';
import { useSettings } from '../contexts/SettingsContext';
import { useNotification } from '../hooks/useNotification';

type SnapshotHistoryRow = HistoricalPrediction;

type BaselineRanges = NonNullable<MachineBaselineResponse['baseline_ranges']>;

function getClientId(): string {
  const key = 'pm_client_id';
  try {
    const existing = window.localStorage.getItem(key);
    if (existing && existing.trim()) return existing;
    const generated = typeof crypto !== 'undefined' && 'randomUUID' in crypto
      ? crypto.randomUUID()
      : `pm_${Math.random().toString(16).slice(2)}_${Date.now()}`;
    window.localStorage.setItem(key, generated);
    return generated;
  } catch {
    return `pm_${Math.random().toString(16).slice(2)}_${Date.now()}`;
  }
}

const ACCESS_TOKEN_KEY = 'pm_access_token';

function getAccessToken(): string | null {
  try {
    const token = window.localStorage.getItem(ACCESS_TOKEN_KEY);
    return token && token.trim() ? token : null;
  } catch {
    return null;
  }
}

function authHeaders(extra?: Record<string, string>): HeadersInit {
  const token = getAccessToken();
  const headers: Record<string, string> = { ...(extra || {}) };
  if (token) headers.Authorization = `Bearer ${token}`;
  return headers;
}

async function fetchWithAuth(input: RequestInfo | URL, init: RequestInit = {}): Promise<Response> {
  const headers = new Headers(init.headers);
  const token = getAccessToken();
  if (token) headers.set('Authorization', `Bearer ${token}`);
  const resp = await fetch(input, { ...init, headers });
  if (resp.status === 401) {
    clearTokens();
    try {
      window.location.reload();
    } catch {
      // ignore
    }
  }
  return resp;
}

function getSessionId(): string {
  const key = 'pm_session_id';
  try {
    const existing = window.sessionStorage.getItem(key);
    if (existing && existing.trim()) return existing;
    const generated = typeof crypto !== 'undefined' && 'randomUUID' in crypto
      ? crypto.randomUUID()
      : `sess_${Math.random().toString(16).slice(2)}_${Date.now()}`;
    window.sessionStorage.setItem(key, generated);
    return generated;
  } catch {
    return `sess_${Math.random().toString(16).slice(2)}_${Date.now()}`;
  }
}

// ============================================================================
// TYPESCRIPT INTERFACES
// ============================================================================

interface Machine {
  machine_id: string;
  display_name: string;
  category: string;
  manufacturer: string;
  model: string;
  sensor_count: number;
  has_classification_model: boolean;
  has_regression_model: boolean;
  has_anomaly_model: boolean;
  has_timeseries_model: boolean;
}

// ============================================================================
// MAIN COMPONENT
// ============================================================================

function MLDashboardPageInner() {
  // Phase 3.7.6.1: Navigation state
  const [drawerOpen, setDrawerOpen] = useState(false);
  const {
    selectedView,
    setSelectedView,
    selectedMachineId,
    setSelectedMachineId,
    connectionStatus,
    setConnectionStatus,
  } = useDashboard();

  // Settings context for auto-prediction and LLM configuration
  const { settings } = useSettings();
  
  // Notification hook for alerts
  const { notifyPredictionComplete, notifyError: _notifyError, notifySuccess: _notifySuccess } = useNotification();
  // _notifyError and _notifySuccess are available for future use in error handling

  // User profile state
  const [currentUser, setCurrentUser] = useState<UserInfo | null>(null);
  const [userMenuAnchor, setUserMenuAnchor] = useState<null | HTMLElement>(null);
  const userMenuOpen = Boolean(userMenuAnchor);

  // Load user on mount
  useEffect(() => {
    const user = getCachedUserInfo();
    setCurrentUser(user);
  }, []);

  const handleLogout = () => {
    setUserMenuAnchor(null);
    clearTokens();
    window.location.reload();
  };

  // State management
  const [machines, setMachines] = useState<Machine[]>([]);
  const [sensorData, setSensorData] = useState<Record<string, number> | null>(null);
  const [dataAgeSeconds, setDataAgeSeconds] = useState<number | null>(null);
  const [sensorHistory, setSensorHistory] = useState<SensorReading[]>([]);
  const [baselineRanges, setBaselineRanges] = useState<BaselineRanges | null>(null);
  const [prediction, setPrediction] = useState<PredictionCardResult | null>(null);
  const [predictionHistory, setPredictionHistory] = useState<SnapshotHistoryRow[]>([]);
  const [selectedPredictionRunId, setSelectedPredictionRunId] = useState<string | null>(null);
  const selectedPredictionRunIdRef = useRef<string | null>(null);
  const [showExplanation, setShowExplanation] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [infoMessage, setInfoMessage] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);

  // Text-only output (current UX focus)
  const [textOutput, setTextOutput] = useState<string>('');
  const [textOutputLoading, setTextOutputLoading] = useState(false);
  const [textOutputError, setTextOutputError] = useState<string | null>(null);

  // Fetch profile baseline ranges for monitoring thresholds (best-effort; never blocks UI)
  useEffect(() => {
    let cancelled = false;

    if (!selectedMachineId) {
      setBaselineRanges(null);
      return;
    }

    (async () => {
      try {
        const resp = await ganApi.getMachineBaseline(selectedMachineId);
        if (cancelled) return;
        const ranges = resp?.baseline_ranges;
        setBaselineRanges(ranges && typeof ranges === 'object' ? (ranges as BaselineRanges) : null);
      } catch {
        if (cancelled) return;
        setBaselineRanges(null);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [selectedMachineId]);
  const [machineSwitchLockRunId, setMachineSwitchLockRunId] = useState<string | null>(null);
  const machineSwitchLockRunIdRef = useRef<string | null>(null);
  const llmWsRef = useRef<WebSocket | null>(null);
  const llmWsRetryTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const llmWsShouldRunRef = useRef(true);

  const [ganResumeState, setGanResumeState] = useState<{ machine_id: string; current_step: number } | null>(null);

  useEffect(() => {
    selectedPredictionRunIdRef.current = selectedPredictionRunId;
  }, [selectedPredictionRunId]);

  useEffect(() => {
    machineSwitchLockRunIdRef.current = machineSwitchLockRunId;
  }, [machineSwitchLockRunId]);
  
  // Day 19.1: Connection & polling state
  // Treat "Online" as "backend reachable" (not browser Internet connectivity).
  const [isOnline, setIsOnline] = useState(true);
  const [isConnected, setIsConnected] = useState(true);
  const [isSyncing, setIsSyncing] = useState(false);
  const [lastSyncTime, setLastSyncTime] = useState<Date | null>(null);
  const [retryCount, setRetryCount] = useState(0);
  const [_isTabVisible, setIsTabVisible] = useState(!document.hidden);
  
  // Refs for cleanup
  const pollingIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const retryTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const syncTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const isFetchingStatusRef = useRef(false);
  const retryCountRef = useRef(0);

  useEffect(() => {
    retryCountRef.current = retryCount;
  }, [retryCount]);
  
  // Day 19.1: Monitor tab visibility for background sync
  useEffect(() => {
    const handleVisibilityChange = () => {
      const visible = !document.hidden;
      setIsTabVisible(visible);
      
      // Sync data when tab becomes visible
      if (visible && selectedMachineId) {
        setIsSyncing(true);
        fetchSensorData(selectedMachineId);
        
        // Clear syncing state after 2 seconds
        if (syncTimeoutRef.current) clearTimeout(syncTimeoutRef.current);
        syncTimeoutRef.current = setTimeout(() => setIsSyncing(false), 2000);
      }
    };
    
    document.addEventListener('visibilitychange', handleVisibilityChange);
    
    return () => {
      document.removeEventListener('visibilitychange', handleVisibilityChange);
      if (syncTimeoutRef.current) clearTimeout(syncTimeoutRef.current);
    };
  }, [selectedMachineId]);

  // Fetch machines on mount
  useEffect(() => {
    fetchMachines();
  }, []);

  // Day 19.1: HTTP Polling for sensor data (5-second intervals)
  useEffect(() => {
    if (!selectedMachineId) {
      setSensorData(null);
      setDataAgeSeconds(null);
      setSensorHistory([]);
      setPrediction(null);
      setTextOutput('');
      setTextOutputError(null);
      setTextOutputLoading(false);
      setIsConnected(true);
      setConnectionStatus('disconnected');
      
      // Clear polling interval
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current);
        pollingIntervalRef.current = null;
      }
      return;
    }

    // Fetch initial sensor data
    fetchSensorData(selectedMachineId);

    // Start HTTP polling (5-second intervals)
    // Only poll when tab is visible or in background sync mode
    if (pollingIntervalRef.current) {
      clearInterval(pollingIntervalRef.current);
      pollingIntervalRef.current = null;
    }

    pollingIntervalRef.current = setInterval(() => {
      fetchSensorData(selectedMachineId);
    }, 5000); // 5 seconds

    return () => {
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current);
        pollingIntervalRef.current = null;
      }
    };
  }, [selectedMachineId]);

  const formatRunDetailsText = (run: any): string => {
    const llm = run?.llm || {};
    const predictions = run?.predictions || {};
    const sensorDataFromRun = run?.sensor_data || {};
    const runType = String(run?.run_type || '').trim().toLowerCase();
    const machineLine = `Machine: ${String(run?.machine_id || '')}`;
    const stampLine = `Stamp: ${String(run?.data_stamp || '')}`;

    const computeHint =
      llm?.combined?.compute ||
      llm?.classification?.compute ||
      llm?.rul?.compute ||
      llm?.anomaly?.compute ||
      llm?.timeseries?.compute ||
      'unknown';

    const computeLine = `LLM: ${String(computeHint)}`;

    // Format sensor data from the run (what the LLM used)
    const sensorLines: string[] = [];
    if (sensorDataFromRun && typeof sensorDataFromRun === 'object') {
      const entries = Object.entries(sensorDataFromRun);
      if (entries.length > 0) {
        for (const [key, value] of entries) {
          const numVal = Number(value);
          if (Number.isFinite(numVal)) {
            sensorLines.push(`  ${key}: ${numVal.toFixed(1)} C`);
          }
        }
      }
    }
    const sensorSection = sensorLines.length > 0
      ? `Sensors at run time:\n${sensorLines.join('\n')}`
      : 'Sensors: (no data)';

    const combinedText = llm?.combined?.summary
      ? String(llm.combined.summary)
      : (runType === 'prediction'
          ? '[Prediction-only run - click "Prediction" button to generate AI explanation]'
          : '[AI explanation pending - the LLM is processing or has not been requested yet]');

    const combinedTextClean = combinedText
      .replaceAll('Â°C', ' C')
      .replaceAll('°C', ' C')
      .replaceAll('Â°F', ' F')
      .replaceAll('°F', ' F');

    const fmtPct = (v: any) => {
      const n = Number(v);
      if (!Number.isFinite(n)) return 'n/a';
      return `${(n * 100).toFixed(2)}%`;
    };

    const fmtNum = (v: any, digits = 2) => {
      const n = Number(v);
      if (!Number.isFinite(n)) return 'n/a';
      return n.toFixed(digits);
    };

    const predictionSummaryLines = (): string[] => {
      const lines: string[] = [];

      const cls = predictions?.classification;
      if (cls?.error) {
        lines.push(`Classification: error: ${String(cls.error)}`);
      } else if (cls) {
        const failureType = String(cls.failure_type ?? 'unknown');
        const labelProb = cls.confidence;
        const failureRisk = cls.failure_probability;
        lines.push(
          `Classification: ${failureType} | p=${fmtPct(labelProb)} | risk=${fmtPct(failureRisk)} | conf=${fmtNum(cls.confidence, 3)}`,
        );
      }

      const rul = predictions?.rul;
      if (rul?.skipped) {
        lines.push('RUL: n/a (no RUL model for this machine)');
      } else if (rul?.error) {
        const msg = String(rul.error);
        if (msg.toLowerCase().includes('rul model not found')) {
          lines.push('RUL: n/a (no RUL model for this machine)');
        } else {
          lines.push(`RUL: error: ${msg}`);
        }
      } else if (rul) {
        lines.push(
          `RUL: ${fmtNum(rul.rul_hours, 2)} hours (${fmtNum(rul.rul_days, 2)} days) | urgency=${String(rul.urgency ?? 'n/a')} | conf=${fmtNum(rul.confidence, 3)}`,
        );
      }

      const ano = predictions?.anomaly;
      if (ano?.error) {
        lines.push(`Anomaly: error: ${String(ano.error)}`);
      } else if (ano) {
        lines.push(
          `Anomaly: is_anomaly=${String(ano.is_anomaly)} | score=${fmtNum(ano.anomaly_score, 4)} | method=${String(ano.detection_method ?? 'unknown')}`,
        );
      }

      const ts = predictions?.timeseries;
      if (ts?.error) {
        lines.push(`Forecast: error: ${String(ts.error)}`);
      } else if (ts?.forecast_summary) {
        const summary = String(ts.forecast_summary);
        const firstTwoLines = summary.split('\n').slice(0, 2).join('\n');
        lines.push(`Forecast: ${firstTwoLines}${summary.includes('\n') ? ' ...' : ''}`);
      }

      return lines;
    };

    return [
      stampLine,
      machineLine,
      computeLine,
      sensorSection,
      '',
      'Notes',
      combinedTextClean,
      '',
      '[Predictions]',
      ...predictionSummaryLines(),
    ].join('\n');
  };

  const loadSnapshots = useCallback(async (machineId: string) => {
    try {
      const resp = await fetchWithAuth(`${API_BASE_URL}/api/ml/machines/${encodeURIComponent(machineId)}/snapshots?limit=500`, {
        method: 'GET',
        headers: authHeaders({ 'Content-Type': 'application/json' }),
        signal: AbortSignal.timeout(30_000),
      });
      if (!resp.ok) return;
      const json = await resp.json();
      const snaps = Array.isArray(json?.snapshots) ? json.snapshots : [];
      const byId = new Map<string, SnapshotHistoryRow>();
      for (const s of snaps) {
        const stamp = String(s?.data_stamp || '').trim();
        if (!stamp) continue;
        const row: SnapshotHistoryRow = {
          id: stamp,
          timestamp: new Date(stamp),
          data_stamp: stamp,
          run_id: s?.run_id ?? null,
          has_run: Boolean(s?.run_id),
          has_explanation: Boolean(s?.has_explanation),
          run_type: s?.run_type ?? null, // "prediction" or "explanation"
          sensor_snapshot: (s?.sensor_data as Record<string, number>) || {},
        };
        // Keep the first occurrence (snapshots are newest-first).
        if (!byId.has(stamp)) byId.set(stamp, row);
      }
      const rows = Array.from(byId.values());

      setPredictionHistory(rows);

      // Auto-select the latest run if none selected.
      if (!selectedPredictionRunId) {
        const firstWithRun = rows.find((r) => Boolean(r.run_id));
        if (firstWithRun?.run_id) {
          setSelectedPredictionRunId(String(firstWithRun.run_id));
        }
      }
    } catch {
      // ignore
    }
  }, [selectedPredictionRunId]);

  const refreshSelectedRunText = useCallback(async (runId: string, opts?: { silent?: boolean }): Promise<boolean> => {
    if (!runId) return false;
    const silent = Boolean(opts?.silent);
    if (!silent) {
      setTextOutputLoading(true);
      setTextOutputError(null);
    }
    try {
      const resp = await fetchWithAuth(`${API_BASE_URL}/api/ml/runs/${encodeURIComponent(runId)}`, {
        method: 'GET',
        headers: authHeaders({ 'Content-Type': 'application/json' }),
        // Under heavy local inference load (model warm-up / Prophet), run reads can exceed 10s.
        signal: AbortSignal.timeout(30_000),
      });
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({ detail: resp.statusText }));
        throw new Error(err?.detail || `HTTP ${resp.status}`);
      }
      const json = await resp.json();
      const nextText = formatRunDetailsText(json);
      setTextOutput(nextText);

      // If this run already has an LLM summary, unlock machine switching.
      const hasSummary = Boolean(json?.llm?.combined?.summary);
      const runType = String(json?.run_type || '').trim().toLowerCase();
      if (hasSummary && machineSwitchLockRunIdRef.current === runId) {
        setMachineSwitchLockRunId(null);
      }

      // Stop polling when we have a summary OR when this run is prediction-only.
      return Boolean(hasSummary || runType === 'prediction');
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      if (!silent) setTextOutputError(msg);
      return false;
    } finally {
      if (!silent) setTextOutputLoading(false);
    }
  }, []);

  // LLM push channel (WebSocket) - connect once per browser client
  useEffect(() => {
    llmWsShouldRunRef.current = true;
    const clientId = getClientId();
    const wsUrl = `${API_BASE_URL}`.replace(/^http/i, 'ws') + `/ws/llm/events?client_id=${encodeURIComponent(clientId)}`;

    const connect = () => {
      if (!llmWsShouldRunRef.current) return;
      try {
        // Ensure we never keep multiple sockets around.
        if (llmWsRef.current) {
          try { llmWsRef.current.close(); } catch { /* ignore */ }
          llmWsRef.current = null;
        }

        const ws = new WebSocket(wsUrl);
        llmWsRef.current = ws;

        ws.onmessage = (evt) => {
          try {
            const msg = JSON.parse(evt.data);
            if (msg?.type !== 'llm_explanation') return;
            const runId = msg?.run_id as string | undefined;
            const useCase = msg?.use_case as 'combined' | 'classification' | 'rul' | 'anomaly' | 'timeseries' | undefined;
            const dataStamp = msg?.data_stamp as string | undefined;
            if (!runId || !useCase) return;

            if (dataStamp) {
              setPredictionHistory((prev) => prev.map((r) => {
                if (r.data_stamp !== dataStamp) return r;
                return { ...r, run_id: runId, has_run: true, has_explanation: true, run_type: 'explanation' as const };
              }));
            }

            if (selectedPredictionRunIdRef.current === runId) {
              void refreshSelectedRunText(runId);
            }

            if (machineSwitchLockRunIdRef.current === runId) {
              setMachineSwitchLockRunId(null);
            }
          } catch {
            // ignore
          }
        };

        ws.onclose = () => {
          if (!llmWsShouldRunRef.current) return;
          if (llmWsRetryTimerRef.current) clearTimeout(llmWsRetryTimerRef.current);
          llmWsRetryTimerRef.current = setTimeout(() => {
            llmWsRetryTimerRef.current = null;
            connect();
          }, 2000);
        };
      } catch {
        // ignore
      }
    };

    connect();

    return () => {
      llmWsShouldRunRef.current = false;
      if (llmWsRetryTimerRef.current) clearTimeout(llmWsRetryTimerRef.current);
      if (llmWsRef.current) {
        try { llmWsRef.current.close(); } catch { /* ignore */ }
      }
      llmWsRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // When machine changes, pull the persisted snapshots.
  useEffect(() => {
    if (!selectedMachineId) return;
    void loadSnapshots(selectedMachineId);
  }, [selectedMachineId, loadSnapshots]);

  // When a run is selected, load its details and (if needed) poll until LLM summary arrives.
  const llmPollTimerRef = useRef<number | null>(null);
  useEffect(() => {
    if (!selectedPredictionRunId) return;

    const runId = selectedPredictionRunId;
    let cancelled = false;

    // Clear any prior poll.
    if (llmPollTimerRef.current) {
      clearInterval(llmPollTimerRef.current);
      llmPollTimerRef.current = null;
    }

    const start = async () => {
      // First fetch (not silent): shows loading if needed.
      const shouldStop = await refreshSelectedRunText(runId);
      if (cancelled) return;
      if (shouldStop) {
        // Prediction complete - notify user
        const machine = machines.find(m => m.machine_id === selectedMachineId);
        notifyPredictionComplete(machine?.display_name || selectedMachineId || 'Machine', 'healthy');
        return;
      }

      // Fallback: poll for up to ~3 minutes in case WebSocket push is missed.
      const deadline = Date.now() + 180_000;
      llmPollTimerRef.current = window.setInterval(async () => {
        if (cancelled) return;
        if (selectedPredictionRunIdRef.current !== runId) return;
        if (Date.now() > deadline) {
          if (llmPollTimerRef.current) clearInterval(llmPollTimerRef.current);
          llmPollTimerRef.current = null;
          return;
        }
        const stop = await refreshSelectedRunText(runId, { silent: true });
        if (stop) {
          // Prediction complete - notify user
          const machine = machines.find(m => m.machine_id === selectedMachineId);
          notifyPredictionComplete(machine?.display_name || selectedMachineId || 'Machine', 'healthy');
          if (llmPollTimerRef.current) clearInterval(llmPollTimerRef.current);
          llmPollTimerRef.current = null;
        }
      }, 5000);
    };

    void start();

    return () => {
      cancelled = true;
      if (llmPollTimerRef.current) {
        clearInterval(llmPollTimerRef.current);
        llmPollTimerRef.current = null;
      }
    };
  }, [selectedPredictionRunId, refreshSelectedRunText]);

  // Day 19.2: Fetch sensor data with real API call and retry logic
  const fetchSensorData = useCallback(async (machineId: string) => {
    if (isFetchingStatusRef.current) return;
    isFetchingStatusRef.current = true;
    
    try {
      const sessionId = getSessionId();
      const clientId = getClientId();
      let data: any;
      try {
        // Prefer passing session/client id (query params) to ensure per-session dataset capture.
        const url = `${API_ENDPOINTS.machineStatus(machineId)}?session_id=${encodeURIComponent(sessionId)}&client_id=${encodeURIComponent(clientId)}`;
        const response = await fetchWithAuth(url, {
          method: 'GET',
          headers: authHeaders({ 'Content-Type': 'application/json' }),
          // Keep this generous to avoid flapping during local CPU-heavy inference.
          signal: AbortSignal.timeout(15_000),
        });
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        data = await response.json();
      } catch {
        // Fall back to the original endpoint if the server doesn't accept query params.
        const response = await fetchWithAuth(API_ENDPOINTS.machineStatus(machineId), {
          method: 'GET',
          headers: authHeaders({ 'Content-Type': 'application/json' }),
          signal: AbortSignal.timeout(15_000),
        });
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        data = await response.json();
      }
      const newData = data.latest_sensors;
      
      // Optimistic UI update
      setSensorData(newData);
      // Track data age from backend (how old the sensor reading is)
      setDataAgeSeconds(typeof data.data_age_seconds === 'number' ? data.data_age_seconds : null);
      // Track if LLM is currently generating for this machine
      if (typeof data.llm_busy === 'boolean') {
        setTextOutputLoading(data.llm_busy);
      }
      setIsOnline(true);
      setIsConnected(true);
      setConnectionStatus('connected');
      setLastSyncTime(new Date());
      setRetryCount(0);
      retryCountRef.current = 0;

      // Add to history (keep last 120 readings = 10 minutes)
      setSensorHistory((prev) => {
        const newReading: SensorReading = {
          timestamp: new Date(data.last_update),
          values: newData,
        };
        return [...prev.slice(-119), newReading];
      });

      // Persisted 5-second snapshot rows for the bottom history.
      const stamp = String(data.data_stamp || data.last_update || '').trim();
      if (stamp) {
        const runIdFromStatus = String(data.run_id || '').trim();
        const runTypeFromStatus = (data.run_type as ('prediction' | 'explanation' | undefined)) ?? undefined;
        const row: SnapshotHistoryRow = {
          id: stamp,
          timestamp: new Date(stamp),
          data_stamp: stamp,
          run_id: runIdFromStatus || null,
          has_run: Boolean(runIdFromStatus),
          has_explanation: Boolean(data.has_explanation),
          run_type: runTypeFromStatus ?? null,
          sensor_snapshot: newData,
        };
        setPredictionHistory((prev) => {
          if (prev.length > 0 && prev[0].id === row.id) return prev;
          const next = [row, ...prev.filter((r) => r.id !== row.id)];
          return next.slice(0, 500);
        });
      }
      
    } catch (err) {
      console.error('Error fetching sensor data:', err);
      // Don't immediately flip to disconnected for transient timeouts.
      // We'll mark disconnected only after retries are exhausted.
      
      // Check if it's a timeout or network error
      const errorMessage = err instanceof Error ? err.message : 'Unknown error';
      
      // Retry logic with exponential backoff
      const currentRetryCount = retryCountRef.current;
      if (currentRetryCount < 3) {
        setRetryCount((prev) => {
          const next = prev + 1;
          retryCountRef.current = next;
          return next;
        });

        // Avoid scheduling extra out-of-band retries while a fixed polling interval is active.
        // The next poll tick will retry automatically.
        if (retryTimeoutRef.current) {
          clearTimeout(retryTimeoutRef.current);
          retryTimeoutRef.current = null;
        }
      } else {
        setIsOnline(false);
        setIsConnected(false);
        setConnectionStatus('disconnected');
        setError(`Failed to connect to server: ${errorMessage}`);
      }
    } finally {
      isFetchingStatusRef.current = false;
    }
  }, [setConnectionStatus]);

  // Day 19.2: Fetch available machines from backend
  const fetchMachines = async () => {
    try {
      const doFetch = async (timeoutMs: number) => {
        return fetchWithAuth(API_ENDPOINTS.machines, {
          method: 'GET',
          headers: authHeaders({ 'Content-Type': 'application/json' }),
          signal: AbortSignal.timeout(timeoutMs),
        });
      };

      // The ML backend can take a bit longer on cold start (model discovery + metadata load).
      // Retry once with a longer timeout before falling back to mock data.
      let response = await doFetch(15000);
      if (!response.ok) {
        response = await doFetch(15000);
      }
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      const data = await response.json();
      setMachines(data.machines);
      setIsOnline(true);
      setIsConnected(true);
      setConnectionStatus('connected');
      
      console.log(`✓ Loaded ${data.total} machines from backend`);
    } catch (err) {
      console.error('Error fetching machines:', err);
      setError('Failed to load machines from server. Using fallback data.');
      setIsOnline(false);
      setIsConnected(false);
      setConnectionStatus('disconnected');
      
      // Fallback to mock data if API fails
      setMachines(getMockMachines());
    }
  };

  // Day 19.2: Run prediction with real API call
  const handleRunPrediction = async () => {
    if (!selectedMachineId) {
      setError('No machine selected');
      return;
    }

    if (loading || machineSwitchLockRunId) {
      setInfoMessage('Prediction/LLM is running. Please wait before starting another run.');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const clientId = getClientId();
      const sessionId = getSessionId();
      const response = await fetchWithAuth(
        `${API_BASE_URL}/api/ml/machines/${encodeURIComponent(selectedMachineId)}/auto/run_once?client_id=${encodeURIComponent(clientId)}&session_id=${encodeURIComponent(sessionId)}`,
        {
          method: 'POST',
          headers: authHeaders({ 'Content-Type': 'application/json' }),
          // Back-end run_once performs synchronous inference (Prophet can take >60s).
          // Keep this long enough to avoid client-side abort while the server is still working.
          signal: AbortSignal.timeout(180_000),
        }
      );

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
        throw new Error(errorData.detail || `HTTP ${response.status}`);
      }

      const apiResult = await response.json();
      const runId = String(apiResult?.run_id || '').trim();
      const machineId = String(apiResult?.machine_id || '').trim();
      if (!runId || !machineId) throw new Error('Invalid run response');

      setMachineSwitchLockRunId(runId);

      setIsOnline(true);
      setIsConnected(true);
      setConnectionStatus('connected');
      setSuccessMessage('Prediction started');

      setPrediction(null);
      setSelectedPredictionRunId(runId);
      await loadSnapshots(machineId);
      await refreshSelectedRunText(runId);

    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error';
      setError(`Prediction failed: ${errorMessage}`);
      console.error('Prediction error:', err);
    } finally {
      setLoading(false);
    }
  };

  // Handle machine selection
  const handleMachineSelect = (machineId: string) => {
    const requested = (machineId || '').trim();
    const current = (selectedMachineId || '').trim();
    if ((machineSwitchLockRunId || loading) && requested && requested !== current) {
      setInfoMessage('Prediction/LLM is running. Please wait before switching machines.');
      return;
    }
    setSelectedMachineId(machineId ? machineId : null);
    setPrediction(null);
    setSensorHistory([]);
    setPredictionHistory([]);
    setSelectedPredictionRunId(null);
  };

  // Get selected machine details
  const selectedMachine = useMemo(() => {
    return machines.find((m) => m.machine_id === selectedMachineId);
  }, [machines, selectedMachineId]);

  // Get available sensors for selected machine
  const availableSensors = useMemo(() => {
    if (!sensorData) return [];
    return Object.keys(sensorData);
  }, [sensorData]);

  // Sensor chart selection (max 5)
  const [selectedSensors, setSelectedSensors] = useState<string[]>([]);

  useEffect(() => {
    if (!availableSensors || availableSensors.length === 0) {
      setSelectedSensors((prev) => (prev.length === 0 ? prev : []));
      return;
    }

    setSelectedSensors((prev) => {
      const filtered = prev.filter((s) => availableSensors.includes(s));
      const next = (filtered.length > 0 ? filtered : availableSensors.slice(0, 3)).slice(0, 5);
      if (next.length === prev.length && next.every((v, i) => v === prev[i])) return prev;
      return next;
    });
  }, [selectedMachineId, availableSensors]);

  const handleSensorToggle = useCallback((sensor: string) => {
    setSelectedSensors((prev) => {
      const has = prev.includes(sensor);
      if (has) return prev.filter((s) => s !== sensor);
      if (prev.length >= 5) return prev;
      return [...prev, sensor];
    });
  }, []);

  // Phase 3.7.6.1: View title helper
  const getViewTitle = (view: typeof selectedView) => {
    switch (view) {
      case 'predictions': return 'Predictions';
      case 'vlm': return 'VLM Integration';
      case 'gan': return 'New Machine Wizard';
      case 'training': return 'Model Training';
      case 'models': return 'Manage Models';
      case 'history': return 'Prediction History';
      case 'reports': return 'Reports';
      case 'tasks': return 'Tasks';
      case 'datasets': return 'Downloads';
      case 'settings': return 'Settings';
      default: return 'Dashboard';
    }
  };

  // ============================================================================
  // RENDER
  // ============================================================================

  const handleContinueWorkflow = useCallback(async () => {
    try {
      const resp = await ganApi.getContinueWorkflow();
      const st = resp?.state;
      if (!resp?.has_state || !st?.machine_id) {
        setSelectedView('gan');
        setInfoMessage('No GAN workflow to continue yet. Start a workflow in New Machine Wizard first.');
        return;
      }

      setGanResumeState({ machine_id: st.machine_id, current_step: st.current_step ?? 0 });
      setSelectedView('gan');
      setSuccessMessage(`Continuing workflow for ${st.machine_id}`);
    } catch (e: any) {
      setError(e?.message || 'Failed to continue workflow');
    }
  }, [setSelectedView]);

  return (
    <TaskSessionProvider>
      <TaskStatusPoller />
      <Box sx={{ display: 'flex' }}>
        <CssBaseline />
      
      {/* Phase 3.7.6.1: Top App Bar with Menu Button */}
      <AppBar 
        position="fixed" 
        sx={{ 
          zIndex: (theme) => theme.zIndex.drawer + 1,
          bgcolor: (theme) => alpha(theme.palette.background.paper, theme.palette.mode === 'dark' ? 0.85 : 0.95),
          backdropFilter: (theme) => (theme.palette.mode === 'dark' ? 'blur(10px)' : undefined),
          borderBottom: 1,
          borderColor: 'divider',
        }}
      >
        <Toolbar>
          <IconButton
            aria-label="open drawer"
            edge="start"
            onClick={() => setDrawerOpen(!drawerOpen)}
            sx={{ mr: 2, color: 'text.primary' }}
          >
            <MenuIcon />
          </IconButton>
          
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', width: '100%' }}>
            <Breadcrumbs aria-label="breadcrumb" sx={{ flexGrow: 1, color: 'inherit' }}>
              <Link
                component="button"
                underline="hover"
                color="inherit"
                onClick={() => setSelectedView('predictions')}
                sx={{ font: 'inherit' }}
              >
                ML Dashboard
              </Link>
              <Typography color="inherit" noWrap>
                {getViewTitle(selectedView)}
              </Typography>
            </Breadcrumbs>

            {/* Day 19.1: Connection Status Indicators */}
            <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
              {selectedView !== 'history' && (
                <Chip
                  icon={<HistoryIcon />}
                  label="View History"
                  size="small"
                  onClick={() => setSelectedView('history')}
                />
              )}

              {isSyncing && (
                <Chip
                  icon={<SyncIcon sx={{ animation: 'spin 1s linear infinite', '@keyframes spin': { from: { transform: 'rotate(0deg)' }, to: { transform: 'rotate(360deg)' } } }} />}
                  label="Syncing"
                  size="small"
                  sx={(theme) => ({
                    bgcolor: alpha(theme.palette.primary.main, theme.palette.mode === 'dark' ? 0.18 : 0.12),
                    color: theme.palette.primary.main,
                  })}
                />
              )}

              <Chip
                icon={isOnline ? <WifiIcon /> : <WifiOffIcon />}
                label={isOnline ? 'Online' : 'Offline'}
                size="small"
                color={isOnline ? 'success' : 'error'}
                sx={{ minWidth: 100 }}
              />

              <Chip
                icon={isConnected ? <CloudDoneIcon /> : <CloudOffIcon />}
                label={connectionStatus === 'connected' ? 'Connected' : connectionStatus === 'offline' ? 'Offline' : 'Disconnected'}
                size="small"
                color={connectionStatus === 'connected' ? 'success' : connectionStatus === 'offline' ? 'error' : 'warning'}
                sx={{ minWidth: 120 }}
              />

              {lastSyncTime && isConnected && (
                <Typography variant="caption" sx={{ color: 'text.secondary', ml: 1 }}>
                  Last sync: {lastSyncTime.toLocaleTimeString()}
                </Typography>
              )}

              {/* User Profile Menu */}
              {currentUser && (
                <>
                  <Divider orientation="vertical" flexItem sx={{ mx: 1, borderColor: 'divider' }} />
                  <Tooltip title={`Signed in as ${currentUser.username}`}>
                    <IconButton
                      onClick={(e) => setUserMenuAnchor(e.currentTarget)}
                      size="small"
                      sx={{ ml: 1 }}
                    >
                      <Avatar
                        sx={(theme) => ({
                          width: 36,
                          height: 36,
                          bgcolor:
                            ROLE_CONFIG[currentUser.role as UserRole]?.color === 'error'
                              ? theme.palette.error.main
                              : ROLE_CONFIG[currentUser.role as UserRole]?.color === 'primary'
                              ? theme.palette.info.main
                              : theme.palette.success.main,
                          fontSize: '0.875rem',
                          fontWeight: 600,
                        })}
                      >
                        {currentUser.username.slice(0, 2).toUpperCase()}
                      </Avatar>
                    </IconButton>
                  </Tooltip>
                  <Menu
                    anchorEl={userMenuAnchor}
                    open={userMenuOpen}
                    onClose={() => setUserMenuAnchor(null)}
                    anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
                    transformOrigin={{ vertical: 'top', horizontal: 'right' }}
                    PaperProps={{
                      sx: {
                        minWidth: 240,
                        mt: 1,
                        bgcolor: 'background.paper',
                        border: 1,
                        borderColor: 'divider',
                      },
                    }}
                  >
                    <Box sx={{ px: 2, py: 1.5 }}>
                      <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
                        {currentUser.username}
                      </Typography>
                      <Typography variant="caption" color="text.secondary" sx={{ display: 'block' }}>
                        {currentUser.email}
                      </Typography>
                      <Box sx={{ mt: 1 }}>
                        <RoleBadge role={currentUser.role as UserRole} />
                      </Box>
                    </Box>
                    <Divider sx={{ borderColor: 'divider' }} />
                    <MenuItem onClick={handleLogout} sx={{ color: 'error.main' }}>
                      <ListItemIcon>
                        <LogoutIcon fontSize="small" sx={{ color: 'error.main' }} />
                      </ListItemIcon>
                      <ListItemText>Sign Out</ListItemText>
                    </MenuItem>
                  </Menu>
                </>
              )}
            </Box>
          </Box>
        </Toolbar>
      </AppBar>

      {/* Phase 3.7.6.1: Side Navigation Drawer */}
      <Drawer
        anchor="left"
        open={drawerOpen}
        onClose={() => setDrawerOpen(false)}
        variant="temporary"
        ModalProps={{
          keepMounted: true, // Better mobile performance
        }}
        sx={{
          '& .MuiDrawer-paper': {
            width: 280,
            boxSizing: 'border-box',
            bgcolor: (theme) => alpha(theme.palette.background.paper, theme.palette.mode === 'dark' ? 0.9 : 0.98),
            backdropFilter: (theme) => (theme.palette.mode === 'dark' ? 'blur(10px)' : undefined),
            borderRight: 1,
            borderColor: 'divider',
          },
        }}
      >
        <Toolbar /> {/* Spacer for AppBar */}
        <NavigationPanel
          selectedView={selectedView}
          onSelectView={(view: string) => {
            setSelectedView(view as typeof selectedView);
            setDrawerOpen(false);
          }}
          onContinueWorkflow={async () => {
            await handleContinueWorkflow();
            setDrawerOpen(false);
          }}
        />
      </Drawer>

      {/* Phase 3.7.6.1: Main Content Area */}
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          p: 3,
          width: '100%',
          minHeight: '100vh',
          mt: 8, // Offset for AppBar
        }}
      >
        {/* Phase 3.7.6.6: Persistent task banner across all views */}
        <GlobalTaskBanner onViewTasks={() => setSelectedView('tasks')} />

        {/* Pending approval alert for operators awaiting approval */}
        {currentUser && currentUser.role === 'operator' && currentUser.is_approved === false && (
          <Alert
            severity="warning"
            icon={<HourglassEmptyIcon />}
            sx={{
              mb: 2,
              background: (theme) => alpha(theme.palette.warning.main, theme.palette.mode === 'dark' ? 0.18 : 0.12),
              border: 1,
              borderColor: (theme) => alpha(theme.palette.warning.main, 0.45),
              '& .MuiAlert-message': { width: '100%' },
            }}
          >
            <Typography variant="body2" sx={{ fontWeight: 600 }}>
              Account Pending Approval
            </Typography>
            <Typography variant="body2" sx={{ mt: 0.5 }}>
              Your operator account is awaiting administrator approval. You will have full access once an admin approves your registration.
              Currently, you have read-only access similar to a viewer.
            </Typography>
          </Alert>
        )}

        {selectedView === 'predictions' && (
          <Container maxWidth="xl">
            {/* Page Header */}
            <Box sx={{ mb: 4 }}>
              <Typography
                variant="h3"
                sx={(theme) => ({
                  fontWeight: 700,
                  background: `linear-gradient(135deg, ${theme.palette.primary.main} 0%, ${theme.palette.secondary.main} 100%)`,
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent',
                  mb: 1,
                })}
              >
                Machine Health Dashboard
              </Typography>
              <Typography variant="body1" sx={{ color: 'text.secondary' }}>
                Real-time monitoring and predictive maintenance for industrial equipment
              </Typography>

              {/* Phase 3.7.6.6: Quick actions */}
              <Box sx={{ mt: 2, display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                <Chip
                  label="Need Training Data?"
                  onClick={() => setSelectedView('gan')}
                  sx={(theme) => ({
                    bgcolor: alpha(theme.palette.primary.main, theme.palette.mode === 'dark' ? 0.18 : 0.12),
                    color: theme.palette.text.primary,
                  })}
                />
                <Chip
                  label="View History"
                  onClick={() => setSelectedView('history')}
                  sx={{ bgcolor: 'action.hover', color: 'text.primary' }}
                />
              </Box>
            </Box>

            {/* Machine Selector */}
            <Paper
              elevation={3}
              sx={{
                p: 3,
                mb: 3,
                bgcolor: (theme) => alpha(theme.palette.background.paper, theme.palette.mode === 'dark' ? 0.75 : 0.95),
                backdropFilter: (theme) => (theme.palette.mode === 'dark' ? 'blur(10px)' : undefined),
                border: 1,
                borderColor: 'divider',
              }}
            >
              <MachineSelector
                machines={machines}
                selectedMachineId={selectedMachineId}
                onSelect={handleMachineSelect}
              />
            </Paper>

            {/* Dashboard Content - Only show when machine is selected */}
            {selectedMachineId && selectedMachine ? (
              <>
                {/* Real-Time Sensor Monitoring */}
                <SensorDashboard
                  machineId={selectedMachineId}
                  sensorData={sensorData || {}}
                  baselineRanges={baselineRanges || undefined}
                  lastUpdated={lastSyncTime}
                  loading={!sensorData}
                  connected={isConnected}
                  error={error}
                  dataAgeSeconds={dataAgeSeconds}
                />

                {/* Prediction Card (contents now show text output) */}
                <PredictionCard
                  machineId={selectedMachineId}
                  prediction={prediction}
                  loading={loading}
                  onRunPrediction={handleRunPrediction}
                  onExplain={() => setShowExplanation(true)}
                  userRole={currentUser?.role as 'admin' | 'operator' | 'viewer' | undefined}
                  isApproved={currentUser?.is_approved ?? true}
                  textOutput={textOutput}
                  textOutputLoading={textOutputLoading}
                  textOutputError={textOutputError}
                  autoRefresh={settings.autoPredictionEnabled}
                  refreshInterval={settings.autoPredictionInterval}
                />

                {/* Sensor Trend Charts */}
                <SensorCharts
                  machineId={selectedMachineId}
                  sensorHistory={sensorHistory}
                  availableSensors={availableSensors}
                  selectedSensors={selectedSensors}
                  onSensorToggle={handleSensorToggle}
                  baselineRanges={baselineRanges || undefined}
                  autoScroll={true}
                  maxDataPoints={120}
                />

                {/* Prediction History */}
                <PredictionHistory
                  machineId={selectedMachineId}
                  predictions={predictionHistory}
                  limit={500}
                  sessionId={getSessionId()}
                  onRowClick={(p) => {
                    if (!p.run_id) return;
                    setSelectedPredictionRunId(String(p.run_id));
                  }}
                  onHistoryCleared={() => {
                    setPredictionHistory([]);
                    setSelectedPredictionRunId(null);
                    setTextOutput('');
                    setTextOutputError(null);
                  }}
                  userRole={currentUser?.role as 'admin' | 'operator' | 'viewer' | undefined}
                />
              </>
            ) : (
              /* Empty State */
              <Paper
                elevation={3}
                sx={(theme) => ({
                  p: 8,
                  textAlign: 'center',
                  bgcolor: alpha(theme.palette.background.paper, theme.palette.mode === 'dark' ? 0.75 : 0.95),
                  backdropFilter: theme.palette.mode === 'dark' ? 'blur(10px)' : undefined,
                  border: 1,
                  borderColor: 'divider',
                })}
              >
                <Typography
                  variant="h5"
                  sx={{ color: 'text.secondary', mb: 2, fontWeight: 600 }}
                >
                  Select a Machine to Begin Monitoring
                </Typography>
                <Typography variant="body1" sx={{ color: 'text.secondary' }}>
                  Choose a machine from the dropdown above to view real-time sensor data,
                  run predictions, and monitor health status
                </Typography>
              </Paper>
            )}

            {/* LLM Explanation Modal */}
            {showExplanation && prediction && sensorData && (
              <LLMExplanationModal
                open={showExplanation}
                onClose={() => setShowExplanation(false)}
                machineId={selectedMachineId!}
                predictionData={{
                  run_id: selectedPredictionRunId || undefined,
                  health_state: getHealthStateFromProbability(prediction.classification.failure_probability),
                  confidence: prediction.classification.confidence,
                  failure_probability: prediction.classification.failure_probability,
                  predicted_failure_type: prediction.classification.failure_type,
                  rul_hours: prediction.rul?.rul_hours,
                  sensor_data: sensorData,
                }}
              />
            )}
          </Container>
        )}

        {/* Phase 3.7.6.2: GAN Wizard View */}
        {selectedView === 'vlm' && (
          <Container maxWidth="xl">
            <VLMIntegrationView />
          </Container>
        )}

        {selectedView === 'gan' && (
          <Container maxWidth="xl">
            <GANWizardView 
              onBack={() => setSelectedView('predictions')} 
              resumeState={ganResumeState}
              userRole={currentUser?.role as 'admin' | 'operator' | 'viewer' | undefined}
            />
          </Container>
        )}

        {/* Phase 3.7.8.4: Model Training View */}
        {selectedView === 'training' && <ModelTrainingView />}

        {/* Phase 3.7.8.5: Manage Models View */}
        {selectedView === 'models' && (
          <ManageModelsView userRole={currentUser?.role as 'admin' | 'operator' | 'viewer' | undefined} />
        )}

        {/* Phase 3.7.6.1: History View (Stub) */}
        {selectedView === 'history' && (
          <Container maxWidth="xl">
            <Paper
              elevation={3}
              sx={(theme) => ({
                p: 8,
                textAlign: 'center',
                bgcolor: alpha(theme.palette.background.paper, theme.palette.mode === 'dark' ? 0.75 : 0.95),
                backdropFilter: theme.palette.mode === 'dark' ? 'blur(10px)' : undefined,
                border: 1,
                borderColor: 'divider',
              })}
            >
              <Typography variant="h4" sx={{ color: 'text.secondary', mb: 2, fontWeight: 600 }}>
                Prediction History
              </Typography>
              <Typography variant="body1" sx={{ color: 'text.secondary' }}>
                Historical prediction analytics coming soon
              </Typography>
            </Paper>
          </Container>
        )}

        {/* Phase 3.7.6.1: Reports View (Stub) */}
        {selectedView === 'reports' && (
          <Container maxWidth="xl">
            <Paper
              elevation={3}
              sx={(theme) => ({
                p: 8,
                textAlign: 'center',
                bgcolor: alpha(theme.palette.background.paper, theme.palette.mode === 'dark' ? 0.75 : 0.95),
                backdropFilter: theme.palette.mode === 'dark' ? 'blur(10px)' : undefined,
                border: 1,
                borderColor: 'divider',
              })}
            >
              <Typography variant="h4" sx={{ color: 'text.secondary', mb: 2, fontWeight: 600 }}>
                Reports
              </Typography>
              <Typography variant="body1" sx={{ color: 'text.secondary' }}>
                Report generation and analytics coming soon
              </Typography>
            </Paper>
          </Container>
        )}

        {selectedView === 'tasks' && (
          <Container maxWidth="xl">
            <TaskSessionView />
          </Container>
        )}

        {/* Phase 3.7.6.1: Dataset Manager View (Stub) */}
        {selectedView === 'datasets' && (
          <Container maxWidth="xl">
            <DatasetDownloadsView userRole={currentUser?.role as 'admin' | 'operator' | 'viewer' | undefined} />
          </Container>
        )}

        {/* Phase 3.7.6.1: Settings View */}
        {selectedView === 'settings' && (
          <Container maxWidth="xl">
            <SettingsView userRole={currentUser?.role as 'admin' | 'operator' | 'viewer' | undefined} />
          </Container>
        )}

        {/* Global Snackbars (visible across all views) */}
        <Snackbar
          open={!!successMessage}
          autoHideDuration={3000}
          onClose={() => setSuccessMessage(null)}
          anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
        >
          <Alert
            onClose={() => setSuccessMessage(null)}
            severity="success"
            sx={{ width: '100%' }}
          >
            {successMessage}
          </Alert>
        </Snackbar>

        <Snackbar
          open={!!infoMessage}
          autoHideDuration={4000}
          onClose={() => setInfoMessage(null)}
          anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
        >
          <Alert
            onClose={() => setInfoMessage(null)}
            severity="info"
            sx={{ width: '100%' }}
          >
            {infoMessage}
          </Alert>
        </Snackbar>

        <Snackbar
          open={!!error}
          autoHideDuration={5000}
          onClose={() => setError(null)}
          anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
        >
          <Alert
            onClose={() => setError(null)}
            severity="error"
            sx={{ width: '100%' }}
          >
            {error}
          </Alert>
        </Snackbar>
      </Box>
      </Box>
    </TaskSessionProvider>
  );
}

export default function MLDashboardPage() {
  return (
    <DashboardProvider>
      <MLDashboardPageInner />
    </DashboardProvider>
  );
}

// ============================================================================
// MOCK DATA GENERATORS (Day 19.2: Kept for fallback when API unavailable)
// ============================================================================

function getMockMachines(): Machine[] {
  const categories = ['Motor', 'Pump', 'Compressor', 'CNC', 'Robot', 'Fan'];
  const manufacturers = ['SIEMENS', 'ABB', 'GRUNDFOS', 'ATLAS COPCO', 'FANUC', 'BROTHER'];
  
  return Array.from({ length: 29 }, (_, i) => ({
    machine_id: `machine_${String(i + 1).padStart(3, '0')}`,
    display_name: `${categories[i % categories.length]} ${manufacturers[i % manufacturers.length]} ${String(i + 1).padStart(3, '0')}`,
    category: categories[i % categories.length],
    manufacturer: manufacturers[i % manufacturers.length],
    model: `MODEL_${String(i + 1).padStart(3, '0')}`,
    sensor_count: 10 + Math.floor(Math.random() * 15),
    has_classification_model: true,
    has_regression_model: i % 2 === 0,
    has_anomaly_model: i % 3 === 0,
    has_timeseries_model: i % 4 === 0,
  }));
}

function getHealthStateFromProbability(failureProb: number): string {
  if (failureProb < 0.15) return 'HEALTHY';
  if (failureProb < 0.40) return 'DEGRADING';
  if (failureProb < 0.70) return 'WARNING';
  return 'CRITICAL';
}
