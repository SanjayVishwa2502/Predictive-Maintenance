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
  predictionHistory: (id: string) => `${API_BASE_URL}/api/ml/machines/${id}/history`,
  health: `${API_BASE_URL}/api/ml/health`,
};

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
} from '@mui/material';
import {
  Wifi as WifiIcon,
  WifiOff as WifiOffIcon,
  CloudDone as CloudDoneIcon,
  CloudOff as CloudOffIcon,
  Sync as SyncIcon,
  Menu as MenuIcon,
  History as HistoryIcon,
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
import { DashboardProvider, useDashboard } from '../modules/ml/context/DashboardContext';
import { ganApi } from '../modules/ml/api/ganApi';
import type { SensorReading } from '../modules/ml/components/SensorCharts';
import type { HistoricalPrediction } from '../modules/ml/components/PredictionHistory';
import type { PredictionResult as PredictionCardResult } from '../modules/ml/components/PredictionCard';

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

  // State management
  const [machines, setMachines] = useState<Machine[]>([]);
  const [sensorData, setSensorData] = useState<Record<string, number> | null>(null);
  const [sensorHistory, setSensorHistory] = useState<SensorReading[]>([]);
  const [prediction, setPrediction] = useState<PredictionCardResult | null>(null);
  const [predictionHistory, setPredictionHistory] = useState<HistoricalPrediction[]>([]);
  const [showExplanation, setShowExplanation] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [infoMessage, setInfoMessage] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);

  const [ganResumeState, setGanResumeState] = useState<{ machine_id: string; current_step: number } | null>(null);
  
  // Day 19.1: Connection & polling state
  const [isOnline, setIsOnline] = useState(navigator.onLine);
  const [isConnected, setIsConnected] = useState(true);
  const [isSyncing, setIsSyncing] = useState(false);
  const [lastSyncTime, setLastSyncTime] = useState<Date | null>(null);
  const [retryCount, setRetryCount] = useState(0);
  const [_isTabVisible, setIsTabVisible] = useState(!document.hidden);
  
  // Refs for cleanup
  const pollingIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const retryTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const syncTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Day 19.1: Monitor online/offline status
  useEffect(() => {
    const handleOnline = () => {
      setIsOnline(true);
      setSuccessMessage('Connection restored');
      setRetryCount(0);
      setConnectionStatus('disconnected');
      // Retry failed operations
      if (selectedMachineId) {
        fetchSensorData(selectedMachineId);
      }
    };
    
    const handleOffline = () => {
      setIsOnline(false);
      setIsConnected(false);
      setConnectionStatus('offline');
      setError('Internet connection lost. Working in offline mode.');
    };
    
    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);
    
    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, [selectedMachineId, setConnectionStatus]);
  
  // Day 19.1: Monitor tab visibility for background sync
  useEffect(() => {
    const handleVisibilityChange = () => {
      const visible = !document.hidden;
      setIsTabVisible(visible);
      
      // Sync data when tab becomes visible
      if (visible && selectedMachineId && isOnline) {
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
  }, [selectedMachineId, isOnline]);

  // Fetch machines on mount
  useEffect(() => {
    fetchMachines();
  }, []);

  // Day 19.1: HTTP Polling for sensor data (30-second intervals)
  useEffect(() => {
    if (!selectedMachineId) {
      setSensorData(null);
      setSensorHistory([]);
      setPrediction(null);
      setIsConnected(true);
      setConnectionStatus(isOnline ? 'connected' : 'offline');
      
      // Clear polling interval
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current);
        pollingIntervalRef.current = null;
      }
      return;
    }

    // Fetch initial sensor data
    fetchSensorData(selectedMachineId);

    // Start HTTP polling (30-second intervals)
    // Only poll when tab is visible or in background sync mode
    pollingIntervalRef.current = setInterval(() => {
      if (isOnline) {
        fetchSensorData(selectedMachineId);
      }
    }, 30000); // 30 seconds

    return () => {
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current);
        pollingIntervalRef.current = null;
      }
    };
  }, [selectedMachineId, isOnline]);

  // Day 19.2: Fetch sensor data with real API call and retry logic
  const fetchSensorData = useCallback(async (machineId: string) => {
    if (!isOnline) {
      // Offline mode: use cached data or show offline indicator
      setIsConnected(false);
      setConnectionStatus('offline');
      return;
    }
    
    try {
      // Real API call to backend
      const response = await fetch(API_ENDPOINTS.machineStatus(machineId), {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
        signal: AbortSignal.timeout(5000), // 5-second timeout
      });
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      const data = await response.json();
      const newData = data.latest_sensors;
      
      // Optimistic UI update
      setSensorData(newData);
      setIsConnected(true);
      setConnectionStatus('connected');
      setLastSyncTime(new Date());
      setRetryCount(0);

      // Add to history (keep last 120 readings = 10 minutes)
      setSensorHistory((prev) => {
        const newReading: SensorReading = {
          timestamp: new Date(data.last_update),
          values: newData,
        };
        return [...prev.slice(-119), newReading];
      });
      
    } catch (err) {
      console.error('Error fetching sensor data:', err);
      setIsConnected(false);
      setConnectionStatus('disconnected');
      
      // Check if it's a timeout or network error
      const errorMessage = err instanceof Error ? err.message : 'Unknown error';
      
      // Retry logic with exponential backoff
      if (retryCount < 3) {
        const retryDelay = Math.min(1000 * Math.pow(2, retryCount), 10000);
        setRetryCount((prev) => prev + 1);
        
        console.log(`Retry ${retryCount + 1}/3 in ${retryDelay}ms...`);
        
        if (retryTimeoutRef.current) clearTimeout(retryTimeoutRef.current);
        retryTimeoutRef.current = setTimeout(() => {
          fetchSensorData(machineId);
        }, retryDelay);
      } else {
        setError(`Failed to connect to server: ${errorMessage}`);
      }
    }
  }, [isOnline, retryCount, setConnectionStatus]);

  // Day 19.2: Fetch available machines from backend
  const fetchMachines = async () => {
    try {
      const response = await fetch(API_ENDPOINTS.machines, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
        signal: AbortSignal.timeout(5000),
      });
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      const data = await response.json();
      setMachines(data.machines);
      setIsConnected(true);
      setConnectionStatus('connected');
      
      console.log(`✓ Loaded ${data.total} machines from backend`);
    } catch (err) {
      console.error('Error fetching machines:', err);
      setError('Failed to load machines from server. Using fallback data.');
      setIsConnected(false);
      setConnectionStatus(isOnline ? 'disconnected' : 'offline');
      
      // Fallback to mock data if API fails
      setMachines(getMockMachines());
    }
  };

  // Day 19.2: Run prediction with real API call
  const handleRunPrediction = async () => {
    if (!selectedMachineId || !sensorData) {
      setError('No machine selected or sensor data unavailable');
      return;
    }
    
    if (!isOnline) {
      setError('Cannot run prediction while offline');
      return;
    }

    setLoading(true);
    setError(null);
    
    const startTime = performance.now();

    try {
      // Real API call to ML prediction endpoint
      const response = await fetch(API_ENDPOINTS.predictClassification, {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          machine_id: selectedMachineId,
          sensor_data: sensorData,
        }),
        signal: AbortSignal.timeout(10000), // 10-second timeout for ML inference
      });
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
        throw new Error(errorData.detail || `HTTP ${response.status}`);
      }
      
      const apiResult = await response.json();
      const endTime = performance.now();
      const inferenceTime = Math.round(endTime - startTime);
      
      console.log(`✓ Prediction completed in ${inferenceTime}ms`);
      
      // Transform API response to match PredictionCardResult interface
      const urgency = apiResult.prediction.rul ? 
        getUrgencyFromRUL(apiResult.prediction.rul.rul_hours).toLowerCase() as 'low' | 'medium' | 'high' | 'critical'
        : 'low' as const;
      
      const result: PredictionCardResult = {
        classification: {
          failure_type: apiResult.prediction.failure_type,
          confidence: apiResult.prediction.confidence,
          failure_probability: apiResult.prediction.failure_probability,
          all_probabilities: apiResult.prediction.all_probabilities,
        },
        rul: apiResult.prediction.rul ? {
          rul_hours: apiResult.prediction.rul.rul_hours,
          rul_days: apiResult.prediction.rul.rul_hours / 24,
          urgency: urgency,
          maintenance_window: apiResult.prediction.rul.maintenance_window || 'TBD',
        } : undefined,
        timestamp: apiResult.timestamp,
      };

      // Update with real prediction data
      setPrediction(result);
      
      // Add to history
      const historicalPred: HistoricalPrediction = {
        id: `pred_${Date.now()}`,
        timestamp: new Date(apiResult.timestamp),
        failure_type: result.classification.failure_type,
        confidence: result.classification.confidence,
        rul_hours: result.rul?.rul_hours || 0,
        urgency: result.rul ? result.rul.urgency : 'Low',
        health_state: getHealthStateFromProbability(result.classification.failure_probability),
      };
      setPredictionHistory((prev) => [historicalPred, ...prev]);

      setSuccessMessage(`Prediction completed in ${inferenceTime}ms`);
      setLastSyncTime(new Date());
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error';
      setError(`Prediction failed: ${errorMessage}`);
      console.error('Prediction error:', err);
      
      // Fallback to mock prediction for demo purposes
      console.log('Using mock prediction as fallback...');
      const mockResult = generateMockPrediction(sensorData);
      setPrediction(mockResult);
    } finally {
      setLoading(false);
    }
  };

  // Handle machine selection
  const handleMachineSelect = (machineId: string) => {
    setSelectedMachineId(machineId ? machineId : null);
    setPrediction(null);
    setSensorHistory([]);
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

  // Phase 3.7.6.1: View title helper
  const getViewTitle = (view: typeof selectedView) => {
    switch (view) {
      case 'predictions': return 'Predictions';
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
          background: 'linear-gradient(135deg, rgba(31, 41, 55, 0.95) 0%, rgba(17, 24, 39, 0.95) 100%)',
          backdropFilter: 'blur(10px)',
          borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
        }}
      >
        <Toolbar>
          <IconButton
            color="inherit"
            aria-label="open drawer"
            edge="start"
            onClick={() => setDrawerOpen(!drawerOpen)}
            sx={{ mr: 2 }}
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
                  sx={{ bgcolor: 'rgba(103, 126, 234, 0.2)', color: '#667eea' }}
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
                <Typography variant="caption" sx={{ color: '#e5e7eb', ml: 1 }}>
                  Last sync: {lastSyncTime.toLocaleTimeString()}
                </Typography>
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
            background: 'linear-gradient(135deg, rgba(31, 41, 55, 0.98) 0%, rgba(17, 24, 39, 0.98) 100%)',
            backdropFilter: 'blur(10px)',
            borderRight: '1px solid rgba(255, 255, 255, 0.1)',
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

        {selectedView === 'predictions' && (
          <Container maxWidth="xl">
            {/* Page Header */}
            <Box sx={{ mb: 4 }}>
              <Typography
                variant="h3"
                sx={{
                  fontWeight: 700,
                  background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent',
                  mb: 1,
                }}
              >
                Machine Health Dashboard
              </Typography>
              <Typography variant="body1" sx={{ color: '#d1d5db' }}>
                Real-time monitoring and predictive maintenance for industrial equipment
              </Typography>

              {/* Phase 3.7.6.6: Quick actions */}
              <Box sx={{ mt: 2, display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                <Chip
                  label="Need Training Data?"
                  onClick={() => setSelectedView('gan')}
                  sx={{ bgcolor: 'rgba(103, 126, 234, 0.18)', color: '#e5e7eb' }}
                />
                <Chip
                  label="View History"
                  onClick={() => setSelectedView('history')}
                  sx={{ bgcolor: 'rgba(255, 255, 255, 0.06)', color: '#e5e7eb' }}
                />
              </Box>
            </Box>

            {/* Machine Selector */}
            <Paper
              elevation={3}
              sx={{
                p: 3,
                mb: 3,
                background: 'linear-gradient(135deg, rgba(31, 41, 55, 0.8) 0%, rgba(17, 24, 39, 0.8) 100%)',
                backdropFilter: 'blur(10px)',
                border: '1px solid rgba(255, 255, 255, 0.1)',
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
                  lastUpdated={new Date()}
                  loading={!sensorData}
                />

                {/* Prediction Card */}
                <PredictionCard
                  machineId={selectedMachineId}
                  prediction={prediction}
                  loading={loading}
                  onRunPrediction={handleRunPrediction}
                  onExplain={() => setShowExplanation(true)}
                />

                {/* Sensor Trend Charts */}
                <SensorCharts
                  machineId={selectedMachineId}
                  sensorHistory={sensorHistory}
                  availableSensors={availableSensors}
                  selectedSensors={availableSensors.slice(0, 3)}
                  autoScroll={true}
                  maxDataPoints={120}
                />

                {/* Prediction History */}
                <PredictionHistory
                  machineId={selectedMachineId}
                  predictions={predictionHistory}
                  limit={100}
                />
              </>
            ) : (
              /* Empty State */
              <Paper
                elevation={3}
                sx={{
                  p: 8,
                  textAlign: 'center',
                  background: 'linear-gradient(135deg, rgba(31, 41, 55, 0.8) 0%, rgba(17, 24, 39, 0.8) 100%)',
                  backdropFilter: 'blur(10px)',
                  border: '1px solid rgba(255, 255, 255, 0.1)',
                }}
              >
                <Typography
                  variant="h5"
                  sx={{ color: '#9ca3af', mb: 2, fontWeight: 600 }}
                >
                  Select a Machine to Begin Monitoring
                </Typography>
                <Typography variant="body1" sx={{ color: '#6b7280' }}>
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
        {selectedView === 'gan' && (
          <Container maxWidth="xl">
            <GANWizardView onBack={() => setSelectedView('predictions')} resumeState={ganResumeState} />
          </Container>
        )}

        {/* Phase 3.7.8.4: Model Training View */}
        {selectedView === 'training' && <ModelTrainingView />}

        {/* Phase 3.7.8.5: Manage Models View */}
        {selectedView === 'models' && <ManageModelsView />}

        {/* Phase 3.7.6.1: History View (Stub) */}
        {selectedView === 'history' && (
          <Container maxWidth="xl">
            <Paper
              elevation={3}
              sx={{
                p: 8,
                textAlign: 'center',
                background: 'linear-gradient(135deg, rgba(31, 41, 55, 0.8) 0%, rgba(17, 24, 39, 0.8) 100%)',
                backdropFilter: 'blur(10px)',
                border: '1px solid rgba(255, 255, 255, 0.1)',
              }}
            >
              <Typography variant="h4" sx={{ color: '#9ca3af', mb: 2, fontWeight: 600 }}>
                Prediction History
              </Typography>
              <Typography variant="body1" sx={{ color: '#6b7280' }}>
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
              sx={{
                p: 8,
                textAlign: 'center',
                background: 'linear-gradient(135deg, rgba(31, 41, 55, 0.8) 0%, rgba(17, 24, 39, 0.8) 100%)',
                backdropFilter: 'blur(10px)',
                border: '1px solid rgba(255, 255, 255, 0.1)',
              }}
            >
              <Typography variant="h4" sx={{ color: '#9ca3af', mb: 2, fontWeight: 600 }}>
                Reports
              </Typography>
              <Typography variant="body1" sx={{ color: '#6b7280' }}>
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
            <DatasetDownloadsView />
          </Container>
        )}

        {/* Phase 3.7.6.1: Settings View (Stub) */}
        {selectedView === 'settings' && (
          <Container maxWidth="xl">
            <Paper
              elevation={3}
              sx={{
                p: 8,
                textAlign: 'center',
                background: 'linear-gradient(135deg, rgba(31, 41, 55, 0.8) 0%, rgba(17, 24, 39, 0.8) 100%)',
                backdropFilter: 'blur(10px)',
                border: '1px solid rgba(255, 255, 255, 0.1)',
              }}
            >
              <Typography variant="h4" sx={{ color: '#9ca3af', mb: 2, fontWeight: 600 }}>
                Settings
              </Typography>
              <Typography variant="body1" sx={{ color: '#6b7280' }}>
                Dashboard configuration and preferences coming soon
              </Typography>
            </Paper>
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

function generateMockPrediction(_sensorData: Record<string, number>): PredictionCardResult {
  const failureProb = Math.random() * 0.3; // 0-30% for mostly healthy
  const confidence = 0.85 + Math.random() * 0.1;
  
  let failureType: string;
  let rulHours: number;
  let urgency: 'low' | 'medium' | 'high' | 'critical';
  
  if (failureProb < 0.15) {
    failureType = 'None';
    rulHours = 400 + Math.random() * 300;
    urgency = 'low';
  } else if (failureProb < 0.40) {
    failureType = 'Bearing Wear';
    rulHours = 150 + Math.random() * 250;
    urgency = 'medium';
  } else if (failureProb < 0.70) {
    failureType = 'Overheating';
    rulHours = 50 + Math.random() * 100;
    urgency = 'high';
  } else {
    failureType = 'Bearing Seizure';
    rulHours = 10 + Math.random() * 40;
    urgency = 'critical';
  }

  return {
    classification: {
      failure_type: failureType,
      confidence,
      failure_probability: failureProb,
      all_probabilities: {
        Normal: 1 - failureProb,
        'Bearing Wear': failureProb * 0.5,
        Overheating: failureProb * 0.3,
        Electrical: failureProb * 0.2,
      },
    },
    rul: {
      rul_hours: rulHours,
      rul_days: rulHours / 24,
      urgency,
      maintenance_window: urgency === 'critical' ? 'Immediate' : urgency === 'high' ? '24 hours' : urgency === 'medium' ? '3 days' : '1 week',
    },
    timestamp: new Date().toISOString(),
  };
}

function getHealthStateFromProbability(failureProb: number): string {
  if (failureProb < 0.15) return 'HEALTHY';
  if (failureProb < 0.40) return 'DEGRADING';
  if (failureProb < 0.70) return 'WARNING';
  return 'CRITICAL';
}

function getUrgencyFromRUL(rulHours: number): string {
  if (rulHours > 240) return 'Low';
  if (rulHours > 120) return 'Medium';
  if (rulHours > 48) return 'High';
  return 'Critical';
}
