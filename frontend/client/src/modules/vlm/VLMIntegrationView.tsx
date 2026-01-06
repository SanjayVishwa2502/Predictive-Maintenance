/**
 * VLM Integration View
 * 
 * Displays live camera feed and inference outputs from an external VLM device.
 * The endpoint is configurable and stored in localStorage - NOT hardcoded.
 * 
 * Features:
 * - Live MJPEG camera feed from {endpoint}/stream or {endpoint}/mjpeg
 * - Real-time inference results from {endpoint}/latest
 * - Connection status monitoring
 * - Configurable endpoint (IP:port)
 */

import { useCallback, useEffect, useState, useRef } from 'react';
import {
  Alert,
  Box,
  Button,
  Card,
  CardContent,
  Chip,
  CircularProgress,
  Collapse,
  Divider,
  Grid,
  IconButton,
  Paper,
  Stack,
  TextField,
  Tooltip,
  Typography,
} from '@mui/material';
import { alpha, useTheme } from '@mui/material/styles';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import ErrorIcon from '@mui/icons-material/Error';
import RefreshIcon from '@mui/icons-material/Refresh';
import LinkIcon from '@mui/icons-material/Link';
import InfoOutlinedIcon from '@mui/icons-material/InfoOutlined';
import VideocamIcon from '@mui/icons-material/Videocam';
import VideocamOffIcon from '@mui/icons-material/VideocamOff';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import StopIcon from '@mui/icons-material/Stop';
import SettingsIcon from '@mui/icons-material/Settings';
import ExpandLessIcon from '@mui/icons-material/ExpandLess';
import VisibilityIcon from '@mui/icons-material/Visibility';
import WarningAmberIcon from '@mui/icons-material/WarningAmber';
import {
  testExternalConnection,
  getServerConfig,
  type ServerConfig,
} from '../../api/configApi';

const STORAGE_KEY = 'pm_vlm_endpoint';

function normalizeEndpoint(raw: string): string {
  const trimmed = raw.trim();
  if (!trimmed) return '';
  if (!/^https?:\/\//i.test(trimmed)) return `http://${trimmed}`;
  return trimmed.replace(/\/$/, ''); // Remove trailing slash
}

// ============================================================================
// Types for VLM inference results
// ============================================================================

interface Detection {
  label: string;
  confidence: number;
  bbox?: [number, number, number, number]; // [x, y, width, height]
}

interface VLMLatestResponse {
  timestamp?: string;
  detections?: Detection[];
  labels?: string[];
  anomaly?: boolean;
  anomaly_score?: number;
  inference_time_ms?: number;
  frame_id?: number;
  status?: string;
  error?: string;
  // Allow any additional fields from the VLM device
  [key: string]: unknown;
}

// ============================================================================
// Main Component
// ============================================================================

export default function VLMIntegrationView() {
  const theme = useTheme();
  
  // Endpoint configuration state
  const [endpointInput, setEndpointInput] = useState<string>('');
  const [savedEndpoint, setSavedEndpoint] = useState<string>('');
  const [savedMsg, setSavedMsg] = useState<string | null>(null);
  const [showSettings, setShowSettings] = useState(false);
  
  // Connection state
  const [testing, setTesting] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState<'unknown' | 'connected' | 'error'>('unknown');
  const [latencyMs, setLatencyMs] = useState<number | null>(null);
  const [connectionError, setConnectionError] = useState<string | null>(null);
  
  // Stream state
  const [streaming, setStreaming] = useState(false);
  const [streamError, setStreamError] = useState<string | null>(null);
  const [streamUrl, setStreamUrl] = useState<string | null>(null);
  const imgRef = useRef<HTMLImageElement>(null);
  
  // Inference results state
  const [latestResults, setLatestResults] = useState<VLMLatestResponse | null>(null);
  const [fetchingResults, setFetchingResults] = useState(false);
  const [resultsError, setResultsError] = useState<string | null>(null);
  const pollIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  
  // Server config
  const [serverConfig, setServerConfig] = useState<ServerConfig | null>(null);

  // ============================================================================
  // Load saved endpoint on mount
  // ============================================================================
  
  useEffect(() => {
    try {
      const existing = window.localStorage.getItem(STORAGE_KEY);
      if (existing && existing.trim()) {
        const normalized = normalizeEndpoint(existing);
        setEndpointInput(normalized);
        setSavedEndpoint(normalized);
      }
    } catch {
      // ignore
    }
    
    // Load server config
    getServerConfig().then(setServerConfig).catch(() => null);
  }, []);

  // ============================================================================
  // Cleanup on unmount
  // ============================================================================
  
  useEffect(() => {
    return () => {
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
      }
    };
  }, []);

  // ============================================================================
  // Save endpoint
  // ============================================================================
  
  const handleSave = useCallback(() => {
    const normalized = normalizeEndpoint(endpointInput);
    setSavedMsg(null);
    setConnectionError(null);

    if (!normalized) {
      setSavedMsg('Enter an endpoint (IP:port or full URL).');
      return;
    }

    try {
      window.localStorage.setItem(STORAGE_KEY, normalized);
    } catch {
      // ignore
    }

    setSavedEndpoint(normalized);
    setSavedMsg('Endpoint saved successfully.');
    setConnectionStatus('unknown');
    
    // Stop current stream if endpoint changed
    if (streaming) {
      handleStopStream();
    }
  }, [endpointInput, streaming]);

  // ============================================================================
  // Test connection
  // ============================================================================
  
  const handleTestConnection = useCallback(async () => {
    const normalized = normalizeEndpoint(endpointInput);
    if (!normalized) {
      setSavedMsg('Enter an endpoint first.');
      return;
    }

    setTesting(true);
    setConnectionError(null);
    setSavedMsg(null);

    try {
      const result = await testExternalConnection(normalized);
      if (result.connected) {
        setConnectionStatus('connected');
        setLatencyMs(result.latency_ms);
        setSavedMsg(`Connected successfully${result.latency_ms ? ` (${result.latency_ms}ms)` : ''}`);
      } else {
        setConnectionStatus('error');
        setConnectionError(result.error);
      }
    } catch (err) {
      setConnectionStatus('error');
      setConnectionError(err instanceof Error ? err.message : 'Connection test failed');
    } finally {
      setTesting(false);
    }
  }, [endpointInput]);

  // ============================================================================
  // Fetch latest results
  // ============================================================================
  
  const fetchLatestResults = useCallback(async (endpoint: string) => {
    if (!endpoint) return;
    
    setFetchingResults(true);
    try {
      const response = await fetch(`${endpoint}/latest`, {
        method: 'GET',
        signal: AbortSignal.timeout(5000),
      });
      
      if (response.ok) {
        const data = await response.json();
        setLatestResults(data);
        setResultsError(null);
      } else {
        setResultsError(`HTTP ${response.status}`);
      }
    } catch (err) {
      // Don't spam errors
      setResultsError(err instanceof Error ? err.message : 'Failed to fetch');
    } finally {
      setFetchingResults(false);
    }
  }, []);

  // ============================================================================
  // Start/Stop stream
  // ============================================================================
  
  const handleStartStream = useCallback(() => {
    if (!savedEndpoint) {
      setStreamError('No endpoint configured. Save an endpoint first.');
      return;
    }

    setStreamError(null);
    setResultsError(null);
    
    // MJPEG stream URL - the VLM device should serve at /stream
    const url = `${savedEndpoint}/stream`;
    
    setStreamUrl(url);
    setStreaming(true);
    
    // Start polling for inference results
    if (pollIntervalRef.current) {
      clearInterval(pollIntervalRef.current);
    }
    
    // Fetch immediately
    fetchLatestResults(savedEndpoint);
    
    // Poll every 500ms for real-time results
    pollIntervalRef.current = setInterval(() => {
      fetchLatestResults(savedEndpoint);
    }, 500);
  }, [savedEndpoint, fetchLatestResults]);

  const handleStopStream = useCallback(() => {
    setStreaming(false);
    setStreamUrl(null);
    setLatestResults(null);
    
    // Stop polling
    if (pollIntervalRef.current) {
      clearInterval(pollIntervalRef.current);
      pollIntervalRef.current = null;
    }
  }, []);

  // ============================================================================
  // Handle stream image events
  // ============================================================================
  
  const handleStreamError = useCallback(() => {
    setStreamError('Failed to load video stream. Check if the VLM device is serving MJPEG at /stream');
  }, []);

  const handleStreamLoad = useCallback(() => {
    setStreamError(null);
  }, []);

  // ============================================================================
  // Render
  // ============================================================================

  return (
    <Box>
      <Stack spacing={3}>
        {/* Header */}
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: 2 }}>
          <Box>
            <Typography
              variant="h4"
              sx={{
                fontWeight: 700,
                background: `linear-gradient(135deg, ${theme.palette.primary.main} 0%, ${theme.palette.secondary.main} 100%)`,
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                mb: 0.5,
              }}
            >
              VLM Integration
            </Typography>
            <Typography variant="body1" sx={{ color: 'text.secondary' }}>
              Live camera feed and AI inference from external VLM device
            </Typography>
          </Box>
          
          <Stack direction="row" spacing={1} alignItems="center">
            {/* Connection Status */}
            <Chip
              icon={savedEndpoint ? (connectionStatus === 'connected' ? <CheckCircleIcon /> : <VideocamIcon />) : <VideocamOffIcon />}
              label={savedEndpoint ? (connectionStatus === 'connected' ? 'Connected' : savedEndpoint) : 'Not Configured'}
              color={connectionStatus === 'connected' ? 'success' : savedEndpoint ? 'default' : 'warning'}
              variant="outlined"
              size="small"
            />
            
            <Tooltip title="Configure Endpoint">
              <IconButton onClick={() => setShowSettings(!showSettings)} size="small">
                <SettingsIcon />
              </IconButton>
            </Tooltip>
          </Stack>
        </Box>

        {/* Settings Panel (Collapsible) */}
        <Collapse in={showSettings}>
          <Paper
            elevation={2}
            sx={{
              p: 3,
              bgcolor: alpha(theme.palette.background.paper, theme.palette.mode === 'dark' ? 0.7 : 0.95),
              border: 1,
              borderColor: 'divider',
            }}
          >
            <Stack spacing={2}>
              <Stack direction="row" spacing={1} alignItems="center">
                <SettingsIcon sx={{ color: 'text.secondary' }} />
                <Typography variant="subtitle1" sx={{ fontWeight: 600, color: 'text.primary' }}>
                  Endpoint Configuration
                </Typography>
                <IconButton size="small" onClick={() => setShowSettings(false)} sx={{ ml: 'auto' }}>
                  <ExpandLessIcon />
                </IconButton>
              </Stack>
              
              <TextField
                label="VLM Endpoint"
                value={endpointInput}
                onChange={(e) => setEndpointInput(e.target.value)}
                placeholder="<device-ip>:<port>"
                fullWidth
                size="small"
                helperText="Enter IP:port or full URL (e.g., your-device:8080). Stored locally."
                InputProps={{
                  startAdornment: <LinkIcon sx={{ mr: 1, color: 'text.secondary', fontSize: 20 }} />,
                }}
              />
              
              <Stack direction="row" spacing={2} alignItems="center" flexWrap="wrap">
                <Button variant="contained" size="small" onClick={handleSave}>
                  Save
                </Button>
                <Button
                  variant="outlined"
                  size="small"
                  onClick={handleTestConnection}
                  disabled={testing || !endpointInput.trim()}
                  startIcon={testing ? <CircularProgress size={14} /> : <RefreshIcon />}
                >
                  {testing ? 'Testing...' : 'Test Connection'}
                </Button>
                
                {connectionStatus !== 'unknown' && (
                  <Chip
                    size="small"
                    icon={connectionStatus === 'connected' ? <CheckCircleIcon /> : <ErrorIcon />}
                    label={connectionStatus === 'connected' ? `OK${latencyMs ? ` (${latencyMs}ms)` : ''}` : 'Failed'}
                    color={connectionStatus === 'connected' ? 'success' : 'error'}
                    variant="outlined"
                  />
                )}
              </Stack>
              
              {savedMsg && <Alert severity="info" sx={{ py: 0.5 }}>{savedMsg}</Alert>}
              {connectionError && <Alert severity="error" sx={{ py: 0.5 }}>{connectionError}</Alert>}
            </Stack>
          </Paper>
        </Collapse>

        {/* No Endpoint Warning */}
        {!savedEndpoint && !showSettings && (
          <Alert
            severity="warning"
            icon={<WarningAmberIcon />}
            action={
              <Button color="inherit" size="small" onClick={() => setShowSettings(true)} startIcon={<SettingsIcon />}>
                Configure
              </Button>
            }
          >
            No VLM endpoint configured. Click Configure to set up the connection to your VLM device.
          </Alert>
        )}

        {/* Main Content - Stream and Results */}
        {savedEndpoint && (
          <Grid container spacing={3}>
            {/* Video Feed Panel */}
            <Grid item xs={12} lg={8}>
              <Paper
                elevation={3}
                sx={{
                  bgcolor: alpha(theme.palette.background.paper, theme.palette.mode === 'dark' ? 0.7 : 0.95),
                  border: 1,
                  borderColor: 'divider',
                  overflow: 'hidden',
                }}
              >
                {/* Stream Header */}
                <Box
                  sx={{
                    p: 2,
                    borderBottom: 1,
                    borderColor: 'divider',
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                  }}
                >
                  <Stack direction="row" spacing={1} alignItems="center">
                    <VideocamIcon sx={{ color: streaming ? 'success.main' : 'text.secondary' }} />
                    <Typography variant="subtitle1" sx={{ fontWeight: 600, color: 'text.primary' }}>
                      Camera Feed
                    </Typography>
                    {streaming && (
                      <Chip
                        size="small"
                        label="LIVE"
                        color="error"
                        sx={{ animation: 'pulse 2s infinite', '@keyframes pulse': { '0%, 100%': { opacity: 1 }, '50%': { opacity: 0.6 } } }}
                      />
                    )}
                  </Stack>
                  
                  <Stack direction="row" spacing={1}>
                    {!streaming ? (
                      <Button
                        variant="contained"
                        color="success"
                        size="small"
                        startIcon={<PlayArrowIcon />}
                        onClick={handleStartStream}
                      >
                        Start Stream
                      </Button>
                    ) : (
                      <Button
                        variant="outlined"
                        color="error"
                        size="small"
                        startIcon={<StopIcon />}
                        onClick={handleStopStream}
                      >
                        Stop
                      </Button>
                    )}
                  </Stack>
                </Box>
                
                {/* Stream Content */}
                <Box
                  sx={{
                    position: 'relative',
                    bgcolor: 'black',
                    minHeight: 400,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                  }}
                >
                  {streaming && streamUrl ? (
                    <>
                      <img
                        ref={imgRef}
                        src={streamUrl}
                        alt="VLM Camera Feed"
                        onError={handleStreamError}
                        onLoad={handleStreamLoad}
                        style={{
                          maxWidth: '100%',
                          maxHeight: '600px',
                          objectFit: 'contain',
                        }}
                      />
                      {streamError && (
                        <Box
                          sx={{
                            position: 'absolute',
                            inset: 0,
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            bgcolor: 'rgba(0,0,0,0.8)',
                          }}
                        >
                          <Alert severity="error" sx={{ maxWidth: 400 }}>
                            {streamError}
                            <Typography variant="caption" display="block" sx={{ mt: 1 }}>
                              Endpoint: {savedEndpoint}/stream
                            </Typography>
                          </Alert>
                        </Box>
                      )}
                    </>
                  ) : (
                    <Stack spacing={2} alignItems="center" sx={{ color: 'grey.500' }}>
                      <VideocamOffIcon sx={{ fontSize: 64, opacity: 0.5 }} />
                      <Typography variant="body1">
                        Stream not active
                      </Typography>
                      <Typography variant="caption" sx={{ color: 'grey.600' }}>
                        Click "Start Stream" to begin receiving video from {savedEndpoint}
                      </Typography>
                    </Stack>
                  )}
                </Box>
              </Paper>
            </Grid>

            {/* Inference Results Panel */}
            <Grid item xs={12} lg={4}>
              <Paper
                elevation={3}
                sx={{
                  bgcolor: alpha(theme.palette.background.paper, theme.palette.mode === 'dark' ? 0.7 : 0.95),
                  border: 1,
                  borderColor: 'divider',
                  height: '100%',
                  minHeight: 400,
                }}
              >
                {/* Results Header */}
                <Box
                  sx={{
                    p: 2,
                    borderBottom: 1,
                    borderColor: 'divider',
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                  }}
                >
                  <Stack direction="row" spacing={1} alignItems="center">
                    <VisibilityIcon sx={{ color: 'primary.main' }} />
                    <Typography variant="subtitle1" sx={{ fontWeight: 600, color: 'text.primary' }}>
                      Inference Results
                    </Typography>
                  </Stack>
                  
                  {fetchingResults && <CircularProgress size={16} />}
                </Box>
                
                {/* Results Content */}
                <CardContent sx={{ p: 2 }}>
                  {!streaming ? (
                    <Box sx={{ textAlign: 'center', py: 4, color: 'text.secondary' }}>
                      <InfoOutlinedIcon sx={{ fontSize: 48, opacity: 0.5, mb: 1 }} />
                      <Typography variant="body2">
                        Start the stream to see inference results
                      </Typography>
                    </Box>
                  ) : resultsError ? (
                    <Alert severity="warning" sx={{ mb: 2 }}>
                      Could not fetch results: {resultsError}
                      <Typography variant="caption" display="block" sx={{ mt: 0.5 }}>
                        Endpoint: {savedEndpoint}/latest
                      </Typography>
                    </Alert>
                  ) : latestResults ? (
                    <Stack spacing={2}>
                      {/* Timestamp */}
                      {latestResults.timestamp && (
                        <Typography variant="caption" sx={{ color: 'text.secondary' }}>
                          Last update: {new Date(latestResults.timestamp).toLocaleTimeString()}
                        </Typography>
                      )}
                      
                      {/* Anomaly Detection */}
                      {latestResults.anomaly !== undefined && (
                        <Card variant="outlined" sx={{ bgcolor: latestResults.anomaly ? alpha(theme.palette.error.main, 0.1) : alpha(theme.palette.success.main, 0.1) }}>
                          <CardContent sx={{ py: 1.5, '&:last-child': { pb: 1.5 } }}>
                            <Stack direction="row" spacing={1} alignItems="center">
                              <Chip
                                label={latestResults.anomaly ? 'ANOMALY DETECTED' : 'NORMAL'}
                                color={latestResults.anomaly ? 'error' : 'success'}
                                size="small"
                              />
                              {latestResults.anomaly_score !== undefined && (
                                <Typography variant="caption" sx={{ color: 'text.secondary' }}>
                                  Score: {(latestResults.anomaly_score * 100).toFixed(1)}%
                                </Typography>
                              )}
                            </Stack>
                          </CardContent>
                        </Card>
                      )}
                      
                      {/* Detections */}
                      {latestResults.detections && latestResults.detections.length > 0 && (
                        <Box>
                          <Typography variant="subtitle2" sx={{ color: 'text.secondary', mb: 1 }}>
                            Detections ({latestResults.detections.length})
                          </Typography>
                          <Stack spacing={1}>
                            {latestResults.detections.map((det, idx) => (
                              <Card key={idx} variant="outlined">
                                <CardContent sx={{ py: 1, '&:last-child': { pb: 1 } }}>
                                  <Stack direction="row" justifyContent="space-between" alignItems="center">
                                    <Typography variant="body2" sx={{ fontWeight: 600, color: 'text.primary' }}>
                                      {det.label}
                                    </Typography>
                                    <Chip
                                      label={`${(det.confidence * 100).toFixed(0)}%`}
                                      size="small"
                                      color={det.confidence > 0.8 ? 'success' : det.confidence > 0.5 ? 'warning' : 'default'}
                                    />
                                  </Stack>
                                </CardContent>
                              </Card>
                            ))}
                          </Stack>
                        </Box>
                      )}
                      
                      {/* Labels (simple list) */}
                      {latestResults.labels && latestResults.labels.length > 0 && (
                        <Box>
                          <Typography variant="subtitle2" sx={{ color: 'text.secondary', mb: 1 }}>
                            Labels
                          </Typography>
                          <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
                            {latestResults.labels.map((label, idx) => (
                              <Chip key={idx} label={label} size="small" variant="outlined" />
                            ))}
                          </Stack>
                        </Box>
                      )}
                      
                      {/* Inference Time */}
                      {latestResults.inference_time_ms !== undefined && (
                        <Typography variant="caption" sx={{ color: 'text.secondary' }}>
                          Inference time: {latestResults.inference_time_ms.toFixed(1)}ms
                        </Typography>
                      )}
                      
                      {/* Raw JSON (expandable) */}
                      <Divider />
                      <Box>
                        <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mb: 1 }}>
                          Raw Response:
                        </Typography>
                        <Box
                          component="pre"
                          sx={{
                            p: 1,
                            bgcolor: 'action.hover',
                            borderRadius: 1,
                            fontSize: '0.7rem',
                            overflow: 'auto',
                            maxHeight: 150,
                            color: 'text.primary',
                            m: 0,
                          }}
                        >
                          {JSON.stringify(latestResults, null, 2)}
                        </Box>
                      </Box>
                    </Stack>
                  ) : (
                    <Box sx={{ textAlign: 'center', py: 4 }}>
                      <CircularProgress size={24} />
                      <Typography variant="body2" sx={{ mt: 1, color: 'text.secondary' }}>
                        Waiting for results...
                      </Typography>
                    </Box>
                  )}
                </CardContent>
              </Paper>
            </Grid>
          </Grid>
        )}

        {/* Info Footer */}
        <Alert severity="info" icon={<InfoOutlinedIcon />}>
          <Typography variant="body2">
            <strong>Endpoint:</strong> {savedEndpoint || 'Not configured'} — 
            The VLM device should serve MJPEG at <code>/stream</code> and JSON results at <code>/latest</code>.
            Backend server is at port {serverConfig?.port || 8000}.
          </Typography>
        </Alert>
      </Stack>
    </Box>
  );
}
