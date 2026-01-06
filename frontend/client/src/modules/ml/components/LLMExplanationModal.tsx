/**
 * LLMExplanationModal Component
 * 
 * Full-screen modal for displaying AI-generated failure explanations
 * Features:
 * - Fetches explanation from backend LLM service
 * - Markdown rendering for formatted content
 * - Structured sections: Summary, Risk Factors, Recommendations
 * - Copy to clipboard functionality
 * - Loading state with skeleton animation
 * - Error handling with retry mechanism
 * - Smooth open/close animations
 * 
 * Design: Professional dark theme with glassmorphism effects
 */

import { useState, useEffect } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  IconButton,
  Box,
  Typography,
  Paper,
  Skeleton,
  Alert,
  Stack,
  Slide,
  Chip,
} from '@mui/material';
import { alpha } from '@mui/material/styles';
import type { TransitionProps } from '@mui/material/transitions';
import {
  Close as CloseIcon,
  ContentCopy as CopyIcon,
  Psychology as BrainIcon,
  Warning as WarningIcon,
  Lightbulb as RecommendIcon,
  Refresh as RetryIcon,
} from '@mui/icons-material';
import ReactMarkdown from 'react-markdown';
import { forwardRef } from 'react';
const ACCESS_TOKEN_KEY = 'pm_access_token';
const REFRESH_TOKEN_KEY = 'pm_refresh_token';

function getAccessToken(): string | null {
  try {
    const token = window.localStorage.getItem(ACCESS_TOKEN_KEY);
    return token && token.trim() ? token : null;
  } catch {
    return null;
  }
}

function clearTokens(): void {
  try {
    window.localStorage.removeItem(ACCESS_TOKEN_KEY);
    window.localStorage.removeItem(REFRESH_TOKEN_KEY);
  } catch {
    // ignore
  }
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

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

// ============================================================================
// TYPESCRIPT INTERFACES
// ============================================================================

export interface LLMExplanationModalProps {
  open: boolean;
  onClose: () => void;
  machineId: string;
  predictionData: PredictionData;
  apiEndpoint?: string;
}

export interface PredictionData {
  // Prefer run-scoped explanations when available.
  // If run_id is provided, the backend will load the exact stored run context
  // (sensor_data + predictions) and generate a consistent explanation.
  run_id?: string;
  health_state: string;
  confidence: number;
  failure_probability: number;
  predicted_failure_type?: string;
  rul_hours?: number;
  sensor_data?: Record<string, number>;
}

interface ExplanationResponse {
  summary: string;
  risk_factors: string[];
  recommendations: string[];
  technical_details?: string;
  confidence_analysis?: string;
}

// ============================================================================
// SLIDE TRANSITION
// ============================================================================

const SlideTransition = forwardRef(function Transition(
  props: TransitionProps & {
    children: React.ReactElement;
  },
  ref: React.Ref<unknown>,
) {
  return <Slide direction="up" ref={ref} {...props} />;
});

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

const fetchExplanation = async (
  machineId: string,
  predictionData: PredictionData,
  apiEndpoint: string
): Promise<ExplanationResponse> => {
  const explainUrl = apiEndpoint.startsWith('http') ? apiEndpoint : `${API_BASE_URL}${apiEndpoint}`;
  const runId = (predictionData.run_id || '').trim();
  const body = runId
    ? {
        machine_id: machineId,
        run_id: runId,
      }
    : {
        machine_id: machineId,
        health_state: predictionData.health_state,
        confidence: predictionData.confidence,
        failure_probability: predictionData.failure_probability,
        predicted_failure_type: predictionData.predicted_failure_type,
        rul_hours: predictionData.rul_hours,
        sensor_data: predictionData.sensor_data,
      };
  const response = await fetchWithAuth(explainUrl, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    throw new Error(`Failed to fetch explanation: ${response.statusText}`);
  }

  const startPayload = await response.json();

  // Async mode: backend returns task_id immediately; we poll until SUCCESS.
  if (startPayload && typeof startPayload.task_id === 'string') {
    const taskId: string = startPayload.task_id;
    const statusUrl = `${API_BASE_URL}/api/llm/explain/${encodeURIComponent(taskId)}`;

    const startedAt = Date.now();
    const maxWaitMs = 150_000; // ~2.5 minutes budget for CPU LLM

    while (Date.now() - startedAt < maxWaitMs) {
      // Wait 2 seconds between polls (keeps load low)
      await new Promise((r) => setTimeout(r, 2000));

      const statusResp = await fetchWithAuth(statusUrl, {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' },
      });

      if (!statusResp.ok) {
        throw new Error(`Failed to poll explanation: ${statusResp.statusText}`);
      }

      const statusJson = await statusResp.json();
      const status = statusJson?.status;

      if (status === 'SUCCESS' && statusJson?.result) {
        return statusJson.result as ExplanationResponse;
      }

      if (status === 'FAILURE') {
        throw new Error(statusJson?.error || 'LLM explanation task failed');
      }
    }

    throw new Error('LLM explanation timed out (CPU inference can take ~2 minutes).');
  }

  // Sync mode: backend returned the explanation directly.
  return startPayload as ExplanationResponse;
};

const copyToClipboard = async (text: string): Promise<boolean> => {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch (error) {
    console.error('Failed to copy to clipboard:', error);
    return false;
  }
};

const formatExplanationText = (explanation: ExplanationResponse): string => {
  let text = '=== AI FAILURE ANALYSIS ===\n\n';
  text += `SUMMARY:\n${explanation.summary}\n\n`;
  text += `RISK FACTORS:\n${explanation.risk_factors.map((r, i) => `${i + 1}. ${r}`).join('\n')}\n\n`;
  text += `RECOMMENDATIONS:\n${explanation.recommendations.map((r, i) => `${i + 1}. ${r}`).join('\n')}`;
  
  if (explanation.technical_details) {
    text += `\n\nTECHNICAL DETAILS:\n${explanation.technical_details}`;
  }
  
  if (explanation.confidence_analysis) {
    text += `\n\nCONFIDENCE ANALYSIS:\n${explanation.confidence_analysis}`;
  }
  
  return text;
};

// ============================================================================
// MAIN COMPONENT
// ============================================================================

export default function LLMExplanationModal({
  open,
  onClose,
  machineId,
  predictionData,
  apiEndpoint = '/api/llm/explain',
}: LLMExplanationModalProps) {
  const [explanation, setExplanation] = useState<ExplanationResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  // Fetch explanation when modal opens
  useEffect(() => {
    if (open) {
      loadExplanation();
    } else {
      // Reset state when modal closes
      setExplanation(null);
      setError(null);
      setCopied(false);
    }
  }, [open]);

  const loadExplanation = async () => {
    setLoading(true);
    setError(null);

    try {
      const result = await fetchExplanation(machineId, predictionData, apiEndpoint);
      setExplanation(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load explanation');
    } finally {
      setLoading(false);
    }
  };

  const handleCopy = async () => {
    if (!explanation) return;

    const text = formatExplanationText(explanation);
    const success = await copyToClipboard(text);

    if (success) {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  const handleRetry = () => {
    loadExplanation();
  };

  // ============================================================================
  // RENDER FUNCTIONS
  // ============================================================================

  const renderLoadingState = () => (
    <Box sx={{ py: 2 }}>
      <Stack spacing={3}>
        {/* Summary Skeleton */}
        <Box>
          <Skeleton
            variant="text"
            width="40%"
            height={32}
            sx={{ bgcolor: 'rgba(255, 255, 255, 0.1)' }}
          />
          <Skeleton
            variant="rectangular"
            height={100}
            sx={{ bgcolor: 'rgba(255, 255, 255, 0.1)', borderRadius: 1, mt: 1 }}
          />
        </Box>

        {/* Risk Factors Skeleton */}
        <Box>
          <Skeleton
            variant="text"
            width="40%"
            height={32}
            sx={{ bgcolor: 'rgba(255, 255, 255, 0.1)' }}
          />
          {[1, 2, 3].map((i) => (
            <Skeleton
              key={i}
              variant="rectangular"
              height={40}
              sx={{ bgcolor: 'rgba(255, 255, 255, 0.1)', borderRadius: 1, mt: 1 }}
            />
          ))}
        </Box>

        {/* Recommendations Skeleton */}
        <Box>
          <Skeleton
            variant="text"
            width="40%"
            height={32}
            sx={{ bgcolor: 'action.hover' }}
          />
          {[1, 2, 3].map((i) => (
            <Skeleton
              key={i}
              variant="rectangular"
              height={40}
              sx={{ bgcolor: 'action.hover', borderRadius: 1, mt: 1 }}
            />
          ))}
        </Box>
      </Stack>
    </Box>
  );

  const renderErrorState = () => (
    <Box sx={{ py: 4, textAlign: 'center' }}>
      <Alert
        severity="error"
        sx={(theme) => ({
          bgcolor: alpha(theme.palette.error.main, theme.palette.mode === 'dark' ? 0.12 : 0.08),
          border: 1,
          borderColor: alpha(theme.palette.error.main, 0.35),
        })}
      >
        {error}
      </Alert>
      <Button
        variant="contained"
        color="primary"
        startIcon={<RetryIcon />}
        onClick={handleRetry}
        sx={{ mt: 3 }}
      >
        Retry
      </Button>
    </Box>
  );

  const renderExplanation = () => {
    if (!explanation) return null;

    return (
      <Stack spacing={3} sx={{ py: 2 }}>
        {/* SUMMARY SECTION */}
        <Paper
          elevation={0}
          sx={{
            p: 3,
            bgcolor: (theme) => alpha(theme.palette.primary.main, theme.palette.mode === 'dark' ? 0.12 : 0.08),
            border: 1,
            borderColor: 'divider',
            borderRadius: 2,
          }}
        >
          <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 2 }}>
            <BrainIcon color="primary" />
            <Typography variant="h6" sx={{ fontWeight: 600, color: 'text.primary' }}>
              AI Analysis Summary
            </Typography>
          </Stack>
          <Typography
            variant="body1"
            sx={{
              color: 'text.secondary',
              lineHeight: 1.7,
              '& p': { mb: 1 },
            }}
          >
            <ReactMarkdown>{explanation.summary}</ReactMarkdown>
          </Typography>
        </Paper>

        {/* RISK FACTORS SECTION */}
        <Paper
          elevation={0}
          sx={{
            p: 3,
            bgcolor: (theme) => alpha(theme.palette.error.main, theme.palette.mode === 'dark' ? 0.12 : 0.08),
            border: 1,
            borderColor: 'divider',
            borderRadius: 2,
          }}
        >
          <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 2 }}>
            <WarningIcon sx={{ color: 'error.main' }} />
            <Typography variant="h6" sx={{ fontWeight: 600, color: 'text.primary' }}>
              Risk Factors
            </Typography>
          </Stack>
          <Stack spacing={1.5}>
            {explanation.risk_factors.map((factor, index) => (
              <Box
                key={index}
                sx={(theme) => ({
                  display: 'flex',
                  gap: 1.5,
                  p: 1.5,
                  bgcolor: alpha(theme.palette.background.paper, theme.palette.mode === 'dark' ? 0.5 : 0.7),
                  borderRadius: 1,
                  border: 1,
                  borderColor: alpha(theme.palette.error.main, 0.25),
                })}
              >
                <Chip
                  label={index + 1}
                  size="small"
                  color="error"
                  sx={{ fontWeight: 600, minWidth: 28 }}
                />
                <Typography variant="body2" sx={{ color: 'text.secondary', flex: 1 }}>
                  <ReactMarkdown>{factor}</ReactMarkdown>
                </Typography>
              </Box>
            ))}
          </Stack>
        </Paper>

        {/* RECOMMENDATIONS SECTION */}
        <Paper
          elevation={0}
          sx={{
            p: 3,
            bgcolor: (theme) => alpha(theme.palette.success.main, theme.palette.mode === 'dark' ? 0.12 : 0.08),
            border: 1,
            borderColor: 'divider',
            borderRadius: 2,
          }}
        >
          <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 2 }}>
            <RecommendIcon sx={{ color: 'success.main' }} />
            <Typography variant="h6" sx={{ fontWeight: 600, color: 'text.primary' }}>
              Recommendations
            </Typography>
          </Stack>
          <Stack spacing={1.5}>
            {explanation.recommendations.map((recommendation, index) => (
              <Box
                key={index}
                sx={(theme) => ({
                  display: 'flex',
                  gap: 1.5,
                  p: 1.5,
                  bgcolor: alpha(theme.palette.background.paper, theme.palette.mode === 'dark' ? 0.5 : 0.7),
                  borderRadius: 1,
                  border: 1,
                  borderColor: alpha(theme.palette.success.main, 0.25),
                })}
              >
                <Chip
                  label={index + 1}
                  size="small"
                  color="success"
                  sx={{ fontWeight: 600, minWidth: 28 }}
                />
                <Typography variant="body2" sx={{ color: 'text.secondary', flex: 1 }}>
                  <ReactMarkdown>{recommendation}</ReactMarkdown>
                </Typography>
              </Box>
            ))}
          </Stack>
        </Paper>

        {/* TECHNICAL DETAILS (Optional) */}
        {explanation.technical_details && (
          <Paper
            elevation={0}
            sx={{
              p: 3,
              bgcolor: 'background.paper',
              border: 1,
              borderColor: 'divider',
              borderRadius: 2,
            }}
          >
            <Typography variant="h6" sx={{ fontWeight: 600, color: 'text.primary', mb: 2 }}>
              Technical Details
            </Typography>
            <Typography variant="body2" sx={{ color: 'text.secondary', lineHeight: 1.7 }}>
              <ReactMarkdown>{explanation.technical_details}</ReactMarkdown>
            </Typography>
          </Paper>
        )}

        {/* CONFIDENCE ANALYSIS (Optional) */}
        {explanation.confidence_analysis && (
          <Paper
            elevation={0}
            sx={{
              p: 3,
              bgcolor: 'background.paper',
              border: 1,
              borderColor: 'divider',
              borderRadius: 2,
            }}
          >
            <Typography variant="h6" sx={{ fontWeight: 600, color: 'text.primary', mb: 2 }}>
              Confidence Analysis
            </Typography>
            <Typography variant="body2" sx={{ color: 'text.secondary', lineHeight: 1.7 }}>
              <ReactMarkdown>{explanation.confidence_analysis}</ReactMarkdown>
            </Typography>
          </Paper>
        )}
      </Stack>
    );
  };

  // ============================================================================
  // MAIN RENDER
  // ============================================================================

  return (
    <Dialog
      open={open}
      onClose={onClose}
      TransitionComponent={SlideTransition}
      maxWidth="md"
      fullWidth
      PaperProps={{
        sx: (theme) => ({
          bgcolor: alpha(theme.palette.background.paper, theme.palette.mode === 'dark' ? 0.9 : 0.98),
          backdropFilter: theme.palette.mode === 'dark' ? 'blur(20px)' : undefined,
          border: 1,
          borderColor: 'divider',
          borderRadius: 3,
          maxHeight: '90vh',
        }),
      }}
      BackdropProps={{
        sx: (theme) => ({
          backdropFilter: theme.palette.mode === 'dark' ? 'blur(8px)' : undefined,
          backgroundColor: alpha('#000', theme.palette.mode === 'dark' ? 0.7 : 0.4),
        }),
      }}
    >
      {/* HEADER */}
      <DialogTitle
        sx={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          borderBottom: 1,
          borderColor: 'divider',
          pb: 2,
        }}
      >
        <Box>
          <Typography variant="h5" sx={{ fontWeight: 700, color: 'text.primary' }}>
            AI Failure Explanation
          </Typography>
          <Typography variant="body2" sx={{ color: 'text.secondary', mt: 0.5 }}>
            Machine: {machineId} • Health: {predictionData.health_state}
          </Typography>
        </Box>
        <IconButton
          onClick={onClose}
          sx={{ color: 'text.secondary', '&:hover': { color: 'text.primary', bgcolor: 'action.hover' } }}
        >
          <CloseIcon />
        </IconButton>
      </DialogTitle>

      {/* CONTENT */}
      <DialogContent sx={{ px: 3 }}>
        {loading && renderLoadingState()}
        {error && renderErrorState()}
        {!loading && !error && explanation && renderExplanation()}
      </DialogContent>

      {/* ACTIONS */}
      <DialogActions
        sx={{
          borderTop: 1,
          borderColor: 'divider',
          px: 3,
          py: 2,
        }}
      >
        <Button
          variant="outlined"
          startIcon={<CopyIcon />}
          onClick={handleCopy}
          disabled={!explanation || loading}
          color={copied ? 'success' : 'primary'}
          sx={(theme) => ({
            '&:hover': {
              bgcolor: copied
                ? alpha(theme.palette.success.main, theme.palette.mode === 'dark' ? 0.14 : 0.1)
                : alpha(theme.palette.primary.main, theme.palette.mode === 'dark' ? 0.14 : 0.1),
            },
          })}
        >
          {copied ? 'Copied!' : 'Copy to Clipboard'}
        </Button>
        <Button
          variant="contained"
          onClick={onClose}
          color="primary"
        >
          Close
        </Button>
      </DialogActions>
    </Dialog>
  );
}
