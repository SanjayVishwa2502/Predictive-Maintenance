/**
 * PredictionCard Component
 * 
 * Displays ML prediction results for a single selected machine including:
 * - Health status with confidence score
 * - Remaining Useful Life (RUL) prediction
 * - Failure type probability distribution
 * - Action buttons for prediction, explanation, and history
 * 
 * Design: Professional dark theme with color-coded status indicators
 * Status Colors: Green (Healthy), Yellow (Degrading), Orange (Warning), Red (Critical)
 */

import {
  Box,
  Card,
  CardContent,
  CardHeader,
  Button,
  Typography,
  LinearProgress,
  Skeleton,
  Grid,
  Chip,
  Divider,
  Stack,
  CircularProgress,
} from '@mui/material';
import {
  PlayArrow as PlayArrowIcon,
  Psychology as PsychologyIcon,
  History as HistoryIcon,
  CheckCircle as CheckCircleIcon,
  Warning as WarningIcon,
  Error as ErrorIcon,
  AccessTime as AccessTimeIcon,
  Refresh as RefreshIcon,
} from '@mui/icons-material';
import ContentCopyIcon from '@mui/icons-material/ContentCopy';
import { useState, useEffect, useMemo } from 'react';

// ============================================================================
// TYPESCRIPT INTERFACES
// ============================================================================

export interface PredictionCardProps {
  machineId: string;
  prediction: PredictionResult | null;
  loading: boolean;
  onRunPrediction: () => void;
  onExplain: () => void;
  onViewHistory?: () => void;
  autoRefresh?: boolean;
  refreshInterval?: number; // seconds

  // Optional: text-output mode (LLM explanation summary)
  // If provided, the card will render this text output panel instead of the full prediction/status UI.
  textOutput?: string;
  textOutputLoading?: boolean;
  textOutputError?: string | null;
  textOutputRefreshSeconds?: number;
}

export interface PredictionResult {
  classification: {
    failure_type: string;
    confidence: number;
    failure_probability: number;
    all_probabilities: Record<string, number>;
  };
  rul?: {
    rul_hours: number;
    rul_days: number;
    urgency: 'low' | 'medium' | 'high' | 'critical';
    maintenance_window: string;
  };
  timestamp: string;
}

interface StatusConfig {
  label: string;
  color: string;
  icon: React.ReactElement;
  bgColor: string;
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/**
 * Get status configuration based on failure probability
 * Status Mapping:
 * - Healthy: failure_prob < 0.15 (Green)
 * - Degrading: 0.15 ≤ failure_prob < 0.40 (Yellow)
 * - Warning: 0.40 ≤ failure_prob < 0.70 (Orange)
 * - Critical: failure_prob ≥ 0.70 (Red)
 */
const getStatusConfig = (failureProbability: number): StatusConfig => {
  if (failureProbability < 0.15) {
    return {
      label: 'Healthy',
      color: '#10b981',
      icon: <CheckCircleIcon sx={{ fontSize: 40 }} />,
      bgColor: 'rgba(16, 185, 129, 0.1)',
    };
  } else if (failureProbability < 0.40) {
    return {
      label: 'Degrading',
      color: '#fbbf24',
      icon: <WarningIcon sx={{ fontSize: 40 }} />,
      bgColor: 'rgba(251, 191, 36, 0.1)',
    };
  } else if (failureProbability < 0.70) {
    return {
      label: 'Warning',
      color: '#f97316',
      icon: <ErrorIcon sx={{ fontSize: 40 }} />,
      bgColor: 'rgba(249, 115, 22, 0.1)',
    };
  } else {
    return {
      label: 'Critical',
      color: '#ef4444',
      icon: <ErrorIcon sx={{ fontSize: 40 }} />,
      bgColor: 'rgba(239, 68, 68, 0.1)',
    };
  }
};

/**
 * Get urgency configuration for RUL prediction
 */
const getUrgencyConfig = (urgency: string) => {
  switch (urgency) {
    case 'low':
      return {
        color: '#10b981',
        label: 'Low',
        description: 'Schedule within 1 week',
      };
    case 'medium':
      return {
        color: '#fbbf24',
        label: 'Medium',
        description: 'Schedule within 3 days',
      };
    case 'high':
      return {
        color: '#f97316',
        label: 'High',
        description: 'Schedule within 24 hours',
      };
    case 'critical':
      return {
        color: '#ef4444',
        label: 'Critical',
        description: 'IMMEDIATE ACTION REQUIRED',
      };
    default:
      return {
        color: '#6b7280',
        label: 'Unknown',
        description: 'Assessment pending',
      };
  }
};

/**
 * Format relative time (e.g., "2 minutes ago")
 */
const getRelativeTime = (timestamp: string): string => {
  const now = new Date();
  const past = new Date(timestamp);
  const diffMs = now.getTime() - past.getTime();
  const diffMins = Math.floor(diffMs / 60000);
  const diffHours = Math.floor(diffMs / 3600000);
  const diffDays = Math.floor(diffMs / 86400000);

  if (diffMins < 1) return 'Just now';
  if (diffMins < 60) return `${diffMins} minute${diffMins > 1 ? 's' : ''} ago`;
  if (diffHours < 24) return `${diffHours} hour${diffHours > 1 ? 's' : ''} ago`;
  return `${diffDays} day${diffDays > 1 ? 's' : ''} ago`;
};

/**
 * Format failure type for display (convert snake_case to Title Case)
 */
const formatFailureType = (type: string): string => {
  return type
    .split('_')
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
};

// ============================================================================
// MAIN COMPONENT
// ============================================================================

export default function PredictionCard({
  machineId,
  prediction,
  loading,
  onRunPrediction,
  onExplain,
  onViewHistory,
  autoRefresh = false,
  refreshInterval = 30,
  textOutput,
  textOutputLoading,
  textOutputError,
  textOutputRefreshSeconds,
}: PredictionCardProps) {
  const [countdown, setCountdown] = useState(refreshInterval);

  const isTextMode =
    typeof textOutput !== 'undefined' ||
    typeof textOutputLoading !== 'undefined' ||
    typeof textOutputError !== 'undefined' ||
    typeof textOutputRefreshSeconds !== 'undefined';

  if (isTextMode) {
    const displayText = (textOutput || '').trim();
    const isBusy = Boolean(textOutputLoading);

    const handleCopy = async () => {
      const txt = displayText;
      if (!txt) return;
      try {
        await navigator.clipboard.writeText(txt);
      } catch {
        try {
          const ta = document.createElement('textarea');
          ta.value = txt;
          ta.style.position = 'fixed';
          ta.style.left = '-10000px';
          document.body.appendChild(ta);
          ta.select();
          document.execCommand('copy');
          document.body.removeChild(ta);
        } catch {
          // ignore
        }
      }
    };

    return (
      <Card
        sx={{
          background: 'linear-gradient(135deg, rgba(31, 41, 55, 0.8) 0%, rgba(17, 24, 39, 0.8) 100%)',
          backdropFilter: 'blur(10px)',
          border: '1px solid rgba(255, 255, 255, 0.1)',
          borderRadius: 2,
          mb: 3,
        }}
      >
        <CardHeader
          title={
            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 2 }}>
              <Typography variant="h6" sx={{ color: '#e5e7eb', fontWeight: 600 }}>
                Prediction Output
              </Typography>
              <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
                <Button
                  size="small"
                  variant="outlined"
                  startIcon={<ContentCopyIcon />}
                  onClick={handleCopy}
                  disabled={!displayText}
                  sx={{ textTransform: 'none' }}
                >
                  Copy
                </Button>
                <Button
                  size="small"
                  variant="contained"
                  startIcon={<PlayArrowIcon />}
                  onClick={onRunPrediction}
                  disabled={isBusy || Boolean(loading)}
                  sx={{ textTransform: 'none' }}
                >
                  Prediction
                </Button>
              </Box>
            </Box>
          }
          subheader={
            <Typography variant="caption" sx={{ color: '#9ca3af' }}>
              Machine: {machineId}
            </Typography>
          }
          sx={{
            '& .MuiCardHeader-content': { overflow: 'hidden' },
            pb: 0,
          }}
        />
        <CardContent>
          {textOutputError && (
            <Typography variant="body2" sx={{ color: '#fca5a5', mb: 1 }}>
              {textOutputError}
            </Typography>
          )}

          <Box
            sx={{
              minHeight: 220,
              p: 2,
              borderRadius: 2,
              bgcolor: 'rgba(0, 0, 0, 0.2)',
              border: '1px solid rgba(255, 255, 255, 0.08)',
              overflow: 'auto',
              whiteSpace: 'pre-wrap',
              color: '#e5e7eb',
              fontSize: 14,
              lineHeight: 1.6,
            }}
          >
            {isBusy ? 'Generating explanation…' : displayText || 'Waiting for explanation…'}
          </Box>
        </CardContent>
      </Card>
    );
  }

  // Auto-refresh countdown timer
  useEffect(() => {
    if (!autoRefresh || !prediction) return;

    const timer = setInterval(() => {
      setCountdown((prev) => {
        if (prev <= 1) {
          onRunPrediction();
          return refreshInterval;
        }
        return prev - 1;
      });
    }, 1000);

    return () => clearInterval(timer);
  }, [autoRefresh, prediction, onRunPrediction, refreshInterval]);

  // Reset countdown when prediction changes
  useEffect(() => {
    setCountdown(refreshInterval);
  }, [prediction, refreshInterval]);

  // Compute status configuration
  const statusConfig = useMemo(() => {
    if (!prediction) return null;
    return getStatusConfig(prediction.classification.failure_probability);
  }, [prediction]);

  // Compute urgency configuration
  const urgencyConfig = useMemo(() => {
    if (!prediction?.rul) return null;
    return getUrgencyConfig(prediction.rul.urgency);
  }, [prediction]);

  // Sort probabilities by value (descending)
  const sortedProbabilities = useMemo(() => {
    if (!prediction) return [];
    return Object.entries(prediction.classification.all_probabilities)
      .sort(([, a], [, b]) => b - a)
      .map(([type, probability]) => ({
        type: formatFailureType(type),
        probability,
      }));
  }, [prediction]);

  // ============================================================================
  // RENDER: EMPTY STATE
  // ============================================================================

  if (!prediction && !loading) {
    return (
      <Card
        sx={{
          background: 'linear-gradient(135deg, rgba(31, 41, 55, 0.8) 0%, rgba(17, 24, 39, 0.8) 100%)',
          backdropFilter: 'blur(10px)',
          border: '1px solid rgba(255, 255, 255, 0.1)',
          borderRadius: 2,
        }}
      >
        <CardHeader
          title={
            <Typography variant="h6" sx={{ fontWeight: 600, color: '#f9fafb' }}>
              MACHINE HEALTH PREDICTION
            </Typography>
          }
        />
        <Divider sx={{ borderColor: 'rgba(255, 255, 255, 0.1)' }} />
        <CardContent>
          <Box
            sx={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              py: 8,
              gap: 3,
            }}
          >
            <PsychologyIcon sx={{ fontSize: 80, color: '#6b7280' }} />
            <Typography variant="h6" sx={{ color: '#9ca3af', textAlign: 'center' }}>
              No prediction yet
            </Typography>
            <Typography variant="body2" sx={{ color: '#6b7280', textAlign: 'center', maxWidth: 400 }}>
              Click the button below to run a health prediction for this machine using the latest sensor data.
            </Typography>
            <Button
              variant="contained"
              size="large"
              startIcon={<PlayArrowIcon />}
              onClick={onRunPrediction}
              sx={{
                mt: 2,
                background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                '&:hover': {
                  background: 'linear-gradient(135deg, #5568d3 0%, #63408b 100%)',
                },
              }}
            >
              Run Prediction
            </Button>
          </Box>
        </CardContent>
      </Card>
    );
  }

  // ============================================================================
  // RENDER: LOADING STATE
  // ============================================================================

  if (loading) {
    return (
      <Card
        sx={{
          background: 'linear-gradient(135deg, rgba(31, 41, 55, 0.8) 0%, rgba(17, 24, 39, 0.8) 100%)',
          backdropFilter: 'blur(10px)',
          border: '1px solid rgba(255, 255, 255, 0.1)',
          borderRadius: 2,
        }}
      >
        <CardHeader
          title={
            <Typography variant="h6" sx={{ fontWeight: 600, color: '#f9fafb' }}>
              MACHINE HEALTH PREDICTION
            </Typography>
          }
          action={
            <CircularProgress size={24} sx={{ color: '#667eea' }} />
          }
        />
        <Divider sx={{ borderColor: 'rgba(255, 255, 255, 0.1)' }} />
        <CardContent>
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Skeleton variant="rectangular" height={200} sx={{ bgcolor: 'rgba(255, 255, 255, 0.05)', borderRadius: 2 }} />
            </Grid>
            <Grid item xs={12} md={6}>
              <Skeleton variant="rectangular" height={200} sx={{ bgcolor: 'rgba(255, 255, 255, 0.05)', borderRadius: 2 }} />
            </Grid>
            <Grid item xs={12}>
              <Skeleton variant="rectangular" height={150} sx={{ bgcolor: 'rgba(255, 255, 255, 0.05)', borderRadius: 2 }} />
            </Grid>
          </Grid>
        </CardContent>
      </Card>
    );
  }

  // ============================================================================
  // RENDER: PREDICTION RESULT
  // ============================================================================

  return (
    <Card
      sx={{
        background: 'linear-gradient(135deg, rgba(31, 41, 55, 0.8) 0%, rgba(17, 24, 39, 0.8) 100%)',
        backdropFilter: 'blur(10px)',
        border: '1px solid rgba(255, 255, 255, 0.1)',
        borderRadius: 2,
      }}
    >
      {/* HEADER */}
      <CardHeader
        title={
          <Typography variant="h6" sx={{ fontWeight: 600, color: '#f9fafb' }}>
            MACHINE HEALTH PREDICTION
          </Typography>
        }
        subheader={
          <Stack direction="row" spacing={2} alignItems="center" sx={{ mt: 1 }}>
            <Typography variant="body2" sx={{ color: '#9ca3af' }}>
              Last prediction: {getRelativeTime(prediction!.timestamp)}
            </Typography>
            {autoRefresh && (
              <Chip
                size="small"
                icon={<RefreshIcon />}
                label={`Auto-refresh in ${countdown}s`}
                sx={{
                  bgcolor: 'rgba(102, 126, 234, 0.2)',
                  color: '#667eea',
                  fontSize: '0.75rem',
                }}
              />
            )}
          </Stack>
        }
        action={
          <Button
            variant="contained"
            size="small"
            startIcon={<PlayArrowIcon />}
            onClick={onRunPrediction}
            sx={{
              background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
              '&:hover': {
                background: 'linear-gradient(135deg, #5568d3 0%, #63408b 100%)',
              },
            }}
          >
            Run Prediction
          </Button>
        }
      />
      <Divider sx={{ borderColor: 'rgba(255, 255, 255, 0.1)' }} />

      <CardContent>
        {/* HEALTH STATUS & RUL SECTION */}
        <Grid container spacing={3} sx={{ mb: 3 }}>
          {/* HEALTH STATUS */}
          <Grid item xs={12} md={6}>
            <Box
              sx={{
                p: 3,
                borderRadius: 2,
                background: statusConfig!.bgColor,
                border: `1px solid ${statusConfig!.color}40`,
                height: '100%',
                display: 'flex',
                flexDirection: 'column',
                justifyContent: 'center',
                alignItems: 'center',
                gap: 2,
                position: 'relative',
                overflow: 'hidden',
                ...(statusConfig!.label === 'Critical' && {
                  animation: 'pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
                  '@keyframes pulse': {
                    '0%, 100%': { opacity: 1 },
                    '50%': { opacity: 0.7 },
                  },
                }),
              }}
            >
              <Typography variant="overline" sx={{ color: '#9ca3af', fontWeight: 600 }}>
                HEALTH STATUS
              </Typography>

              <Box sx={{ color: statusConfig!.color }}>
                {statusConfig!.icon}
              </Box>

              <Typography variant="h4" sx={{ color: statusConfig!.color, fontWeight: 700 }}>
                {statusConfig!.label}
              </Typography>

              {/* Confidence Score with Progress Ring */}
              <Box sx={{ position: 'relative', display: 'inline-flex' }}>
                <CircularProgress
                  variant="determinate"
                  value={prediction!.classification.confidence * 100}
                  size={80}
                  thickness={4}
                  sx={{
                    color: statusConfig!.color,
                    '& .MuiCircularProgress-circle': {
                      strokeLinecap: 'round',
                    },
                  }}
                />
                <Box
                  sx={{
                    top: 0,
                    left: 0,
                    bottom: 0,
                    right: 0,
                    position: 'absolute',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                  }}
                >
                  <Typography variant="h6" sx={{ color: '#f9fafb', fontWeight: 700 }}>
                    {Math.round(prediction!.classification.confidence * 100)}%
                  </Typography>
                </Box>
              </Box>

              <Typography variant="body2" sx={{ color: '#d1d5db' }}>
                Confidence Score
              </Typography>

              <Typography
                variant="body1"
                sx={{
                  color: '#f9fafb',
                  fontWeight: 500,
                  textAlign: 'center',
                  mt: 1,
                }}
              >
                {formatFailureType(prediction!.classification.failure_type)}
              </Typography>
            </Box>
          </Grid>

          {/* REMAINING USEFUL LIFE */}
          {prediction!.rul && (
            <Grid item xs={12} md={6}>
              <Box
                sx={{
                  p: 3,
                  borderRadius: 2,
                  background: `${urgencyConfig!.color}10`,
                  border: `1px solid ${urgencyConfig!.color}40`,
                  height: '100%',
                  display: 'flex',
                  flexDirection: 'column',
                  justifyContent: 'center',
                  alignItems: 'center',
                  gap: 2,
                }}
              >
                <Typography variant="overline" sx={{ color: '#9ca3af', fontWeight: 600 }}>
                  REMAINING USEFUL LIFE
                </Typography>

                <AccessTimeIcon sx={{ fontSize: 40, color: urgencyConfig!.color }} />

                <Typography variant="h3" sx={{ color: urgencyConfig!.color, fontWeight: 700 }}>
                  {Math.round(prediction!.rul.rul_hours)} hours
                </Typography>

                <Typography variant="h6" sx={{ color: '#d1d5db' }}>
                  ({prediction!.rul.rul_days.toFixed(1)} days)
                </Typography>

                <Chip
                  label={urgencyConfig!.label}
                  sx={{
                    bgcolor: urgencyConfig!.color,
                    color: '#fff',
                    fontWeight: 600,
                    fontSize: '0.875rem',
                  }}
                />

                <Typography
                  variant="body2"
                  sx={{
                    color: '#d1d5db',
                    textAlign: 'center',
                    mt: 1,
                  }}
                >
                  {urgencyConfig!.description}
                </Typography>

                <Typography variant="body2" sx={{ color: '#9ca3af' }}>
                  {prediction!.rul.maintenance_window}
                </Typography>
              </Box>
            </Grid>
          )}
        </Grid>

        {/* FAILURE TYPE PROBABILITIES */}
        <Box
          sx={{
            p: 3,
            borderRadius: 2,
            background: 'rgba(31, 41, 55, 0.5)',
            border: '1px solid rgba(255, 255, 255, 0.1)',
          }}
        >
          <Typography
            variant="subtitle1"
            sx={{ color: '#f9fafb', fontWeight: 600, mb: 2 }}
          >
            FAILURE TYPE PROBABILITIES
          </Typography>

          <Stack spacing={2}>
            {sortedProbabilities.map(({ type, probability }) => (
              <Box key={type}>
                <Box
                  sx={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    mb: 0.5,
                  }}
                >
                  <Typography variant="body2" sx={{ color: '#d1d5db', fontWeight: 500 }}>
                    {type}
                  </Typography>
                  <Typography variant="body2" sx={{ color: '#f9fafb', fontWeight: 600 }}>
                    {Math.round(probability * 100)}%
                  </Typography>
                </Box>
                <LinearProgress
                  variant="determinate"
                  value={probability * 100}
                  sx={{
                    height: 8,
                    borderRadius: 4,
                    bgcolor: 'rgba(255, 255, 255, 0.1)',
                    '& .MuiLinearProgress-bar': {
                      borderRadius: 4,
                      background:
                        probability >= 0.5
                          ? 'linear-gradient(90deg, #ef4444 0%, #dc2626 100%)'
                          : probability >= 0.25
                          ? 'linear-gradient(90deg, #f97316 0%, #ea580c 100%)'
                          : probability >= 0.10
                          ? 'linear-gradient(90deg, #fbbf24 0%, #f59e0b 100%)'
                          : 'linear-gradient(90deg, #10b981 0%, #059669 100%)',
                    },
                  }}
                />
              </Box>
            ))}
          </Stack>
        </Box>

        {/* ACTION BUTTONS */}
        <Stack
          direction={{ xs: 'column', sm: 'row' }}
          spacing={2}
          sx={{ mt: 3 }}
        >
          <Button
            variant="outlined"
            size="large"
            startIcon={<PsychologyIcon />}
            onClick={onExplain}
            fullWidth
            sx={{
              borderColor: '#667eea',
              color: '#667eea',
              '&:hover': {
                borderColor: '#5568d3',
                bgcolor: 'rgba(102, 126, 234, 0.1)',
              },
            }}
          >
            Get AI Explanation
          </Button>

          {onViewHistory && (
            <Button
              variant="outlined"
              size="large"
              startIcon={<HistoryIcon />}
              onClick={onViewHistory}
              fullWidth
              sx={{
                borderColor: '#764ba2',
                color: '#764ba2',
                '&:hover': {
                  borderColor: '#63408b',
                  bgcolor: 'rgba(118, 75, 162, 0.1)',
                },
              }}
            >
              View History
            </Button>
          )}
        </Stack>
      </CardContent>
    </Card>
  );
}
