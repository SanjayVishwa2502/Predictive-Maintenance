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
  const response = await fetch(apiEndpoint, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      machine_id: machineId,
      health_state: predictionData.health_state,
      confidence: predictionData.confidence,
      failure_probability: predictionData.failure_probability,
      predicted_failure_type: predictionData.predicted_failure_type,
      rul_hours: predictionData.rul_hours,
      sensor_data: predictionData.sensor_data,
    }),
  });

  if (!response.ok) {
    throw new Error(`Failed to fetch explanation: ${response.statusText}`);
  }

  return await response.json();
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
      </Stack>
    </Box>
  );

  const renderErrorState = () => (
    <Box sx={{ py: 4, textAlign: 'center' }}>
      <Alert
        severity="error"
        sx={{
          bgcolor: 'rgba(239, 68, 68, 0.1)',
          border: '1px solid rgba(239, 68, 68, 0.3)',
          color: '#fca5a5',
        }}
      >
        {error}
      </Alert>
      <Button
        variant="contained"
        startIcon={<RetryIcon />}
        onClick={handleRetry}
        sx={{
          mt: 3,
          bgcolor: '#667eea',
          '&:hover': { bgcolor: '#5568d3' },
        }}
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
            background: 'linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%)',
            border: '1px solid rgba(255, 255, 255, 0.1)',
            borderRadius: 2,
          }}
        >
          <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 2 }}>
            <BrainIcon sx={{ color: '#667eea' }} />
            <Typography variant="h6" sx={{ fontWeight: 600, color: '#f9fafb' }}>
              AI Analysis Summary
            </Typography>
          </Stack>
          <Typography
            variant="body1"
            sx={{
              color: '#d1d5db',
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
            background: 'linear-gradient(135deg, rgba(239, 68, 68, 0.1) 0%, rgba(220, 38, 38, 0.1) 100%)',
            border: '1px solid rgba(255, 255, 255, 0.1)',
            borderRadius: 2,
          }}
        >
          <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 2 }}>
            <WarningIcon sx={{ color: '#ef4444' }} />
            <Typography variant="h6" sx={{ fontWeight: 600, color: '#f9fafb' }}>
              Risk Factors
            </Typography>
          </Stack>
          <Stack spacing={1.5}>
            {explanation.risk_factors.map((factor, index) => (
              <Box
                key={index}
                sx={{
                  display: 'flex',
                  gap: 1.5,
                  p: 1.5,
                  bgcolor: 'rgba(31, 41, 55, 0.5)',
                  borderRadius: 1,
                  border: '1px solid rgba(239, 68, 68, 0.2)',
                }}
              >
                <Chip
                  label={index + 1}
                  size="small"
                  sx={{
                    bgcolor: '#ef4444',
                    color: '#fff',
                    fontWeight: 600,
                    minWidth: 28,
                  }}
                />
                <Typography variant="body2" sx={{ color: '#d1d5db', flex: 1 }}>
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
            background: 'linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(5, 150, 105, 0.1) 100%)',
            border: '1px solid rgba(255, 255, 255, 0.1)',
            borderRadius: 2,
          }}
        >
          <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 2 }}>
            <RecommendIcon sx={{ color: '#10b981' }} />
            <Typography variant="h6" sx={{ fontWeight: 600, color: '#f9fafb' }}>
              Recommendations
            </Typography>
          </Stack>
          <Stack spacing={1.5}>
            {explanation.recommendations.map((recommendation, index) => (
              <Box
                key={index}
                sx={{
                  display: 'flex',
                  gap: 1.5,
                  p: 1.5,
                  bgcolor: 'rgba(31, 41, 55, 0.5)',
                  borderRadius: 1,
                  border: '1px solid rgba(16, 185, 129, 0.2)',
                }}
              >
                <Chip
                  label={index + 1}
                  size="small"
                  sx={{
                    bgcolor: '#10b981',
                    color: '#fff',
                    fontWeight: 600,
                    minWidth: 28,
                  }}
                />
                <Typography variant="body2" sx={{ color: '#d1d5db', flex: 1 }}>
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
              bgcolor: 'rgba(31, 41, 55, 0.5)',
              border: '1px solid rgba(255, 255, 255, 0.1)',
              borderRadius: 2,
            }}
          >
            <Typography variant="h6" sx={{ fontWeight: 600, color: '#f9fafb', mb: 2 }}>
              Technical Details
            </Typography>
            <Typography variant="body2" sx={{ color: '#9ca3af', lineHeight: 1.7 }}>
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
              bgcolor: 'rgba(31, 41, 55, 0.5)',
              border: '1px solid rgba(255, 255, 255, 0.1)',
              borderRadius: 2,
            }}
          >
            <Typography variant="h6" sx={{ fontWeight: 600, color: '#f9fafb', mb: 2 }}>
              Confidence Analysis
            </Typography>
            <Typography variant="body2" sx={{ color: '#9ca3af', lineHeight: 1.7 }}>
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
        sx: {
          background: 'linear-gradient(135deg, rgba(31, 41, 55, 0.95) 0%, rgba(17, 24, 39, 0.95) 100%)',
          backdropFilter: 'blur(20px)',
          border: '1px solid rgba(255, 255, 255, 0.1)',
          borderRadius: 3,
          maxHeight: '90vh',
        },
      }}
      BackdropProps={{
        sx: {
          backdropFilter: 'blur(8px)',
          backgroundColor: 'rgba(0, 0, 0, 0.7)',
        },
      }}
    >
      {/* HEADER */}
      <DialogTitle
        sx={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
          pb: 2,
        }}
      >
        <Box>
          <Typography variant="h5" sx={{ fontWeight: 700, color: '#f9fafb' }}>
            AI Failure Explanation
          </Typography>
          <Typography variant="body2" sx={{ color: '#9ca3af', mt: 0.5 }}>
            Machine: {machineId} • Health: {predictionData.health_state}
          </Typography>
        </Box>
        <IconButton
          onClick={onClose}
          sx={{
            color: '#9ca3af',
            '&:hover': { color: '#f9fafb', bgcolor: 'rgba(255, 255, 255, 0.1)' },
          }}
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
          borderTop: '1px solid rgba(255, 255, 255, 0.1)',
          px: 3,
          py: 2,
        }}
      >
        <Button
          variant="outlined"
          startIcon={<CopyIcon />}
          onClick={handleCopy}
          disabled={!explanation || loading}
          sx={{
            borderColor: copied ? '#10b981' : '#667eea',
            color: copied ? '#10b981' : '#667eea',
            '&:hover': {
              borderColor: copied ? '#059669' : '#5568d3',
              bgcolor: copied ? 'rgba(16, 185, 129, 0.1)' : 'rgba(102, 126, 234, 0.1)',
            },
          }}
        >
          {copied ? 'Copied!' : 'Copy to Clipboard'}
        </Button>
        <Button
          variant="contained"
          onClick={onClose}
          sx={{
            bgcolor: '#667eea',
            '&:hover': { bgcolor: '#5568d3' },
          }}
        >
          Close
        </Button>
      </DialogActions>
    </Dialog>
  );
}
