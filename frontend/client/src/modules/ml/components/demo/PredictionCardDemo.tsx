/**
 * PredictionCard Demo Component
 * 
 * Interactive demo to test PredictionCard component with mock data
 * Simulates different health states and prediction scenarios
 */

import { useState } from 'react';
import {
  Container,
  Box,
  Typography,
  Button,
  Stack,
  Paper,
  Chip,
  Alert,
  AlertTitle,
  ToggleButtonGroup,
  ToggleButton,
} from '@mui/material';
import PredictionCard from '../PredictionCard';
import type { PredictionResult } from '../PredictionCard';

// ============================================================================
// MOCK DATA
// ============================================================================

const MOCK_PREDICTIONS: Record<string, PredictionResult> = {
  healthy: {
    classification: {
      failure_type: 'normal',
      confidence: 0.95,
      failure_probability: 0.05,
      all_probabilities: {
        normal: 0.95,
        bearing_wear: 0.03,
        overheating: 0.01,
        electrical_fault: 0.01,
      },
    },
    rul: {
      rul_hours: 720,
      rul_days: 30,
      urgency: 'low',
      maintenance_window: 'Schedule within 1 week',
    },
    timestamp: new Date(Date.now() - 2 * 60 * 1000).toISOString(), // 2 minutes ago
  },

  degrading: {
    classification: {
      failure_type: 'bearing_wear',
      confidence: 0.87,
      failure_probability: 0.25,
      all_probabilities: {
        normal: 0.75,
        bearing_wear: 0.18,
        overheating: 0.05,
        electrical_fault: 0.02,
      },
    },
    rul: {
      rul_hours: 156,
      rul_days: 6.5,
      urgency: 'medium',
      maintenance_window: 'Schedule within 3 days',
    },
    timestamp: new Date(Date.now() - 5 * 60 * 1000).toISOString(), // 5 minutes ago
  },

  warning: {
    classification: {
      failure_type: 'overheating',
      confidence: 0.82,
      failure_probability: 0.55,
      all_probabilities: {
        normal: 0.45,
        bearing_wear: 0.20,
        overheating: 0.30,
        electrical_fault: 0.05,
      },
    },
    rul: {
      rul_hours: 36,
      rul_days: 1.5,
      urgency: 'high',
      maintenance_window: 'Schedule within 24 hours',
    },
    timestamp: new Date(Date.now() - 10 * 60 * 1000).toISOString(), // 10 minutes ago
  },

  critical: {
    classification: {
      failure_type: 'electrical_fault',
      confidence: 0.92,
      failure_probability: 0.85,
      all_probabilities: {
        normal: 0.15,
        bearing_wear: 0.05,
        overheating: 0.10,
        electrical_fault: 0.70,
      },
    },
    rul: {
      rul_hours: 8,
      rul_days: 0.33,
      urgency: 'critical',
      maintenance_window: 'IMMEDIATE MAINTENANCE REQUIRED',
    },
    timestamp: new Date(Date.now() - 1 * 60 * 1000).toISOString(), // 1 minute ago
  },

  noRul: {
    classification: {
      failure_type: 'normal',
      confidence: 0.93,
      failure_probability: 0.07,
      all_probabilities: {
        normal: 0.93,
        bearing_wear: 0.04,
        overheating: 0.02,
        electrical_fault: 0.01,
      },
    },
    timestamp: new Date(Date.now() - 30 * 1000).toISOString(), // 30 seconds ago
  },
};

// ============================================================================
// DEMO COMPONENT
// ============================================================================

export default function PredictionCardDemo() {
  const [selectedScenario, setSelectedScenario] = useState<string>('healthy');
  const [currentPrediction, setCurrentPrediction] = useState<PredictionResult | null>(
    MOCK_PREDICTIONS.healthy
  );
  const [loading, setLoading] = useState(false);
  const [autoRefresh, setAutoRefresh] = useState(false);

  // Handle scenario change
  const handleScenarioChange = (
    _event: React.MouseEvent<HTMLElement>,
    newScenario: string | null
  ) => {
    if (newScenario !== null) {
      setSelectedScenario(newScenario);
      setCurrentPrediction(MOCK_PREDICTIONS[newScenario]);
    }
  };

  // Simulate running prediction
  const handleRunPrediction = () => {
    setLoading(true);
    setTimeout(() => {
      setCurrentPrediction({
        ...MOCK_PREDICTIONS[selectedScenario],
        timestamp: new Date().toISOString(),
      });
      setLoading(false);
    }, 2000);
  };

  // Handle AI explanation
  const handleExplain = () => {
    alert('AI Explanation Modal would open here (LLM integration in Day 18.1)');
  };

  // Handle view history
  const handleViewHistory = () => {
    alert('Prediction History Modal would open here (Day 18.1)');
  };

  // Clear prediction
  const handleClear = () => {
    setCurrentPrediction(null);
  };

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      {/* Header */}
      <Box sx={{ mb: 4 }}>
        <Typography variant="h4" sx={{ color: '#f9fafb', fontWeight: 700, mb: 1 }}>
          PredictionCard Component Demo
        </Typography>
        <Typography variant="body1" sx={{ color: '#9ca3af' }}>
          Interactive demonstration of the PredictionCard component with different health states
        </Typography>
      </Box>

      {/* Architectural Rule Alert */}
      <Alert severity="info" sx={{ mb: 4 }}>
        <AlertTitle>🏗️ Architectural Rule: Single-Machine Monitoring</AlertTitle>
        This component displays ML prediction results for <strong>ONE selected machine at a time</strong>.
        No fleet-wide operations per Phase 3.7.3 specifications.
      </Alert>

      {/* Demo Controls */}
      <Paper
        sx={{
          p: 3,
          mb: 4,
          background: 'linear-gradient(135deg, rgba(31, 41, 55, 0.8) 0%, rgba(17, 24, 39, 0.8) 100%)',
          backdropFilter: 'blur(10px)',
          border: '1px solid rgba(255, 255, 255, 0.1)',
        }}
      >
        <Typography variant="h6" sx={{ color: '#f9fafb', mb: 2 }}>
          Demo Controls
        </Typography>

        <Stack spacing={3}>
          {/* Scenario Selection */}
          <Box>
            <Typography variant="subtitle2" sx={{ color: '#d1d5db', mb: 1 }}>
              Select Scenario:
            </Typography>
            <ToggleButtonGroup
              value={selectedScenario}
              exclusive
              onChange={handleScenarioChange}
              aria-label="scenario selection"
              sx={{
                flexWrap: 'wrap',
                gap: 1,
              }}
            >
              <ToggleButton
                value="healthy"
                sx={{
                  color: '#10b981',
                  borderColor: '#10b981',
                  '&.Mui-selected': {
                    bgcolor: 'rgba(16, 185, 129, 0.2)',
                    color: '#10b981',
                  },
                }}
              >
                🟢 Healthy (5% failure)
              </ToggleButton>
              <ToggleButton
                value="degrading"
                sx={{
                  color: '#fbbf24',
                  borderColor: '#fbbf24',
                  '&.Mui-selected': {
                    bgcolor: 'rgba(251, 191, 36, 0.2)',
                    color: '#fbbf24',
                  },
                }}
              >
                🟡 Degrading (25% failure)
              </ToggleButton>
              <ToggleButton
                value="warning"
                sx={{
                  color: '#f97316',
                  borderColor: '#f97316',
                  '&.Mui-selected': {
                    bgcolor: 'rgba(249, 115, 22, 0.2)',
                    color: '#f97316',
                  },
                }}
              >
                🟠 Warning (55% failure)
              </ToggleButton>
              <ToggleButton
                value="critical"
                sx={{
                  color: '#ef4444',
                  borderColor: '#ef4444',
                  '&.Mui-selected': {
                    bgcolor: 'rgba(239, 68, 68, 0.2)',
                    color: '#ef4444',
                  },
                }}
              >
                🔴 Critical (85% failure)
              </ToggleButton>
              <ToggleButton
                value="noRul"
                sx={{
                  color: '#6b7280',
                  borderColor: '#6b7280',
                  '&.Mui-selected': {
                    bgcolor: 'rgba(107, 114, 128, 0.2)',
                    color: '#6b7280',
                  },
                }}
              >
                No RUL Data
              </ToggleButton>
            </ToggleButtonGroup>
          </Box>

          {/* Action Buttons */}
          <Stack direction="row" spacing={2} flexWrap="wrap">
            <Button
              variant="contained"
              onClick={handleRunPrediction}
              disabled={loading}
              sx={{
                background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
              }}
            >
              Simulate Run Prediction
            </Button>

            <Button
              variant="outlined"
              onClick={handleClear}
              sx={{
                borderColor: '#ef4444',
                color: '#ef4444',
                '&:hover': {
                  borderColor: '#dc2626',
                  bgcolor: 'rgba(239, 68, 68, 0.1)',
                },
              }}
            >
              Clear Prediction
            </Button>

            <Button
              variant={autoRefresh ? 'contained' : 'outlined'}
              onClick={() => setAutoRefresh(!autoRefresh)}
              sx={{
                borderColor: '#10b981',
                color: autoRefresh ? '#fff' : '#10b981',
                bgcolor: autoRefresh ? '#10b981' : 'transparent',
                '&:hover': {
                  borderColor: '#059669',
                  bgcolor: autoRefresh ? '#059669' : 'rgba(16, 185, 129, 0.1)',
                },
              }}
            >
              {autoRefresh ? '✓ Auto-Refresh ON' : 'Auto-Refresh OFF'}
            </Button>
          </Stack>
        </Stack>
      </Paper>

      {/* Statistics */}
      <Paper
        sx={{
          p: 3,
          mb: 4,
          background: 'rgba(31, 41, 55, 0.5)',
          border: '1px solid rgba(255, 255, 255, 0.1)',
        }}
      >
        <Typography variant="h6" sx={{ color: '#f9fafb', mb: 2 }}>
          Component Statistics
        </Typography>
        <Stack direction="row" spacing={3} flexWrap="wrap">
          <Chip
            label={`Current Scenario: ${selectedScenario}`}
            sx={{ bgcolor: 'rgba(102, 126, 234, 0.2)', color: '#667eea' }}
          />
          <Chip
            label={`Loading: ${loading ? 'Yes' : 'No'}`}
            sx={{ bgcolor: 'rgba(251, 191, 36, 0.2)', color: '#fbbf24' }}
          />
          <Chip
            label={`Auto-Refresh: ${autoRefresh ? 'Enabled (30s)' : 'Disabled'}`}
            sx={{ bgcolor: 'rgba(16, 185, 129, 0.2)', color: '#10b981' }}
          />
          <Chip
            label={`Prediction: ${currentPrediction ? 'Available' : 'Empty'}`}
            sx={{ bgcolor: 'rgba(118, 75, 162, 0.2)', color: '#764ba2' }}
          />
        </Stack>
      </Paper>

      {/* PredictionCard Component */}
      <PredictionCard
        machineId="motor_siemens_1la7_001"
        prediction={currentPrediction}
        loading={loading}
        onRunPrediction={handleRunPrediction}
        onExplain={handleExplain}
        onViewHistory={handleViewHistory}
        autoRefresh={autoRefresh}
        refreshInterval={30}
      />

      {/* Feature Checklist */}
      <Paper
        sx={{
          p: 3,
          mt: 4,
          background: 'rgba(31, 41, 55, 0.5)',
          border: '1px solid rgba(255, 255, 255, 0.1)',
        }}
      >
        <Typography variant="h6" sx={{ color: '#f9fafb', mb: 2 }}>
          ✅ Features Implemented
        </Typography>
        <Stack spacing={1}>
          <Typography variant="body2" sx={{ color: '#10b981' }}>
            ✓ Status Badge with color coding (🟢 Healthy, 🟡 Degrading, 🟠 Warning, 🔴 Critical)
          </Typography>
          <Typography variant="body2" sx={{ color: '#10b981' }}>
            ✓ Confidence Display with progress ring
          </Typography>
          <Typography variant="body2" sx={{ color: '#10b981' }}>
            ✓ RUL Countdown (hours + days) with urgency indicator
          </Typography>
          <Typography variant="body2" sx={{ color: '#10b981' }}>
            ✓ Probability Bars for all failure types (sorted by probability)
          </Typography>
          <Typography variant="body2" sx={{ color: '#10b981' }}>
            ✓ Run Prediction Button (triggers new prediction)
          </Typography>
          <Typography variant="body2" sx={{ color: '#10b981' }}>
            ✓ AI Explanation Button (opens LLM modal)
          </Typography>
          <Typography variant="body2" sx={{ color: '#10b981' }}>
            ✓ View History Button (opens prediction history)
          </Typography>
          <Typography variant="body2" sx={{ color: '#10b981' }}>
            ✓ Loading State with skeleton and spinner
          </Typography>
          <Typography variant="body2" sx={{ color: '#10b981' }}>
            ✓ Empty State ("No prediction yet")
          </Typography>
          <Typography variant="body2" sx={{ color: '#10b981' }}>
            ✓ Auto-Refresh with countdown timer (30s interval)
          </Typography>
          <Typography variant="body2" sx={{ color: '#10b981' }}>
            ✓ Pulse animation for Critical status
          </Typography>
          <Typography variant="body2" sx={{ color: '#10b981' }}>
            ✓ Relative time display ("2 minutes ago")
          </Typography>
          <Typography variant="body2" sx={{ color: '#10b981' }}>
            ✓ Responsive layout (mobile/tablet/desktop)
          </Typography>
          <Typography variant="body2" sx={{ color: '#10b981' }}>
            ✓ Professional dark theme with glassmorphism
          </Typography>
        </Stack>
      </Paper>
    </Container>
  );
}
