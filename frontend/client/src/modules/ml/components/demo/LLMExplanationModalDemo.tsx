/**
 * LLMExplanationModalDemo Component
 * 
 * Interactive demo for LLM Explanation Modal with:
 * - Multiple mock scenarios (Healthy, Warning, Critical)
 * - Simulated API responses
 * - Network delay simulation
 * - Error state testing
 */

import { useState } from 'react';
import {
  Box,
  Container,
  Paper,
  Typography,
  Button,
  Stack,
  Grid,
  Card,
  CardContent,
  Chip,
  Divider,
} from '@mui/material';
import {
  Psychology as BrainIcon,
  CheckCircle as HealthyIcon,
  Warning as WarningIcon,
  Error as CriticalIcon,
  BugReport as ErrorIcon,
} from '@mui/icons-material';
import LLMExplanationModal from '../LLMExplanationModal';
import type { PredictionData } from '../LLMExplanationModal';

// ============================================================================
// MOCK DATA
// ============================================================================

const MOCK_SCENARIOS = {
  healthy: {
    data: {
      health_state: 'HEALTHY',
      confidence: 0.94,
      failure_probability: 0.06,
      predicted_failure_type: 'None',
      rul_hours: 720,
      sensor_data: {
        bearing_temperature: 52.3,
        vibration_x: 1.8,
        spindle_current: 8.5,
      },
    } as PredictionData,
    explanation: {
      summary: 'The machine is operating within **normal parameters**. All sensor readings are stable and within acceptable ranges. No immediate maintenance actions are required. The predicted Remaining Useful Life (RUL) of **720 hours** indicates excellent operational health.',
      risk_factors: [
        'Minor bearing temperature fluctuation detected (±2°C), but within normal range',
        'Vibration levels are low (1.8 mm/s), indicating stable mechanical condition',
        'Current draw is optimal, suggesting efficient motor operation',
      ],
      recommendations: [
        '**Continue routine monitoring** - Maintain current inspection schedule',
        '**Track temperature trends** - Monitor for gradual increases over time',
        '**Lubrication check** - Verify bearing lubrication at next scheduled maintenance',
        '**Document baseline** - Record current sensor values as healthy reference points',
      ],
      technical_details: 'Statistical analysis shows all sensor values are within 1.5 standard deviations of historical mean. No anomalous patterns detected in time-series data. Fourier analysis of vibration signals shows no harmonic distortion.',
      confidence_analysis: 'High confidence (94%) based on: (1) Consistent sensor readings over 48-hour window, (2) Strong correlation with healthy training data, (3) Low prediction variance across ensemble models.',
    },
  },
  warning: {
    data: {
      health_state: 'WARNING',
      confidence: 0.82,
      failure_probability: 0.53,
      predicted_failure_type: 'Bearing Wear',
      rul_hours: 168,
      sensor_data: {
        bearing_temperature: 78.5,
        vibration_x: 5.2,
        spindle_current: 15.3,
      },
    } as PredictionData,
    explanation: {
      summary: 'The machine shows **early warning signs** of potential bearing degradation. Elevated temperature (78.5°C) and increased vibration (5.2 mm/s) indicate developing mechanical issues. **Immediate inspection recommended** within the next 7 days. Predicted RUL: **168 hours (1 week)**.',
      risk_factors: [
        '**Bearing temperature elevated by 26°C** above baseline - indicates increased friction',
        '**Vibration amplitude 2.9x normal** - suggests bearing cage wear or misalignment',
        '**Current draw increased by 80%** - motor compensating for mechanical resistance',
        'Temperature-vibration correlation pattern matches known bearing failure signatures',
        'Trend analysis shows 15% degradation rate over past 72 hours',
      ],
      recommendations: [
        '**Schedule inspection within 7 days** - Visual and thermographic examination of bearing assemblies',
        '**Increase monitoring frequency** - Switch to hourly data collection instead of 5-second intervals',
        '**Prepare replacement parts** - Order bearing kits for CNC Brother Speedio model',
        '**Reduce operational load** - Limit high-speed cutting operations to 70% of rated capacity',
        '**Verify lubrication system** - Check for oil contamination or inadequate flow',
        '**Plan maintenance window** - Coordinate downtime for proactive bearing replacement',
      ],
      technical_details: 'Spectral analysis reveals bearing frequency components at 1.8x and 2.6x Ball Pass Frequency Outer Race (BPFO). Time-domain kurtosis exceeds 4.5, indicating impulse events. Temperature gradient analysis shows localized hotspot at bearing housing.',
      confidence_analysis: 'Moderate-high confidence (82%) based on: (1) Clear degradation trend over 3-day window, (2) Multiple correlated sensor anomalies, (3) Pattern matching with historical bearing failure cases. Lower confidence due to variability in current measurements.',
    },
  },
  critical: {
    data: {
      health_state: 'CRITICAL',
      confidence: 0.91,
      failure_probability: 0.89,
      predicted_failure_type: 'Bearing Seizure',
      rul_hours: 24,
      sensor_data: {
        bearing_temperature: 95.8,
        vibration_x: 9.7,
        spindle_current: 22.1,
      },
    } as PredictionData,
    explanation: {
      summary: '**CRITICAL ALERT**: Machine exhibits severe degradation symptoms consistent with **imminent bearing failure**. Temperature at 95.8°C (near bearing material limits), extreme vibration (9.7 mm/s), and excessive current draw (22.1A) indicate advanced damage. **IMMEDIATE ACTION REQUIRED**. Predicted RUL: **24 hours or less**.',
      risk_factors: [
        '**SEVERE: Bearing temperature 43°C above normal** - approaching material degradation threshold (100°C)',
        '**SEVERE: Vibration 5.4x baseline** - indicates advanced bearing cage failure or ball spalling',
        '**SEVERE: Current draw 2.6x rated** - motor overheating risk, thermal protection may trip',
        '**HIGH: Rapid temperature escalation** - 18°C increase in last 6 hours (exponential trend)',
        '**HIGH: Vibration shocks detected** - intermittent impact events suggesting loose components',
        '**MODERATE: Acoustic signature anomalies** - grinding noises reported by operators',
      ],
      recommendations: [
        '**URGENT: Stop machine immediately** - Risk of catastrophic failure outweighs production loss',
        '**IMMEDIATE: Safety inspection** - Check for oil leaks, smoke, or unusual odors',
        '**PRIORITY 1: Emergency maintenance** - Replace all spindle bearings and inspect shaft for damage',
        '**Thermal imaging scan** - Identify exact failure location before disassembly',
        '**Root cause analysis** - Investigate lubrication failure, contamination, or installation errors',
        '**Quality check completed parts** - Inspect recent workpieces for dimensional errors',
        '**Update maintenance procedures** - Review why early warning signs were not detected',
      ],
      technical_details: 'Advanced failure indicators: (1) Bearing BPFO harmonics up to 8th order, (2) Envelope acceleration exceeds 50g RMS, (3) Oil debris sensor triggered (metallic particles >100µm), (4) Spindle runout increased to 0.015mm TIR, (5) Temperature rise rate: 3°C/hour (critical threshold).',
      confidence_analysis: 'Very high confidence (91%) based on: (1) Multiple independent failure indicators, (2) Consistent with textbook bearing seizure progression, (3) Low model variance (±4% across ensemble), (4) Historical validation: 89% of similar cases resulted in failure within 36 hours.',
    },
  },
};

// ============================================================================
// MOCK API HANDLER
// ============================================================================

const mockFetchExplanation = (
  scenario: keyof typeof MOCK_SCENARIOS,
  shouldError: boolean = false
): Promise<any> => {
  return new Promise((resolve, reject) => {
    // Simulate network delay (1-3 seconds)
    const delay = 1000 + Math.random() * 2000;

    setTimeout(() => {
      if (shouldError) {
        reject(new Error('Network error: Unable to connect to LLM service'));
      } else {
        resolve(MOCK_SCENARIOS[scenario].explanation);
      }
    }, delay);
  });
};

// ============================================================================
// MAIN DEMO COMPONENT
// ============================================================================

export default function LLMExplanationModalDemo() {
  const [modalOpen, setModalOpen] = useState(false);
  const [currentScenario, setCurrentScenario] = useState<keyof typeof MOCK_SCENARIOS>('warning');
  const [simulateError, setSimulateError] = useState(false);

  const handleOpenModal = (scenario: keyof typeof MOCK_SCENARIOS, withError: boolean = false) => {
    setCurrentScenario(scenario);
    setSimulateError(withError);
    setModalOpen(true);
  };

  const handleCloseModal = () => {
    setModalOpen(false);
  };

  // Custom fetch handler for demo
  const demoApiEndpoint = async (_url: string, _options: RequestInit) => {
    // Intercept fetch call and return mock data
    return {
      ok: !simulateError,
      statusText: simulateError ? 'Service Unavailable' : 'OK',
      json: async () => mockFetchExplanation(currentScenario, simulateError),
    } as Response;
  };

  // ============================================================================
  // RENDER
  // ============================================================================

  return (
    <Box
      sx={{
        minHeight: '100vh',
        background: 'linear-gradient(135deg, #0f172a 0%, #1e293b 100%)',
        py: 4,
      }}
    >
      <Container maxWidth="lg">
        {/* HEADER */}
        <Paper
          elevation={3}
          sx={{
            p: 3,
            mb: 3,
            background: 'linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%)',
            backdropFilter: 'blur(10px)',
            border: '1px solid rgba(255, 255, 255, 0.1)',
          }}
        >
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
            LLM EXPLANATION MODAL DEMO
          </Typography>
          <Typography variant="body1" sx={{ color: '#d1d5db' }}>
            Phase 3.7.3 - Day 18.1: AI-generated failure explanations with Markdown rendering
          </Typography>
        </Paper>

        {/* SCENARIO CARDS */}
        <Grid container spacing={3} sx={{ mb: 3 }}>
          {/* HEALTHY SCENARIO */}
          <Grid item xs={12} md={4}>
            <Card
              sx={{
                background: 'linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(5, 150, 105, 0.1) 100%)',
                border: '1px solid rgba(16, 185, 129, 0.3)',
                height: '100%',
              }}
            >
              <CardContent>
                <Stack spacing={2}>
                  <Stack direction="row" spacing={1} alignItems="center">
                    <HealthyIcon sx={{ color: '#10b981', fontSize: 32 }} />
                    <Typography variant="h6" sx={{ color: '#f9fafb', fontWeight: 600 }}>
                      Healthy Machine
                    </Typography>
                  </Stack>
                  <Divider sx={{ borderColor: 'rgba(255, 255, 255, 0.1)' }} />
                  <Stack spacing={1}>
                    <Chip
                      label="Confidence: 94%"
                      size="small"
                      sx={{ bgcolor: 'rgba(16, 185, 129, 0.2)', color: '#10b981' }}
                    />
                    <Chip
                      label="RUL: 720 hours"
                      size="small"
                      sx={{ bgcolor: 'rgba(16, 185, 129, 0.2)', color: '#10b981' }}
                    />
                    <Chip
                      label="Risk: Low (6%)"
                      size="small"
                      sx={{ bgcolor: 'rgba(16, 185, 129, 0.2)', color: '#10b981' }}
                    />
                  </Stack>
                  <Button
                    variant="contained"
                    fullWidth
                    startIcon={<BrainIcon />}
                    onClick={() => handleOpenModal('healthy')}
                    sx={{
                      bgcolor: '#10b981',
                      '&:hover': { bgcolor: '#059669' },
                    }}
                  >
                    View Explanation
                  </Button>
                </Stack>
              </CardContent>
            </Card>
          </Grid>

          {/* WARNING SCENARIO */}
          <Grid item xs={12} md={4}>
            <Card
              sx={{
                background: 'linear-gradient(135deg, rgba(251, 191, 36, 0.1) 0%, rgba(245, 158, 11, 0.1) 100%)',
                border: '1px solid rgba(251, 191, 36, 0.3)',
                height: '100%',
              }}
            >
              <CardContent>
                <Stack spacing={2}>
                  <Stack direction="row" spacing={1} alignItems="center">
                    <WarningIcon sx={{ color: '#fbbf24', fontSize: 32 }} />
                    <Typography variant="h6" sx={{ color: '#f9fafb', fontWeight: 600 }}>
                      Warning State
                    </Typography>
                  </Stack>
                  <Divider sx={{ borderColor: 'rgba(255, 255, 255, 0.1)' }} />
                  <Stack spacing={1}>
                    <Chip
                      label="Confidence: 82%"
                      size="small"
                      sx={{ bgcolor: 'rgba(251, 191, 36, 0.2)', color: '#fbbf24' }}
                    />
                    <Chip
                      label="RUL: 168 hours"
                      size="small"
                      sx={{ bgcolor: 'rgba(251, 191, 36, 0.2)', color: '#fbbf24' }}
                    />
                    <Chip
                      label="Risk: Medium (53%)"
                      size="small"
                      sx={{ bgcolor: 'rgba(251, 191, 36, 0.2)', color: '#fbbf24' }}
                    />
                  </Stack>
                  <Button
                    variant="contained"
                    fullWidth
                    startIcon={<BrainIcon />}
                    onClick={() => handleOpenModal('warning')}
                    sx={{
                      bgcolor: '#fbbf24',
                      color: '#1f2937',
                      '&:hover': { bgcolor: '#f59e0b' },
                    }}
                  >
                    View Explanation
                  </Button>
                </Stack>
              </CardContent>
            </Card>
          </Grid>

          {/* CRITICAL SCENARIO */}
          <Grid item xs={12} md={4}>
            <Card
              sx={{
                background: 'linear-gradient(135deg, rgba(239, 68, 68, 0.1) 0%, rgba(220, 38, 38, 0.1) 100%)',
                border: '1px solid rgba(239, 68, 68, 0.3)',
                height: '100%',
              }}
            >
              <CardContent>
                <Stack spacing={2}>
                  <Stack direction="row" spacing={1} alignItems="center">
                    <CriticalIcon sx={{ color: '#ef4444', fontSize: 32 }} />
                    <Typography variant="h6" sx={{ color: '#f9fafb', fontWeight: 600 }}>
                      Critical State
                    </Typography>
                  </Stack>
                  <Divider sx={{ borderColor: 'rgba(255, 255, 255, 0.1)' }} />
                  <Stack spacing={1}>
                    <Chip
                      label="Confidence: 91%"
                      size="small"
                      sx={{ bgcolor: 'rgba(239, 68, 68, 0.2)', color: '#ef4444' }}
                    />
                    <Chip
                      label="RUL: 24 hours"
                      size="small"
                      sx={{ bgcolor: 'rgba(239, 68, 68, 0.2)', color: '#ef4444' }}
                    />
                    <Chip
                      label="Risk: High (89%)"
                      size="small"
                      sx={{ bgcolor: 'rgba(239, 68, 68, 0.2)', color: '#ef4444' }}
                    />
                  </Stack>
                  <Button
                    variant="contained"
                    fullWidth
                    startIcon={<BrainIcon />}
                    onClick={() => handleOpenModal('critical')}
                    sx={{
                      bgcolor: '#ef4444',
                      '&:hover': { bgcolor: '#dc2626' },
                    }}
                  >
                    View Explanation
                  </Button>
                </Stack>
              </CardContent>
            </Card>
          </Grid>
        </Grid>

        {/* ERROR STATE TEST */}
        <Paper
          elevation={3}
          sx={{
            p: 3,
            background: 'linear-gradient(135deg, rgba(31, 41, 55, 0.8) 0%, rgba(17, 24, 39, 0.8) 100%)',
            backdropFilter: 'blur(10px)',
            border: '1px solid rgba(255, 255, 255, 0.1)',
          }}
        >
          <Stack direction="row" spacing={2} alignItems="center">
            <ErrorIcon sx={{ color: '#f97316' }} />
            <Typography variant="h6" sx={{ color: '#f9fafb', fontWeight: 600 }}>
              Error State Testing
            </Typography>
            <Button
              variant="outlined"
              startIcon={<BrainIcon />}
              onClick={() => handleOpenModal('warning', true)}
              sx={{
                borderColor: '#f97316',
                color: '#f97316',
                '&:hover': {
                  borderColor: '#ea580c',
                  bgcolor: 'rgba(249, 115, 22, 0.1)',
                },
              }}
            >
              Trigger Error
            </Button>
          </Stack>
        </Paper>

        {/* FEATURES LIST */}
        <Paper
          elevation={3}
          sx={{
            p: 3,
            mt: 3,
            background: 'linear-gradient(135deg, rgba(31, 41, 55, 0.8) 0%, rgba(17, 24, 39, 0.8) 100%)',
            backdropFilter: 'blur(10px)',
            border: '1px solid rgba(255, 255, 255, 0.1)',
          }}
        >
          <Typography variant="h6" sx={{ fontWeight: 600, color: '#f9fafb', mb: 2 }}>
            IMPLEMENTED FEATURES
          </Typography>
          <Divider sx={{ borderColor: 'rgba(255, 255, 255, 0.1)', mb: 2 }} />
          <Grid container spacing={2}>
            <Grid item xs={12} sm={6}>
              <Stack spacing={1}>
                <Typography variant="body2" sx={{ color: '#10b981' }}>
                  [OK] Full-screen modal with backdrop blur
                </Typography>
                <Typography variant="body2" sx={{ color: '#10b981' }}>
                  [OK] Fetch from /api/llm/explain
                </Typography>
                <Typography variant="body2" sx={{ color: '#10b981' }}>
                  [OK] Markdown rendering with react-markdown
                </Typography>
                <Typography variant="body2" sx={{ color: '#10b981' }}>
                  [OK] Structured sections (Summary, Risks, Recommendations)
                </Typography>
              </Stack>
            </Grid>
            <Grid item xs={12} sm={6}>
              <Stack spacing={1}>
                <Typography variant="body2" sx={{ color: '#10b981' }}>
                  [OK] Copy to clipboard button
                </Typography>
                <Typography variant="body2" sx={{ color: '#10b981' }}>
                  [OK] Loading state with skeleton animation
                </Typography>
                <Typography variant="body2" sx={{ color: '#10b981' }}>
                  [OK] Error state with retry button
                </Typography>
                <Typography variant="body2" sx={{ color: '#10b981' }}>
                  [OK] Smooth slide-up animation
                </Typography>
              </Stack>
            </Grid>
          </Grid>
        </Paper>
      </Container>

      {/* MODAL */}
      <LLMExplanationModal
        open={modalOpen}
        onClose={handleCloseModal}
        machineId="CNC_BROTHER_SPEEDIO_001"
        predictionData={MOCK_SCENARIOS[currentScenario].data}
        apiEndpoint={demoApiEndpoint as any}
      />
    </Box>
  );
}
