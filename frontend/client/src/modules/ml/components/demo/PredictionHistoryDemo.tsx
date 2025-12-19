/**
 * PredictionHistoryDemo Component
 * 
 * Interactive demo for PredictionHistory with:
 * - 50 mock historical predictions
 * - Various health states and urgency levels
 * - Realistic progression over time
 * - Control panel for data regeneration
 */

import { useState, useMemo } from 'react';
import {
  Box,
  Container,
  Paper,
  Typography,
  Button,
  Stack,
  Grid,
  Chip,
  Divider,
} from '@mui/material';
import {
  Refresh as RefreshIcon,
  CheckCircle as HealthyIcon,
  Warning as WarningIcon,
  Error as CriticalIcon,
} from '@mui/icons-material';
import PredictionHistory from '../PredictionHistory';
import type { HistoricalPrediction } from '../PredictionHistory';

// ============================================================================
// MOCK DATA GENERATION
// ============================================================================

const FAILURE_TYPES = [
  'Bearing Wear',
  'Bearing Seizure',
  'Spindle Imbalance',
  'Motor Overload',
  'Coolant System Failure',
  'Tool Breakage',
  'None',
];

const getRandomHealthState = (trendFactor: number): string => {
  // Simulate degradation trend over time
  const random = Math.random() + trendFactor;
  if (random < 0.5) return 'HEALTHY';
  if (random < 0.7) return 'DEGRADING';
  if (random < 0.85) return 'WARNING';
  return 'CRITICAL';
};

const getFailureTypeForState = (healthState: string): string => {
  if (healthState === 'HEALTHY') return 'None';
  const types = FAILURE_TYPES.filter(t => t !== 'None');
  return types[Math.floor(Math.random() * types.length)];
};

const getUrgencyForRUL = (rulHours: number): string => {
  if (rulHours > 240) return 'Low';
  if (rulHours > 120) return 'Medium';
  if (rulHours > 48) return 'High';
  return 'Critical';
};

const generateMockPredictions = (count: number): HistoricalPrediction[] => {
  const predictions: HistoricalPrediction[] = [];
  const now = new Date();

  for (let i = 0; i < count; i++) {
    // Generate timestamps going back in time (15 minutes apart)
    const timestamp = new Date(now.getTime() - i * 15 * 60 * 1000);
    
    // Add trend factor (increases with older data to simulate improvement over time)
    const trendFactor = i / count * 0.2;
    
    const healthState = getRandomHealthState(trendFactor);
    const failureType = getFailureTypeForState(healthState);
    
    // Generate RUL based on health state
    let rulHours: number;
    if (healthState === 'HEALTHY') {
      rulHours = 400 + Math.random() * 300;
    } else if (healthState === 'DEGRADING') {
      rulHours = 150 + Math.random() * 250;
    } else if (healthState === 'WARNING') {
      rulHours = 50 + Math.random() * 100;
    } else {
      rulHours = 10 + Math.random() * 40;
    }

    // Generate confidence based on health state
    let confidence: number;
    if (healthState === 'HEALTHY') {
      confidence = 0.85 + Math.random() * 0.1;
    } else if (healthState === 'DEGRADING') {
      confidence = 0.75 + Math.random() * 0.15;
    } else if (healthState === 'WARNING') {
      confidence = 0.70 + Math.random() * 0.15;
    } else {
      confidence = 0.80 + Math.random() * 0.15;
    }

    const urgency = getUrgencyForRUL(rulHours);

    predictions.push({
      id: `pred_${i + 1}`,
      timestamp,
      failure_type: failureType,
      confidence,
      rul_hours: rulHours,
      urgency,
      health_state: healthState,
    });
  }

  return predictions;
};

// ============================================================================
// MAIN DEMO COMPONENT
// ============================================================================

export default function PredictionHistoryDemo() {
  const [predictions, setPredictions] = useState<HistoricalPrediction[]>(
    () => generateMockPredictions(50)
  );

  const handleRefresh = () => {
    setPredictions(generateMockPredictions(50));
  };

  // Calculate statistics
  const stats = useMemo(() => {
    const healthyCount = predictions.filter(p => p.health_state === 'HEALTHY').length;
    const degradingCount = predictions.filter(p => p.health_state === 'DEGRADING').length;
    const warningCount = predictions.filter(p => p.health_state === 'WARNING').length;
    const criticalCount = predictions.filter(p => p.health_state === 'CRITICAL').length;

    return {
      total: predictions.length,
      healthy: healthyCount,
      degrading: degradingCount,
      warning: warningCount,
      critical: criticalCount,
      avgConfidence: (predictions.reduce((sum, p) => sum + p.confidence, 0) / predictions.length * 100).toFixed(1),
      avgRUL: (predictions.reduce((sum, p) => sum + p.rul_hours, 0) / predictions.length).toFixed(1),
    };
  }, [predictions]);

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
      <Container maxWidth="xl">
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
            PREDICTION HISTORY DEMO
          </Typography>
          <Typography variant="body1" sx={{ color: '#d1d5db' }}>
            Phase 3.7.3 - Day 18.1: Paginated table with historical ML predictions
          </Typography>
        </Paper>

        {/* STATISTICS & CONTROLS */}
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
          <Grid container spacing={3} alignItems="center">
            {/* Statistics */}
            <Grid item xs={12} md={9}>
              <Grid container spacing={2}>
                <Grid item xs={6} sm={3}>
                  <Stack spacing={0.5}>
                    <Typography variant="caption" sx={{ color: '#9ca3af' }}>
                      Total Predictions
                    </Typography>
                    <Chip
                      label={stats.total}
                      sx={{
                        bgcolor: 'rgba(102, 126, 234, 0.2)',
                        color: '#667eea',
                        fontWeight: 600,
                        fontSize: 16,
                      }}
                    />
                  </Stack>
                </Grid>

                <Grid item xs={6} sm={3}>
                  <Stack spacing={0.5}>
                    <Stack direction="row" spacing={0.5} alignItems="center">
                      <HealthyIcon sx={{ fontSize: 16, color: '#10b981' }} />
                      <Typography variant="caption" sx={{ color: '#9ca3af' }}>
                        Healthy
                      </Typography>
                    </Stack>
                    <Chip
                      label={stats.healthy}
                      sx={{
                        bgcolor: 'rgba(16, 185, 129, 0.2)',
                        color: '#10b981',
                        fontWeight: 600,
                        fontSize: 16,
                      }}
                    />
                  </Stack>
                </Grid>

                <Grid item xs={6} sm={3}>
                  <Stack spacing={0.5}>
                    <Stack direction="row" spacing={0.5} alignItems="center">
                      <WarningIcon sx={{ fontSize: 16, color: '#fbbf24' }} />
                      <Typography variant="caption" sx={{ color: '#9ca3af' }}>
                        Warning
                      </Typography>
                    </Stack>
                    <Chip
                      label={stats.warning}
                      sx={{
                        bgcolor: 'rgba(251, 191, 36, 0.2)',
                        color: '#fbbf24',
                        fontWeight: 600,
                        fontSize: 16,
                      }}
                    />
                  </Stack>
                </Grid>

                <Grid item xs={6} sm={3}>
                  <Stack spacing={0.5}>
                    <Stack direction="row" spacing={0.5} alignItems="center">
                      <CriticalIcon sx={{ fontSize: 16, color: '#ef4444' }} />
                      <Typography variant="caption" sx={{ color: '#9ca3af' }}>
                        Critical
                      </Typography>
                    </Stack>
                    <Chip
                      label={stats.critical}
                      sx={{
                        bgcolor: 'rgba(239, 68, 68, 0.2)',
                        color: '#ef4444',
                        fontWeight: 600,
                        fontSize: 16,
                      }}
                    />
                  </Stack>
                </Grid>

                <Grid item xs={6} sm={3}>
                  <Stack spacing={0.5}>
                    <Typography variant="caption" sx={{ color: '#9ca3af' }}>
                      Avg Confidence
                    </Typography>
                    <Typography variant="body1" sx={{ color: '#d1d5db', fontWeight: 600 }}>
                      {stats.avgConfidence}%
                    </Typography>
                  </Stack>
                </Grid>

                <Grid item xs={6} sm={3}>
                  <Stack spacing={0.5}>
                    <Typography variant="caption" sx={{ color: '#9ca3af' }}>
                      Avg RUL
                    </Typography>
                    <Typography variant="body1" sx={{ color: '#d1d5db', fontWeight: 600 }}>
                      {stats.avgRUL} hrs
                    </Typography>
                  </Stack>
                </Grid>
              </Grid>
            </Grid>

            {/* Controls */}
            <Grid item xs={12} md={3}>
              <Stack spacing={2}>
                <Button
                  variant="outlined"
                  fullWidth
                  startIcon={<RefreshIcon />}
                  onClick={handleRefresh}
                  sx={{
                    borderColor: '#667eea',
                    color: '#667eea',
                    '&:hover': {
                      borderColor: '#5568d3',
                      bgcolor: 'rgba(102, 126, 234, 0.1)',
                    },
                  }}
                >
                  Regenerate Data
                </Button>
              </Stack>
            </Grid>
          </Grid>
        </Paper>

        {/* PREDICTION HISTORY TABLE */}
        <PredictionHistory
          machineId="CNC_BROTHER_SPEEDIO_001"
          predictions={predictions}
          limit={100}
        />

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
                  [OK] Paginated DataGrid (10/25/50 rows per page)
                </Typography>
                <Typography variant="body2" sx={{ color: '#10b981' }}>
                  [OK] Sortable columns (click headers)
                </Typography>
                <Typography variant="body2" sx={{ color: '#10b981' }}>
                  [OK] Date range filter (hour/day/week/month/all)
                </Typography>
                <Typography variant="body2" sx={{ color: '#10b981' }}>
                  [OK] Search by failure type or urgency
                </Typography>
                <Typography variant="body2" sx={{ color: '#10b981' }}>
                  [OK] CSV export functionality
                </Typography>
              </Stack>
            </Grid>
            <Grid item xs={12} sm={6}>
              <Stack spacing={1}>
                <Typography variant="body2" sx={{ color: '#10b981' }}>
                  [OK] Row click to view details (modal)
                </Typography>
                <Typography variant="body2" sx={{ color: '#10b981' }}>
                  [OK] Color-coded status icons
                </Typography>
                <Typography variant="body2" sx={{ color: '#10b981' }}>
                  [OK] Confidence progress bars
                </Typography>
                <Typography variant="body2" sx={{ color: '#10b981' }}>
                  [OK] Urgency badges with colors
                </Typography>
                <Typography variant="body2" sx={{ color: '#10b981' }}>
                  [OK] Responsive design
                </Typography>
              </Stack>
            </Grid>
          </Grid>
        </Paper>
      </Container>
    </Box>
  );
}
