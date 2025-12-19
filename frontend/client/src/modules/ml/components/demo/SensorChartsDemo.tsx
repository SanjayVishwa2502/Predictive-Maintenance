/**
 * SensorChartsDemo Component
 * 
 * Interactive demo for SensorCharts with:
 * - Real-time data generation (simulates 5-second intervals)
 * - Multiple sensor types with realistic value ranges
 * - Control panel for data generation speed
 * - Statistics display
 */

import { useState, useEffect, useCallback } from 'react';
import {
  Box,
  Container,
  Paper,
  Typography,
  Button,
  Stack,
  Chip,
  Grid,
  Divider,
} from '@mui/material';
import {
  PlayArrow as PlayArrowIcon,
  Pause as PauseIcon,
  Refresh as RefreshIcon,
  Speed as SpeedIcon,
} from '@mui/icons-material';
import SensorCharts from '../SensorCharts';
import type { SensorReading } from '../SensorCharts';

// ============================================================================
// MOCK DATA GENERATION
// ============================================================================

const AVAILABLE_SENSORS = [
  'bearing_temperature',
  'spindle_temperature',
  'vibration_x',
  'vibration_y',
  'vibration_z',
  'spindle_current',
  'feed_current',
  'coolant_pressure',
  'spindle_speed',
  'feed_rate',
];

const getSensorBaseline = (sensor: string): number => {
  if (sensor.includes('temp')) return 55;
  if (sensor.includes('vibration')) return 2.5;
  if (sensor.includes('current')) return 10;
  if (sensor.includes('pressure')) return 80;
  if (sensor.includes('speed')) return 3000;
  if (sensor.includes('rate')) return 500;
  return 50;
};

const getSensorVariance = (sensor: string): number => {
  if (sensor.includes('temp')) return 8;
  if (sensor.includes('vibration')) return 1.2;
  if (sensor.includes('current')) return 3;
  if (sensor.includes('pressure')) return 10;
  if (sensor.includes('speed')) return 200;
  if (sensor.includes('rate')) return 50;
  return 5;
};

const generateSensorValue = (
  sensor: string,
  time: number,
  previousValue?: number
): number => {
  const baseline = getSensorBaseline(sensor);
  const variance = getSensorVariance(sensor);
  
  // Simulate trend over time (gradual increase for degradation)
  const trendFactor = Math.sin(time / 50) * 0.3;
  
  // Random walk from previous value (smoother transitions)
  const randomWalk = previousValue
    ? previousValue + (Math.random() - 0.5) * variance * 0.3
    : baseline;
  
  // Combine baseline, trend, and random walk
  const value = baseline + trendFactor * variance + (randomWalk - baseline) * 0.7;
  
  return Math.max(0, value);
};

const generateInitialData = (count: number): SensorReading[] => {
  const now = new Date();
  const data: SensorReading[] = [];
  const previousValues: Record<string, number> = {};

  for (let i = 0; i < count; i++) {
    const timestamp = new Date(now.getTime() - (count - i) * 5000); // 5 seconds apart
    const values: Record<string, number> = {};

    AVAILABLE_SENSORS.forEach((sensor) => {
      values[sensor] = generateSensorValue(sensor, i, previousValues[sensor]);
      previousValues[sensor] = values[sensor];
    });

    data.push({ timestamp, values });
  }

  return data;
};

// ============================================================================
// MAIN DEMO COMPONENT
// ============================================================================

export default function SensorChartsDemo() {
  const [sensorHistory, setSensorHistory] = useState<SensorReading[]>(
    () => generateInitialData(60) // Start with 5 minutes of data
  );
  const [isGenerating, setIsGenerating] = useState(true);
  const [updateInterval, setUpdateInterval] = useState(1000); // 1 second for demo (faster than real 5s)
  const [selectedSensors, setSelectedSensors] = useState<string[]>([
    'bearing_temperature',
    'vibration_x',
    'spindle_current',
  ]);

  // Generate new data point
  const generateNewDataPoint = useCallback(() => {
    setSensorHistory((prev) => {
      const lastReading = prev[prev.length - 1];
      const newTimestamp = new Date(lastReading.timestamp.getTime() + 5000);
      const newValues: Record<string, number> = {};

      AVAILABLE_SENSORS.forEach((sensor) => {
        newValues[sensor] = generateSensorValue(
          sensor,
          prev.length,
          lastReading.values[sensor]
        );
      });

      return [...prev, { timestamp: newTimestamp, values: newValues }];
    });
  }, []);

  // Auto-generate data
  useEffect(() => {
    if (!isGenerating) return;

    const interval = setInterval(generateNewDataPoint, updateInterval);
    return () => clearInterval(interval);
  }, [isGenerating, updateInterval, generateNewDataPoint]);

  // Handle sensor toggle
  const handleSensorToggle = (sensor: string) => {
    setSelectedSensors((prev) => {
      if (prev.includes(sensor)) {
        return prev.filter((s) => s !== sensor);
      } else if (prev.length < 5) {
        return [...prev, sensor];
      }
      return prev;
    });
  };

  // Reset data
  const handleReset = () => {
    setSensorHistory(generateInitialData(60));
    setIsGenerating(true);
  };

  // Calculate statistics
  const stats = {
    dataPoints: sensorHistory.length,
    timeSpan: sensorHistory.length * 5, // seconds
    selectedCount: selectedSensors.length,
    updateRate: `${updateInterval / 1000}s`,
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
            SENSOR CHARTS DEMO
          </Typography>
          <Typography variant="body1" sx={{ color: '#d1d5db' }}>
            Phase 3.7.3 - Day 17.4: Multi-line time-series visualization with real-time updates
          </Typography>
        </Paper>

        {/* CONTROL PANEL */}
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
            {/* Controls */}
            <Grid item xs={12} md={6}>
              <Stack direction="row" spacing={2} flexWrap="wrap">
                <Button
                  variant="contained"
                  startIcon={isGenerating ? <PauseIcon /> : <PlayArrowIcon />}
                  onClick={() => setIsGenerating(!isGenerating)}
                  sx={{
                    bgcolor: isGenerating ? '#f97316' : '#10b981',
                    '&:hover': {
                      bgcolor: isGenerating ? '#ea580c' : '#059669',
                    },
                  }}
                >
                  {isGenerating ? 'Pause' : 'Resume'}
                </Button>

                <Button
                  variant="outlined"
                  startIcon={<RefreshIcon />}
                  onClick={handleReset}
                  sx={{
                    borderColor: '#667eea',
                    color: '#667eea',
                    '&:hover': {
                      borderColor: '#5568d3',
                      bgcolor: 'rgba(102, 126, 234, 0.1)',
                    },
                  }}
                >
                  Reset
                </Button>

                <Button
                  variant="outlined"
                  startIcon={<SpeedIcon />}
                  onClick={() =>
                    setUpdateInterval((prev) => (prev === 1000 ? 500 : prev === 500 ? 2000 : 1000))
                  }
                  sx={{
                    borderColor: '#667eea',
                    color: '#667eea',
                    '&:hover': {
                      borderColor: '#5568d3',
                      bgcolor: 'rgba(102, 126, 234, 0.1)',
                    },
                  }}
                >
                  Speed: {updateInterval === 500 ? '2x' : updateInterval === 1000 ? '1x' : '0.5x'}
                </Button>
              </Stack>
            </Grid>

            {/* Statistics */}
            <Grid item xs={12} md={6}>
              <Stack direction="row" spacing={1} flexWrap="wrap" justifyContent="flex-end">
                <Chip
                  label={`${stats.dataPoints} Points`}
                  size="small"
                  sx={{
                    bgcolor: 'rgba(102, 126, 234, 0.2)',
                    color: '#667eea',
                  }}
                />
                <Chip
                  label={`${Math.floor(stats.timeSpan / 60)} Minutes`}
                  size="small"
                  sx={{
                    bgcolor: 'rgba(16, 185, 129, 0.2)',
                    color: '#10b981',
                  }}
                />
                <Chip
                  label={`${stats.selectedCount}/5 Sensors`}
                  size="small"
                  sx={{
                    bgcolor: 'rgba(168, 85, 247, 0.2)',
                    color: '#a855f7',
                  }}
                />
                <Chip
                  label={`Update: ${stats.updateRate}`}
                  size="small"
                  sx={{
                    bgcolor: 'rgba(249, 115, 22, 0.2)',
                    color: '#f97316',
                  }}
                />
              </Stack>
            </Grid>
          </Grid>
        </Paper>

        {/* SENSOR CHARTS */}
        <SensorCharts
          machineId="CNC_BROTHER_SPEEDIO_001"
          sensorHistory={sensorHistory}
          availableSensors={AVAILABLE_SENSORS}
          selectedSensors={selectedSensors}
          onSensorToggle={handleSensorToggle}
          autoScroll={isGenerating}
          maxDataPoints={120}
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
          <Typography
            variant="h6"
            sx={{ fontWeight: 600, color: '#f9fafb', mb: 2 }}
          >
            IMPLEMENTED FEATURES
          </Typography>
          <Divider sx={{ borderColor: 'rgba(255, 255, 255, 0.1)', mb: 2 }} />
          <Grid container spacing={2}>
            <Grid item xs={12} sm={6}>
              <Stack spacing={1}>
                <Typography variant="body2" sx={{ color: '#10b981' }}>
                  [OK] Multi-line chart with Recharts
                </Typography>
                <Typography variant="body2" sx={{ color: '#10b981' }}>
                  [OK] Sensor selection (max 5)
                </Typography>
                <Typography variant="body2" sx={{ color: '#10b981' }}>
                  [OK] Color-coded by sensor category
                </Typography>
                <Typography variant="body2" sx={{ color: '#10b981' }}>
                  [OK] Auto-scroll with pause control
                </Typography>
                <Typography variant="body2" sx={{ color: '#10b981' }}>
                  [OK] Zoom in/out/reset controls
                </Typography>
              </Stack>
            </Grid>
            <Grid item xs={12} sm={6}>
              <Stack spacing={1}>
                <Typography variant="body2" sx={{ color: '#10b981' }}>
                  [OK] Hover tooltips with exact values
                </Typography>
                <Typography variant="body2" sx={{ color: '#10b981' }}>
                  [OK] Interactive legend
                </Typography>
                <Typography variant="body2" sx={{ color: '#10b981' }}>
                  [OK] Threshold reference lines
                </Typography>
                <Typography variant="body2" sx={{ color: '#10b981' }}>
                  [OK] CSV export functionality
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
