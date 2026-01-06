/**
 * SensorCharts Component
 * 
 * Multi-line time-series chart for sensor data visualization
 * Features:
 * - Real-time data streaming (5-second intervals)
 * - Multi-sensor selection (max 5 concurrent)
 * - Auto-scroll as new data arrives
 * - Zoom and pan capabilities
 * - Threshold indicators (warning/critical)
 * - CSV export functionality
 * - Responsive design
 * 
 * Design: Professional dark theme with color-coded sensor categories
 */

import { useState, useMemo, useRef, useEffect } from 'react';
import { alpha } from '@mui/material/styles';
import {
  Box,
  Card,
  CardContent,
  CardHeader,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Chip,
  Button,
  Stack,
  Typography,
  IconButton,
  Tooltip,
  Divider,
} from '@mui/material';
import type { SelectChangeEvent } from '@mui/material';
import {
  Download as DownloadIcon,
  ZoomIn as ZoomInIcon,
  ZoomOut as ZoomOutIcon,
  RestartAlt as RestartAltIcon,
  PlayArrow as PlayArrowIcon,
  Pause as PauseIcon,
} from '@mui/icons-material';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';

// ============================================================================
// TYPESCRIPT INTERFACES
// ============================================================================

export interface SensorChartsProps {
  machineId: string;
  sensorHistory: SensorReading[];
  availableSensors: string[];
  selectedSensors?: string[];
  onSensorToggle?: (sensor: string) => void;
  baselineRanges?: Record<
    string,
    {
      min?: number | null;
      typical?: number | null;
      max?: number | null;
      alarm?: number | null;
      trip?: number | null;
    }
  >;
  autoScroll?: boolean;
  maxDataPoints?: number;
}

export interface SensorReading {
  timestamp: Date;
  values: Record<string, number>;
}

// ============================================================================
// SENSOR COLORS
// ============================================================================

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

const formatSensorName = (name: string): string => {
  return name
    .split('_')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
};

const formatTimestamp = (timestamp: Date): string => {
  return new Intl.DateTimeFormat('en-US', {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false,
  }).format(timestamp);
};

const stripMachinePrefix = (machineId: string, sensorName: string): string => {
  const mid = (machineId || '').trim().toLowerCase();
  if (!mid) return sensorName;
  const prefix = `${mid}_`;
  return sensorName.toLowerCase().startsWith(prefix) ? sensorName.slice(prefix.length) : sensorName;
};

const exportToCSV = (data: SensorReading[], sensors: string[], machineId: string) => {
  const headers = ['Timestamp', ...sensors.map(s => formatSensorName(s))];
  const rows = data.map(reading => [
    reading.timestamp.toISOString(),
    ...sensors.map(sensor => reading.values[sensor]?.toFixed(2) || ''),
  ]);
  
  const csv = [
    headers.join(','),
    ...rows.map(row => row.join(',')),
  ].join('\n');
  
  const blob = new Blob([csv], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = `${machineId}_sensor_data_${Date.now()}.csv`;
  link.click();
  URL.revokeObjectURL(url);
};

// ============================================================================
// MAIN COMPONENT
// ============================================================================

export default function SensorCharts({
  machineId,
  sensorHistory,
  availableSensors,
  selectedSensors: externalSelectedSensors,
  onSensorToggle,
  baselineRanges,
  autoScroll = true,
  maxDataPoints = 120,
}: SensorChartsProps) {
  const [internalSelectedSensors, setInternalSelectedSensors] = useState<string[]>([]);
  const [isPaused, setIsPaused] = useState(false);
  const [zoomLevel, setZoomLevel] = useState(1);
  const chartContainerRef = useRef<HTMLDivElement>(null);

  // Distinct colors for up to 5 sensor lines (bright, high-contrast palette)
  const seriesPalette = useMemo(
    () => [
      '#3b82f6', // Blue
      '#10b981', // Emerald
      '#f59e0b', // Amber
      '#ef4444', // Red
      '#8b5cf6', // Violet
    ],
    []
  );

  // Position-based color: first selected sensor -> color[0], second -> color[1], etc.
  const getSensorColorByIndex = (index: number): string => {
    return seriesPalette[index % seriesPalette.length] || seriesPalette[0];
  };

  const isExternallyControlled =
    Array.isArray(externalSelectedSensors) && typeof onSensorToggle === 'function';

  // Use external selection only when a change handler exists; otherwise fall back to internal.
  const selectedSensors = isExternallyControlled ? externalSelectedSensors : internalSelectedSensors;

  // Keep a sensible internal default selection as sensors become available/change.
  useEffect(() => {
    if (!availableSensors || availableSensors.length === 0) {
      setInternalSelectedSensors((prev) => (prev.length === 0 ? prev : []));
      return;
    }

    setInternalSelectedSensors((prev) => {
      const filtered = prev.filter((s) => availableSensors.includes(s));
      const next = (filtered.length > 0 ? filtered : availableSensors.slice(0, 3)).slice(0, 5);
      if (next.length === prev.length && next.every((v, i) => v === prev[i])) return prev;
      return next;
    });
  }, [availableSensors]);

  // Handle sensor selection change
  const handleSensorChange = (event: SelectChangeEvent<string[]>) => {
    const value = event.target.value as string[];
    
    // Limit to 5 sensors max
    if (value.length > 5) return;

    if (isExternallyControlled) {
      // External control: compute delta and toggle only changed sensors.
      const current = externalSelectedSensors;
      const removed = current.filter((s) => !value.includes(s));
      const added = value.filter((s) => !current.includes(s));
      [...removed, ...added].forEach((sensor) => onSensorToggle(sensor));
      return;
    }

    // Internal control
    setInternalSelectedSensors(value);
  };

  // Prepare chart data (limit to maxDataPoints)
  const chartData = useMemo(() => {
    const limited = sensorHistory.slice(-maxDataPoints);
    return limited.map(reading => ({
      time: formatTimestamp(reading.timestamp),
      timestamp: reading.timestamp.getTime(),
      ...reading.values,
    }));
  }, [sensorHistory, maxDataPoints]);

  // Get min/max values for Y-axis domain
  const getYAxisDomain = useMemo(() => {
    const toNum = (v: unknown): number | undefined => {
      if (typeof v === 'number' && !isNaN(v)) return v;
      return undefined;
    };

    if (selectedSensors.length === 0) {
      return [0, 100];
    }

    // Seed with baseline-derived domain when there is no chart data yet.
    if (chartData.length === 0) {
      const first = selectedSensors[0];
      const baseKey = stripMachinePrefix(machineId, first);
      const range =
        baselineRanges?.[baseKey] ||
        baselineRanges?.[baseKey.toLowerCase()] ||
        baselineRanges?.[baseKey.replace(/\s+/g, '_')];
      const minCandidate = toNum(range?.min) ?? 0;
      const maxCandidate = toNum(range?.trip) ?? toNum(range?.alarm) ?? toNum(range?.max) ?? 100;
      const padding = Math.max((maxCandidate - minCandidate) * 0.1, 1);
      return [Math.floor(minCandidate - padding), Math.ceil(maxCandidate + padding)];
    }

    let min = Infinity;
    let max = -Infinity;

    chartData.forEach((point) => {
      selectedSensors.forEach((sensor) => {
        const value = (point as Record<string, unknown>)[sensor] as number;
        if (typeof value === 'number' && !isNaN(value)) {
          min = Math.min(min, value);
          max = Math.max(max, value);
        }
      });
    });

    // Include baseline ranges to keep thresholds in view and avoid weird defaults.
    selectedSensors.forEach((sensor) => {
      const baseKey = stripMachinePrefix(machineId, sensor);
      const range =
        baselineRanges?.[baseKey] ||
        baselineRanges?.[baseKey.toLowerCase()] ||
        baselineRanges?.[baseKey.replace(/\s+/g, '_')];
      const candidates = [range?.min, range?.typical, range?.max, range?.alarm, range?.trip]
        .map(toNum)
        .filter((v): v is number => v !== undefined);
      candidates.forEach((v) => {
        min = Math.min(min, v);
        max = Math.max(max, v);
      });
    });

    if (!isFinite(min) || !isFinite(max)) {
      return [0, 100];
    }

    const padding = Math.max((max - min) * 0.1, 1);
    return [Math.floor(min - padding), Math.ceil(max + padding)];
  }, [chartData, selectedSensors, baselineRanges, machineId]);

  // Handle zoom
  const handleZoomIn = () => {
    setZoomLevel(prev => Math.min(prev * 1.5, 5));
  };

  const handleZoomOut = () => {
    setZoomLevel(prev => Math.max(prev / 1.5, 1));
  };

  const handleResetZoom = () => {
    setZoomLevel(1);
  };

  // Handle CSV export
  const handleExport = () => {
    exportToCSV(sensorHistory, selectedSensors, machineId);
  };

  // Auto-scroll to end when new data arrives
  useEffect(() => {
    if (autoScroll && !isPaused && chartContainerRef.current) {
      chartContainerRef.current.scrollLeft = chartContainerRef.current.scrollWidth;
    }
  }, [chartData, autoScroll, isPaused]);

  // ============================================================================
  // RENDER
  // ============================================================================

  return (
    <Card
      sx={(theme) => ({
        bgcolor: alpha(theme.palette.background.paper, theme.palette.mode === 'dark' ? 0.75 : 0.95),
        backdropFilter: theme.palette.mode === 'dark' ? 'blur(10px)' : undefined,
        border: 1,
        borderColor: 'divider',
        borderRadius: 2,
      })}
    >
      {/* HEADER */}
      <CardHeader
        title={
          <Typography variant="h6" sx={{ fontWeight: 600, color: 'text.primary' }}>
            SENSOR TREND ANALYSIS
          </Typography>
        }
        subheader={
          <Typography variant="body2" sx={{ color: 'text.secondary', mt: 0.5 }}>
            Last {Math.floor(maxDataPoints * 5 / 60)} minutes • {chartData.length} data points
          </Typography>
        }
      />
      <Divider sx={{ borderColor: 'divider' }} />

      <CardContent>
        {/* SENSOR SELECTION */}
        <FormControl
          fullWidth
          sx={(theme) => ({
            mb: 3,
            '& .MuiOutlinedInput-root': {
              bgcolor: alpha(theme.palette.background.paper, theme.palette.mode === 'dark' ? 0.5 : 0.9),
            },
          })}
        >
          <InputLabel>Select Sensors (Max 5)</InputLabel>
          <Select
            multiple
            value={selectedSensors}
            onChange={handleSensorChange}
            label="Select Sensors (Max 5)"
            renderValue={(selected) => (
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                {selected.map((value, idx) => (
                  <Chip
                    key={value}
                    label={formatSensorName(value)}
                    size="small"
                    sx={{
                      bgcolor: alpha(getSensorColorByIndex(idx), 0.16),
                      color: getSensorColorByIndex(idx),
                      borderColor: getSensorColorByIndex(idx),
                      border: '1px solid',
                    }}
                  />
                ))}
              </Box>
            )}
          >
            {availableSensors.map((sensor) => {
              const sensorIdx = selectedSensors.indexOf(sensor);
              const dotColor = sensorIdx >= 0 ? getSensorColorByIndex(sensorIdx) : '#6b7280';
              return (
                <MenuItem
                  key={sensor}
                  value={sensor}
                  disabled={
                    !selectedSensors.includes(sensor) && selectedSensors.length >= 5
                  }
                >
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <Box
                      sx={{
                        width: 12,
                        height: 12,
                        borderRadius: '50%',
                        bgcolor: dotColor,
                      }}
                    />
                    {formatSensorName(sensor)}
                  </Box>
                </MenuItem>
              );
            })}
          </Select>
        </FormControl>

        {/* CHART CONTROLS */}
        <Stack
          direction="row"
          spacing={2}
          sx={{ mb: 2 }}
          flexWrap="wrap"
        >
          <Tooltip title="Zoom In">
            <IconButton
              size="small"
              onClick={handleZoomIn}
              disabled={zoomLevel >= 5}
              sx={{
                color: '#667eea',
                '&:hover': { bgcolor: 'rgba(102, 126, 234, 0.1)' },
              }}
            >
              <ZoomInIcon />
            </IconButton>
          </Tooltip>

          <Tooltip title="Zoom Out">
            <IconButton
              size="small"
              onClick={handleZoomOut}
              disabled={zoomLevel <= 1}
              sx={{
                color: '#667eea',
                '&:hover': { bgcolor: 'rgba(102, 126, 234, 0.1)' },
              }}
            >
              <ZoomOutIcon />
            </IconButton>
          </Tooltip>

          <Tooltip title="Reset Zoom">
            <IconButton
              size="small"
              onClick={handleResetZoom}
              disabled={zoomLevel === 1}
              sx={{
                color: '#667eea',
                '&:hover': { bgcolor: 'rgba(102, 126, 234, 0.1)' },
              }}
            >
              <RestartAltIcon />
            </IconButton>
          </Tooltip>

          <Tooltip title={isPaused ? 'Resume Auto-scroll' : 'Pause Auto-scroll'}>
            <IconButton
              size="small"
              onClick={() => setIsPaused(!isPaused)}
              sx={{
                color: isPaused ? '#f97316' : '#10b981',
                '&:hover': {
                  bgcolor: isPaused ? 'rgba(249, 115, 22, 0.1)' : 'rgba(16, 185, 129, 0.1)',
                },
              }}
            >
              {isPaused ? <PlayArrowIcon /> : <PauseIcon />}
            </IconButton>
          </Tooltip>

          <Button
            variant="outlined"
            size="small"
            startIcon={<DownloadIcon />}
            onClick={handleExport}
            disabled={chartData.length === 0 || selectedSensors.length === 0}
            sx={{
              borderColor: '#667eea',
              color: '#667eea',
              '&:hover': {
                borderColor: '#5568d3',
                bgcolor: 'rgba(102, 126, 234, 0.1)',
              },
            }}
          >
            Export CSV
          </Button>

          <Chip
            label={`Zoom: ${(zoomLevel * 100).toFixed(0)}%`}
            size="small"
            sx={{
              bgcolor: 'rgba(102, 126, 234, 0.2)',
              color: '#667eea',
            }}
          />
        </Stack>

        {/* CHART */}
        <Box
          ref={chartContainerRef}
          sx={{
            width: '100%',
            height: 400,
            overflowX: autoScroll ? 'hidden' : 'auto',
            overflowY: 'hidden',
          }}
        >
          <ResponsiveContainer width={`${100 * zoomLevel}%`} height="100%">
            <LineChart
              data={chartData}
              margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255, 255, 255, 0.1)" />
              
              <XAxis
                dataKey="time"
                stroke="#9ca3af"
                style={{ fontSize: 12 }}
              />
              
              <YAxis
                stroke="#9ca3af"
                style={{ fontSize: 12 }}
                domain={getYAxisDomain}
              />
              
              <RechartsTooltip
                contentStyle={{
                  backgroundColor: 'rgba(17, 24, 39, 0.95)',
                  border: '1px solid rgba(255, 255, 255, 0.2)',
                  borderRadius: 8,
                  color: '#f9fafb',
                }}
                labelStyle={{ color: '#d1d5db', marginBottom: 8 }}
              />
              
              <Legend
                wrapperStyle={{ color: '#d1d5db' }}
                formatter={(value) => formatSensorName(value)}
              />

              {/* Lines for each selected sensor */}
              {selectedSensors.map((sensor, idx) => (
                <Line
                  key={sensor}
                  type="monotone"
                  dataKey={sensor}
                  name={formatSensorName(sensor)}
                  stroke={getSensorColorByIndex(idx)}
                  strokeWidth={2}
                  dot={false}
                  activeDot={{ r: 6 }}
                  animationDuration={300}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </Box>

        {/* EMPTY STATE */}
        {chartData.length === 0 && (
          <Box
            sx={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              py: 8,
              gap: 2,
            }}
          >
            <Typography variant="h6" sx={{ color: '#9ca3af' }}>
              No sensor data available
            </Typography>
            <Typography variant="body2" sx={{ color: '#6b7280' }}>
              Waiting for sensor readings...
            </Typography>
          </Box>
        )}

        {/* NO SENSORS SELECTED */}
        {chartData.length > 0 && selectedSensors.length === 0 && (
          <Box
            sx={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              py: 8,
              gap: 2,
            }}
          >
            <Typography variant="h6" sx={{ color: '#9ca3af' }}>
              No sensors selected
            </Typography>
            <Typography variant="body2" sx={{ color: '#6b7280' }}>
              Select up to 5 sensors from the dropdown above
            </Typography>
          </Box>
        )}
      </CardContent>
    </Card>
  );
}
