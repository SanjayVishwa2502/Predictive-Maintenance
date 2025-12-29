/**
 * Sensor Dashboard Component
 * Phase 3.7.3 Day 17.2
 * 
 * Real-time sensor monitoring dashboard for single machine.
 * ⚠️ ARCHITECTURAL RULE: Displays all sensors for ONE machine at a time
 * 
 * Features:
 * - Responsive grid layout (4 cols desktop, 2 tablet, 1 mobile)
 * - Live update indicator with pulsing animation
 * - Color-coded sensor values (green/yellow/red)
 * - Threshold visualization bars
 * - Skeleton loading state
 * - Empty state (no machine selected)
 * - Error state (connection lost)
 */

import React, { useMemo } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
  Stack,
  Chip,
  LinearProgress,
  Skeleton,
  Alert,
  Tooltip,
} from '@mui/material';
import ThermostatIcon from '@mui/icons-material/Thermostat';
import VibrationIcon from '@mui/icons-material/Vibration';
import BoltIcon from '@mui/icons-material/Bolt';
import SpeedIcon from '@mui/icons-material/Speed';
import CompressIcon from '@mui/icons-material/Compress';
import WaterDropIcon from '@mui/icons-material/WaterDrop';
import AirIcon from '@mui/icons-material/Air';
import SettingsIcon from '@mui/icons-material/Settings';
import CircleIcon from '@mui/icons-material/Circle';
import SignalWifiOffIcon from '@mui/icons-material/SignalWifiOff';

// ============================================================
// TypeScript Interfaces
// ============================================================

export interface SensorDashboardProps {
  machineId: string;
  machineName?: string;
  sensorData: Record<string, number> | null;
  lastUpdated: Date | null;
  loading?: boolean;
  connected?: boolean;
  error?: string | null;
}

interface SensorCardProps {
  name: string;
  value: number;
  unit: string;
  threshold?: { warning: number; critical: number };
  icon: React.ReactNode;
}

// ============================================================
// Sensor Parsing & Icon Mapping
// ============================================================

const getSensorIcon = (sensorName: string): React.ReactNode => {
  const nameLower = sensorName.toLowerCase();
  
  if (nameLower.includes('temp') || nameLower.includes('temperature')) {
    return <ThermostatIcon sx={{ fontSize: 32 }} />;
  }
  if (nameLower.includes('vibration') || nameLower.includes('velocity') || nameLower.includes('rms')) {
    return <VibrationIcon sx={{ fontSize: 32 }} />;
  }
  if (nameLower.includes('current') || nameLower.includes('amp')) {
    return <BoltIcon sx={{ fontSize: 32 }} />;
  }
  if (nameLower.includes('voltage') || nameLower.includes('volt')) {
    return <BoltIcon sx={{ fontSize: 32 }} />;
  }
  if (nameLower.includes('speed') || nameLower.includes('rpm')) {
    return <SpeedIcon sx={{ fontSize: 32 }} />;
  }
  if (nameLower.includes('pressure')) {
    return <CompressIcon sx={{ fontSize: 32 }} />;
  }
  if (nameLower.includes('flow') || nameLower.includes('liquid')) {
    return <WaterDropIcon sx={{ fontSize: 32 }} />;
  }
  if (nameLower.includes('air') || nameLower.includes('gas')) {
    return <AirIcon sx={{ fontSize: 32 }} />;
  }
  
  return <SettingsIcon sx={{ fontSize: 32 }} />;
};

const parseSensorUnit = (sensorName: string): string => {
  const nameLower = sensorName.toLowerCase();
  
  if (nameLower.includes('_c') || nameLower.includes('temp')) return '°C';
  if (nameLower.includes('_f')) return '°F';
  if (nameLower.includes('_mm') || nameLower.includes('velocity')) return 'mm/s';
  if (nameLower.includes('_a') || nameLower.includes('current')) return 'A';
  if (nameLower.includes('_v') || nameLower.includes('voltage')) return 'V';
  if (nameLower.includes('_rpm') || nameLower.includes('speed')) return 'RPM';
  if (nameLower.includes('_bar') || nameLower.includes('pressure')) return 'bar';
  if (nameLower.includes('_psi')) return 'PSI';
  if (nameLower.includes('_hz') || nameLower.includes('frequency')) return 'Hz';
  if (nameLower.includes('_percent') || nameLower.includes('load')) return '%';
  
  return '';
};

const getDefaultThreshold = (sensorName: string): { warning: number; critical: number } | undefined => {
  const nameLower = sensorName.toLowerCase();
  
  // Temperature sensors
  if (nameLower.includes('temp') || nameLower.includes('_c')) {
    if (nameLower.includes('bearing')) {
      return { warning: 70, critical: 85 };
    }
    if (nameLower.includes('winding') || nameLower.includes('stator')) {
      return { warning: 80, critical: 100 };
    }
    if (nameLower.includes('ambient') || nameLower.includes('casing')) {
      return { warning: 50, critical: 65 };
    }
    return { warning: 75, critical: 90 };
  }
  
  // Vibration sensors
  if (nameLower.includes('vibration') || nameLower.includes('velocity')) {
    return { warning: 4.5, critical: 7.1 };
  }
  
  // Current sensors
  if (nameLower.includes('current') || nameLower.includes('_a')) {
    return { warning: 15, critical: 20 };
  }
  
  // Voltage sensors
  if (nameLower.includes('voltage') || nameLower.includes('_v')) {
    return { warning: 450, critical: 480 };
  }
  
  // Pressure sensors
  if (nameLower.includes('pressure')) {
    return { warning: 8, critical: 10 };
  }
  
  return undefined;
};

const parseSensorName = (sensorName: string): string => {
  // Remove machine ID prefix if present
  let cleaned = sensorName;
  const parts = sensorName.split('_');
  
  // If sensor name has many parts, try to extract the meaningful portion
  if (parts.length > 3) {
    const keywords = ['bearing', 'temp', 'temperature', 'vibration', 'current', 'voltage', 
                     'speed', 'pressure', 'flow', 'winding', 'stator', 'rotor', 'ambient'];
    
    const keywordIndex = parts.findIndex(part => 
      keywords.some(keyword => part.toLowerCase().includes(keyword))
    );
    
    if (keywordIndex >= 0) {
      cleaned = parts.slice(keywordIndex).join('_');
    }
  }
  
  // Convert to title case and replace underscores with spaces
  return cleaned
    .split('_')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
};

// ============================================================
// Sensor Card Component
// ============================================================

const SensorCard: React.FC<SensorCardProps> = ({
  name,
  value,
  unit,
  threshold,
  icon,
}) => {
  // Determine status color based on thresholds
  const getStatusColor = (): string => {
    if (!threshold) return '#10b981'; // Green (no threshold = normal)
    
    if (value >= threshold.critical) return '#ef4444'; // Red
    if (value >= threshold.warning) return '#fbbf24'; // Yellow
    return '#10b981'; // Green
  };

  const getStatusLabel = (): string => {
    if (!threshold) return 'Normal';
    
    if (value >= threshold.critical) return 'Critical';
    if (value >= threshold.warning) return 'Warning';
    return 'Normal';
  };

  // Calculate progress bar percentage (0-100%)
  const getProgressPercentage = (): number => {
    if (!threshold) return 50; // Default to middle if no threshold
    
    const max = threshold.critical * 1.2; // 120% of critical as max
    return Math.min((value / max) * 100, 100);
  };

  const statusColor = getStatusColor();
  const statusLabel = getStatusLabel();
  const progressPercentage = getProgressPercentage();

  return (
    <Card
      sx={{
        height: '100%',
        minHeight: 160,
        background: 'rgba(31, 41, 55, 0.8)',
        backdropFilter: 'blur(10px)',
        border: `1px solid ${statusColor}40`,
        borderRadius: 2,
        transition: 'all 0.3s ease',
        '&:hover': {
          transform: 'translateY(-4px)',
          boxShadow: `0 8px 16px ${statusColor}30`,
          borderColor: `${statusColor}80`,
        },
      }}
    >
      <CardContent sx={{ height: '100%', display: 'flex', flexDirection: 'column', p: 2 }}>
        {/* Icon & Value */}
        <Stack direction="row" alignItems="center" spacing={1} sx={{ mb: 1 }}>
          <Box sx={{ color: statusColor, display: 'flex', alignItems: 'center' }}>
            {icon}
          </Box>
          <Box sx={{ flex: 1 }}>
            <Typography
              variant="h4"
              sx={{
                fontWeight: 700,
                color: statusColor,
                fontSize: { xs: '1.5rem', sm: '2rem' },
                lineHeight: 1,
              }}
            >
              {value.toFixed(1)}
              <Typography
                component="span"
                variant="body2"
                sx={{ ml: 0.5, color: 'text.secondary', fontSize: '0.875rem' }}
              >
                {unit}
              </Typography>
            </Typography>
          </Box>
        </Stack>

        {/* Sensor Name */}
        <Typography
          variant="body2"
          sx={{
            color: 'text.primary',
            fontWeight: 500,
            mb: 0.5,
            lineHeight: 1.3,
          }}
        >
          {name}
        </Typography>

        {/* Status Label */}
        <Typography
          variant="caption"
          sx={{
            color: 'text.secondary',
            fontSize: '0.7rem',
            mb: 'auto',
          }}
        >
          {statusLabel}
        </Typography>

        {/* Threshold Bar */}
        <Box sx={{ mt: 2 }}>
          <LinearProgress
            variant="determinate"
            value={progressPercentage}
            sx={{
              height: 6,
              borderRadius: 3,
              backgroundColor: 'rgba(148, 163, 184, 0.2)',
              '& .MuiLinearProgress-bar': {
                backgroundColor: statusColor,
                borderRadius: 3,
                transition: 'transform 0.4s ease',
              },
            }}
          />
          {threshold && (
            <Stack direction="row" justifyContent="space-between" sx={{ mt: 0.5 }}>
              <Typography variant="caption" sx={{ fontSize: '0.65rem', color: 'text.disabled' }}>
                0
              </Typography>
              <Typography variant="caption" sx={{ fontSize: '0.65rem', color: '#fbbf24' }}>
                ⚠ {threshold.warning}
              </Typography>
              <Typography variant="caption" sx={{ fontSize: '0.65rem', color: '#ef4444' }}>
                🔴 {threshold.critical}
              </Typography>
            </Stack>
          )}
        </Box>
      </CardContent>
    </Card>
  );
};

// ============================================================
// Sensor Dashboard Component
// ============================================================

const SensorDashboard: React.FC<SensorDashboardProps> = ({
  machineId,
  machineName,
  sensorData,
  lastUpdated,
  loading = false,
  connected = false,
  error = null,
}) => {
  // Calculate time since last update
  const timeSinceUpdate = useMemo(() => {
    if (!lastUpdated) return 'Never';
    
    const seconds = Math.floor((Date.now() - lastUpdated.getTime()) / 1000);
    if (seconds < 60) return `${seconds}s ago`;
    
    const minutes = Math.floor(seconds / 60);
    if (minutes < 60) return `${minutes}m ago`;
    
    const hours = Math.floor(minutes / 60);
    return `${hours}h ago`;
  }, [lastUpdated]);

  // Parse sensor data into cards
  const sensorCards = useMemo(() => {
    if (!sensorData) return [];
    
    return Object.entries(sensorData).map(([sensorName, value]) => {
      const displayName = parseSensorName(sensorName);
      const unit = parseSensorUnit(sensorName);
      const icon = getSensorIcon(sensorName);
      const threshold = getDefaultThreshold(sensorName);
      
      return {
        key: sensorName,
        name: displayName,
        value,
        unit,
        icon,
        threshold,
      };
    });
  }, [sensorData]);

  // Empty State: No machine selected
  if (!machineId) {
    return (
      <Box sx={{ p: 3 }}>
        <Alert severity="info" sx={{ maxWidth: 600, mx: 'auto' }}>
          <Typography variant="body2">
            No machine selected. Please select a machine from the dropdown above to view real-time sensor data.
          </Typography>
        </Alert>
      </Box>
    );
  }

  // Error State: Connection lost
  if (error) {
    return (
      <Box sx={{ p: 3 }}>
        <Alert
          severity="error"
          icon={<SignalWifiOffIcon />}
          sx={{ maxWidth: 600, mx: 'auto' }}
        >
          <Typography variant="body2" sx={{ fontWeight: 600, mb: 0.5 }}>
            Connection Error
          </Typography>
          <Typography variant="caption" sx={{ color: 'text.secondary' }}>
            {error}
          </Typography>
        </Alert>
      </Box>
    );
  }

  return (
    <Box sx={{ width: '100%' }}>
      {/* Header */}
      <Stack
        direction={{ xs: 'column', sm: 'row' }}
        justifyContent="space-between"
        alignItems={{ xs: 'flex-start', sm: 'center' }}
        spacing={2}
        sx={{ mb: 3 }}
      >
        <Box>
          <Typography variant="h5" sx={{ fontWeight: 600, mb: 0.5 }}>
            Real-Time Sensor Monitoring
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Machine: {machineName || machineId}
          </Typography>
        </Box>

        <Stack direction="row" spacing={2} alignItems="center">
          {/* Live Indicator */}
          <Tooltip title={connected ? 'Backend connected' : 'Disconnected'}>
            <Chip
              icon={
                <CircleIcon
                  sx={{
                    fontSize: 12,
                    animation: connected ? 'pulse 2s infinite' : 'none',
                    '@keyframes pulse': {
                      '0%, 100%': { opacity: 1 },
                      '50%': { opacity: 0.5 },
                    },
                  }}
                />
              }
              label={connected ? 'Live' : 'Offline'}
              size="small"
              sx={{
                backgroundColor: connected ? 'rgba(16, 185, 129, 0.15)' : 'rgba(100, 116, 139, 0.15)',
                color: connected ? '#10b981' : '#64748b',
                borderColor: connected ? 'rgba(16, 185, 129, 0.3)' : 'rgba(100, 116, 139, 0.3)',
                border: '1px solid',
                fontWeight: 600,
                fontSize: '0.75rem',
              }}
            />
          </Tooltip>

          {/* Last Updated */}
          <Typography variant="caption" sx={{ color: 'text.disabled', fontSize: '0.75rem' }}>
            Updated: {timeSinceUpdate}
          </Typography>
        </Stack>
      </Stack>

      {/* Sensor Grid */}
      {loading ? (
        // Loading State: Skeleton Cards
        <Grid container spacing={2}>
          {[1, 2, 3, 4, 5, 6, 7, 8].map((i) => (
            <Grid item xs={12} sm={6} md={4} lg={3} key={i}>
              <Card
                sx={{
                  height: 160,
                  background: 'rgba(31, 41, 55, 0.8)',
                  borderRadius: 2,
                }}
              >
                <CardContent>
                  <Stack spacing={1}>
                    <Skeleton variant="circular" width={32} height={32} />
                    <Skeleton variant="text" width="60%" height={40} />
                    <Skeleton variant="text" width="80%" />
                    <Skeleton variant="text" width="50%" />
                    <Skeleton variant="rectangular" height={6} sx={{ borderRadius: 3 }} />
                  </Stack>
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>
      ) : sensorCards.length === 0 ? (
        // Empty State: No sensor data
        <Alert severity="warning" sx={{ maxWidth: 600, mx: 'auto' }}>
          <Typography variant="body2">
            No sensor data available for this machine. The machine may not be transmitting data yet.
          </Typography>
        </Alert>
      ) : (
        // Normal State: Display Sensor Cards
        <Grid container spacing={2}>
          {sensorCards.map((sensor) => (
            <Grid
              item
              xs={12}
              sm={6}
              md={4}
              lg={3}
              key={sensor.key}
              sx={{
                minWidth: 250,
              }}
            >
              <SensorCard
                name={sensor.name}
                value={sensor.value}
                unit={sensor.unit}
                threshold={sensor.threshold}
                icon={sensor.icon}
              />
            </Grid>
          ))}
        </Grid>
      )}

      {/* Sensor Count Summary */}
      {!loading && sensorCards.length > 0 && (
        <Typography
          variant="caption"
          sx={{
            display: 'block',
            mt: 2,
            textAlign: 'center',
            color: 'text.disabled',
            fontSize: '0.75rem',
          }}
        >
          Monitoring {sensorCards.length} sensor{sensorCards.length !== 1 ? 's' : ''}
        </Typography>
      )}
    </Box>
  );
};

export default SensorDashboard;
