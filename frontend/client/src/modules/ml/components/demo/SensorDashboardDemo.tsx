/**
 * Sensor Dashboard Demo Component
 * Phase 3.7.3 Day 17.2
 * 
 * Demonstrates the SensorDashboard component with simulated real-time data
 */

import React, { useState, useEffect } from 'react';
import {
  Container,
  Paper,
  Typography,
  Box,
  Divider,
  Stack,
  Button,
  Chip,
  Switch,
  FormControlLabel,
} from '@mui/material';
import SensorDashboard from '../SensorDashboard';

// ============================================================
// Mock Sensor Data Generator
// ============================================================

const generateMockSensorData = (machineType: 'motor' | 'pump' | 'cnc'): Record<string, number> => {
  const baseData: Record<string, Record<string, number>> = {
    motor: {
      'motor_bearing_de_temp_C': 55 + Math.random() * 20,
      'motor_bearing_nde_temp_C': 52 + Math.random() * 18,
      'motor_winding_temp_C': 65 + Math.random() * 15,
      'motor_stator_temp_C': 70 + Math.random() * 12,
      'motor_rotor_temp_C': 68 + Math.random() * 14,
      'motor_casing_temp_C': 45 + Math.random() * 10,
      'motor_vibration_rms_mm_s': 2.5 + Math.random() * 2,
      'motor_vibration_peak_mm_s': 3.5 + Math.random() * 3,
      'motor_current_phase_a_A': 10 + Math.random() * 5,
      'motor_current_phase_b_A': 10 + Math.random() * 5,
      'motor_current_phase_c_A': 10 + Math.random() * 5,
      'motor_voltage_phase_a_V': 400 + Math.random() * 20,
      'motor_voltage_phase_b_V': 400 + Math.random() * 20,
      'motor_voltage_phase_c_V': 400 + Math.random() * 20,
      'motor_speed_rpm': 1450 + Math.random() * 50,
      'motor_power_kw': 45 + Math.random() * 10,
      'motor_frequency_hz': 50 + Math.random() * 0.5,
      'motor_load_percent': 70 + Math.random() * 20,
      'motor_efficiency_percent': 92 + Math.random() * 3,
      'motor_ambient_temp_C': 25 + Math.random() * 5,
      'motor_bearing_noise_db': 65 + Math.random() * 10,
      'motor_insulation_resistance_mohm': 95 + Math.random() * 5,
    },
    pump: {
      'pump_inlet_temp_C': 35 + Math.random() * 10,
      'pump_outlet_temp_C': 40 + Math.random() * 12,
      'pump_bearing_temp_C': 55 + Math.random() * 15,
      'pump_motor_temp_C': 60 + Math.random() * 18,
      'pump_casing_temp_C': 42 + Math.random() * 8,
      'pump_vibration_mm_s': 3.0 + Math.random() * 2.5,
      'pump_inlet_pressure_bar': 2.0 + Math.random() * 1.0,
      'pump_outlet_pressure_bar': 6.0 + Math.random() * 2.0,
      'pump_flow_rate_m3_h': 180 + Math.random() * 40,
      'pump_current_A': 15 + Math.random() * 5,
    },
    cnc: {
      'cnc_spindle_temp_C': 45 + Math.random() * 20,
      'cnc_spindle_speed_rpm': 4500 + Math.random() * 500,
      'cnc_spindle_load_percent': 60 + Math.random() * 30,
      'cnc_coolant_temp_C': 28 + Math.random() * 7,
      'cnc_coolant_flow_l_min': 15 + Math.random() * 5,
      'cnc_x_axis_position_mm': 250 + Math.random() * 100,
      'cnc_y_axis_position_mm': 180 + Math.random() * 80,
      'cnc_z_axis_position_mm': 120 + Math.random() * 60,
      'cnc_x_axis_vibration_mm_s': 1.5 + Math.random() * 1.0,
      'cnc_y_axis_vibration_mm_s': 1.3 + Math.random() * 0.8,
      'cnc_z_axis_vibration_mm_s': 1.1 + Math.random() * 0.6,
      'cnc_hydraulic_pressure_bar': 180 + Math.random() * 30,
      'cnc_tool_wear_mm': 0.05 + Math.random() * 0.15,
      'cnc_ambient_temp_C': 22 + Math.random() * 3,
      'cnc_power_consumption_kw': 25 + Math.random() * 10,
    },
  };

  return baseData[machineType];
};

// ============================================================
// Machine Profiles
// ============================================================

const MOCK_MACHINES = [
  {
    id: 'motor_siemens_1la7_001',
    name: 'Motor Siemens 1LA7 001',
    type: 'motor' as const,
  },
  {
    id: 'pump_grundfos_cr3_004',
    name: 'Pump Grundfos CR3 004',
    type: 'pump' as const,
  },
  {
    id: 'cnc_brother_speedio_009',
    name: 'CNC Brother Speedio 009',
    type: 'cnc' as const,
  },
];

// ============================================================
// Sensor Dashboard Demo Component
// ============================================================

const SensorDashboardDemo: React.FC = () => {
  const [selectedMachine, setSelectedMachine] = useState(MOCK_MACHINES[0]);
  const [sensorData, setSensorData] = useState<Record<string, number> | null>(null);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const [loading, setLoading] = useState(false);
  const [connected, setConnected] = useState(true);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Initial data load
  useEffect(() => {
    setLoading(true);
    setTimeout(() => {
      setSensorData(generateMockSensorData(selectedMachine.type));
      setLastUpdated(new Date());
      setLoading(false);
    }, 1000);
  }, [selectedMachine]);

  // Auto-refresh simulation (5 seconds)
  useEffect(() => {
    if (!autoRefresh || !connected) return;

    const interval = setInterval(() => {
      setSensorData(generateMockSensorData(selectedMachine.type));
      setLastUpdated(new Date());
    }, 5000);

    return () => clearInterval(interval);
  }, [autoRefresh, connected, selectedMachine]);

  // Manual refresh
  const handleRefresh = () => {
    setLoading(true);
    setTimeout(() => {
      setSensorData(generateMockSensorData(selectedMachine.type));
      setLastUpdated(new Date());
      setLoading(false);
    }, 500);
  };

  // Toggle connection
  const handleToggleConnection = () => {
    setConnected(!connected);
    if (connected) {
      setError('WebSocket connection lost');
    } else {
      setError(null);
      setSensorData(generateMockSensorData(selectedMachine.type));
      setLastUpdated(new Date());
    }
  };

  // Clear machine
  const handleClearMachine = () => {
    setSensorData(null);
    setLastUpdated(null);
  };

  return (
    <Container maxWidth="xl" sx={{ py: 4 }}>
      <Paper
        elevation={2}
        sx={{
          p: 3,
          background: 'rgba(30, 41, 59, 0.8)',
          backdropFilter: 'blur(10px)',
          border: '1px solid rgba(148, 163, 184, 0.2)',
        }}
      >
        {/* Header */}
        <Box sx={{ mb: 4 }}>
          <Typography variant="h4" gutterBottom sx={{ fontWeight: 600 }}>
            Sensor Dashboard Component
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            Phase 3.7.3 Day 17.2 - Real-Time Sensor Monitoring
          </Typography>

          {/* Demo Controls */}
          <Stack direction="row" spacing={2} alignItems="center" flexWrap="wrap">
            <Typography variant="body2" sx={{ fontWeight: 500 }}>
              Select Machine:
            </Typography>
            {MOCK_MACHINES.map((machine) => (
              <Chip
                key={machine.id}
                label={machine.name}
                onClick={() => setSelectedMachine(machine)}
                color={selectedMachine.id === machine.id ? 'primary' : 'default'}
                variant={selectedMachine.id === machine.id ? 'filled' : 'outlined'}
              />
            ))}
          </Stack>

          <Stack direction="row" spacing={2} alignItems="center" sx={{ mt: 2 }} flexWrap="wrap">
            <FormControlLabel
              control={
                <Switch
                  checked={autoRefresh}
                  onChange={(e) => setAutoRefresh(e.target.checked)}
                  color="primary"
                />
              }
              label="Auto-refresh (5s)"
            />
            <FormControlLabel
              control={
                <Switch
                  checked={connected}
                  onChange={handleToggleConnection}
                  color="success"
                />
              }
              label="WebSocket Connected"
            />
            <Button variant="outlined" size="small" onClick={handleRefresh}>
              Refresh Now
            </Button>
            <Button variant="outlined" size="small" onClick={handleClearMachine} color="warning">
              Clear Machine
            </Button>
          </Stack>
        </Box>

        <Divider sx={{ mb: 3 }} />

        {/* Component Features */}
        <Box sx={{ mb: 3 }}>
          <Typography variant="h6" gutterBottom sx={{ mb: 2 }}>
            Features Demonstrated
          </Typography>
          <Stack direction="row" spacing={1} flexWrap="wrap" gap={1}>
            <Chip label="✅ Responsive Grid (4/2/1 cols)" size="small" color="success" />
            <Chip label="✅ Live Update Indicator" size="small" color="success" />
            <Chip label="✅ Color-Coded Values" size="small" color="success" />
            <Chip label="✅ Threshold Bars" size="small" color="success" />
            <Chip label="✅ Auto-Refresh (5s)" size="small" color="success" />
            <Chip label="✅ Skeleton Loading" size="small" color="success" />
            <Chip label="✅ Empty State" size="small" color="success" />
            <Chip label="✅ Error State" size="small" color="success" />
            <Chip label="✅ Real-Time Updates" size="small" color="success" />
          </Stack>
        </Box>

        <Divider sx={{ mb: 3 }} />

        {/* Sensor Dashboard Component */}
        <Box sx={{ mt: 3 }}>
          <SensorDashboard
            machineId={sensorData ? selectedMachine.id : ''}
            machineName={selectedMachine.name}
            sensorData={sensorData}
            lastUpdated={lastUpdated}
            loading={loading}
            connected={connected}
            error={error}
          />
        </Box>

        {/* Legend */}
        {sensorData && (
          <Box sx={{ mt: 4, p: 2, backgroundColor: 'rgba(51, 65, 85, 0.5)', borderRadius: 2 }}>
            <Typography variant="body2" sx={{ fontWeight: 600, mb: 1 }}>
              Color Coding Guide:
            </Typography>
            <Stack direction="row" spacing={3} flexWrap="wrap">
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Box sx={{ width: 16, height: 16, borderRadius: '50%', backgroundColor: '#10b981' }} />
                <Typography variant="caption">Green: Normal (below warning)</Typography>
              </Box>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Box sx={{ width: 16, height: 16, borderRadius: '50%', backgroundColor: '#fbbf24' }} />
                <Typography variant="caption">Yellow: Warning threshold reached</Typography>
              </Box>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Box sx={{ width: 16, height: 16, borderRadius: '50%', backgroundColor: '#ef4444' }} />
                <Typography variant="caption">Red: Critical threshold exceeded</Typography>
              </Box>
            </Stack>
          </Box>
        )}
      </Paper>
    </Container>
  );
};

export default SensorDashboardDemo;
