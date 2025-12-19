/**
 * Machine Selector Demo Component
 * Phase 3.7.3 Day 17.1
 * 
 * Demonstrates the MachineSelector component with mock data
 */

import React, { useState } from 'react';
import {
  Container,
  Paper,
  Typography,
  Box,
  Divider,
  Stack,
  Chip,
  Alert,
} from '@mui/material';
import MachineSelector from '../MachineSelector';
import type { Machine } from '../MachineSelector';

// ============================================================
// Mock Machine Data (29 machines from GAN system)
// ============================================================

const MOCK_MACHINES: Machine[] = [
  // Motors
  {
    machine_id: 'motor_siemens_1la7_001',
    display_name: 'Motor Siemens 1LA7 001',
    category: 'Motor',
    manufacturer: 'SIEMENS',
    model: '1LA7',
    sensor_count: 22,
    has_classification_model: true,
    has_regression_model: true,
    has_anomaly_model: true,
    has_timeseries_model: true,
  },
  {
    machine_id: 'motor_abb_m3bp_002',
    display_name: 'Motor ABB M3BP 002',
    category: 'Motor',
    manufacturer: 'ABB',
    model: 'M3BP',
    sensor_count: 20,
    has_classification_model: true,
    has_regression_model: true,
    has_anomaly_model: false,
    has_timeseries_model: true,
  },
  {
    machine_id: 'motor_weg_w22_003',
    display_name: 'Motor WEG W22 003',
    category: 'Motor',
    manufacturer: 'WEG',
    model: 'W22',
    sensor_count: 18,
    has_classification_model: true,
    has_regression_model: false,
    has_anomaly_model: true,
    has_timeseries_model: false,
  },

  // Pumps
  {
    machine_id: 'pump_grundfos_cr3_004',
    display_name: 'Pump Grundfos CR3 004',
    category: 'Pump',
    manufacturer: 'GRUNDFOS',
    model: 'CR3',
    sensor_count: 10,
    has_classification_model: true,
    has_regression_model: true,
    has_anomaly_model: true,
    has_timeseries_model: false,
  },
  {
    machine_id: 'pump_ksb_etanorm_005',
    display_name: 'Pump KSB Etanorm 005',
    category: 'Pump',
    manufacturer: 'KSB',
    model: 'ETANORM',
    sensor_count: 12,
    has_classification_model: true,
    has_regression_model: false,
    has_anomaly_model: true,
    has_timeseries_model: true,
  },

  // Compressors
  {
    machine_id: 'compressor_atlas_copco_ga30_006',
    display_name: 'Compressor Atlas Copco GA30 006',
    category: 'Compressor',
    manufacturer: 'ATLAS COPCO',
    model: 'GA30',
    sensor_count: 10,
    has_classification_model: true,
    has_regression_model: true,
    has_anomaly_model: false,
    has_timeseries_model: true,
  },
  {
    machine_id: 'compressor_ingersoll_rand_r75_007',
    display_name: 'Compressor Ingersoll Rand R75 007',
    category: 'Compressor',
    manufacturer: 'INGERSOLL RAND',
    model: 'R75',
    sensor_count: 15,
    has_classification_model: true,
    has_regression_model: true,
    has_anomaly_model: true,
    has_timeseries_model: true,
  },

  // CNC Machines
  {
    machine_id: 'cnc_haas_vf2_008',
    display_name: 'CNC Haas VF2 008',
    category: 'CNC Machine',
    manufacturer: 'HAAS',
    model: 'VF-2',
    sensor_count: 25,
    has_classification_model: true,
    has_regression_model: false,
    has_anomaly_model: true,
    has_timeseries_model: false,
  },
  {
    machine_id: 'cnc_brother_speedio_009',
    display_name: 'CNC Brother Speedio 009',
    category: 'CNC Machine',
    manufacturer: 'BROTHER',
    model: 'SPEEDIO',
    sensor_count: 30,
    has_classification_model: true,
    has_regression_model: true,
    has_anomaly_model: true,
    has_timeseries_model: true,
  },

  // Robots
  {
    machine_id: 'robot_fanuc_r2000ic_010',
    display_name: 'Robot Fanuc R2000iC 010',
    category: 'Robotic Arm',
    manufacturer: 'FANUC',
    model: 'R-2000iC',
    sensor_count: 18,
    has_classification_model: true,
    has_regression_model: true,
    has_anomaly_model: false,
    has_timeseries_model: true,
  },
  {
    machine_id: 'robot_abb_irb6700_011',
    display_name: 'Robot ABB IRB6700 011',
    category: 'Robotic Arm',
    manufacturer: 'ABB',
    model: 'IRB 6700',
    sensor_count: 20,
    has_classification_model: false,
    has_regression_model: false,
    has_anomaly_model: false,
    has_timeseries_model: false,
  },

  // Fans
  {
    machine_id: 'fan_ebm_papst_w2e200_012',
    display_name: 'Fan EBM-Papst W2E200 012',
    category: 'Fan',
    manufacturer: 'EBM-PAPST',
    model: 'W2E200',
    sensor_count: 8,
    has_classification_model: true,
    has_regression_model: false,
    has_anomaly_model: true,
    has_timeseries_model: false,
  },

  // Transformers
  {
    machine_id: 'transformer_abb_tmax_013',
    display_name: 'Transformer ABB Tmax 013',
    category: 'Transformer',
    manufacturer: 'ABB',
    model: 'TMAX',
    sensor_count: 14,
    has_classification_model: true,
    has_regression_model: true,
    has_anomaly_model: true,
    has_timeseries_model: false,
  },

  // Hydraulic Systems
  {
    machine_id: 'hydraulic_press_schuler_014',
    display_name: 'Hydraulic Press Schuler 014',
    category: 'Hydraulic Press',
    manufacturer: 'SCHULER',
    model: 'HP-500',
    sensor_count: 16,
    has_classification_model: true,
    has_regression_model: false,
    has_anomaly_model: true,
    has_timeseries_model: true,
  },

  // Conveyors
  {
    machine_id: 'conveyor_siemens_mc100_015',
    display_name: 'Conveyor Siemens MC100 015',
    category: 'Conveyor',
    manufacturer: 'SIEMENS',
    model: 'MC100',
    sensor_count: 12,
    has_classification_model: true,
    has_regression_model: true,
    has_anomaly_model: false,
    has_timeseries_model: false,
  },

  // Cooling Towers
  {
    machine_id: 'cooling_tower_bac_vxt_016',
    display_name: 'Cooling Tower BAC VXT 016',
    category: 'Cooling Tower',
    manufacturer: 'BAC',
    model: 'VXT',
    sensor_count: 10,
    has_classification_model: true,
    has_regression_model: true,
    has_anomaly_model: true,
    has_timeseries_model: true,
  },

  // Additional machines to reach 29
  {
    machine_id: 'motor_nidec_017',
    display_name: 'Motor Nidec Ultra IE5 017',
    category: 'Motor',
    manufacturer: 'NIDEC',
    model: 'ULTRA IE5',
    sensor_count: 22,
    has_classification_model: true,
    has_regression_model: true,
    has_anomaly_model: false,
    has_timeseries_model: true,
  },
  {
    machine_id: 'pump_flowserve_018',
    display_name: 'Pump Flowserve 3196 018',
    category: 'Pump',
    manufacturer: 'FLOWSERVE',
    model: '3196',
    sensor_count: 11,
    has_classification_model: false,
    has_regression_model: false,
    has_anomaly_model: false,
    has_timeseries_model: false,
  },
  {
    machine_id: 'compressor_sullair_019',
    display_name: 'Compressor Sullair LS20 019',
    category: 'Compressor',
    manufacturer: 'SULLAIR',
    model: 'LS-20',
    sensor_count: 13,
    has_classification_model: true,
    has_regression_model: false,
    has_anomaly_model: true,
    has_timeseries_model: false,
  },
  {
    machine_id: 'cnc_mazak_020',
    display_name: 'CNC Mazak Integrex 020',
    category: 'CNC Machine',
    manufacturer: 'MAZAK',
    model: 'INTEGREX',
    sensor_count: 28,
    has_classification_model: true,
    has_regression_model: true,
    has_anomaly_model: true,
    has_timeseries_model: false,
  },
  {
    machine_id: 'robot_kuka_021',
    display_name: 'Robot KUKA KR210 021',
    category: 'Robotic Arm',
    manufacturer: 'KUKA',
    model: 'KR 210',
    sensor_count: 19,
    has_classification_model: true,
    has_regression_model: false,
    has_anomaly_model: false,
    has_timeseries_model: true,
  },
  {
    machine_id: 'fan_ziehl_abegg_022',
    display_name: 'Fan Ziehl-Abegg EC 022',
    category: 'Fan',
    manufacturer: 'ZIEHL-ABEGG',
    model: 'EC',
    sensor_count: 9,
    has_classification_model: false,
    has_regression_model: false,
    has_anomaly_model: true,
    has_timeseries_model: false,
  },
  {
    machine_id: 'transformer_schneider_023',
    display_name: 'Transformer Schneider TriHal 023',
    category: 'Transformer',
    manufacturer: 'SCHNEIDER',
    model: 'TRIHAL',
    sensor_count: 15,
    has_classification_model: true,
    has_regression_model: true,
    has_anomaly_model: false,
    has_timeseries_model: false,
  },
  {
    machine_id: 'hydraulic_bosch_024',
    display_name: 'Hydraulic Bosch Rexroth 024',
    category: 'Hydraulic System',
    manufacturer: 'BOSCH REXROTH',
    model: 'A4VSO',
    sensor_count: 17,
    has_classification_model: true,
    has_regression_model: false,
    has_anomaly_model: true,
    has_timeseries_model: true,
  },
  {
    machine_id: 'conveyor_interroll_025',
    display_name: 'Conveyor Interroll RollerDrive 025',
    category: 'Conveyor',
    manufacturer: 'INTERROLL',
    model: 'ROLLERDRIVE',
    sensor_count: 11,
    has_classification_model: false,
    has_regression_model: false,
    has_anomaly_model: false,
    has_timeseries_model: false,
  },
  {
    machine_id: 'cooling_tower_evapco_026',
    display_name: 'Cooling Tower EVAPCO AT 026',
    category: 'Cooling Tower',
    manufacturer: 'EVAPCO',
    model: 'AT',
    sensor_count: 12,
    has_classification_model: true,
    has_regression_model: true,
    has_anomaly_model: true,
    has_timeseries_model: false,
  },
  {
    machine_id: 'motor_teco_027',
    display_name: 'Motor TECO AEEF 027',
    category: 'Motor',
    manufacturer: 'TECO',
    model: 'AEEF',
    sensor_count: 21,
    has_classification_model: true,
    has_regression_model: false,
    has_anomaly_model: false,
    has_timeseries_model: true,
  },
  {
    machine_id: 'pump_sulzer_028',
    display_name: 'Pump Sulzer XFP 028',
    category: 'Pump',
    manufacturer: 'SULZER',
    model: 'XFP',
    sensor_count: 13,
    has_classification_model: true,
    has_regression_model: true,
    has_anomaly_model: true,
    has_timeseries_model: true,
  },
  {
    machine_id: 'compressor_kaeser_029',
    display_name: 'Compressor Kaeser ASD 029',
    category: 'Compressor',
    manufacturer: 'KAESER',
    model: 'ASD',
    sensor_count: 14,
    has_classification_model: true,
    has_regression_model: true,
    has_anomaly_model: false,
    has_timeseries_model: true,
  },
];

// ============================================================
// Machine Selector Demo Component
// ============================================================

const MachineSelectorDemo: React.FC = () => {
  const [selectedMachineId, setSelectedMachineId] = useState<string | null>(null);
  const [loading] = useState(false);

  const selectedMachine = MOCK_MACHINES.find(m => m.machine_id === selectedMachineId);

  // Count statistics
  const totalMachines = MOCK_MACHINES.length;
  const trainedMachines = MOCK_MACHINES.filter(m =>
    m.has_classification_model ||
    m.has_regression_model ||
    m.has_anomaly_model ||
    m.has_timeseries_model
  ).length;
  const categories = [...new Set(MOCK_MACHINES.map(m => m.category))].length;

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
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
            Machine Selector Component
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            Phase 3.7.3 Day 17.1 - Single Machine Selection (No Fleet Overview)
          </Typography>

          {/* Statistics */}
          <Stack direction="row" spacing={2}>
            <Chip
              label={`${totalMachines} Total Machines`}
              color="primary"
              variant="outlined"
            />
            <Chip
              label={`${trainedMachines} Trained Models`}
              sx={{
                backgroundColor: 'rgba(16, 185, 129, 0.15)',
                color: '#10b981',
                borderColor: 'rgba(16, 185, 129, 0.3)',
              }}
              variant="outlined"
            />
            <Chip
              label={`${categories} Categories`}
              color="secondary"
              variant="outlined"
            />
          </Stack>
        </Box>

        <Divider sx={{ mb: 3 }} />

        {/* Architectural Rule Alert */}
        <Alert severity="info" sx={{ mb: 3 }}>
          <Typography variant="body2" sx={{ fontWeight: 500 }}>
            ⚠️ Architectural Rule: Only one machine operates at a time
          </Typography>
          <Typography variant="caption" color="text.secondary">
            This system monitors a single machine per session. Fleet-wide operations are not supported.
          </Typography>
        </Alert>

        {/* Machine Selector */}
        <Box sx={{ mb: 4 }}>
          <MachineSelector
            machines={MOCK_MACHINES}
            selectedMachineId={selectedMachineId}
            onSelect={setSelectedMachineId}
            loading={loading}
          />
        </Box>

        {/* Selected Machine Details */}
        {selectedMachine && (
          <>
            <Divider sx={{ mb: 3 }} />
            <Box>
              <Typography variant="h6" gutterBottom sx={{ mb: 2 }}>
                Selected Machine Details
              </Typography>

              <Stack spacing={2}>
                <Box>
                  <Typography variant="caption" color="text.secondary">
                    Machine ID
                  </Typography>
                  <Typography variant="body1" sx={{ fontFamily: 'monospace' }}>
                    {selectedMachine.machine_id}
                  </Typography>
                </Box>

                <Box>
                  <Typography variant="caption" color="text.secondary">
                    Category
                  </Typography>
                  <Typography variant="body1">
                    {selectedMachine.category}
                  </Typography>
                </Box>

                <Box>
                  <Typography variant="caption" color="text.secondary">
                    Manufacturer & Model
                  </Typography>
                  <Typography variant="body1">
                    {selectedMachine.manufacturer} - {selectedMachine.model}
                  </Typography>
                </Box>

                <Box>
                  <Typography variant="caption" color="text.secondary" sx={{ mb: 1, display: 'block' }}>
                    Available Models
                  </Typography>
                  <Stack direction="row" spacing={1}>
                    {selectedMachine.has_classification_model && (
                      <Chip
                        label="Classification"
                        size="small"
                        sx={{ backgroundColor: 'rgba(102, 126, 234, 0.2)', color: '#667eea' }}
                      />
                    )}
                    {selectedMachine.has_regression_model && (
                      <Chip
                        label="Regression (RUL)"
                        size="small"
                        sx={{ backgroundColor: 'rgba(16, 185, 129, 0.2)', color: '#10b981' }}
                      />
                    )}
                    {selectedMachine.has_anomaly_model && (
                      <Chip
                        label="Anomaly Detection"
                        size="small"
                        sx={{ backgroundColor: 'rgba(251, 191, 36, 0.2)', color: '#fbbf24' }}
                      />
                    )}
                    {selectedMachine.has_timeseries_model && (
                      <Chip
                        label="Timeseries"
                        size="small"
                        sx={{ backgroundColor: 'rgba(59, 130, 246, 0.2)', color: '#3b82f6' }}
                      />
                    )}
                  </Stack>
                </Box>

                <Box>
                  <Typography variant="caption" color="text.secondary">
                    Sensor Count
                  </Typography>
                  <Typography variant="body1">
                    {selectedMachine.sensor_count} sensors configured
                  </Typography>
                </Box>
              </Stack>
            </Box>
          </>
        )}

        {!selectedMachine && (
          <Alert severity="warning" sx={{ mt: 2 }}>
            No machine selected. Use the dropdown above to select a machine for monitoring.
          </Alert>
        )}
      </Paper>
    </Container>
  );
};

export default MachineSelectorDemo;
