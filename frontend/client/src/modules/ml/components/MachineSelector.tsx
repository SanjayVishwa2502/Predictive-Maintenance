/**
 * Machine Selector Component
 * Phase 3.7.3 Day 17.1
 * 
 * Searchable autocomplete dropdown for selecting a single machine.
 * ⚠️ ARCHITECTURAL RULE: Only one machine operates at a time (no fleet overview)
 * 
 * Features:
 * - Searchable dropdown with all 29 machines
 * - Grouped by category (Motors, Pumps, CNCs, etc.)
 * - Model availability badges (classification, regression, anomaly, timeseries)
 * - Sensor count display
 * - Recent selections history (last 5)
 * - Keyboard navigation support
 */

import React, { useState, useEffect } from 'react';
import {
  Autocomplete,
  TextField,
  Box,
  Typography,
  Chip,
  Paper,
  InputAdornment,
  IconButton,
  Tooltip,
  Stack,
} from '@mui/material';
import SearchIcon from '@mui/icons-material/Search';
import ClearIcon from '@mui/icons-material/Clear';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import SensorsIcon from '@mui/icons-material/Sensors';

// ============================================================
// TypeScript Interfaces
// ============================================================

export interface Machine {
  machine_id: string;
  display_name: string;
  category: string;
  manufacturer: string;
  model: string;
  sensor_count: number;
  has_classification_model: boolean;
  has_regression_model: boolean;
  has_anomaly_model: boolean;
  has_timeseries_model: boolean;
}

export interface MachineSelectorProps {
  machines: Machine[];
  selectedMachineId: string | null;
  onSelect: (machineId: string) => void;
  loading?: boolean;
}

// ============================================================
// Category Icons Mapping
// ============================================================

const CATEGORY_ICONS: Record<string, string> = {
  'Motor': '🔧',
  'Motors': '🔧',
  'Pump': '💧',
  'Pumps': '💧',
  'Compressor': '🌀',
  'Compressors': '🌀',
  'Robot': '🤖',
  'Robots': '🤖',
  'Robotic Arm': '🤖',
  'CNC': '🎯',
  'CNC Machine': '🎯',
  'Fan': '🌬️',
  'Fans': '🌬️',
  'Transformer': '🔌',
  'Transformers': '🔌',
  'Hydraulic System': '🏗️',
  'Hydraulic Press': '🏗️',
  'Conveyor': '📦',
  'Conveyors': '📦',
  'Cooling Tower': '❄️',
  'Cooling': '❄️',
};

const getCategoryIcon = (category: string): string => {
  return CATEGORY_ICONS[category] || '⚙️';
};

// ============================================================
// Recent Selections Storage (LocalStorage)
// ============================================================

const RECENT_SELECTIONS_KEY = 'ml_dashboard_recent_machines';
const MAX_RECENT = 5;

const getRecentSelections = (): string[] => {
  try {
    const stored = localStorage.getItem(RECENT_SELECTIONS_KEY);
    return stored ? JSON.parse(stored) : [];
  } catch {
    return [];
  }
};

const saveRecentSelection = (machineId: string): void => {
  try {
    const recent = getRecentSelections().filter(id => id !== machineId);
    recent.unshift(machineId);
    localStorage.setItem(RECENT_SELECTIONS_KEY, JSON.stringify(recent.slice(0, MAX_RECENT)));
  } catch {
    // Ignore localStorage errors
  }
};

// ============================================================
// Machine Selector Component
// ============================================================

const MachineSelector: React.FC<MachineSelectorProps> = ({
  machines,
  selectedMachineId,
  onSelect,
  loading = false,
}) => {
  const [recentIds, setRecentIds] = useState<string[]>([]);
  const [inputValue, setInputValue] = useState('');

  // Load recent selections on mount
  useEffect(() => {
    setRecentIds(getRecentSelections());
  }, []);

  // Find selected machine object
  const selectedMachine = machines.find(m => m.machine_id === selectedMachineId) || null;

  // Handle selection
  const handleSelect = (_event: React.SyntheticEvent, value: Machine | null) => {
    if (value) {
      onSelect(value.machine_id);
      saveRecentSelection(value.machine_id);
      setRecentIds(getRecentSelections());
    }
  };

  // Handle clear
  const handleClear = () => {
    onSelect('');
    setInputValue('');
  };

  // Check if machine has any trained model
  const hasTrainedModel = (machine: Machine): boolean => {
    return (
      machine.has_classification_model ||
      machine.has_regression_model ||
      machine.has_anomaly_model ||
      machine.has_timeseries_model
    );
  };

  // Count trained models
  const countTrainedModels = (machine: Machine): number => {
    let count = 0;
    if (machine.has_classification_model) count++;
    if (machine.has_regression_model) count++;
    if (machine.has_anomaly_model) count++;
    if (machine.has_timeseries_model) count++;
    return count;
  };

  return (
    <Box sx={{ width: '100%', maxWidth: 600 }}>
      <Autocomplete
        value={selectedMachine}
        onChange={handleSelect}
        inputValue={inputValue}
        onInputChange={(_event, newInputValue) => setInputValue(newInputValue)}
        options={machines}
        groupBy={(option) => option.category || 'Other'}
        getOptionLabel={(option) => option.display_name}
        isOptionEqualToValue={(option, value) => option.machine_id === value.machine_id}
        loading={loading}
        disabled={loading}
        clearOnEscape
        openOnFocus
        PaperComponent={(props) => (
          <Paper
            {...props}
            elevation={8}
            sx={{
              maxHeight: 400,
              overflowY: 'auto',
              background: 'rgba(31, 41, 55, 0.95)',
              backdropFilter: 'blur(10px)',
              border: '1px solid rgba(255, 255, 255, 0.1)',
              '& .MuiAutocomplete-listbox': {
                '& .MuiAutocomplete-groupLabel': {
                  backgroundColor: 'rgba(51, 65, 85, 0.8)',
                  color: '#f1f5f9',
                  fontWeight: 600,
                  fontSize: '0.875rem',
                  padding: '8px 16px',
                  position: 'sticky',
                  top: 0,
                  zIndex: 1,
                },
              },
            }}
          />
        )}
        renderInput={(params) => (
          <TextField
            {...params}
            label="Select Machine"
            placeholder="Search by name, category, or manufacturer..."
            variant="outlined"
            InputProps={{
              ...params.InputProps,
              startAdornment: (
                <InputAdornment position="start">
                  <SearchIcon sx={{ color: 'text.secondary' }} />
                </InputAdornment>
              ),
              endAdornment: (
                <>
                  {selectedMachineId && (
                    <InputAdornment position="end">
                      <Tooltip title="Clear selection">
                        <IconButton
                          size="small"
                          onClick={handleClear}
                          edge="end"
                          sx={{ mr: 1 }}
                        >
                          <ClearIcon />
                        </IconButton>
                      </Tooltip>
                    </InputAdornment>
                  )}
                  {params.InputProps.endAdornment}
                </>
              ),
            }}
            sx={{
              '& .MuiOutlinedInput-root': {
                background: 'rgba(31, 41, 55, 0.8)',
                backdropFilter: 'blur(10px)',
                '&:hover': {
                  background: 'rgba(31, 41, 55, 0.9)',
                },
                '&.Mui-focused': {
                  background: 'rgba(31, 41, 55, 0.95)',
                },
              },
            }}
          />
        )}
        renderOption={(props, option) => {
          const isTrained = hasTrainedModel(option);
          const modelCount = countTrainedModels(option);
          const isRecent = recentIds.includes(option.machine_id);

          return (
            <Box
              component="li"
              {...props}
              sx={{
                display: 'flex !important',
                flexDirection: 'column',
                alignItems: 'flex-start !important',
                gap: 0.5,
                padding: '12px 16px !important',
                borderBottom: '1px solid rgba(148, 163, 184, 0.1)',
                '&:hover': {
                  background: 'rgba(102, 126, 234, 0.1) !important',
                },
                '&[aria-selected="true"]': {
                  background: 'rgba(102, 126, 234, 0.2) !important',
                },
              }}
            >
              {/* Main Row: Name, Sensor Count, Status */}
              <Stack
                direction="row"
                spacing={1}
                alignItems="center"
                sx={{ width: '100%' }}
              >
                {/* Category Icon */}
                <Typography variant="h6" sx={{ fontSize: '1.25rem', minWidth: 24 }}>
                  {getCategoryIcon(option.category)}
                </Typography>

                {/* Machine Name */}
                <Typography
                  variant="body1"
                  sx={{
                    flex: 1,
                    fontWeight: 500,
                    color: 'text.primary',
                  }}
                >
                  {option.display_name}
                  {isRecent && (
                    <Chip
                      label="Recent"
                      size="small"
                      sx={{
                        ml: 1,
                        height: 20,
                        fontSize: '0.65rem',
                        backgroundColor: 'rgba(59, 130, 246, 0.2)',
                        color: '#60a5fa',
                      }}
                    />
                  )}
                </Typography>

                {/* Sensor Count Badge */}
                <Chip
                  icon={<SensorsIcon sx={{ fontSize: '0.875rem' }} />}
                  label={`${option.sensor_count} sensors`}
                  size="small"
                  variant="outlined"
                  sx={{
                    borderColor: 'rgba(148, 163, 184, 0.3)',
                    color: 'text.secondary',
                    fontSize: '0.75rem',
                  }}
                />

                {/* Trained Model Badge */}
                {isTrained ? (
                  <Tooltip title={`${modelCount} trained model${modelCount > 1 ? 's' : ''}`}>
                    <Chip
                      icon={<CheckCircleIcon sx={{ fontSize: '0.875rem' }} />}
                      label="Trained"
                      size="small"
                      sx={{
                        backgroundColor: 'rgba(16, 185, 129, 0.15)',
                        color: '#10b981',
                        fontWeight: 600,
                        fontSize: '0.75rem',
                        border: '1px solid rgba(16, 185, 129, 0.3)',
                      }}
                    />
                  </Tooltip>
                ) : (
                  <Chip
                    label="No Model"
                    size="small"
                    sx={{
                      backgroundColor: 'rgba(100, 116, 139, 0.15)',
                      color: '#64748b',
                      fontSize: '0.75rem',
                      border: '1px solid rgba(100, 116, 139, 0.3)',
                    }}
                  />
                )}
              </Stack>

              {/* Secondary Row: Manufacturer | Model */}
              <Stack direction="row" spacing={1} alignItems="center" sx={{ width: '100%', pl: 4 }}>
                <Typography
                  variant="caption"
                  sx={{
                    color: 'text.secondary',
                    fontSize: '0.75rem',
                    textTransform: 'uppercase',
                    letterSpacing: '0.5px',
                  }}
                >
                  {option.manufacturer}
                </Typography>
                <Typography
                  variant="caption"
                  sx={{ color: 'text.disabled', fontSize: '0.75rem' }}
                >
                  |
                </Typography>
                <Typography
                  variant="caption"
                  sx={{ color: 'text.secondary', fontSize: '0.75rem' }}
                >
                  {option.model}
                </Typography>

                {/* Model Type Indicators */}
                <Box sx={{ flex: 1 }} />
                <Stack direction="row" spacing={0.5}>
                  {option.has_classification_model && (
                    <Tooltip title="Classification Model">
                      <Box
                        sx={{
                          width: 6,
                          height: 6,
                          borderRadius: '50%',
                          backgroundColor: '#667eea',
                        }}
                      />
                    </Tooltip>
                  )}
                  {option.has_regression_model && (
                    <Tooltip title="Regression Model (RUL)">
                      <Box
                        sx={{
                          width: 6,
                          height: 6,
                          borderRadius: '50%',
                          backgroundColor: '#10b981',
                        }}
                      />
                    </Tooltip>
                  )}
                  {option.has_anomaly_model && (
                    <Tooltip title="Anomaly Detection Model">
                      <Box
                        sx={{
                          width: 6,
                          height: 6,
                          borderRadius: '50%',
                          backgroundColor: '#fbbf24',
                        }}
                      />
                    </Tooltip>
                  )}
                  {option.has_timeseries_model && (
                    <Tooltip title="Timeseries Model">
                      <Box
                        sx={{
                          width: 6,
                          height: 6,
                          borderRadius: '50%',
                          backgroundColor: '#3b82f6',
                        }}
                      />
                    </Tooltip>
                  )}
                </Stack>
              </Stack>
            </Box>
          );
        }}
        filterOptions={(options, { inputValue }) => {
          const searchLower = inputValue.toLowerCase();
          return options.filter((option) => {
            return (
              option.display_name.toLowerCase().includes(searchLower) ||
              option.category.toLowerCase().includes(searchLower) ||
              option.manufacturer.toLowerCase().includes(searchLower) ||
              option.model.toLowerCase().includes(searchLower) ||
              option.machine_id.toLowerCase().includes(searchLower)
            );
          });
        }}
        sx={{
          '& .MuiAutocomplete-popupIndicator': {
            color: 'text.secondary',
          },
          '& .MuiAutocomplete-clearIndicator': {
            color: 'text.secondary',
          },
        }}
      />

      {/* Recent Selections Helper Text */}
      {!selectedMachineId && recentIds.length > 0 && (
        <Typography
          variant="caption"
          sx={{
            display: 'block',
            mt: 1,
            ml: 2,
            color: 'text.disabled',
            fontSize: '0.75rem',
          }}
        >
          💡 Tip: Your recent selections appear with a "Recent" badge
        </Typography>
      )}
    </Box>
  );
};

export default MachineSelector;
