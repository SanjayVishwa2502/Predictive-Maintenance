/**
 * PredictionHistory Component
 * 
 * Paginated table displaying historical ML predictions for a machine
 * Features:
 * - Material-UI DataGrid with pagination
 * - Sortable columns
 * - Date range filtering
 * - Search functionality
 * - CSV export
 * - Row click for details
 * - Color-coded status indicators
 * 
 * Design: Professional dark theme with responsive layout
 */

import { useState, useMemo } from 'react';
import {
  Box,
  Card,
  CardContent,
  CardHeader,
  TextField,
  Button,
  IconButton,
  Chip,
  Stack,
  Typography,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  InputAdornment,
  Tooltip,
  LinearProgress,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
} from '@mui/material';
import type { SelectChangeEvent } from '@mui/material';
import { DataGrid } from '@mui/x-data-grid';
import type { GridColDef, GridRenderCellParams } from '@mui/x-data-grid';
import {
  Search as SearchIcon,
  Download as DownloadIcon,
  Visibility as ViewIcon,
  CheckCircle as HealthyIcon,
  Warning as WarningIcon,
  Error as CriticalIcon,
  Circle as DegradingIcon,
} from '@mui/icons-material';

// ============================================================================
// TYPESCRIPT INTERFACES
// ============================================================================

export interface PredictionHistoryProps {
  machineId: string;
  predictions?: HistoricalPrediction[];
  limit?: number;
  onRowClick?: (prediction: HistoricalPrediction) => void;
}

export interface HistoricalPrediction {
  id: string;
  timestamp: Date;
  failure_type: string;
  confidence: number;
  rul_hours: number;
  urgency: string;
  health_state: string;
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

const getStatusConfig = (healthState: string) => {
  const configs: Record<string, { icon: React.ReactElement; color: string; label: string }> = {
    HEALTHY: {
      icon: <HealthyIcon sx={{ fontSize: 20 }} />,
      color: '#10b981',
      label: 'Healthy',
    },
    DEGRADING: {
      icon: <DegradingIcon sx={{ fontSize: 20 }} />,
      color: '#fbbf24',
      label: 'Degrading',
    },
    WARNING: {
      icon: <WarningIcon sx={{ fontSize: 20 }} />,
      color: '#f97316',
      label: 'Warning',
    },
    CRITICAL: {
      icon: <CriticalIcon sx={{ fontSize: 20 }} />,
      color: '#ef4444',
      label: 'Critical',
    },
  };
  return configs[healthState] || configs.HEALTHY;
};

const getUrgencyColor = (urgency: string): string => {
  const colors: Record<string, string> = {
    Low: '#10b981',
    Medium: '#fbbf24',
    High: '#f97316',
    Critical: '#ef4444',
  };
  return colors[urgency] || '#6b7280';
};

const formatTimestamp = (date: Date): string => {
  return new Intl.DateTimeFormat('en-US', {
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    hour12: true,
  }).format(date);
};

const exportToCSV = (predictions: HistoricalPrediction[], machineId: string) => {
  const headers = ['Timestamp', 'Status', 'Failure Type', 'Confidence (%)', 'RUL (hours)', 'RUL (days)', 'Urgency'];
  const rows = predictions.map(pred => [
    pred.timestamp.toISOString(),
    pred.health_state,
    pred.failure_type,
    (pred.confidence * 100).toFixed(1),
    pred.rul_hours.toFixed(1),
    (pred.rul_hours / 24).toFixed(1),
    pred.urgency,
  ]);

  const csv = [
    headers.join(','),
    ...rows.map(row => row.join(',')),
  ].join('\n');

  const blob = new Blob([csv], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = `${machineId}_prediction_history_${Date.now()}.csv`;
  link.click();
  URL.revokeObjectURL(url);
};

const filterByDateRange = (predictions: HistoricalPrediction[], range: string): HistoricalPrediction[] => {
  const now = new Date();
  const cutoff = new Date();

  switch (range) {
    case 'hour':
      cutoff.setHours(now.getHours() - 1);
      break;
    case 'day':
      cutoff.setDate(now.getDate() - 1);
      break;
    case 'week':
      cutoff.setDate(now.getDate() - 7);
      break;
    case 'month':
      cutoff.setMonth(now.getMonth() - 1);
      break;
    case 'all':
    default:
      return predictions;
  }

  return predictions.filter(pred => pred.timestamp >= cutoff);
};

// ============================================================================
// MAIN COMPONENT
// ============================================================================

export default function PredictionHistory({
  machineId,
  predictions = [],
  limit = 100,
  onRowClick,
}: PredictionHistoryProps) {
  const [searchQuery, setSearchQuery] = useState('');
  const [dateRange, setDateRange] = useState('all');
  const [selectedPrediction, setSelectedPrediction] = useState<HistoricalPrediction | null>(null);
  const [detailsOpen, setDetailsOpen] = useState(false);

  // Filter predictions
  const filteredPredictions = useMemo(() => {
    let filtered = predictions.slice(0, limit);

    // Apply date filter
    filtered = filterByDateRange(filtered, dateRange);

    // Apply search filter
    if (searchQuery.trim()) {
      const query = searchQuery.toLowerCase();
      filtered = filtered.filter(
        pred =>
          pred.failure_type.toLowerCase().includes(query) ||
          pred.urgency.toLowerCase().includes(query) ||
          pred.health_state.toLowerCase().includes(query)
      );
    }

    return filtered;
  }, [predictions, limit, dateRange, searchQuery]);

  // Handle row click
  const handleRowClick = (prediction: HistoricalPrediction) => {
    if (onRowClick) {
      onRowClick(prediction);
    } else {
      setSelectedPrediction(prediction);
      setDetailsOpen(true);
    }
  };

  // Handle CSV export
  const handleExport = () => {
    exportToCSV(filteredPredictions, machineId);
  };

  // Handle date range change
  const handleDateRangeChange = (event: SelectChangeEvent) => {
    setDateRange(event.target.value);
  };

  // DataGrid columns
  const columns: GridColDef[] = [
    {
      field: 'timestamp',
      headerName: 'Timestamp',
      width: 150,
      renderCell: (params: GridRenderCellParams) => (
        <Typography variant="body2" sx={{ color: '#d1d5db' }}>
          {formatTimestamp(params.row.timestamp)}
        </Typography>
      ),
    },
    {
      field: 'health_state',
      headerName: 'Status',
      width: 140,
      renderCell: (params: GridRenderCellParams) => {
        const config = getStatusConfig(params.row.health_state);
        return (
          <Stack direction="row" spacing={1} alignItems="center">
            <Box sx={{ color: config.color }}>{config.icon}</Box>
            <Typography variant="body2" sx={{ color: config.color, fontWeight: 500 }}>
              {config.label}
            </Typography>
          </Stack>
        );
      },
    },
    {
      field: 'confidence',
      headerName: 'Confidence',
      width: 150,
      renderCell: (params: GridRenderCellParams) => {
        const percentage = params.row.confidence * 100;
        return (
          <Box sx={{ width: '100%' }}>
            <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 0.5 }}>
              <Typography variant="body2" sx={{ color: '#d1d5db', fontWeight: 500 }}>
                {percentage.toFixed(0)}%
              </Typography>
            </Stack>
            <LinearProgress
              variant="determinate"
              value={percentage}
              sx={{
                height: 4,
                borderRadius: 2,
                bgcolor: 'rgba(255, 255, 255, 0.1)',
                '& .MuiLinearProgress-bar': {
                  bgcolor: percentage >= 80 ? '#10b981' : percentage >= 60 ? '#fbbf24' : '#f97316',
                },
              }}
            />
          </Box>
        );
      },
    },
    {
      field: 'rul_hours',
      headerName: 'RUL (hours)',
      width: 120,
      type: 'number',
      renderCell: (params: GridRenderCellParams) => (
        <Typography variant="body2" sx={{ color: '#d1d5db', fontWeight: 500 }}>
          {params.row.rul_hours.toFixed(1)}
        </Typography>
      ),
    },
    {
      field: 'rul_days',
      headerName: 'RUL (days)',
      width: 120,
      type: 'number',
      renderCell: (params: GridRenderCellParams) => (
        <Typography variant="body2" sx={{ color: '#9ca3af' }}>
          {(params.row.rul_hours / 24).toFixed(1)}
        </Typography>
      ),
    },
    {
      field: 'urgency',
      headerName: 'Urgency',
      width: 120,
      renderCell: (params: GridRenderCellParams) => (
        <Chip
          label={params.row.urgency}
          size="small"
          sx={{
            bgcolor: `${getUrgencyColor(params.row.urgency)}20`,
            color: getUrgencyColor(params.row.urgency),
            borderColor: getUrgencyColor(params.row.urgency),
            border: '1px solid',
            fontWeight: 600,
          }}
        />
      ),
    },
    {
      field: 'actions',
      headerName: 'Actions',
      width: 100,
      sortable: false,
      renderCell: (params: GridRenderCellParams) => (
        <Tooltip title="View Details">
          <IconButton
            size="small"
            onClick={(e) => {
              e.stopPropagation();
              handleRowClick(params.row);
            }}
            sx={{
              color: '#667eea',
              '&:hover': { bgcolor: 'rgba(102, 126, 234, 0.1)' },
            }}
          >
            <ViewIcon />
          </IconButton>
        </Tooltip>
      ),
    },
  ];

  // ============================================================================
  // RENDER
  // ============================================================================

  return (
    <Card
      sx={{
        background: 'linear-gradient(135deg, rgba(31, 41, 55, 0.8) 0%, rgba(17, 24, 39, 0.8) 100%)',
        backdropFilter: 'blur(10px)',
        border: '1px solid rgba(255, 255, 255, 0.1)',
        borderRadius: 2,
      }}
    >
      {/* HEADER */}
      <CardHeader
        title={
          <Typography variant="h6" sx={{ fontWeight: 600, color: '#f9fafb' }}>
            PREDICTION HISTORY
          </Typography>
        }
        subheader={
          <Typography variant="body2" sx={{ color: '#9ca3af', mt: 0.5 }}>
            Last {limit} predictions • {filteredPredictions.length} shown
          </Typography>
        }
      />

      <CardContent>
        {/* FILTERS & ACTIONS */}
        <Stack direction="row" spacing={2} sx={{ mb: 3 }} flexWrap="wrap">
          {/* Search */}
          <TextField
            size="small"
            placeholder="Search by failure type or urgency..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            InputProps={{
              startAdornment: (
                <InputAdornment position="start">
                  <SearchIcon sx={{ color: '#9ca3af' }} />
                </InputAdornment>
              ),
            }}
            sx={{
              flex: 1,
              minWidth: 200,
              '& .MuiOutlinedInput-root': {
                bgcolor: 'rgba(31, 41, 55, 0.5)',
                '& fieldset': { borderColor: 'rgba(255, 255, 255, 0.2)' },
                '&:hover fieldset': { borderColor: 'rgba(255, 255, 255, 0.3)' },
                '& input': { color: '#d1d5db' },
              },
            }}
          />

          {/* Date Range Filter */}
          <FormControl
            size="small"
            sx={{
              minWidth: 150,
              '& .MuiOutlinedInput-root': {
                bgcolor: 'rgba(31, 41, 55, 0.5)',
                '& fieldset': { borderColor: 'rgba(255, 255, 255, 0.2)' },
                '&:hover fieldset': { borderColor: 'rgba(255, 255, 255, 0.3)' },
              },
              '& .MuiInputLabel-root': { color: '#d1d5db' },
              '& .MuiSelect-select': { color: '#d1d5db' },
            }}
          >
            <InputLabel>Date Range</InputLabel>
            <Select value={dateRange} onChange={handleDateRangeChange} label="Date Range">
              <MenuItem value="all">All Time</MenuItem>
              <MenuItem value="hour">Last Hour</MenuItem>
              <MenuItem value="day">Last 24 Hours</MenuItem>
              <MenuItem value="week">Last Week</MenuItem>
              <MenuItem value="month">Last Month</MenuItem>
            </Select>
          </FormControl>

          {/* Export Button */}
          <Button
            variant="outlined"
            startIcon={<DownloadIcon />}
            onClick={handleExport}
            disabled={filteredPredictions.length === 0}
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
        </Stack>

        {/* DATA GRID */}
        <Box
          sx={{
            height: 600,
            width: '100%',
            '& .MuiDataGrid-root': {
              border: '1px solid rgba(255, 255, 255, 0.1)',
              borderRadius: 1,
              bgcolor: 'rgba(31, 41, 55, 0.5)',
            },
            '& .MuiDataGrid-cell': {
              borderColor: 'rgba(255, 255, 255, 0.1)',
              color: '#d1d5db',
            },
            '& .MuiDataGrid-columnHeaders': {
              bgcolor: 'rgba(102, 126, 234, 0.1)',
              borderColor: 'rgba(255, 255, 255, 0.1)',
              color: '#f9fafb',
            },
            '& .MuiDataGrid-columnHeaderTitle': {
              fontWeight: 600,
            },
            '& .MuiDataGrid-row': {
              cursor: 'pointer',
              '&:hover': {
                bgcolor: 'rgba(102, 126, 234, 0.1)',
              },
            },
            '& .MuiDataGrid-footerContainer': {
              borderColor: 'rgba(255, 255, 255, 0.1)',
              bgcolor: 'rgba(31, 41, 55, 0.5)',
            },
            '& .MuiTablePagination-root': {
              color: '#d1d5db',
            },
            '& .MuiDataGrid-overlay': {
              bgcolor: 'rgba(31, 41, 55, 0.95)',
            },
          }}
        >
          <DataGrid
            rows={filteredPredictions}
            columns={columns}
            initialState={{
              pagination: {
                paginationModel: { pageSize: 10, page: 0 },
              },
            }}
            pageSizeOptions={[10, 25, 50]}
            disableRowSelectionOnClick
            onRowClick={(params) => handleRowClick(params.row)}
            sx={{
              '& .MuiDataGrid-virtualScroller': {
                minHeight: filteredPredictions.length === 0 ? 200 : 'auto',
              },
            }}
          />
        </Box>
      </CardContent>

      {/* DETAILS MODAL */}
      <Dialog
        open={detailsOpen}
        onClose={() => setDetailsOpen(false)}
        maxWidth="sm"
        fullWidth
        PaperProps={{
          sx: {
            background: 'linear-gradient(135deg, rgba(31, 41, 55, 0.95) 0%, rgba(17, 24, 39, 0.95) 100%)',
            backdropFilter: 'blur(20px)',
            border: '1px solid rgba(255, 255, 255, 0.1)',
            borderRadius: 2,
          },
        }}
      >
        {selectedPrediction && (
          <>
            <DialogTitle sx={{ borderBottom: '1px solid rgba(255, 255, 255, 0.1)' }}>
              <Typography variant="h6" sx={{ fontWeight: 600, color: '#f9fafb' }}>
                Prediction Details
              </Typography>
              <Typography variant="body2" sx={{ color: '#9ca3af', mt: 0.5 }}>
                {formatTimestamp(selectedPrediction.timestamp)}
              </Typography>
            </DialogTitle>
            <DialogContent sx={{ mt: 2 }}>
              <Stack spacing={2}>
                <Box>
                  <Typography variant="caption" sx={{ color: '#9ca3af' }}>
                    Health State
                  </Typography>
                  <Stack direction="row" spacing={1} alignItems="center" sx={{ mt: 0.5 }}>
                    <Box sx={{ color: getStatusConfig(selectedPrediction.health_state).color }}>
                      {getStatusConfig(selectedPrediction.health_state).icon}
                    </Box>
                    <Typography variant="body1" sx={{ color: '#f9fafb', fontWeight: 500 }}>
                      {getStatusConfig(selectedPrediction.health_state).label}
                    </Typography>
                  </Stack>
                </Box>

                <Box>
                  <Typography variant="caption" sx={{ color: '#9ca3af' }}>
                    Failure Type
                  </Typography>
                  <Typography variant="body1" sx={{ color: '#f9fafb', mt: 0.5 }}>
                    {selectedPrediction.failure_type}
                  </Typography>
                </Box>

                <Box>
                  <Typography variant="caption" sx={{ color: '#9ca3af' }}>
                    Confidence
                  </Typography>
                  <Typography variant="body1" sx={{ color: '#f9fafb', mt: 0.5 }}>
                    {(selectedPrediction.confidence * 100).toFixed(1)}%
                  </Typography>
                </Box>

                <Box>
                  <Typography variant="caption" sx={{ color: '#9ca3af' }}>
                    Remaining Useful Life
                  </Typography>
                  <Typography variant="body1" sx={{ color: '#f9fafb', mt: 0.5 }}>
                    {selectedPrediction.rul_hours.toFixed(1)} hours ({(selectedPrediction.rul_hours / 24).toFixed(1)} days)
                  </Typography>
                </Box>

                <Box>
                  <Typography variant="caption" sx={{ color: '#9ca3af' }}>
                    Urgency Level
                  </Typography>
                  <Box sx={{ mt: 0.5 }}>
                    <Chip
                      label={selectedPrediction.urgency}
                      sx={{
                        bgcolor: `${getUrgencyColor(selectedPrediction.urgency)}20`,
                        color: getUrgencyColor(selectedPrediction.urgency),
                        borderColor: getUrgencyColor(selectedPrediction.urgency),
                        border: '1px solid',
                        fontWeight: 600,
                      }}
                    />
                  </Box>
                </Box>
              </Stack>
            </DialogContent>
            <DialogActions sx={{ borderTop: '1px solid rgba(255, 255, 255, 0.1)', p: 2 }}>
              <Button
                variant="contained"
                onClick={() => setDetailsOpen(false)}
                sx={{
                  bgcolor: '#667eea',
                  '&:hover': { bgcolor: '#5568d3' },
                }}
              >
                Close
              </Button>
            </DialogActions>
          </>
        )}
      </Dialog>
    </Card>
  );
}
