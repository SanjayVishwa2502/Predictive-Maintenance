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
  Visibility as ViewIcon,
} from '@mui/icons-material';
import SecureDeleteDialog from '../../../components/SecureDeleteDialog';

const ACCESS_TOKEN_KEY = 'pm_access_token';
const REFRESH_TOKEN_KEY = 'pm_refresh_token';

function getAccessToken(): string | null {
  try {
    const token = window.localStorage.getItem(ACCESS_TOKEN_KEY);
    return token && token.trim() ? token : null;
  } catch {
    return null;
  }
}

function clearTokens(): void {
  try {
    window.localStorage.removeItem(ACCESS_TOKEN_KEY);
    window.localStorage.removeItem(REFRESH_TOKEN_KEY);
  } catch {
    // ignore
  }
}

async function fetchWithAuth(input: RequestInfo | URL, init: RequestInit = {}): Promise<Response> {
  const headers = new Headers(init.headers);
  const token = getAccessToken();
  if (token) headers.set('Authorization', `Bearer ${token}`);
  const resp = await fetch(input, { ...init, headers });
  if (resp.status === 401) {
    clearTokens();
    try {
      window.location.reload();
    } catch {
      // ignore
    }
  }
  return resp;
}

// ============================================================================
// TYPESCRIPT INTERFACES
// ============================================================================

export interface PredictionHistoryProps {
  machineId: string;
  predictions?: HistoricalPrediction[];
  limit?: number;
  sessionId?: string;
  onRowClick?: (prediction: HistoricalPrediction) => void;
  onHistoryCleared?: () => void;
  userRole?: 'admin' | 'operator' | 'viewer';
}

export interface HistoricalPrediction {
  id: string;
  timestamp: Date;

  // Optional fields used by demos/older code paths
  failure_type?: string;
  confidence?: number;
  rul_hours?: number;
  urgency?: string;
  health_state?: string;

  // 5-second snapshot metadata
  data_stamp?: string;
  run_id?: string | null;
  has_run?: boolean;
  has_explanation?: boolean;
  run_type?: 'prediction' | 'explanation' | null;  // "prediction" = ML only, "explanation" = full with LLM
  sensor_snapshot?: Record<string, number>;
}

type RunDetails = {
  run_id: string;
  machine_id: string;
  data_stamp: string;
  created_at: string;
  sensor_data: Record<string, number>;
  predictions: Record<string, any>;
  llm: Record<string, any>;
};

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

const formatTimestamp = (date: Date): string => {
  return new Intl.DateTimeFormat('en-US', {
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: true,
  }).format(date);
};

const viewLatestDataset = (machineId: string, sessionId?: string) => {
  const qs = sessionId ? `?session_id=${encodeURIComponent(sessionId)}` : '';
  const url = `${API_BASE}/api/ml/machines/${encodeURIComponent(machineId)}/audit/wide/latest${qs}`;
  window.open(url, '_blank', 'noopener,noreferrer');
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
  sessionId,
  onRowClick,
  onHistoryCleared,
  userRole,
}: PredictionHistoryProps) {
  const [searchQuery, setSearchQuery] = useState('');
  const [dateRange, setDateRange] = useState('all');
  const [explanationFilter, setExplanationFilter] = useState<'all' | 'no' | 'with'>('all');
  const [selectedPrediction, setSelectedPrediction] = useState<HistoricalPrediction | null>(null);
  const [detailsOpen, setDetailsOpen] = useState(false);
  const [detailsLoading, setDetailsLoading] = useState(false);
  const [detailsError, setDetailsError] = useState<string | null>(null);
  const [runDetails, setRunDetails] = useState<RunDetails | null>(null);
  const [clearingHistory, setClearingHistory] = useState(false);
  
  // Secure delete dialog state
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);

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
          (pred.data_stamp || '').toLowerCase().includes(query) ||
          (pred.run_id || '').toLowerCase().includes(query)
      );
    }

    // Apply explanation filter
    if (explanationFilter !== 'all') {
      filtered = filtered.filter((pred) => {
        const hasExplanation = Boolean(pred.has_explanation);
        return explanationFilter === 'with' ? hasExplanation : !hasExplanation;
      });
    }

    return filtered;
  }, [predictions, limit, dateRange, searchQuery, explanationFilter]);

  const openDetails = async (prediction: HistoricalPrediction) => {
    setSelectedPrediction(prediction);
    setDetailsOpen(true);
    setDetailsLoading(Boolean((prediction.run_id || '').trim()));
    setDetailsError(null);
    setRunDetails(null);

    const runId = (prediction.run_id || '').trim();
    if (!runId) {
      setDetailsLoading(false);
      return;
    }

    try {
      const resp = await fetchWithAuth(`${API_BASE}/api/ml/runs/${encodeURIComponent(runId)}`, {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' },
        signal: AbortSignal.timeout(10_000),
      });
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({ detail: resp.statusText }));
        throw new Error(err?.detail || `HTTP ${resp.status}`);
      }
      const json = await resp.json();
      setRunDetails(json as RunDetails);
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setDetailsError(msg);
    } finally {
      setDetailsLoading(false);
    }
  };

  // Row click = select/run callback (used by dashboard to show text output)
  const handleRowSelect = (prediction: HistoricalPrediction) => {
    const hasRun = Boolean((prediction.run_id || '').trim());
    if (onRowClick && hasRun) {
      onRowClick(prediction);
      return;
    }
    void openDetails(prediction);
  };

  // Handle date range change
  const handleDateRangeChange = (event: SelectChangeEvent) => {
    setDateRange(event.target.value);
  };

  const handleExplanationFilterChange = (event: SelectChangeEvent) => {
    const v = event.target.value as 'all' | 'no' | 'with';
    setExplanationFilter(v);
  };

  const handleDeleteClick = () => {
    setDeleteDialogOpen(true);
  };

  const clearAllHistory = async () => {
    const mid = (machineId || '').trim();
    if (!mid || clearingHistory) return;

    setClearingHistory(true);
    try {
      const resp = await fetchWithAuth(`${API_BASE}/api/ml/machines/${encodeURIComponent(mid)}/history`, {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' },
        signal: AbortSignal.timeout(30_000),
      });
      if (!resp.ok) {
        const errorText = await resp.text().catch(() => 'Delete failed');
        throw new Error(errorText || 'Delete failed');
      }
      onHistoryCleared?.();
    } finally {
      setClearingHistory(false);
    }
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
      field: 'has_run',
      headerName: 'Status',
      width: 150,
      renderCell: (params: GridRenderCellParams) => {
        const hasRun = Boolean(params.row.run_id);
        const hasExplanation = Boolean(params.row.has_explanation);
        const runType = params.row.run_type as string | undefined;
        
        // Determine label and palette based on run_type
        let label: string;
        let palette: { bg: string; fg: string; border: string };
        
        if (!hasRun) {
          label = 'NO RUN';
          palette = { bg: 'rgba(156,163,175,0.12)', fg: '#9ca3af', border: 'rgba(156,163,175,0.25)' };
        } else if (hasExplanation) {
          // Full with LLM explanation (green)
          label = 'EXPLAINED';
          palette = { bg: 'rgba(16,185,129,0.15)', fg: '#10b981', border: 'rgba(16,185,129,0.35)' };
        } else if (runType === 'prediction') {
          // ML-only auto prediction (blue)
          label = 'PREDICTION';
          palette = { bg: 'rgba(59,130,246,0.15)', fg: '#3b82f6', border: 'rgba(59,130,246,0.35)' };
        } else {
          // Run exists but no explanation yet (yellow)
          label = 'PENDING';
          palette = { bg: 'rgba(251,191,36,0.12)', fg: '#fbbf24', border: 'rgba(251,191,36,0.35)' };
        }
        
        return (
          <Chip
            size="small"
            label={label}
            sx={{
              bgcolor: palette.bg,
              color: palette.fg,
              border: '1px solid',
              borderColor: palette.border,
              fontWeight: 700,
            }}
          />
        );
      },
    },
    {
      field: 'run_id',
      headerName: 'Run ID',
      width: 260,
      renderCell: (params: GridRenderCellParams) => (
        <Typography variant="body2" sx={{ color: params.row.run_id ? '#d1d5db' : '#6b7280' }}>
          {params.row.run_id ? String(params.row.run_id) : '—'}
        </Typography>
      ),
    },
    {
      field: 'sensor_snapshot',
      headerName: 'Sensor Readings',
      flex: 1,
      minWidth: 320,
      sortable: false,
      renderCell: (params: GridRenderCellParams) => {
        const snap = (params.row.sensor_snapshot || {}) as Record<string, number>;
        const entries = Object.entries(snap);
        if (!entries.length) {
          return (
            <Typography variant="body2" sx={{ color: '#6b7280' }}>
              —
            </Typography>
          );
        }

        // Render a readable inline summary but preserve full data via title tooltip.
        const full = entries
          .map(([k, v]) => `${k}=${Number.isFinite(Number(v)) ? Number(v).toFixed(3) : String(v)}`)
          .join(', ');

        return (
          <Typography
            variant="body2"
            title={full}
            sx={{
              color: '#d1d5db',
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              whiteSpace: 'nowrap',
              width: '100%',
            }}
          >
            {full}
          </Typography>
        );
      },
    },
    {
      field: 'actions',
      headerName: 'Actions',
      width: 100,
      sortable: false,
      renderCell: (params: GridRenderCellParams) => {
        const canView = true;
        return (
          <Tooltip title={'View Details'}>
            <span>
              <IconButton
                size="small"
                disabled={!canView}
                onClick={(e) => {
                  e.stopPropagation();
                  void openDetails(params.row);
                }}
                sx={{
                  color: canView ? '#667eea' : '#6b7280',
                  '&:hover': { bgcolor: canView ? 'rgba(102, 126, 234, 0.1)' : 'transparent' },
                }}
              >
                <ViewIcon />
              </IconButton>
            </span>
          </Tooltip>
        );
      },
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
            Last {limit} snapshots • {filteredPredictions.length} shown
          </Typography>
        }
      />

      <CardContent>
        {/* FILTERS & ACTIONS */}
        <Stack direction="row" spacing={2} sx={{ mb: 3 }} flexWrap="wrap">
          {/* Search */}
          <TextField
            size="small"
            placeholder="Search by data stamp or run id..."
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

          {/* Explanation Filter */}
          <FormControl
            size="small"
            sx={{
              minWidth: 190,
              '& .MuiOutlinedInput-root': {
                bgcolor: 'rgba(31, 41, 55, 0.5)',
                '& fieldset': { borderColor: 'rgba(255, 255, 255, 0.2)' },
                '&:hover fieldset': { borderColor: 'rgba(255, 255, 255, 0.3)' },
              },
              '& .MuiInputLabel-root': { color: '#d1d5db' },
              '& .MuiSelect-select': { color: '#d1d5db' },
            }}
          >
            <InputLabel>Explanation</InputLabel>
            <Select value={explanationFilter} onChange={handleExplanationFilterChange} label="Explanation">
              <MenuItem value="all">All points</MenuItem>
              <MenuItem value="no">No explanation</MenuItem>
              <MenuItem value="with">With explanation</MenuItem>
            </Select>
          </FormControl>

          {/* View dataset (server-collected CSV) */}
          <Button
            variant="outlined"
            onClick={() => viewLatestDataset(machineId, sessionId)}
            sx={{
              borderColor: '#667eea',
              color: '#667eea',
              '&:hover': {
                borderColor: '#5568d3',
                bgcolor: 'rgba(102, 126, 234, 0.1)',
              },
            }}
          >
            View Dataset
          </Button>

          {userRole === 'admin' && (
            <Button
              variant="outlined"
              color="error"
              disabled={clearingHistory || !machineId.trim()}
              onClick={handleDeleteClick}
            >
              Delete All Prediction History
            </Button>
          )}
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
            onRowClick={(params) => handleRowSelect(params.row)}
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
        onClose={() => {
          setDetailsOpen(false);
          setDetailsLoading(false);
          setDetailsError(null);
          setRunDetails(null);
        }}
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
                Run Details
              </Typography>
              <Typography variant="body2" sx={{ color: '#9ca3af', mt: 0.5 }}>
                {formatTimestamp(selectedPrediction.timestamp)}
              </Typography>
            </DialogTitle>
            <DialogContent sx={{ mt: 2 }}>
              <Stack spacing={2}>
                <Box>
                  <Typography variant="caption" sx={{ color: '#9ca3af' }}>
                    Data Stamp
                  </Typography>
                  <Typography variant="body2" sx={{ color: '#d1d5db', mt: 0.5 }}>
                    {selectedPrediction.data_stamp || '—'}
                  </Typography>
                </Box>

                <Box>
                  <Typography variant="caption" sx={{ color: '#9ca3af' }}>
                    Run ID
                  </Typography>
                  <Typography variant="body2" sx={{ color: '#d1d5db', mt: 0.5 }}>
                    {selectedPrediction.run_id || '—'}
                  </Typography>
                </Box>

                {detailsLoading && (
                  <Typography variant="body2" sx={{ color: '#9ca3af' }}>
                    Loading run details...
                  </Typography>
                )}

                {detailsError && (
                  <Typography variant="body2" sx={{ color: '#f97316' }}>
                    {detailsError}
                  </Typography>
                )}

                {(selectedPrediction.sensor_snapshot || runDetails?.sensor_data) && (
                  <Box>
                    <Typography variant="caption" sx={{ color: '#9ca3af' }}>
                      Observed Data (Snapshot)
                    </Typography>
                    <Typography
                      variant="body2"
                      component="pre"
                      sx={{
                        color: '#d1d5db',
                        mt: 0.5,
                        p: 1,
                        borderRadius: 1,
                        bgcolor: 'rgba(0,0,0,0.25)',
                        maxHeight: 180,
                        overflow: 'auto',
                        whiteSpace: 'pre-wrap',
                        fontFamily:
                          'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace',
                      }}
                    >
                      {JSON.stringify((runDetails?.sensor_data || selectedPrediction.sensor_snapshot || {}), null, 2)}
                    </Typography>
                  </Box>
                )}

                {runDetails && (
                  <>
                    <Box>
                      <Typography variant="caption" sx={{ color: '#9ca3af' }}>
                        Predictions (All)
                      </Typography>
                      <Typography
                        variant="body2"
                        component="pre"
                        sx={{
                          color: '#d1d5db',
                          mt: 0.5,
                          p: 1,
                          borderRadius: 1,
                          bgcolor: 'rgba(0,0,0,0.25)',
                          maxHeight: 200,
                          overflow: 'auto',
                          whiteSpace: 'pre-wrap',
                          fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace',
                        }}
                      >
                        {JSON.stringify(runDetails.predictions || {}, null, 2)}
                      </Typography>
                    </Box>

                    <Box>
                      <Typography variant="caption" sx={{ color: '#9ca3af' }}>
                        LLM Outputs (All)
                      </Typography>
                      <Typography
                        variant="body2"
                        component="pre"
                        sx={{
                          color: '#d1d5db',
                          mt: 0.5,
                          p: 1,
                          borderRadius: 1,
                          bgcolor: 'rgba(0,0,0,0.25)',
                          maxHeight: 220,
                          overflow: 'auto',
                          whiteSpace: 'pre-wrap',
                          fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace',
                        }}
                      >
                        {JSON.stringify(runDetails.llm || {}, null, 2)}
                      </Typography>
                    </Box>
                  </>
                )}
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

      {/* Secure Delete Dialog for Admin */}
      <SecureDeleteDialog
        open={deleteDialogOpen}
        onClose={() => setDeleteDialogOpen(false)}
        onConfirm={clearAllHistory}
        title="Delete All Prediction History"
        description="This will permanently delete all prediction history, snapshots, and runs for this machine."
        itemName={machineId}
        confirmButtonText={clearingHistory ? 'Deleting...' : 'Delete All History'}
      />
    </Card>
  );
}
