import { useEffect, useMemo, useState } from 'react';
import {
  Alert,
  Box,
  Button,
  Chip,
  CircularProgress,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  IconButton,
  InputAdornment,
  Paper,
  Stack,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TextField,
  Tooltip,
  Typography,
} from '@mui/material';
import {
  Delete as DeleteIcon,
  PlayArrow as PlayIcon,
  Refresh as RefreshIcon,
  Search as SearchIcon,
  Visibility as ViewIcon,
} from '@mui/icons-material';

import { ganApi } from '../../api/ganApi';
import type { MachineDetails } from '../../types/gan.types';

interface MachineListProps {
  onMachineSelect: (machineId: string) => void;
  onRefresh?: () => void;
}

type StageChipColor = 'default' | 'primary' | 'success' | 'warning' | 'error' | 'info';

const getStageInfo = (m: MachineDetails): { label: string; color: StageChipColor; next: string } => {
  const flags = m.status;

  if (flags?.status === 'failed') {
    return { label: 'Failed', color: 'error', next: 'Check logs / retry' };
  }

  // If there is no metadata/profile, we can't run any workflow steps.
  if (!flags?.has_metadata) {
    return { label: 'Profile Needed', color: 'default', next: 'Upload/validate profile' };
  }

  if (flags.has_synthetic_data) {
    return { label: 'Synthetic Ready', color: 'success', next: 'Validate & download' };
  }

  if (flags.status === 'training') {
    return { label: 'Training', color: 'warning', next: 'Wait for training to finish' };
  }

  if (flags.has_trained_model) {
    return { label: 'Model Ready', color: 'warning', next: 'Generate synthetic data' };
  }

  if (flags.has_seed_data) {
    return { label: 'Seed Ready', color: 'primary', next: 'Train TVAE model' };
  }

  return { label: 'Profile Ready', color: 'info', next: 'Generate seed data' };
};

const getStatusTooltip = (m: MachineDetails): string => {
  const s = m.status;
  if (!s) return 'No workflow status available.';
  return `has_metadata=${String(!!s.has_metadata)} | has_seed_data=${String(!!s.has_seed_data)} | has_trained_model=${String(
    !!s.has_trained_model
  )} | has_synthetic_data=${String(!!s.has_synthetic_data)}`;
};

export default function MachineList({ onMachineSelect, onRefresh }: MachineListProps) {
  const [machines, setMachines] = useState<MachineDetails[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');

  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [machineToDelete, setMachineToDelete] = useState<string | null>(null);
  const [deleting, setDeleting] = useState(false);

  const filteredMachines = useMemo(() => {
    const q = searchQuery.trim().toLowerCase();
    if (!q) return machines;
    return machines.filter(
      (m) =>
        m.machine_id.toLowerCase().includes(q) ||
        m.machine_type.toLowerCase().includes(q) ||
        m.manufacturer.toLowerCase().includes(q) ||
        m.model.toLowerCase().includes(q)
    );
  }, [machines, searchQuery]);

  const stageSummary = useMemo(() => {
    const summary = {
      profileNeeded: 0,
      profileReady: 0,
      seedReady: 0,
      trainingOrModelReady: 0,
      syntheticReady: 0,
      failed: 0,
    };

    for (const m of machines) {
      const stage = getStageInfo(m).label;
      switch (stage) {
        case 'Profile Needed':
          summary.profileNeeded++;
          break;
        case 'Profile Ready':
          summary.profileReady++;
          break;
        case 'Seed Ready':
          summary.seedReady++;
          break;
        case 'Training':
        case 'Model Ready':
          summary.trainingOrModelReady++;
          break;
        case 'Synthetic Ready':
          summary.syntheticReady++;
          break;
        case 'Failed':
          summary.failed++;
          break;
        default:
          break;
      }
    }

    return summary;
  }, [machines]);

  const fetchMachines = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await ganApi.getMachines();

      // Preferred: single-call summaries from the backend.
      if (response.machine_details && response.machine_details.length > 0) {
        setMachines(response.machine_details);
        return;
      }

      // Backward-compatible fallback: fetch details per machine.
      // Do not drop machines if some detail requests fail; render with safe defaults.
      const detailResults = await Promise.allSettled(
        response.machines.map((machineId: string) => ganApi.getMachineDetails(machineId))
      );

      const details: MachineDetails[] = detailResults.map((r, idx) => {
        if (r.status === 'fulfilled') return r.value;
        const machineId = response.machines[idx];
        return {
          machine_id: machineId,
          machine_type: machineId.split('_')[0] || 'unknown',
          manufacturer: 'unknown',
          model: 'unknown',
          num_sensors: 0,
          degradation_states: 4,
          status: {
            machine_id: machineId,
            status: 'not_started',
            has_metadata: false,
            has_seed_data: false,
            has_trained_model: false,
            has_synthetic_data: false,
            can_generate_seed: false,
            can_train_model: false,
            can_generate_synthetic: false,
            last_updated: undefined,
          },
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
        };
      });

      setMachines(details);
    } catch (e: unknown) {
      const err = e as any;
      setMachines([]);
      setError(err?.response?.data?.detail || err?.message || 'Failed to fetch machines');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchMachines();
  }, []);

  const handleRefresh = () => {
    fetchMachines();
    onRefresh?.();
  };

  const handleDeleteClick = (machineId: string) => {
    setMachineToDelete(machineId);
    setDeleteDialogOpen(true);
  };

  const handleDeleteConfirm = async () => {
    if (!machineToDelete) return;
    setDeleting(true);
    try {
      await ganApi.deleteMachine(machineToDelete);
      setMachines((prev) => prev.filter((m) => m.machine_id !== machineToDelete));
      setDeleteDialogOpen(false);
      setMachineToDelete(null);
    } catch (e: unknown) {
      const err = e as any;
      setError(err?.response?.data?.detail || err?.message || 'Failed to delete machine');
    } finally {
      setDeleting(false);
    }
  };

  if (loading) {
    return (
      <Box sx={{ textAlign: 'center', py: 8 }}>
        <CircularProgress size={48} />
        <Typography variant="body1" sx={{ mt: 2 }}>
          Loading machines...
        </Typography>
      </Box>
    );
  }

  return (
    <Box>
      <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: 3 }}>
        <Box>
          <Typography variant="h5" sx={{ fontWeight: 600, mb: 0.5 }}>
            Machine Profiles
          </Typography>
          <Typography variant="body2">
            {filteredMachines.length} of {machines.length} machines
          </Typography>
        </Box>
        <Button variant="outlined" startIcon={<RefreshIcon />} onClick={handleRefresh}>
          Refresh
        </Button>
      </Stack>

      <Stack direction="row" spacing={1} sx={{ mb: 2, flexWrap: 'wrap' }}>
        <Chip label={`Profile Needed: ${stageSummary.profileNeeded}`} size="small" variant="outlined" />
        <Chip label={`Profile Ready: ${stageSummary.profileReady}`} size="small" color="info" variant="outlined" />
        <Chip label={`Seed Ready: ${stageSummary.seedReady}`} size="small" color="primary" variant="outlined" />
        <Chip
          label={`Model/Training: ${stageSummary.trainingOrModelReady}`}
          size="small"
          color="warning"
          variant="outlined"
        />
        <Chip label={`Synthetic Ready: ${stageSummary.syntheticReady}`} size="small" color="success" variant="outlined" />
        {stageSummary.failed > 0 && <Chip label={`Failed: ${stageSummary.failed}`} size="small" color="error" />}
      </Stack>

      <TextField
        fullWidth
        placeholder="Search by ID, type, manufacturer, or model..."
        value={searchQuery}
        onChange={(e) => setSearchQuery(e.target.value)}
        InputProps={{
          startAdornment: (
            <InputAdornment position="start">
              <SearchIcon />
            </InputAdornment>
          ),
        }}
        sx={{ mb: 3 }}
      />

      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      <TableContainer component={Paper}>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell sx={{ fontWeight: 600 }}>Machine ID</TableCell>
              <TableCell sx={{ fontWeight: 600 }}>Type</TableCell>
              <TableCell sx={{ fontWeight: 600 }}>Manufacturer</TableCell>
              <TableCell sx={{ fontWeight: 600 }}>Model</TableCell>
              <TableCell sx={{ fontWeight: 600 }}>Stage</TableCell>
              <TableCell sx={{ fontWeight: 600 }}>Next Step</TableCell>
              <TableCell align="center" sx={{ fontWeight: 600 }}>
                Actions
              </TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {filteredMachines.length === 0 ? (
              <TableRow>
                <TableCell colSpan={7} align="center" sx={{ py: 8 }}>
                  <Typography variant="body1">
                    {searchQuery ? 'No machines match your search.' : 'No machines found.'}
                  </Typography>
                </TableCell>
              </TableRow>
            ) : (
              filteredMachines.map((m) => {
                const stage = getStageInfo(m);
                return (
                  <TableRow key={m.machine_id} hover>
                    <TableCell sx={{ fontWeight: 500 }}>{m.machine_id}</TableCell>
                    <TableCell>{m.machine_type}</TableCell>
                    <TableCell>{m.manufacturer}</TableCell>
                    <TableCell>{m.model}</TableCell>
                    <TableCell>
                      <Tooltip title={getStatusTooltip(m)}>
                        <Chip size="small" color={stage.color} label={stage.label} variant="outlined" />
                      </Tooltip>
                    </TableCell>
                    <TableCell>
                      <Typography variant="body2" color="text.secondary">
                        {stage.next}
                      </Typography>
                    </TableCell>
                    <TableCell align="center">
                      <Stack direction="row" spacing={1} justifyContent="center">
                        <Tooltip title="View workflow">
                          <IconButton size="small" onClick={() => onMachineSelect(m.machine_id)}>
                            <ViewIcon fontSize="small" />
                          </IconButton>
                        </Tooltip>
                        <Tooltip title="Start workflow">
                          <IconButton size="small" onClick={() => onMachineSelect(m.machine_id)}>
                            <PlayIcon fontSize="small" />
                          </IconButton>
                        </Tooltip>
                        <Tooltip title="Delete machine">
                          <IconButton size="small" onClick={() => handleDeleteClick(m.machine_id)}>
                            <DeleteIcon fontSize="small" />
                          </IconButton>
                        </Tooltip>
                      </Stack>
                    </TableCell>
                  </TableRow>
                );
              })
            )}
          </TableBody>
        </Table>
      </TableContainer>

      <Dialog open={deleteDialogOpen} onClose={() => !deleting && setDeleteDialogOpen(false)}>
        <DialogTitle>Delete Machine Profile?</DialogTitle>
        <DialogContent>
          <Typography>
            Are you sure you want to delete machine <strong>{machineToDelete}</strong>? This will remove all associated data
            including seed data, trained models, and generated datasets. This action cannot be undone.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeleteDialogOpen(false)} disabled={deleting}>
            Cancel
          </Button>
          <Button
            onClick={handleDeleteConfirm}
            color="error"
            variant="contained"
            disabled={deleting}
            startIcon={deleting ? <CircularProgress size={16} /> : <DeleteIcon />}
          >
            {deleting ? 'Deleting...' : 'Delete'}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}
