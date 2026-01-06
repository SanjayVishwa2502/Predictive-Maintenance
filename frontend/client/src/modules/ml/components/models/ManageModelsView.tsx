import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  Alert,
  Box,
  Button,
  CircularProgress,
  Container,
  Divider,
  Paper,
  Select,
  MenuItem,
  Stack,
  TextField,
  Typography,
} from '@mui/material';
import { alpha } from '@mui/material/styles';

import { useTaskSession } from '../../context/TaskSessionContext';
import { mlTrainingApi } from '../../api/mlTrainingApi';
import type { ModelType as TrainingModelType } from '../../api/mlTrainingApi';

import { mlModelsApi } from '../../api/mlModelsApi';
import type { MachineModelInventory, ModelType, InventoryStatus } from '../../api/mlModelsApi';
import SecureDeleteDialog from '../../../../components/SecureDeleteDialog';

interface ManageModelsViewProps {
  userRole?: 'admin' | 'operator' | 'viewer';
}

const MODEL_TYPES: ModelType[] = ['classification', 'regression', 'anomaly', 'timeseries'];

function statusLabel(s: InventoryStatus): string {
  if (s === 'available') return 'Available';
  if (s === 'missing') return 'Missing';
  return 'Corrupted';
}

function canDelete(s: InventoryStatus): boolean {
  return s !== 'missing';
}

function inferCategory(m: MachineModelInventory): string {
  const explicit = (m.category || '').trim();
  if (explicit) return explicit;
  const id = (m.machine_id || '').trim();
  const first = id.split('_')[0] || '';
  return first;
}

function trainAllLabel(m: MachineModelInventory): string {
  return isFullyTrained(m.models) ? 'Retrain All Models' : 'Train All Models';
}

function isFullyTrained(models: MachineModelInventory['models'] | undefined | null): boolean {
  const m = models || ({} as any);
  return MODEL_TYPES.every((t) => (m?.[t]?.status || 'missing') === 'available');
}

export default function ManageModelsView({ userRole }: ManageModelsViewProps) {
  const { registerRunningTask } = useTaskSession();

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [inventory, setInventory] = useState<MachineModelInventory[]>([]);

  const [actionMessage, setActionMessage] = useState<string | null>(null);

  // Secure delete dialog state
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [machineToDelete, setMachineToDelete] = useState<string | null>(null);

  const [search, setSearch] = useState('');
  const [trainedFilter, setTrainedFilter] = useState<'all' | 'trained' | 'untrained'>('all');
  const [categoryFilter, setCategoryFilter] = useState<string>('all');

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    setActionMessage(null);
    try {
      const resp = await mlModelsApi.getInventory();
      setInventory(resp.machines || []);
    } catch (e: any) {
      setError(e?.message || 'Failed to load model inventory');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const categories = useMemo(() => {
    const set = new Set<string>();
    for (const m of inventory) {
      const c = inferCategory(m);
      if (c) set.add(c);
    }
    return Array.from(set).sort((a, b) => a.localeCompare(b));
  }, [inventory]);

  const rows = useMemo(() => {
    const q = search.trim().toLowerCase();

    return inventory.filter((m) => {
      if (q) {
        const hay = `${m.machine_id} ${m.display_name || ''} ${m.manufacturer || ''} ${m.model || ''} ${m.category || ''}`
          .toLowerCase();
        if (!hay.includes(q)) return false;
      }

      const trained = isFullyTrained(m.models);
      if (trainedFilter === 'trained' && !trained) return false;
      if (trainedFilter === 'untrained' && trained) return false;

      if (categoryFilter !== 'all') {
        const c = inferCategory(m);
        if (c !== categoryFilter) return false;
      }
      return true;
    });
  }, [inventory, search, trainedFilter, categoryFilter]);

  const handleRetrainAll = async (machineId: string) => {
    setActionMessage(null);
    setError(null);
    try {
      const resp = await mlTrainingApi.startTraining({
        machine_id: machineId,
        model_types: MODEL_TYPES as unknown as TrainingModelType[],
      });
      registerRunningTask({ task_id: resp.task_id, machine_id: machineId, kind: 'ml_train' });
      setActionMessage(`Started retraining all 4 models for ${machineId} (task ${resp.task_id})`);
    } catch (e: any) {
      setError(e?.message || 'Failed to start retrain');
    }
  };

  const handleDeleteAll = (machineId: string) => {
    setMachineToDelete(machineId);
    setDeleteDialogOpen(true);
  };

  const handleDeleteConfirm = async () => {
    if (!machineToDelete) return;
    setActionMessage(null);
    setError(null);
    try {
      const resp = await mlModelsApi.deleteAllModels(machineToDelete);
      const errorCount = Object.keys(resp.errors || {}).length;
      setActionMessage(
        errorCount > 0
          ? `Deleted all models for ${machineToDelete} (with ${errorCount} errors)`
          : `Deleted all models for ${machineToDelete}`
      );
      setMachineToDelete(null);
      await load();
    } catch (e: any) {
      setError(e?.message || 'Delete all failed');
      throw e; // Re-throw so SecureDeleteDialog knows it failed
    }
  };

  return (
    <Container maxWidth="xl">
      <Paper
        elevation={3}
        sx={(theme) => ({
          p: 3,
          bgcolor: alpha(theme.palette.background.paper, theme.palette.mode === 'dark' ? 0.75 : 0.95),
          backdropFilter: theme.palette.mode === 'dark' ? 'blur(10px)' : undefined,
          border: 1,
          borderColor: 'divider',
        })}
      >
        <Stack direction={{ xs: 'column', sm: 'row' }} justifyContent="space-between" spacing={2}>
          <Box>
            <Typography variant="h5" sx={{ fontWeight: 700, color: 'text.primary' }}>
              Manage Models
            </Typography>
            <Typography variant="body2" sx={{ color: 'text.secondary' }}>
              Manage trained model artifacts per machine. Retraining always runs the full 4-model system.
            </Typography>
          </Box>

          <Stack direction={{ xs: 'column', sm: 'row' }} spacing={1} alignItems={{ sm: 'center' }}>
            <TextField
              size="small"
              label="Search"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
            />

            <Select
              size="small"
              value={trainedFilter}
              onChange={(e) => setTrainedFilter(e.target.value as any)}
            >
              <MenuItem value="all">All</MenuItem>
              <MenuItem value="trained">Trained</MenuItem>
              <MenuItem value="untrained">Not trained</MenuItem>
            </Select>

            <Select
              size="small"
              value={categoryFilter}
              onChange={(e) => setCategoryFilter(String(e.target.value))}
            >
              <MenuItem value="all">All types</MenuItem>
              {categories.map((c) => (
                <MenuItem key={c} value={c}>
                  {c}
                </MenuItem>
              ))}
            </Select>

            <Button variant="outlined" onClick={load} disabled={loading}>
              Refresh
            </Button>
          </Stack>
        </Stack>

        <Divider sx={{ my: 2, bgcolor: 'rgba(255, 255, 255, 0.08)' }} />

        {loading && (
          <Stack direction="row" spacing={2} alignItems="center">
            <CircularProgress size={18} />
            <Typography variant="body2" sx={{ color: '#9ca3af' }}>
              Loading inventory…
            </Typography>
          </Stack>
        )}

        {error && (
          <Alert severity="error" sx={{ mt: 2 }}>
            {error}
          </Alert>
        )}

        {actionMessage && (
          <Alert severity="info" sx={{ mt: 2 }}>
            {actionMessage}
          </Alert>
        )}

        {!loading && !error && rows.length === 0 && (
          <Alert severity="info" sx={{ mt: 2 }}>
            No machines found.
          </Alert>
        )}

        <Stack spacing={2} sx={{ mt: 2 }}>
          {rows.map((m) => (
            <Paper
              key={m.machine_id}
              elevation={0}
              sx={{
                p: 2,
                background: 'rgba(255, 255, 255, 0.03)',
                border: '1px solid rgba(255, 255, 255, 0.08)',
              }}
            >
              <Stack direction={{ xs: 'column', sm: 'row' }} spacing={1} alignItems={{ sm: 'center' }} justifyContent="space-between">
                <Typography variant="subtitle1" sx={{ color: '#e5e7eb', fontWeight: 600 }}>
                  {m.machine_id}
                </Typography>
                <Stack direction="row" spacing={1} alignItems="center">
                  <Button size="small" variant="contained" onClick={() => handleRetrainAll(m.machine_id)}>
                    {trainAllLabel(m)}
                  </Button>
                  {userRole === 'admin' && (
                    <Button
                      size="small"
                      color="error"
                      variant="outlined"
                      onClick={() => handleDeleteAll(m.machine_id)}
                    >
                      Delete All Models
                    </Button>
                  )}
                </Stack>
              </Stack>

              {(m.display_name || m.category || m.manufacturer || m.model) && (
                <Typography variant="caption" sx={{ color: '#9ca3af' }}>
                  {(m.display_name || '').trim() || `${m.manufacturer || ''} ${m.model || ''}`.trim()}
                  {inferCategory(m) ? ` • ${inferCategory(m)}` : ''}
                  {isFullyTrained(m.models) ? ' • Trained' : ' • Not trained'}
                </Typography>
              )}

              <Stack direction={{ xs: 'column', md: 'row' }} spacing={2} sx={{ mt: 1 }}>
                {MODEL_TYPES.map((mt) => {
                  const status = m.models?.[mt]?.status || 'missing';
                  return (
                    <Box
                      key={mt}
                      sx={{
                        flex: 1,
                        p: 1.5,
                        borderRadius: 1,
                        border: '1px solid rgba(255, 255, 255, 0.08)',
                      }}
                    >
                      <Typography variant="caption" sx={{ color: '#9ca3af', display: 'block' }}>
                        {mt}
                      </Typography>
                      <Typography variant="body2" sx={{ color: '#d1d5db', fontWeight: 600 }}>
                        {statusLabel(status)}
                      </Typography>

                      <Typography variant="caption" sx={{ color: '#6b7280' }}>
                        {canDelete(status) ? 'Included in Delete All Models' : 'No artifacts to delete'}
                      </Typography>
                    </Box>
                  );
                })}
              </Stack>
            </Paper>
          ))}
        </Stack>
      </Paper>

      <SecureDeleteDialog
        open={deleteDialogOpen}
        onClose={() => setDeleteDialogOpen(false)}
        onConfirm={handleDeleteConfirm}
        title="Delete All Models"
        description="This will permanently delete all trained model artifacts for this machine."
        itemName={machineToDelete || undefined}
        confirmButtonText="Delete All Models"
      />
    </Container>
  );
}
