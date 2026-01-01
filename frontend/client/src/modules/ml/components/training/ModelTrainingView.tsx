import { useEffect, useMemo, useState } from 'react';
import { Alert, Box, Button, Container, Paper, Typography } from '@mui/material';

import MachineSelector from '../MachineSelector';
import type { Machine } from '../MachineSelector';

import { ganApi } from '../../api/ganApi';
import { mlTrainingApi } from '../../api/mlTrainingApi';
import type { ModelType } from '../../api/mlTrainingApi';

import { useDashboard } from '../../context/DashboardContext';
import { useTaskSession } from '../../context/TaskSessionContext';
import { ALL_MODELS, ModelTypeSelector } from './ModelTypeSelector';
import { DEFAULT_TRAINING_CONFIG, TrainingConfigForm } from './TrainingConfigForm';
import { TrainingProgressMonitor } from './TrainingProgressMonitor';

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

type InventoryStatus = 'missing' | 'available' | 'corrupted';

type InventoryEntry = {
  status: InventoryStatus;
};

type ModelInventoryResponse = {
  machines: Array<{
    machine_id: string;
    models: Record<string, InventoryEntry>;
  }>;
  total: number;
};

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

function hasAnyUntrainedModel(models: Record<string, InventoryEntry> | null | undefined): boolean {
  const m = models || {};
  const keys: ModelType[] = ['classification', 'regression', 'anomaly', 'timeseries'];
  return keys.some((k) => {
    const st = m[k]?.status;
    return st !== 'available';
  });
}

function isModelAvailable(models: Record<string, InventoryEntry> | null | undefined, mt: ModelType): boolean {
  return (models || {})[mt]?.status === 'available';
}

export default function ModelTrainingView() {
  const { setSelectedView } = useDashboard();
  const { registerRunningTask, runningTasks, completedTasks, focusedTaskId } = useTaskSession();

  const [machines, setMachines] = useState<Machine[]>([]);
  const [machinesLoading, setMachinesLoading] = useState(false);
  const [machinesError, setMachinesError] = useState<string | null>(null);

  const [selectedMachineId, setSelectedMachineId] = useState<string | null>(null);
  const [selectedModels, setSelectedModels] = useState<Set<ModelType>>(new Set(ALL_MODELS));
  const [config, setConfig] = useState(DEFAULT_TRAINING_CONFIG);

  const [starting, setStarting] = useState(false);
  const [startError, setStartError] = useState<string | null>(null);
  const [activeTaskId, setActiveTaskId] = useState<string | null>(null);

  const allModelsSelected = useMemo(() => selectedModels.size === ALL_MODELS.length, [selectedModels]);
  const canStart = Boolean(selectedMachineId) && allModelsSelected && !starting;

  useEffect(() => {
    let cancelled = false;

    (async () => {
      setMachinesLoading(true);
      setMachinesError(null);
      try {
        const invResp = await fetchWithAuth(`${API_BASE}/api/ml/models/inventory`, {
          method: 'GET',
          headers: { 'Content-Type': 'application/json' },
        });
        if (!invResp.ok) {
          throw new Error(`Failed to load model inventory (HTTP ${invResp.status})`);
        }
        const inventory: ModelInventoryResponse = await invResp.json();
        const invByMachine = new Map<string, Record<string, InventoryEntry>>(
          (inventory.machines || []).map((m) => [m.machine_id, m.models || {}])
        );

        const machineList = await ganApi.getMachines();
        const withSynthetic = (machineList.machine_details || [])
          .filter((m) => Boolean(m.status?.has_synthetic_data))
          .map((m) => ({
            machine_id: m.machine_id,
            display_name: `${m.manufacturer} ${m.model}`,
            category: m.machine_type,
            manufacturer: m.manufacturer,
            model: m.model,
            sensor_count: m.num_sensors,
            has_classification_model: isModelAvailable(invByMachine.get(m.machine_id), 'classification'),
            has_regression_model: isModelAvailable(invByMachine.get(m.machine_id), 'regression'),
            has_anomaly_model: isModelAvailable(invByMachine.get(m.machine_id), 'anomaly'),
            has_timeseries_model: isModelAvailable(invByMachine.get(m.machine_id), 'timeseries'),
          })) satisfies Machine[];

        // Only show machines that still need training (i.e., at least one model is missing/corrupted).
        const needsTraining = withSynthetic.filter((m) => hasAnyUntrainedModel(invByMachine.get(m.machine_id)));

        if (!cancelled) {
          setMachines(needsTraining);
        }
      } catch (e: any) {
        if (!cancelled) {
          setMachinesError(e?.message || 'Failed to load machines');
        }
      } finally {
        if (!cancelled) {
          setMachinesLoading(false);
        }
      }
    })();

    return () => {
      cancelled = true;
    };
  }, []);

  // If the user clicks a task in the Tasks view, resume it here.
  useEffect(() => {
    if (!focusedTaskId) return;

    const all = [...runningTasks, ...completedTasks];
    const t = all.find((x) => x.task_id === focusedTaskId);
    if (!t) return;
    if (t.kind !== 'ml_train') return;

    setActiveTaskId(t.task_id);
    setSelectedMachineId(t.machine_id || null);
  }, [focusedTaskId, runningTasks, completedTasks]);

  const handleStartTraining = async () => {
    if (!selectedMachineId) return;

    setStarting(true);
    setStartError(null);

    try {
      const resp = await mlTrainingApi.startTraining({
        machine_id: selectedMachineId,
        // System constraint: train the full 4-model set as one combined system.
        model_types: [...ALL_MODELS],
        time_limit_per_model: config.timeLimitSeconds,
      });

      registerRunningTask({ task_id: resp.task_id, machine_id: selectedMachineId, kind: 'ml_train' });
      setActiveTaskId(resp.task_id);
    } catch (e: any) {
      setStartError(e?.message || 'Failed to start training');
    } finally {
      setStarting(false);
    }
  };

  if (machinesError) {
    return (
      <Container maxWidth="xl">
        <Alert severity="error">{machinesError}</Alert>
      </Container>
    );
  }

  if (!machinesLoading && machines.length === 0) {
    return (
      <Container maxWidth="xl">
        <Alert
          severity="info"
          action={
            <Button color="inherit" onClick={() => setSelectedView('gan')}>
              Open GAN Wizard
            </Button>
          }
        >
          No machines need training right now (all models already exist), or there are no machines with synthetic training data yet.
        </Alert>
      </Container>
    );
  }

  return (
    <Container maxWidth="xl">
      <Paper sx={{ p: 3 }}>
        <Typography variant="h5" sx={{ mb: 1 }}>
          Model Training
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Training runs the complete 4-model system (classification, regression, anomaly, timeseries).
        </Typography>

        <Box sx={{ mt: 3 }}>
          <MachineSelector
            machines={machines}
            selectedMachineId={selectedMachineId}
            onSelect={(id) => setSelectedMachineId(id || null)}
            loading={machinesLoading}
          />
        </Box>

        <ModelTypeSelector selected={selectedModels} onChange={setSelectedModels} />

        {!allModelsSelected && (
          <Alert severity="warning" sx={{ mt: 2 }}>
            To retrain the system correctly, all 4 models must be selected.
          </Alert>
        )}

        <TrainingConfigForm config={config} onChange={setConfig} />

        {startError && (
          <Alert severity="error" sx={{ mt: 2 }}>
            {startError}
          </Alert>
        )}

        <Alert severity="info" sx={{ mt: 2 }}>
          This trains all 4 models sequentially as one system.
        </Alert>

        <Box sx={{ mt: 3, display: 'flex', gap: 2, alignItems: 'center', flexWrap: 'wrap' }}>
          <Button variant="contained" disabled={!canStart} onClick={handleStartTraining}>
            Train All 4 Models
          </Button>
          {!selectedMachineId && (
            <Typography variant="body2" color="text.secondary">
              Select a machine to enable training.
            </Typography>
          )}
        </Box>

        {activeTaskId && (
          <TrainingProgressMonitor taskId={activeTaskId} />
        )}
      </Paper>
    </Container>
  );
}
