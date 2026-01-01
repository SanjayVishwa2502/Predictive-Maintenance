/**
 * GAN Workflow Stepper - Phase 3.7.6.3
 *
 * Implements the 4-step workflow:
 * 1) Seed generation
 * 2) TVAE training (async via Celery task polling)
 * 3) Synthetic generation
 * 4) Validation
 */

import { useEffect, useMemo, useRef, useState } from 'react';
import {
  Box,
  Button,
  Divider,
  Paper,
  Stack,
  Step,
  StepContent,
  StepLabel,
  Stepper,
  TextField,
  Typography,
  LinearProgress,
  Alert,
  Chip,
} from '@mui/material';
import {
  PlayArrow as PlayIcon,
  Refresh as RefreshIcon,
} from '@mui/icons-material';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
} from 'recharts';

import TrainingProgressChart from './TrainingProgressChart';

import { ganApi } from '../../api/ganApi';
import { useTaskSession } from '../../context/TaskSessionContext';
import type {
  GenerationRequest,
  GenerationResponse,
  MachineDetails,
  SeedGenerationRequest,
  SeedGenerationResponse,
  TaskStatusResponse,
  TrainingRequest,
  TrainingResponse,
  ValidateDataQualityResponse,
} from '../../types/gan.types';

interface GANWorkflowStepperProps {
  machineId: string;
  onBackToList: () => void;
}

type LossPoint = { epoch: number; loss: number };

const statusChipColor = (
  status: string
): 'default' | 'primary' | 'success' | 'warning' | 'error' => {
  switch (status) {
    case 'not_started':
      return 'default';
    case 'seed_generated':
      return 'primary';
    case 'training':
      return 'warning';
    case 'trained':
      return 'warning';
    case 'synthetic_generated':
      return 'success';
    case 'failed':
      return 'error';
    default:
      return 'default';
  }
};

function makeIllustrativeRulDecay(points = 50) {
  // Minimal, deterministic plot to satisfy the UX requirement without assuming
  // backend returns a time series. This is an illustrative decay curve.
  const maxRul = 100;
  return Array.from({ length: points }, (_, i) => {
    const t = i / Math.max(1, points - 1);
    return { t: i, rul: Math.round(maxRul * (1 - t)) };
  });
}

export default function GANWorkflowStepper({ machineId, onBackToList }: GANWorkflowStepperProps) {
  const { registerRunningTask, updateTaskFromStatus } = useTaskSession();
  const [activeStep, setActiveStep] = useState(0);

  const [machineDetails, setMachineDetails] = useState<MachineDetails | null>(null);
  const [detailsError, setDetailsError] = useState<string | null>(null);

  // Step 1: seed
  const [seedRequest, setSeedRequest] = useState<SeedGenerationRequest>({ samples: 10000 });
  const [seedRunning, setSeedRunning] = useState(false);
  const [seedResult, setSeedResult] = useState<SeedGenerationResponse | null>(null);
  const [seedError, setSeedError] = useState<string | null>(null);

  // Step 2: training
  const [trainingRequest, setTrainingRequest] = useState<TrainingRequest>({ epochs: 300, batch_size: 500 });
  const [trainingStart, setTrainingStart] = useState<TrainingResponse | null>(null);
  const [trainingStatus, setTrainingStatus] = useState<TaskStatusResponse | null>(null);
  const [trainingError, setTrainingError] = useState<string | null>(null);
  const [lossPoints, setLossPoints] = useState<LossPoint[]>([]);
  const lastLossEpochRef = useRef<number | null>(null);

  // Step 3: generation
  const [generationRequest, setGenerationRequest] = useState<GenerationRequest>({
    train_samples: 35000,
    val_samples: 7500,
    test_samples: 7500,
  });
  const [generationRunning, setGenerationRunning] = useState(false);
  const [generationResult, setGenerationResult] = useState<GenerationResponse | null>(null);
  const [generationError, setGenerationError] = useState<string | null>(null);

  // Step 4: validation
  const [validationRunning, setValidationRunning] = useState(false);
  const [validationResult, setValidationResult] = useState<ValidateDataQualityResponse | null>(null);
  const [validationError, setValidationError] = useState<string | null>(null);

  const rulDecay = useMemo(() => makeIllustrativeRulDecay(60), []);

  const hasSeedData = !!machineDetails?.status?.has_seed_data;
  const hasTrainedModel = !!machineDetails?.status?.has_trained_model;
  const hasSyntheticData = !!machineDetails?.status?.has_synthetic_data;

  const refreshMachineDetails = async () => {
    setDetailsError(null);
    try {
      const details = await ganApi.getMachineDetails(machineId);
      setMachineDetails(details);
    } catch (e: any) {
      setMachineDetails(null);
      setDetailsError(e?.response?.data?.detail || e?.message || 'Failed to load machine details');
    }
  };

  useEffect(() => {
    refreshMachineDetails();
    // Reset step-specific state on machine change
    setActiveStep(0);
    setSeedResult(null);
    setSeedError(null);
    setTrainingStart(null);
    setTrainingStatus(null);
    setTrainingError(null);
    setLossPoints([]);
    lastLossEpochRef.current = null;
    setGenerationResult(null);
    setGenerationError(null);
    setValidationResult(null);
    setValidationError(null);
  }, [machineId]);

  // Poll training task status if we have an active task
  useEffect(() => {
    if (!trainingStart?.task_id) return;

    let cancelled = false;
    let timer: number | undefined;

    const poll = async () => {
      try {
        const status = await ganApi.getTaskStatus(trainingStart.task_id);
        if (cancelled) return;

        setTrainingStatus(status);
        updateTaskFromStatus(trainingStart.task_id, status);

        const epoch = status.progress?.epoch;
        const loss = status.progress?.loss;
        if (typeof epoch === 'number' && typeof loss === 'number' && Number.isFinite(epoch) && Number.isFinite(loss)) {
          const epochInt = Math.floor(epoch);
          const lastEpoch = lastLossEpochRef.current;
          if (lastEpoch === null || epochInt > lastEpoch) {
            lastLossEpochRef.current = epochInt;
            setLossPoints((prev) => {
              const next = [...prev, { epoch: epochInt, loss }];
              // Keep memory bounded during long trainings
              return next.length > 5000 ? next.slice(next.length - 5000) : next;
            });
          } else if (epochInt === lastEpoch) {
            // Some backends report multiple loss updates per epoch; update last point.
            setLossPoints((prev) => {
              if (!prev.length) return [{ epoch: epochInt, loss }];
              const last = prev[prev.length - 1];
              if (last.epoch !== epochInt) return prev;
              const next = prev.slice(0, -1);
              next.push({ epoch: epochInt, loss });
              return next;
            });
          }
        }

        if (status.status === 'SUCCESS' || status.status === 'FAILURE') {
          return;
        }
      } catch (e: any) {
        if (!cancelled) {
          setTrainingError(e?.response?.data?.detail || e?.message || 'Failed to poll training status');
        }
      }

      timer = window.setTimeout(poll, 2500);
    };

    poll();

    return () => {
      cancelled = true;
      if (timer) window.clearTimeout(timer);
    };
  }, [trainingStart?.task_id]);

  const handleSeedRun = async () => {
    setSeedError(null);
    setSeedRunning(true);
    try {
      const result = await ganApi.generateSeed(machineId, seedRequest);
      setSeedResult(result);
      await refreshMachineDetails();
    } catch (e: any) {
      setSeedResult(null);
      setSeedError(e?.response?.data?.detail || e?.message || 'Seed generation failed');
    } finally {
      setSeedRunning(false);
    }
  };

  const handleTrainStart = async () => {
    setTrainingError(null);
    setTrainingStart(null);
    setTrainingStatus(null);
    setLossPoints([]);
    lastLossEpochRef.current = null;

    try {
      const resp = await ganApi.trainModel(machineId, trainingRequest);
      setTrainingStart(resp);
      registerRunningTask({ task_id: resp.task_id, machine_id: machineId, kind: 'gan' });
      await refreshMachineDetails();
    } catch (e: any) {
      setTrainingError(e?.response?.data?.detail || e?.message || 'Failed to start training');
    }
  };

  const handleGenerateRun = async () => {
    setGenerationError(null);
    setGenerationRunning(true);
    try {
      const result = await ganApi.generateSynthetic(machineId, generationRequest);
      setGenerationResult(result);
      await refreshMachineDetails();
    } catch (e: any) {
      setGenerationResult(null);
      setGenerationError(e?.response?.data?.detail || e?.message || 'Synthetic generation failed');
    } finally {
      setGenerationRunning(false);
    }
  };

  const handleValidateRun = async () => {
    setValidationError(null);
    setValidationRunning(true);
    try {
      const result = await ganApi.validateDataQuality(machineId);
      setValidationResult(result);
    } catch (e: any) {
      setValidationResult(null);
      setValidationError(e?.response?.data?.detail || e?.message || 'Validation failed');
    } finally {
      setValidationRunning(false);
    }
  };

  const trainingProgressPct = useMemo(() => {
    const p = trainingStatus?.progress?.progress_percent;
    return typeof p === 'number' ? Math.max(0, Math.min(100, p)) : 0;
  }, [trainingStatus?.progress?.progress_percent]);

  return (
    <Box>
      <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: 2 }}>
        <Box>
          <Typography variant="h5" sx={{ fontWeight: 700 }}>
            Workflow: {machineId}
          </Typography>
          <Stack direction="row" spacing={1} alignItems="center" sx={{ mt: 0.5, flexWrap: 'wrap' }}>
            {machineDetails?.machine_type && (
              <Typography variant="body2" sx={{ color: '#9ca3af' }}>
                {machineDetails.machine_type} • {machineDetails.manufacturer} • {machineDetails.model}
              </Typography>
            )}
            {machineDetails?.status?.status && (
              <Chip
                size="small"
                label={machineDetails.status.status}
                color={statusChipColor(machineDetails.status.status)}
                variant="outlined"
              />
            )}
          </Stack>
        </Box>
        <Stack direction="row" spacing={1}>
          <Button variant="outlined" onClick={onBackToList}>
            Back to Machines
          </Button>
          <Button
            variant="outlined"
            startIcon={<RefreshIcon />}
            onClick={refreshMachineDetails}
          >
            Refresh
          </Button>
        </Stack>
      </Stack>

      {detailsError && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {detailsError}
        </Alert>
      )}

      <Paper
        sx={{
          bgcolor: 'rgba(255, 255, 255, 0.02)',
          border: '1px solid rgba(255, 255, 255, 0.1)',
          p: 2,
        }}
      >
        <Stepper activeStep={activeStep} orientation="vertical">
          {/* STEP 1 */}
          <Step>
            <StepLabel>Seed generation</StepLabel>
            <StepContent>
              <Stack spacing={2}>
                <Typography variant="body2" sx={{ color: '#9ca3af' }}>
                  Generate temporal seed data used for TVAE training.
                </Typography>

                <Stack direction={{ xs: 'column', sm: 'row' }} spacing={2}>
                  <TextField
                    label="Samples"
                    type="number"
                    value={seedRequest.samples}
                    onChange={(e) => setSeedRequest({ samples: Number(e.target.value) })}
                    inputProps={{ min: 1000, max: 100000 }}
                    sx={{ maxWidth: 240 }}
                  />
                  <Button
                    variant="contained"
                    startIcon={<PlayIcon />}
                    onClick={handleSeedRun}
                    disabled={seedRunning}
                    sx={{ alignSelf: { xs: 'stretch', sm: 'center' } }}
                  >
                    {hasSeedData ? 'Regenerate Seed' : 'Generate Seed'}
                  </Button>
                </Stack>

                {seedRunning && <LinearProgress />}
                {seedError && <Alert severity="error">{seedError}</Alert>}
                {seedResult && (
                  <Alert severity="success">
                    Seed generated: {seedResult.samples_generated} samples • {seedResult.file_size_mb.toFixed(2)} MB • {seedResult.generation_time_seconds.toFixed(1)}s
                  </Alert>
                )}

                <Divider />

                <Typography variant="subtitle2">RUL decay (illustrative)</Typography>
                <Box sx={{ height: 220 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={rulDecay}>
                      <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
                      <XAxis dataKey="t" />
                      <YAxis />
                      <Tooltip />
                      <Line type="monotone" dataKey="rul" stroke="#667eea" dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </Box>

                <Stack direction="row" spacing={1}>
                  <Button variant="outlined" onClick={() => setActiveStep(1)} disabled={seedRunning}>
                    Next
                  </Button>
                </Stack>
              </Stack>
            </StepContent>
          </Step>

          {/* STEP 2 */}
          <Step>
            <StepLabel>TVAE training</StepLabel>
            <StepContent>
              <Stack spacing={2}>
                <Typography variant="body2" sx={{ color: '#9ca3af' }}>
                  Starts an asynchronous Celery task. Progress is polled from the server.
                </Typography>

                <Stack direction={{ xs: 'column', sm: 'row' }} spacing={2}>
                  <TextField
                    label="Epochs"
                    type="number"
                    value={trainingRequest.epochs}
                    onChange={(e) => setTrainingRequest((p) => ({ ...p, epochs: Number(e.target.value) }))}
                    inputProps={{ min: 50, max: 1000 }}
                    sx={{ maxWidth: 200 }}
                  />
                  <TextField
                    label="Batch size"
                    type="number"
                    value={trainingRequest.batch_size}
                    onChange={(e) => setTrainingRequest((p) => ({ ...p, batch_size: Number(e.target.value) }))}
                    inputProps={{ min: 100, max: 2000 }}
                    sx={{ maxWidth: 200 }}
                  />
                  <Button
                    variant="contained"
                    startIcon={<PlayIcon />}
                    onClick={handleTrainStart}
                    sx={{ alignSelf: { xs: 'stretch', sm: 'center' } }}
                  >
                    {hasTrainedModel ? 'Retrain TVAE Model' : 'Train TVAE Model'}
                  </Button>
                </Stack>

                {trainingError && <Alert severity="error">{trainingError}</Alert>}

                {trainingStart && (
                  <Alert severity="info">
                    Training started: task {trainingStart.task_id} • est. {trainingStart.estimated_time_minutes} min
                  </Alert>
                )}

                {trainingStatus?.status === 'PROGRESS' && (
                  <Box>
                    <LinearProgress variant="determinate" value={trainingProgressPct} />
                    <Typography variant="caption" sx={{ color: '#9ca3af' }}>
                      {trainingStatus.progress?.message || 'Training...'}
                    </Typography>
                  </Box>
                )}

                {trainingStatus?.status === 'SUCCESS' && (
                  <Alert severity="success">Training completed successfully.</Alert>
                )}
                {trainingStatus?.status === 'FAILURE' && (
                  <Alert severity="error">Training failed: {trainingStatus.error || 'unknown error'}</Alert>
                )}

                <Divider />

                <Typography variant="subtitle2">Loss curve</Typography>
                <TrainingProgressChart data={lossPoints} height={240} smoothingWindow={5} />

                <Stack direction="row" spacing={1}>
                  <Button
                    variant="outlined"
                    onClick={() => setActiveStep(2)}
                    disabled={Boolean(trainingStart?.task_id) && trainingStatus?.status !== 'SUCCESS' && trainingStatus?.status !== 'FAILURE'}
                  >
                    Next
                  </Button>
                </Stack>
              </Stack>
            </StepContent>
          </Step>

          {/* STEP 3 */}
          <Step>
            <StepLabel>Synthetic generation</StepLabel>
            <StepContent>
              <Stack spacing={2}>
                <Typography variant="body2" sx={{ color: '#9ca3af' }}>
                  Generates train/val/test parquet files from the trained model.
                </Typography>

                <Stack direction={{ xs: 'column', sm: 'row' }} spacing={2}>
                  <TextField
                    label="Train samples"
                    type="number"
                    value={generationRequest.train_samples}
                    onChange={(e) => setGenerationRequest((p) => ({ ...p, train_samples: Number(e.target.value) }))}
                    inputProps={{ min: 1000, max: 100000 }}
                    sx={{ maxWidth: 220 }}
                  />
                  <TextField
                    label="Val samples"
                    type="number"
                    value={generationRequest.val_samples}
                    onChange={(e) => setGenerationRequest((p) => ({ ...p, val_samples: Number(e.target.value) }))}
                    inputProps={{ min: 100, max: 20000 }}
                    sx={{ maxWidth: 220 }}
                  />
                  <TextField
                    label="Test samples"
                    type="number"
                    value={generationRequest.test_samples}
                    onChange={(e) => setGenerationRequest((p) => ({ ...p, test_samples: Number(e.target.value) }))}
                    inputProps={{ min: 100, max: 20000 }}
                    sx={{ maxWidth: 220 }}
                  />
                </Stack>

                <Button
                  variant="contained"
                  startIcon={<PlayIcon />}
                  onClick={handleGenerateRun}
                  disabled={generationRunning}
                  sx={{ alignSelf: 'flex-start' }}
                >
                  {hasSyntheticData ? 'Regenerate Synthetic Data' : 'Generate Synthetic Data'}
                </Button>

                {generationRunning && <LinearProgress />}
                {generationError && <Alert severity="error">{generationError}</Alert>}

                {generationResult && (
                  <Alert severity="success">
                    Generated in {generationResult.generation_time_seconds.toFixed(1)}s (≈{(
                      (generationResult.train_samples + generationResult.val_samples + generationResult.test_samples) /
                      Math.max(0.001, generationResult.generation_time_seconds)
                    ).toFixed(0)} samples/sec)
                  </Alert>
                )}

                {generationResult && (
                  <Box sx={{ color: '#9ca3af' }}>
                    <Typography variant="body2">Train: {generationResult.train_file}</Typography>
                    <Typography variant="body2">Val: {generationResult.val_file}</Typography>
                    <Typography variant="body2">Test: {generationResult.test_file}</Typography>
                  </Box>
                )}

                <Stack direction="row" spacing={1}>
                  <Button variant="outlined" onClick={() => setActiveStep(3)} disabled={generationRunning}>
                    Next
                  </Button>
                </Stack>
              </Stack>
            </StepContent>
          </Step>

          {/* STEP 4 */}
          <Step>
            <StepLabel>Validation</StepLabel>
            <StepContent>
              <Stack spacing={2}>
                <Typography variant="body2" sx={{ color: '#9ca3af' }}>
                  Runs basic quality checks.
                </Typography>

                <Button
                  variant="contained"
                  startIcon={<PlayIcon />}
                  onClick={handleValidateRun}
                  disabled={validationRunning}
                  sx={{ alignSelf: 'flex-start' }}
                >
                  Validate Data Quality
                </Button>

                {validationRunning && <LinearProgress />}
                {validationError && <Alert severity="error">{validationError}</Alert>}
                {validationResult && (
                  <Alert severity={validationResult.valid ? 'success' : 'warning'}>
                    {validationResult.message} • quality={validationResult.quality_score}
                  </Alert>
                )}

                {validationResult && (
                  <Box sx={{ color: '#9ca3af' }}>
                    <Typography variant="body2">Samples: {validationResult.num_samples}</Typography>
                    <Typography variant="body2">Features: {validationResult.num_features}</Typography>
                    <Typography variant="body2">Nulls: {validationResult.null_values}</Typography>
                  </Box>
                )}

                <Stack direction="row" spacing={1}>
                  <Button variant="outlined" onClick={() => setActiveStep(0)}>
                    Back to Start
                  </Button>
                </Stack>
              </Stack>
            </StepContent>
          </Step>
        </Stepper>
      </Paper>

      <Typography variant="caption" sx={{ display: 'block', mt: 1, color: '#9ca3af' }}>
        Note: The RUL decay chart is illustrative unless the backend exposes a seed preview series.
      </Typography>
    </Box>
  );
}
