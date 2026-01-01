/**
 * WorkflowStepper - Phase 3.7.6.3 (3.2)
 *
 * Implements the 4-step workflow:
 * 1) Generate Seed
 * 2) Train TVAE
 * 3) Generate Synthetic
 * 4) Validate
 */

import { useEffect, useMemo, useRef, useState } from 'react';
import {
  Alert,
  Box,
  Button,
  Chip,
  Divider,
  LinearProgress,
  Paper,
  Stack,
  Step,
  StepLabel,
  Stepper,
  TextField,
  Typography,
} from '@mui/material';
import {
  PlayArrow as PlayIcon,
  Refresh as RefreshIcon,
} from '@mui/icons-material';

import CorrelationHeatmap from './CorrelationHeatmap';
import FeatureDistributionChart from './FeatureDistributionChart';
import SeedDataChart from './SeedDataChart';
import TrainingProgressChart from './TrainingProgressChart';

import { ganApi } from '../../api/ganApi';
import { useTaskSession } from '../../context/TaskSessionContext';
import type {
  MachineDetails,
  SeedGenerationRequest,
  SeedGenerationResponse,
  TaskStatusResponse,
  TrainingRequest,
  TrainingResponse,
  ValidateDataQualityResponse,
  VisualizationSummaryResponse,
} from '../../types/gan.types';

interface WorkflowStepperProps {
  machineId: string;
  onBackToList: () => void;
  initialStep?: number;
}

type LossPoint = { epoch: number; loss: number };

type SplitRatio = { train: number; val: number; test: number };

const statusChipColor = (
  status: string
): 'default' | 'primary' | 'success' | 'warning' | 'error' => {
  switch (status) {
    case 'not_started':
      return 'default';
    case 'seed_generated':
      return 'primary';
    case 'training':
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

function clampRatio(n: number) {
  if (!Number.isFinite(n)) return 0;
  return Math.max(0, Math.min(100, n));
}

function normalizeSplitRatio(split: SplitRatio): SplitRatio {
  const train = clampRatio(split.train);
  const val = clampRatio(split.val);
  const test = clampRatio(split.test);
  const sum = train + val + test;
  if (sum <= 0) return { train: 70, val: 15, test: 15 };
  return {
    train: (train / sum) * 100,
    val: (val / sum) * 100,
    test: (test / sum) * 100,
  };
}

function splitToCounts(total: number, split: SplitRatio) {
  const safeTotal = Number.isFinite(total) ? Math.max(0, Math.floor(total)) : 0;
  const norm = normalizeSplitRatio(split);
  const train = Math.round(safeTotal * (norm.train / 100));
  const val = Math.round(safeTotal * (norm.val / 100));
  const test = Math.max(0, safeTotal - train - val);
  return { train, val, test };
}

function makeIllustrativeSeedSeries(points = 80, sensorCount = 3) {
  const now = Date.now();
  const start = now - points * 1000;

  return Array.from({ length: points }, (_, i) => {
    const t = i / Math.max(1, points - 1);
    const ts = new Date(start + i * 1000).toISOString();
    const rul = Math.max(0, Math.round(100 * (1 - t)));

    const sensors: Record<string, number> = {};
    for (let s = 1; s <= sensorCount; s++) {
      sensors[`sensor_${s}`] =
        Math.sin((i / 8) * s) * (1 + t) + Math.cos((i / 15) * (s + 1)) * 0.5 + t * s;
    }

    return { timestamp: ts, rul, ...sensors };
  });
}

export default function WorkflowStepper({ machineId, onBackToList, initialStep }: WorkflowStepperProps) {
  const { registerRunningTask, updateTaskFromStatus } = useTaskSession();
  const [currentStep, setCurrentStep] = useState(0);

  const [machineDetails, setMachineDetails] = useState<MachineDetails | null>(null);
  const [detailsError, setDetailsError] = useState<string | null>(null);

  // Step 1: seed
  const [seedRequest, setSeedRequest] = useState<SeedGenerationRequest>({ samples: 10000 });
  const [seedRunning, setSeedRunning] = useState(false);
  const [seedResult, setSeedResult] = useState<SeedGenerationResponse | null>(null);
  const [seedError, setSeedError] = useState<string | null>(null);

  // Step 2: training
  const [trainingRequest, setTrainingRequest] = useState<TrainingRequest>({
    epochs: 300,
    batch_size: 500,
  });
  const [trainingStart, setTrainingStart] = useState<TrainingResponse | null>(null);
  const [trainingStatus, setTrainingStatus] = useState<TaskStatusResponse | null>(null);
  const [trainingError, setTrainingError] = useState<string | null>(null);
  const [lossPoints, setLossPoints] = useState<LossPoint[]>([]);
  const lastLossEpochRef = useRef<number | null>(null);

  // Step 3: generation
  const [totalSamples, setTotalSamples] = useState(50000);
  const [splitRatio, setSplitRatio] = useState<SplitRatio>({ train: 70, val: 15, test: 15 });
  const [generationRunning, setGenerationRunning] = useState(false);
  const [generationError, setGenerationError] = useState<string | null>(null);
  const [generationInfo, setGenerationInfo] = useState<{
    trainSamples: number;
    valSamples: number;
    testSamples: number;
    trainFile: string;
    valFile: string;
    testFile: string;
    generationTimeSeconds: number;
  } | null>(null);

  // Step 4: validation
  const [validationRunning, setValidationRunning] = useState(false);
  const [validationResult, setValidationResult] = useState<ValidateDataQualityResponse | null>(null);
  const [validationError, setValidationError] = useState<string | null>(null);

  // Phase 3.7.6.5: visualization summaries for charts
  const [vizSummary, setVizSummary] = useState<VisualizationSummaryResponse | null>(null);
  const [vizError, setVizError] = useState<string | null>(null);

  const illustrativeSeedSeries = useMemo(() => makeIllustrativeSeedSeries(90, 3), []);

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

  const refreshVisualizationSummary = async () => {
    setVizError(null);
    try {
      const summary = await ganApi.getVisualizationSummary(machineId, { points: 300, bins: 20, max_features: 8 });
      setVizSummary(summary);
    } catch (e: any) {
      setVizSummary(null);
      setVizError(e?.response?.data?.detail || e?.message || 'Failed to load visualization summary');
    }
  };

  useEffect(() => {
    refreshMachineDetails();
    setCurrentStep(typeof initialStep === 'number' && Number.isFinite(initialStep) ? Math.max(0, Math.min(3, Math.floor(initialStep))) : 0);

    setSeedResult(null);
    setSeedError(null);

    setTrainingStart(null);
    setTrainingStatus(null);
    setTrainingError(null);
    setLossPoints([]);
    lastLossEpochRef.current = null;

    setGenerationInfo(null);
    setGenerationError(null);

    setValidationResult(null);
    setValidationError(null);

    setVizSummary(null);
    setVizError(null);
  }, [machineId]);

  // Persist workflow continuation state so it survives page refresh/navigation.
  useEffect(() => {
    (async () => {
      try {
        await ganApi.setContinueWorkflow({ workflow: 'gan_profile', machine_id: machineId, current_step: currentStep });
      } catch {
        // Best-effort; don't block the workflow UI if persistence fails.
      }
    })();
  }, [machineId, currentStep]);

  // If validation succeeds, consider the workflow complete and clear the continuation marker.
  useEffect(() => {
    if (!validationResult) return;
    if (!validationResult.valid) return;
    (async () => {
      try {
        await ganApi.clearContinueWorkflow();
      } catch {
        // best-effort
      }
    })();
  }, [validationResult]);

  // Poll training task status (Celery) if we have an active task
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
              return next.length > 5000 ? next.slice(next.length - 5000) : next;
            });
          } else if (epochInt === lastEpoch) {
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

  const trainingProgressPct = useMemo(() => {
    const p = trainingStatus?.progress?.progress_percent;
    return typeof p === 'number' ? Math.max(0, Math.min(100, p)) : 0;
  }, [trainingStatus?.progress?.progress_percent]);

  const handleSeedRun = async () => {
    setSeedError(null);
    setSeedRunning(true);
    try {
      const result = await ganApi.generateSeed(machineId, seedRequest);
      setSeedResult(result);
      await refreshMachineDetails();
      await refreshVisualizationSummary();
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
      const counts = splitToCounts(totalSamples, splitRatio);
      const result = await ganApi.generateSynthetic(machineId, {
        train_samples: counts.train,
        val_samples: counts.val,
        test_samples: counts.test,
      });

      setGenerationInfo({
        trainSamples: result.train_samples,
        valSamples: result.val_samples,
        testSamples: result.test_samples,
        trainFile: result.train_file,
        valFile: result.val_file,
        testFile: result.test_file,
        generationTimeSeconds: result.generation_time_seconds,
      });

      await refreshMachineDetails();
    } catch (e: any) {
      setGenerationInfo(null);
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
      await refreshVisualizationSummary();
    } catch (e: any) {
      setValidationResult(null);
      setValidationError(e?.response?.data?.detail || e?.message || 'Validation failed');
    } finally {
      setValidationRunning(false);
    }
  };

  const SeedGenerationStep = () => (
    <Stack spacing={2}>
      <Typography variant="body2" sx={{ color: '#9ca3af' }}>
        Generate temporal seed data used for TVAE training.
      </Typography>

      <Stack direction={{ xs: 'column', sm: 'row' }} spacing={2}>
        <TextField
          label="Number of samples"
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
          {hasSeedData ? 'Regenerate Physics-Based Seed' : 'Generate Physics-Based Seed'}
        </Button>
      </Stack>

      {seedRunning && <LinearProgress />}
      {seedError && <Alert severity="error">{seedError}</Alert>}
      {seedResult && (
        <Alert severity="success">
          Output: {seedResult.file_path} • row count: {seedResult.samples_generated}
        </Alert>
      )}

      <Divider />

      <SeedDataChart
        data={vizSummary?.seed_series?.points?.length ? (vizSummary.seed_series.points as any) : (illustrativeSeedSeries as any)}
        sensorKeys={
          vizSummary?.seed_series?.sensor_keys?.length
            ? vizSummary.seed_series.sensor_keys
            : ['sensor_1', 'sensor_2', 'sensor_3']
        }
        height={260}
      />

      {vizError && <Alert severity="warning">{vizError}</Alert>}

      <Stack direction="row" spacing={1}>
        <Button variant="outlined" onClick={() => setCurrentStep(1)} disabled={seedRunning}>
          Next
        </Button>
      </Stack>
    </Stack>
  );

  const TVAETrainingStep = () => (
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
          Training started: job {trainingStart.task_id} • est. {trainingStart.estimated_time_minutes} min
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

      <TrainingProgressChart data={lossPoints} height={260} smoothingWindow={5} />

      <Stack direction="row" spacing={1}>
        <Button variant="outlined" onClick={() => setCurrentStep(0)}>
          Back
        </Button>
        <Button
          variant="outlined"
          onClick={() => setCurrentStep(2)}
          disabled={Boolean(trainingStart?.task_id) && trainingStatus?.status !== 'SUCCESS' && trainingStatus?.status !== 'FAILURE'}
        >
          Next
        </Button>
      </Stack>
    </Stack>
  );

  const SyntheticGenerationStep = () => {
    const counts = splitToCounts(totalSamples, splitRatio);
    const samplesPerSecond = generationInfo
      ? (generationInfo.trainSamples + generationInfo.valSamples + generationInfo.testSamples) /
        Math.max(0.001, generationInfo.generationTimeSeconds)
      : 0;

    return (
      <Stack spacing={2}>
        <Typography variant="body2" sx={{ color: '#9ca3af' }}>
          Generate a synthetic dataset split into train/val/test.
        </Typography>

        <Stack direction={{ xs: 'column', sm: 'row' }} spacing={2}>
          <TextField
            label="Total samples"
            type="number"
            value={totalSamples}
            onChange={(e) => setTotalSamples(Number(e.target.value))}
            inputProps={{ min: 1000, max: 200000 }}
            sx={{ maxWidth: 220 }}
          />
          <TextField
            label="Split train %"
            type="number"
            value={Math.round(splitRatio.train)}
            onChange={(e) => setSplitRatio((p) => ({ ...p, train: Number(e.target.value) }))}
            inputProps={{ min: 0, max: 100 }}
            sx={{ maxWidth: 160 }}
          />
          <TextField
            label="Val %"
            type="number"
            value={Math.round(splitRatio.val)}
            onChange={(e) => setSplitRatio((p) => ({ ...p, val: Number(e.target.value) }))}
            inputProps={{ min: 0, max: 100 }}
            sx={{ maxWidth: 140 }}
          />
          <TextField
            label="Test %"
            type="number"
            value={Math.round(splitRatio.test)}
            onChange={(e) => setSplitRatio((p) => ({ ...p, test: Number(e.target.value) }))}
            inputProps={{ min: 0, max: 100 }}
            sx={{ maxWidth: 140 }}
          />
        </Stack>

        <Typography variant="body2" sx={{ color: '#9ca3af' }}>
          Computed split: train={counts.train}, val={counts.val}, test={counts.test}
        </Typography>

        <Button
          variant="contained"
          startIcon={<PlayIcon />}
          onClick={handleGenerateRun}
          disabled={generationRunning}
          sx={{ alignSelf: 'flex-start' }}
        >
          {hasSyntheticData ? 'Regenerate Synthetic Dataset' : 'Generate Synthetic Dataset'}
        </Button>

        {generationRunning && <LinearProgress />}
        {generationError && <Alert severity="error">{generationError}</Alert>}

        {generationInfo && (
          <Alert severity="success">
            Output: 3 parquet files • {samplesPerSecond.toFixed(0)} samples/sec
          </Alert>
        )}

        {generationInfo && (
          <Box sx={{ color: '#9ca3af' }}>
            <Typography variant="body2">Train ({generationInfo.trainSamples}): {generationInfo.trainFile}</Typography>
            <Typography variant="body2">Val ({generationInfo.valSamples}): {generationInfo.valFile}</Typography>
            <Typography variant="body2">Test ({generationInfo.testSamples}): {generationInfo.testFile}</Typography>
          </Box>
        )}

        <Stack direction="row" spacing={1}>
          <Button variant="outlined" onClick={() => setCurrentStep(1)}>
            Back
          </Button>
          <Button variant="outlined" onClick={() => setCurrentStep(3)} disabled={generationRunning}>
            Next
          </Button>
        </Stack>
      </Stack>
    );
  };

  const ValidationStep = () => (
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

      {vizError && <Alert severity="warning">{vizError}</Alert>}

      {vizSummary?.distributions?.length ? (
        <>
          <Divider />
          <FeatureDistributionChart distributions={vizSummary.distributions as any} />
        </>
      ) : null}

      {vizSummary?.correlation?.features?.length && vizSummary?.correlation?.matrix?.length ? (
        <>
          <Divider />
          <CorrelationHeatmap
            features={vizSummary.correlation.features}
            matrix={vizSummary.correlation.matrix}
          />
        </>
      ) : null}

      <Stack direction="row" spacing={1}>
        <Button variant="outlined" onClick={() => setCurrentStep(2)}>
          Back
        </Button>
        <Button variant="outlined" onClick={() => setCurrentStep(0)}>
          Back to Start
        </Button>
      </Stack>
    </Stack>
  );

  const steps = [
    { label: 'Generate Seed', component: <SeedGenerationStep /> },
    { label: 'Train TVAE', component: <TVAETrainingStep /> },
    { label: 'Generate Synthetic', component: <SyntheticGenerationStep /> },
    { label: 'Validate', component: <ValidationStep /> },
  ];

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
          <Button variant="outlined" startIcon={<RefreshIcon />} onClick={refreshMachineDetails}>
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
        <Stepper activeStep={currentStep}>
          {steps.map((step) => (
            <Step key={step.label}>
              <StepLabel>{step.label}</StepLabel>
            </Step>
          ))}
        </Stepper>

        <Box sx={{ mt: 3 }}>{steps[currentStep]?.component}</Box>
      </Paper>

      <Typography variant="caption" sx={{ display: 'block', mt: 1, color: '#9ca3af' }}>
        Note: Charts may fall back to illustrative data until datasets exist.
      </Typography>
    </Box>
  );
}
