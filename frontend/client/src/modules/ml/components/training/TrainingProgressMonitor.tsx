import { useEffect, useMemo, useRef, useState } from 'react';
import {
  Alert,
  Box,
  Button,
  Chip,
  LinearProgress,
  Paper,
  Typography,
} from '@mui/material';

import type { TaskStatusResponse } from '../../types/gan.types';
import { mlTrainingApi } from '../../api/mlTrainingApi';
import { useTaskSession } from '../../context/TaskSessionContext';
import { TrainingResultsDashboard } from './TrainingResultsDashboard';
import { useNotification } from '../../../../hooks/useNotification';

function isTerminal(status: TaskStatusResponse['status']): boolean {
  return status === 'SUCCESS' || status === 'FAILURE' || status === 'REVOKED';
}

function tryExtractModelFromMessage(message: string | undefined | null): string | null {
  if (!message) return null;
  const m = message.match(/\b(classification|regression|anomaly|timeseries)\b/i);
  return m ? m[1].toLowerCase() : null;
}

function fmtDuration(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return '';
  const s = Math.round(seconds);
  const mins = Math.floor(s / 60);
  const rem = s % 60;
  if (mins <= 0) return `${rem}s`;
  return `${mins}m ${rem.toString().padStart(2, '0')}s`;
}

type ExtractedMetric = { label: string; value: number };

function extractMetricsFromResult(result: any): Array<{ modelType: string; metrics: ExtractedMetric[] }> {
  const out: Array<{ modelType: string; metrics: ExtractedMetric[] }> = [];

  const pull = (modelType: string, report: any) => {
    if (!report || typeof report !== 'object') return;

    const candidates: Array<{ key: string; label: string }> = [
      { key: 'f1', label: 'F1' },
      { key: 'f1_score', label: 'F1' },
      { key: 'accuracy', label: 'Accuracy' },
      { key: 'r2', label: 'R²' },
      { key: 'r2_score', label: 'R²' },
      { key: 'rmse', label: 'RMSE' },
      { key: 'mae', label: 'MAE' },
      { key: 'mape', label: 'MAPE' },
    ];

    const direct = (key: string): number | null => {
      const v = (report as any)[key];
      return typeof v === 'number' ? v : null;
    };

    const nested = (key: string): number | null => {
      const paths = [
        ['metrics', key],
        ['summary', key],
        ['best_model', key],
      ];
      for (const p of paths) {
        let cur: any = report;
        for (const seg of p) cur = cur && typeof cur === 'object' ? cur[seg] : undefined;
        if (typeof cur === 'number') return cur;
      }
      return null;
    };

    const metrics: ExtractedMetric[] = [];
    for (const c of candidates) {
      const v = direct(c.key) ?? nested(c.key);
      if (typeof v === 'number' && Number.isFinite(v)) {
        metrics.push({ label: c.label, value: v });
      }
    }

    if (metrics.length > 0) {
      // Keep stable order / no duplicates
      const seen = new Set<string>();
      const uniq = metrics.filter((m) => (seen.has(m.label) ? false : (seen.add(m.label), true)));
      out.push({ modelType, metrics: uniq });
    }
  };

  if (result && typeof result === 'object') {
    if (result.results && typeof result.results === 'object') {
      for (const [modelType, entry] of Object.entries(result.results)) {
        const report = (entry as any)?.report;
        pull(String(modelType), report);
      }
    }

    // Single-model task shape
    if (typeof result.model_type === 'string') {
      pull(result.model_type, result.report);
    }
  }

  return out;
}

export function TrainingProgressMonitor({ taskId }: { taskId: string }) {
  const [status, setStatus] = useState<TaskStatusResponse | null>(null);
  const [pollError, setPollError] = useState<string | null>(null);
  const [cancelling, setCancelling] = useState(false);
  const [showLogs, setShowLogs] = useState(false);
  const [hasNotified, setHasNotified] = useState(false);

  const logsRef = useRef<HTMLDivElement | null>(null);
  const { updateTaskFromStatus } = useTaskSession();
  const { notifyMLTrained, notifyError } = useNotification();

  // Notify when training completes
  useEffect(() => {
    if (!status || hasNotified) return;
    
    if (status.status === 'SUCCESS') {
      notifyMLTrained('ML Models', undefined);
      setHasNotified(true);
    } else if (status.status === 'FAILURE') {
      notifyError('ML Training Failed', status.progress?.message || 'Training task failed');
      setHasNotified(true);
    }
  }, [status, hasNotified, notifyMLTrained, notifyError]);

  useEffect(() => {
    let mounted = true;

    const tick = async () => {
      try {
        const next = await mlTrainingApi.getTaskStatus(taskId);
        if (!mounted) return;
        setStatus(next);
        setPollError(null);
        updateTaskFromStatus(taskId, next);
        return next;
      } catch (e: any) {
        if (!mounted) return null;
        setPollError(e?.message || 'Failed to poll training status');
        return null;
      }
    };

    let interval: ReturnType<typeof setInterval> | null = null;

    (async () => {
      const first = await tick();
      if (!mounted) return;
      if (first && isTerminal(first.status)) return;

      interval = setInterval(async () => {
        const next = await tick();
        if (next && isTerminal(next.status) && interval) {
          clearInterval(interval);
          interval = null;
        }
      }, 2000);
    })();

    return () => {
      mounted = false;
      if (interval) clearInterval(interval);
    };
  }, [taskId, updateTaskFromStatus]);

  // Auto-scroll logs to bottom when updated
  useEffect(() => {
    const el = logsRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
  }, [status?.logs]);

  const progress = status?.progress?.progress_percent ?? 0;
  const message = status?.progress?.message || status?.progress?.stage || '';
  const derivedModel = tryExtractModelFromMessage(status?.progress?.message);

  const eta = useMemo(() => {
    const startedAtStr = status?.started_at;
    if (!startedAtStr) return null;

    const started = new Date(startedAtStr).getTime();
    if (!Number.isFinite(started)) return null;

    const elapsedSeconds = (Date.now() - started) / 1000;
    if (!Number.isFinite(elapsedSeconds) || elapsedSeconds <= 0) return null;

    if (progress <= 1) return null;
    const remainingSeconds = (elapsedSeconds * (100 - progress)) / progress;
    if (!Number.isFinite(remainingSeconds) || remainingSeconds < 0) return null;

    return fmtDuration(remainingSeconds);
  }, [status?.started_at, progress]);

  const extractedMetrics = useMemo(() => extractMetricsFromResult(status?.result), [status?.result]);

  const handleCancel = async () => {
    setCancelling(true);
    try {
      await mlTrainingApi.cancelTask(taskId);
    } finally {
      setCancelling(false);
    }
  };

  return (
    <Paper sx={{ p: 2, mt: 2 }}>
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 2, flexWrap: 'wrap' }}>
        <Box>
          <Typography variant="h6">Training Progress</Typography>
          <Typography variant="body2" color="text.secondary">
            Task: {taskId}
          </Typography>
        </Box>

        <Button
          variant="outlined"
          disabled={cancelling || (status ? isTerminal(status.status) : false)}
          onClick={handleCancel}
        >
          Cancel
        </Button>
      </Box>

      <Box sx={{ mt: 2 }}>
        <Typography variant="body2" color="text.secondary">
          Training: {status?.machine_id || '—'}
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Model: {derivedModel || '—'}
        </Typography>

        <LinearProgress variant="determinate" value={progress} sx={{ my: 2 }} />
        <Typography variant="body2" color="text.secondary">
          {message}
        </Typography>
        {eta && (
          <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5 }}>
            Estimated time remaining: {eta}
          </Typography>
        )}
      </Box>

      {typeof status?.logs === 'string' && status.logs.length > 0 && (
        <Box sx={{ mt: 2, display: 'flex', justifyContent: 'flex-end' }}>
          <Button variant="outlined" onClick={() => setShowLogs((v) => !v)}>
            {showLogs ? 'Hide Training Logs' : 'View Training Logs'}
          </Button>
        </Box>
      )}

      {showLogs && typeof status?.logs === 'string' && status.logs.length > 0 && (
        <Paper variant="outlined" sx={{ p: 2, mt: 2 }}>
          <Typography variant="subtitle2" sx={{ mb: 1 }}>
            Live Training Logs
          </Typography>
          <Box
            ref={logsRef}
            sx={{
              maxHeight: 300,
              overflow: 'auto',
              whiteSpace: 'pre-wrap',
              fontFamily: 'monospace',
              fontSize: '0.8rem',
              lineHeight: 1.4,
            }}
          >
            {status.logs}
          </Box>
        </Paper>
      )}

      {extractedMetrics.length > 0 && (
        <Paper variant="outlined" sx={{ p: 2, mt: 2 }}>
          <Typography variant="subtitle2" sx={{ mb: 1 }}>
            Metrics
          </Typography>
          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
            {extractedMetrics.flatMap((m) =>
              m.metrics.map((metric) => (
                <Chip
                  key={`${m.modelType}:${metric.label}`}
                  label={`${m.modelType}: ${metric.label} ${metric.value.toFixed(4)}`}
                  size="small"
                />
              ))
            )}
          </Box>
        </Paper>
      )}

      {pollError && (
        <Alert severity="warning" sx={{ mt: 2 }}>
          {pollError}
        </Alert>
      )}

      {status?.status === 'SUCCESS' && (
        <>
          <Alert severity="success" sx={{ mt: 2 }}>
            Training completed.
          </Alert>
          <TrainingResultsDashboard taskId={taskId} machineId={status.machine_id || undefined} />
        </>
      )}

      {status?.status === 'FAILURE' && (
        <Alert severity="error" sx={{ mt: 2 }}>
          Training failed: {status.error || 'Unknown error'}
        </Alert>
      )}

      {status?.status === 'REVOKED' && (
        <Alert severity="info" sx={{ mt: 2 }}>
          Training cancelled.
        </Alert>
      )}
    </Paper>
  );
}
