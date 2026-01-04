import { useEffect, useMemo, useState, useRef } from 'react';
import {
  Alert,
  Box,
  Button,
  Chip,
  Divider,
  LinearProgress,
  Paper,
  Stack,
  TextField,
  Typography,
} from '@mui/material';

import { ganApi } from '../../api/ganApi';
import { mlTrainingApi } from '../../api/mlTrainingApi';
import { useTaskSession } from '../../context/TaskSessionContext';
import { useDashboard } from '../../context/DashboardContext';
import { useNotification } from '../../../../hooks/useNotification';

function formatEta(ms: number): string {
  if (!Number.isFinite(ms) || ms <= 0) return '—';
  const totalSeconds = Math.max(0, Math.round(ms / 1000));
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}m ${String(seconds).padStart(2, '0')}s`;
}

function computeEtaMs(progressPercent: number | undefined, startedAt: number): number | null {
  if (typeof progressPercent !== 'number') return null;
  if (!Number.isFinite(progressPercent) || progressPercent <= 0) return null;
  if (progressPercent >= 100) return 0;

  const elapsed = Date.now() - startedAt;
  const remaining = elapsed * ((100 - progressPercent) / progressPercent);
  return Number.isFinite(remaining) ? Math.max(0, remaining) : null;
}

export default function TaskMonitor() {
  const { runningTasks, completedTasks, updateTaskFromStatus, focusTask } = useTaskSession();
  const { setSelectedView, setSelectedMachineId } = useDashboard();
  const [error, setError] = useState<string | null>(null);
  const [cancelling, setCancelling] = useState<Record<string, boolean>>({});
  const { notifyGANTrained, notifyMLTrained, notifyError } = useNotification();
  
  // Track which tasks we've already notified about
  const notifiedTasksRef = useRef<Set<string>>(new Set());

  const allTasks = useMemo(() => {
    const byId = new Map<string, any>();
    for (const t of runningTasks) byId.set(t.task_id, t);
    for (const t of completedTasks) if (!byId.has(t.task_id)) byId.set(t.task_id, t);
    return Array.from(byId.values());
  }, [runningTasks, completedTasks]);

  const pendingTasks = useMemo(() => allTasks.filter((t) => t.status === 'PENDING'), [allTasks]);
  const activeTasks = useMemo(() => allTasks.filter((t) => t.status === 'RUNNING' || t.status === 'PENDING'), [allTasks]);
  const doneTasks = useMemo(() => allTasks.filter((t) => t.status === 'SUCCESS'), [allTasks]);
  const failedTasks = useMemo(() => allTasks.filter((t) => t.status === 'FAILURE'), [allTasks]);

  // Notify when tasks complete
  useEffect(() => {
    for (const task of allTasks) {
      if (notifiedTasksRef.current.has(task.task_id)) continue;
      
      if (task.status === 'SUCCESS') {
        notifiedTasksRef.current.add(task.task_id);
        if (task.kind === 'gan') {
          notifyGANTrained(task.machine_id || 'Machine', task.progress_percent || 100);
        } else if (task.kind === 'ml_train') {
          notifyMLTrained('ML Models', undefined);
        }
      } else if (task.status === 'FAILURE') {
        notifiedTasksRef.current.add(task.task_id);
        notifyError(
          `${task.kind.toUpperCase()} Task Failed`,
          task.message || `Task ${task.task_id} failed`
        );
      }
    }
  }, [allTasks, notifyGANTrained, notifyMLTrained, notifyError]);

  const refresh = async () => {
    setError(null);
    const tasksToPoll = activeTasks;
    if (tasksToPoll.length === 0) return;

    await Promise.all(
      tasksToPoll.map(async (t) => {
        try {
          const status = t.kind === 'ml_train'
            ? await mlTrainingApi.getTaskStatus(t.task_id)
            : await ganApi.getTaskStatus(t.task_id);
          updateTaskFromStatus(t.task_id, status);
        } catch (e: any) {
          // Preserve first error but keep polling the rest.
          setError((prev) => prev ?? (e?.response?.data?.detail || e?.message || 'Failed to refresh task status'));
        }
      })
    );
  };

  // Auto-refresh every 2 seconds
  useEffect(() => {
    if (activeTasks.length === 0) return;

    let cancelled = false;
    const tick = async () => {
      if (cancelled) return;
      await refresh();
    };

    tick();
    const timer = window.setInterval(tick, 2000);

    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [activeTasks.map((t) => t.task_id).join('|')]);

  const handleCancel = async (taskId: string) => {
    setError(null);
    setCancelling((prev) => ({ ...prev, [taskId]: true }));
    try {
      const task = activeTasks.find((t) => t.task_id === taskId) || allTasks.find((t) => t.task_id === taskId);
      if (task?.kind === 'ml_train') {
        await mlTrainingApi.cancelTask(taskId);
      } else {
        await ganApi.cancelTask(taskId);
      }
      // Task may not stop immediately; we rely on polling to reflect REVOKED/FAILURE.
      await refresh();
    } catch (e: any) {
      setError(e?.response?.data?.detail || e?.message || 'Failed to cancel task');
    } finally {
      setCancelling((prev) => ({ ...prev, [taskId]: false }));
    }
  };

  const openTask = (t: { task_id: string; machine_id: string; kind: 'gan' | 'ml_train' }) => {
    focusTask(t.task_id);
    setSelectedMachineId(t.machine_id || null);
    setSelectedView(t.kind === 'ml_train' ? 'training' : 'gan');
  };

  const TaskCard = ({ t }: { t: any }) => {
    const pct = typeof t.progress_percent === 'number' ? Math.max(0, Math.min(100, t.progress_percent)) : undefined;
    const etaMs = computeEtaMs(pct, t.started_at);
    const logsText = typeof t.logs === 'string' && t.logs.trim().length > 0 ? t.logs : '';

    return (
      <Paper
        role="button"
        tabIndex={0}
        onClick={() => openTask(t)}
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            openTask(t);
          }
        }}
        sx={{
          p: 2,
          bgcolor: 'rgba(255, 255, 255, 0.02)',
          border: '1px solid rgba(255, 255, 255, 0.08)',
          cursor: 'pointer',
          '&:hover': { borderColor: 'rgba(102, 126, 234, 0.55)' },
        }}
      >
        <Stack direction={{ xs: 'column', sm: 'row' }} justifyContent="space-between" spacing={1}>
          <Box>
            <Typography variant="body2" sx={{ color: '#e5e7eb', fontWeight: 600 }}>
              {t.kind.toUpperCase()} • {t.machine_id}
            </Typography>
            <Typography variant="caption" sx={{ color: '#9ca3af' }}>
              {t.task_id}
            </Typography>
          </Box>

          <Stack direction="row" spacing={1} alignItems="center" justifyContent={{ xs: 'flex-start', sm: 'flex-end' }}>
            <Chip
              size="small"
              label={t.status}
              sx={{
                bgcolor: 'rgba(255, 255, 255, 0.06)',
                borderColor: 'rgba(255, 255, 255, 0.08)',
                color: '#d1d5db',
              }}
              variant="outlined"
            />
            <Typography variant="body2" sx={{ color: '#9ca3af' }}>
              {typeof pct === 'number' ? `${Math.round(pct)}%` : '…'} • ETA: {etaMs === null ? '—' : formatEta(etaMs)}
            </Typography>
            {t.status === 'RUNNING' && (
              <Button
                size="small"
                variant="outlined"
                disabled={Boolean(cancelling[t.task_id])}
                onClick={(e) => {
                  e.stopPropagation();
                  handleCancel(t.task_id);
                }}
              >
                {cancelling[t.task_id] ? 'Cancelling…' : 'Cancel'}
              </Button>
            )}
          </Stack>
        </Stack>

        <Box sx={{ mt: 1 }}>
          <LinearProgress
            variant={typeof pct === 'number' ? 'determinate' : 'indeterminate'}
            value={typeof pct === 'number' ? pct : 0}
          />
        </Box>

        <Box sx={{ mt: 1 }}>
          <Typography variant="caption" sx={{ color: '#6b7280' }}>
            {t.message || '—'}
          </Typography>
        </Box>

        <Box sx={{ mt: 1 }}>
          <TextField
            fullWidth
            size="small"
            multiline
            minRows={2}
            maxRows={6}
            value={logsText}
            placeholder="Logs will appear here…"
            InputProps={{ readOnly: true }}
            onClick={(e) => e.stopPropagation()}
          />
        </Box>
      </Paper>
    );
  };

  return (
    <Box>
      <Paper
        sx={{
          p: 3,
          bgcolor: 'rgba(255, 255, 255, 0.02)',
          border: '1px solid rgba(255, 255, 255, 0.1)',
        }}
      >
        <Stack direction={{ xs: 'column', sm: 'row' }} justifyContent="space-between" spacing={2}>
          <Box>
            <Typography variant="h6" sx={{ fontWeight: 700, color: '#e5e7eb' }}>
              Task Monitor
            </Typography>
            <Typography variant="body2" sx={{ color: '#9ca3af' }}>
              Live task list (running, pending, completed, failed). Auto-refreshes every 2 seconds.
            </Typography>
          </Box>

          <Stack direction="row" spacing={1} alignItems="center">
            <Chip size="small" label={`Pending: ${pendingTasks.length}`} />
            <Chip size="small" label={`Running: ${runningTasks.length}`} />
            <Chip size="small" label={`Done: ${doneTasks.length}`} />
            <Chip size="small" label={`Failed: ${failedTasks.length}`} />
            <Button size="small" variant="outlined" onClick={refresh}>
              Refresh
            </Button>
          </Stack>
        </Stack>

        {error && (
          <Alert severity="error" sx={{ mt: 2 }}>
            {error}
          </Alert>
        )}

        <Divider sx={{ my: 2, bgcolor: 'rgba(255, 255, 255, 0.08)' }} />

        {allTasks.length === 0 && (
          <Typography variant="body2" sx={{ color: '#6b7280' }}>
            No tasks started yet.
          </Typography>
        )}

        {pendingTasks.length > 0 && (
          <Box sx={{ mb: 2 }}>
            <Typography variant="subtitle2" sx={{ color: '#d1d5db', mb: 1 }}>
              Pending
            </Typography>
            <Stack spacing={1.5}>
              {pendingTasks.map((t) => (
                <TaskCard key={t.task_id} t={t} />
              ))}
            </Stack>
          </Box>
        )}

        {runningTasks.length > 0 && (
          <Box sx={{ mb: 2 }}>
            <Typography variant="subtitle2" sx={{ color: '#d1d5db', mb: 1 }}>
              Running
            </Typography>
            <Stack spacing={1.5}>
              {runningTasks.map((t) => (
                <TaskCard key={t.task_id} t={t} />
              ))}
            </Stack>
          </Box>
        )}

        {doneTasks.length > 0 && (
          <Box sx={{ mb: failedTasks.length > 0 ? 2 : 0 }}>
            <Typography variant="subtitle2" sx={{ color: '#d1d5db', mb: 1 }}>
              Completed
            </Typography>
            <Stack spacing={1}>
              {doneTasks.map((t) => (
                <TaskCard key={t.task_id} t={t} />
              ))}
            </Stack>
          </Box>
        )}

        {failedTasks.length > 0 && (
          <Box>
            <Typography variant="subtitle2" sx={{ color: '#d1d5db', mb: 1 }}>
              Failed
            </Typography>
            <Stack spacing={1}>
              {failedTasks.map((t) => (
                <TaskCard key={t.task_id} t={t} />
              ))}
            </Stack>
          </Box>
        )}
      </Paper>
    </Box>
  );
}
