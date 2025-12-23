import { useEffect } from 'react';

import { ganApi } from '../api/ganApi';
import { mlTrainingApi } from '../api/mlTrainingApi';
import { useTaskSession } from '../context/TaskSessionContext';

function sleep(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Background poller that keeps session task statuses up to date.
 *
 * Why this exists:
 * - Tasks can be started from multiple views (GAN wizard, Manage Models, Training view)
 * - Without a global poller, tasks started outside TrainingProgressMonitor never transition to SUCCESS/FAILURE.
 */
export default function TaskStatusPoller({ intervalMs = 2000 }: { intervalMs?: number }) {
  const { runningTasks, updateTaskFromStatus } = useTaskSession();

  useEffect(() => {
    let mounted = true;
    let timer: ReturnType<typeof setInterval> | null = null;
    let polling = false;

    const tick = async () => {
      if (polling) return;
      polling = true;

      try {
        // Snapshot to avoid chasing state changes mid-loop.
        const snapshot = runningTasks;

        for (const t of snapshot) {
          if (!mounted) return;
          try {
            const status = t.kind === 'gan'
              ? await ganApi.getTaskStatus(t.task_id)
              : await mlTrainingApi.getTaskStatus(t.task_id);

            if (!mounted) return;
            updateTaskFromStatus(t.task_id, status);
          } catch {
            // Ignore transient polling errors; user can view details in task-specific UI.
          }

          // Small yield between tasks to keep UI responsive.
          await sleep(25);
        }
      } finally {
        polling = false;
      }
    };

    // Only poll when we actually have running tasks.
    if (runningTasks.length > 0) {
      tick();
      timer = setInterval(tick, intervalMs);
    }

    return () => {
      mounted = false;
      if (timer) clearInterval(timer);
    };
  }, [intervalMs, runningTasks, updateTaskFromStatus]);

  return null;
}
