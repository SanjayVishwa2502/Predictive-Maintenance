import { createContext, useCallback, useContext, useMemo, useState } from 'react';
import type { TaskStatusResponse } from '../types/gan.types';

type TaskKind = 'gan' | 'ml_train';

type SessionTaskStatus = 'PENDING' | 'RUNNING' | 'SUCCESS' | 'FAILURE';

export interface SessionTask {
  task_id: string;
  machine_id: string;
  kind: TaskKind;
  status: SessionTaskStatus;
  progress_percent?: number;
  message?: string;
  logs?: string;
  started_at: number;
  completed_at?: number;
}

interface TaskSessionContextValue {
  runningTasks: SessionTask[];
  completedTasks: SessionTask[];
  focusedTaskId: string | null;
  registerRunningTask: (task: { task_id: string; machine_id: string; kind: TaskKind }) => void;
  updateTaskFromStatus: (task_id: string, status: TaskStatusResponse) => void;
  focusTask: (taskId: string | null) => void;
}

const TaskSessionContext = createContext<TaskSessionContextValue | null>(null);

export function TaskSessionProvider({ children }: { children: React.ReactNode }) {
  const [runningById, setRunningById] = useState<Record<string, SessionTask>>({});
  const [completed, setCompleted] = useState<SessionTask[]>([]);
  const [focusedTaskId, setFocusedTaskId] = useState<string | null>(null);

  const registerRunningTask = useCallback(
    (task: { task_id: string; machine_id: string; kind: TaskKind }) => {
      setRunningById((prev) => {
        if (prev[task.task_id]) return prev;
        return {
          ...prev,
          [task.task_id]: {
            task_id: task.task_id,
            machine_id: task.machine_id,
            kind: task.kind,
            status: 'PENDING',
            started_at: Date.now(),
          },
        };
      });
    },
    []
  );

  const focusTask = useCallback((taskId: string | null) => {
    setFocusedTaskId(taskId);
  }, []);

  const updateTaskFromStatus = useCallback((task_id: string, status: TaskStatusResponse) => {
    const nextProgress = status.progress?.progress_percent;
    const nextMessage = status.progress?.message || status.progress?.stage || undefined;
    const nextLogs = typeof status.logs === 'string' ? status.logs : undefined;

    const nextSessionStatus: SessionTaskStatus =
      status.status === 'SUCCESS'
        ? 'SUCCESS'
        : status.status === 'FAILURE' || status.status === 'REVOKED'
          ? 'FAILURE'
          : status.status === 'PENDING'
            ? 'PENDING'
            : 'RUNNING';

    const isTerminal = nextSessionStatus === 'SUCCESS' || nextSessionStatus === 'FAILURE';
    const terminalStatus: SessionTaskStatus | null = isTerminal ? nextSessionStatus : null;

    setRunningById((prev) => {
      const existing = prev[task_id];
      if (!existing) return prev;

      const updated: SessionTask = {
        ...existing,
        status: nextSessionStatus,
        progress_percent: typeof nextProgress === 'number' ? nextProgress : existing.progress_percent,
        message: nextMessage ?? existing.message,
        logs: nextLogs ?? existing.logs,
      };

      if (!isTerminal || !terminalStatus) {
        return { ...prev, [task_id]: updated };
      }

      const { [task_id]: _, ...rest } = prev;
      setCompleted((cPrev) => [
        {
          ...updated,
          status: terminalStatus,
          completed_at: Date.now(),
        },
        ...cPrev,
      ].slice(0, 25));

      // If the user was focused on this task, clear focus once it completes.
      setFocusedTaskId((cur) => (cur === task_id ? null : cur));

      return rest;
    });
  }, []);

  const runningTasks = useMemo(() => {
    return Object.values(runningById).sort((a, b) => b.started_at - a.started_at);
  }, [runningById]);

  const value = useMemo(
    () => ({
      runningTasks,
      completedTasks: completed,
      focusedTaskId,
      registerRunningTask,
      updateTaskFromStatus,
      focusTask,
    }),
    [runningTasks, completed, focusedTaskId, registerRunningTask, updateTaskFromStatus, focusTask]
  );

  return <TaskSessionContext.Provider value={value}>{children}</TaskSessionContext.Provider>;
}

export function useTaskSession() {
  const ctx = useContext(TaskSessionContext);
  if (!ctx) {
    throw new Error('useTaskSession must be used within TaskSessionProvider');
  }
  return ctx;
}
