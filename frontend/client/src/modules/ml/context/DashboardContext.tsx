import { createContext, useContext, useMemo, useState } from 'react';

export type DashboardView =
  | 'predictions'
  | 'gan'
  | 'training'
  | 'models'
  | 'history'
  | 'reports'
  | 'tasks'
  | 'datasets'
  | 'settings';

export type ConnectionStatus = 'connected' | 'disconnected' | 'offline';

interface DashboardContextValue {
  selectedView: DashboardView;
  setSelectedView: (view: DashboardView) => void;

  selectedMachineId: string | null;
  setSelectedMachineId: (machineId: string | null) => void;

  connectionStatus: ConnectionStatus;
  setConnectionStatus: (status: ConnectionStatus) => void;
}

const DashboardContext = createContext<DashboardContextValue | undefined>(undefined);

export function DashboardProvider({ children }: { children: React.ReactNode }) {
  const [selectedView, setSelectedView] = useState<DashboardView>('predictions');
  const [selectedMachineId, setSelectedMachineId] = useState<string | null>(null);
  const [connectionStatus, setConnectionStatus] = useState<ConnectionStatus>('connected');

  const value = useMemo(
    () => ({
      selectedView,
      setSelectedView,
      selectedMachineId,
      setSelectedMachineId,
      connectionStatus,
      setConnectionStatus,
    }),
    [selectedView, selectedMachineId, connectionStatus]
  );

  return <DashboardContext.Provider value={value}>{children}</DashboardContext.Provider>;
}

export function useDashboard() {
  const ctx = useContext(DashboardContext);
  if (!ctx) throw new Error('useDashboard must be used within DashboardProvider');
  return ctx;
}
