/**
 * GAN Wizard View - Phase 3.7.6.2
 * 
 * Main view for GAN workflow integrating:
 * - MachineProfileUpload (create new profiles)
 * - MachineList (browse existing machines)
 * - Tab-based navigation between upload and list
 */

import { useEffect, useState } from 'react';
import {
  Box,
  Tabs,
  Tab,
  Paper,
  Typography,
  Button,
  Stack,
} from '@mui/material';
import {
  Add as AddIcon,
  List as ListIcon,
  ArrowBack as ArrowBackIcon,
} from '@mui/icons-material';
import MachineProfileUpload from './MachineProfileUpload';
import MachineList from './MachineList';
import WorkflowStepper from './WorkflowStepper';
import TaskMonitor from './TaskMonitor';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel({ children, value, index }: TabPanelProps) {
  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`gan-tabpanel-${index}`}
      aria-labelledby={`gan-tab-${index}`}
    >
      {value === index && <Box sx={{ py: 3 }}>{children}</Box>}
    </div>
  );
}

interface GANWizardViewProps {
  onBack?: () => void;
  resumeState?: { machine_id: string; current_step: number } | null;
  userRole?: 'admin' | 'operator' | 'viewer';
}

export default function GANWizardView({ onBack, resumeState, userRole }: GANWizardViewProps) {
  const [currentTab, setCurrentTab] = useState(0);
  const [refreshTrigger, setRefreshTrigger] = useState(0);
  const [selectedMachineId, setSelectedMachineId] = useState<string | null>(null);
  const [resumeStep, setResumeStep] = useState<number>(0);

  const handleTabChange = (_event: React.SyntheticEvent, newValue: number) => {
    setCurrentTab(newValue);
  };

  const handleProfileCreated = (machineId: string) => {
    console.log('Profile created:', machineId);
    // Switch to machine list tab
    setCurrentTab(1);
    // Trigger refresh
    setRefreshTrigger(prev => prev + 1);
  };

  const handleMachineSelect = (machineId: string) => {
    setSelectedMachineId(machineId);
  };

  useEffect(() => {
    const mid = String(resumeState?.machine_id || '').trim();
    if (!mid) return;

    const step = typeof resumeState?.current_step === 'number' && Number.isFinite(resumeState.current_step)
      ? Math.max(0, Math.min(3, Math.floor(resumeState.current_step)))
      : 0;

    setResumeStep(step);
    setSelectedMachineId(mid);
  }, [resumeState?.machine_id, resumeState?.current_step]);

  if (selectedMachineId) {
    return (
      <WorkflowStepper
        machineId={selectedMachineId}
        initialStep={resumeState?.machine_id === selectedMachineId ? resumeStep : 0}
        onBackToList={() => setSelectedMachineId(null)}
      />
    );
  }

  return (
    <Box>
      {/* Header with Back Button */}
      <Stack direction="row" alignItems="center" spacing={2} sx={{ mb: 3 }}>
        {onBack && (
          <Button
            variant="outlined"
            startIcon={<ArrowBackIcon />}
            onClick={onBack}
            sx={{
              borderColor: 'rgba(255, 255, 255, 0.2)',
              color: '#9ca3af',
              '&:hover': {
                borderColor: '#667eea',
                bgcolor: 'rgba(102, 126, 234, 0.1)',
              },
            }}
          >
            Back to Dashboard
          </Button>
        )}
        <Box sx={{ flex: 1 }}>
          <Typography
            variant="h4"
            sx={{
              fontWeight: 700,
              mb: 1,
              background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
            }}
          >
            New Machine Wizard
          </Typography>
          <Typography variant="body1" sx={{ color: '#9ca3af' }}>
            Generate synthetic training data for machines without historical failure data
          </Typography>
        </Box>
      </Stack>

      {/* Tabs Navigation */}
      <Paper
        sx={{
          bgcolor: 'rgba(255, 255, 255, 0.02)',
          border: '1px solid rgba(255, 255, 255, 0.1)',
        }}
      >
        <Tabs
          value={currentTab}
          onChange={handleTabChange}
          sx={{
            borderBottom: 1,
            borderColor: 'rgba(255, 255, 255, 0.1)',
            '& .MuiTab-root': {
              textTransform: 'none',
              fontSize: '1rem',
              fontWeight: 500,
            },
          }}
        >
          <Tab
            icon={<AddIcon />}
            iconPosition="start"
            label="Create New Profile"
            id="gan-tab-0"
            aria-controls="gan-tabpanel-0"
          />
          <Tab
            icon={<ListIcon />}
            iconPosition="start"
            label="Existing Machines"
            id="gan-tab-1"
            aria-controls="gan-tabpanel-1"
          />
        </Tabs>

        {/* Tab Content */}
        <TabPanel value={currentTab} index={0}>
          <MachineProfileUpload onProfileCreated={handleProfileCreated} />
        </TabPanel>

        <TabPanel value={currentTab} index={1}>
          <MachineList
            key={refreshTrigger}
            onMachineSelect={handleMachineSelect}
            onRefresh={() => setRefreshTrigger(prev => prev + 1)}
            userRole={userRole}
          />
        </TabPanel>
      </Paper>

      <Box sx={{ mt: 3 }}>
        <TaskMonitor />
      </Box>
    </Box>
  );
}
