/**
 * Navigation Panel Component - Phase 3.7.6.1
 * 
 * Side navigation menu for ML Dashboard with multiple views:
 * - Predictions (existing functionality)
 * - New Machine Wizard (GAN workflow)
 * - Prediction History
 * - Reports
 * - Dataset Manager
 * - Settings
 */

import {
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Divider,
  Box,
  Typography,
  Chip,
  Stack,
  Badge,
} from '@mui/material';
import {
  Timeline as TimelineIcon,
  AutoFixHigh as AutoFixHighIcon,
  History as HistoryIcon,
  Assessment as AssessmentIcon,
  PlaylistPlay as PlaylistPlayIcon,
  Storage as StorageIcon,
  Settings as SettingsIcon,
} from '@mui/icons-material';

import { useTaskSession } from '../context/TaskSessionContext';

interface NavOption {
  id: string;
  label: string;
  icon: React.ReactNode;
  description: string;
  dividerAfter?: boolean;
}

interface NavigationPanelProps {
  selectedView: string;
  onSelectView: (view: string) => void;
}

const navOptions: NavOption[] = [
  {
    id: 'predictions',
    label: 'Predictions',
    icon: <TimelineIcon />,
    description: 'Run ML predictions on machines',
  },
  {
    id: 'gan',
    label: 'New Machine Wizard',
    icon: <AutoFixHighIcon />,
    description: 'Generate synthetic training data',
    dividerAfter: true,
  },
  {
    id: 'history',
    label: 'Prediction History',
    icon: <HistoryIcon />,
    description: 'View past predictions and trends',
  },
  {
    id: 'reports',
    label: 'Reports',
    icon: <AssessmentIcon />,
    description: 'Generate analysis reports',
  },
  {
    id: 'tasks',
    label: 'Tasks',
    icon: <PlaylistPlayIcon />,
    description: 'Running & completed tasks',
  },
  {
    id: 'datasets',
     label: 'Downloads',
    icon: <StorageIcon />,
     description: 'Download synthetic parquet splits',
    dividerAfter: true,
  },
  {
    id: 'settings',
    label: 'Settings',
    icon: <SettingsIcon />,
    description: 'Configure dashboard preferences',
  },
];

export default function NavigationPanel({ selectedView, onSelectView }: NavigationPanelProps) {
  const { runningTasks, completedTasks } = useTaskSession();
  const runningCount = runningTasks.length;

  return (
    <Box sx={{ width: 280, height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* Header */}
      <Box sx={{ p: 2, textAlign: 'center' }}>
        <Typography
          variant="h6"
          sx={{
            fontWeight: 700,
            background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
          }}
        >
          Dashboard Menu
        </Typography>
        <Typography variant="caption" sx={{ color: '#9ca3af' }}>
          Navigate between features
        </Typography>
      </Box>

      <Divider sx={{ bgcolor: 'rgba(255, 255, 255, 0.1)' }} />

      {/* Navigation List */}
      <Box sx={{ flex: 1, overflow: 'auto' }}>
        <List sx={{ pt: 2 }}>
          {navOptions.map((option) => (
            <Box key={option.id}>
              <ListItem disablePadding>
                <ListItemButton
                  selected={selectedView === option.id}
                  onClick={() => onSelectView(option.id)}
                  sx={{
                    mx: 1,
                    borderRadius: 2,
                    '&.Mui-selected': {
                      bgcolor: 'rgba(103, 126, 234, 0.15)',
                      borderLeft: '3px solid #667eea',
                      '&:hover': {
                        bgcolor: 'rgba(103, 126, 234, 0.2)',
                      },
                    },
                    '&:hover': {
                      bgcolor: 'rgba(255, 255, 255, 0.05)',
                    },
                  }}
                >
                  <ListItemIcon
                    sx={{
                      color: selectedView === option.id ? '#667eea' : '#9ca3af',
                      minWidth: 40,
                    }}
                  >
                    {option.id === 'tasks' && runningCount > 0 ? (
                      <Badge color="primary" badgeContent={runningCount} max={99}>
                        {option.icon}
                      </Badge>
                    ) : (
                      option.icon
                    )}
                  </ListItemIcon>
                  <ListItemText
                    primary={option.label}
                    secondary={option.description}
                    primaryTypographyProps={{
                      fontSize: '0.95rem',
                      fontWeight: selectedView === option.id ? 600 : 400,
                      color: selectedView === option.id ? '#e5e7eb' : '#d1d5db',
                    }}
                    secondaryTypographyProps={{
                      fontSize: '0.75rem',
                      color: '#9ca3af',
                    }}
                  />
                </ListItemButton>
              </ListItem>
              {option.dividerAfter && (
                <Divider sx={{ my: 1, mx: 2, bgcolor: 'rgba(255, 255, 255, 0.05)' }} />
              )}
            </Box>
          ))}
        </List>

        {(runningTasks.length > 0 || completedTasks.length > 0) && (
          <Box sx={{ px: 2, pb: 2 }}>
            <Divider sx={{ my: 1, bgcolor: 'rgba(255, 255, 255, 0.08)' }} />
            <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: 1 }}>
              <Typography variant="subtitle2" sx={{ color: '#d1d5db' }}>
                Tasks (session)
              </Typography>
              <Stack direction="row" spacing={1}>
                <Chip size="small" label={`Running: ${runningTasks.length}`} />
                <Chip size="small" label={`Done: ${completedTasks.length}`} />
              </Stack>
            </Stack>

            {runningTasks.slice(0, 4).map((t) => (
              <Box key={t.task_id} sx={{ mb: 1 }}>
                <Typography variant="caption" sx={{ color: '#9ca3af', display: 'block' }}>
                  {t.kind.toUpperCase()} • {t.machine_id}
                </Typography>
                <Typography variant="caption" sx={{ color: '#6b7280' }}>
                  {typeof t.progress_percent === 'number' ? `${Math.round(t.progress_percent)}%` : '…'}
                  {t.message ? ` • ${t.message}` : ''}
                </Typography>
              </Box>
            ))}

            {runningTasks.length === 0 && completedTasks.length > 0 && (
              <Typography variant="caption" sx={{ color: '#6b7280' }}>
                No running tasks.
              </Typography>
            )}

            {completedTasks.slice(0, 4).map((t) => (
              <Box key={t.task_id} sx={{ mt: 1 }}>
                <Typography variant="caption" sx={{ color: '#9ca3af', display: 'block' }}>
                  {t.kind.toUpperCase()} • {t.machine_id}
                </Typography>
                <Typography variant="caption" sx={{ color: t.status === 'SUCCESS' ? '#86efac' : '#fca5a5' }}>
                  {t.status}
                </Typography>
              </Box>
            ))}
          </Box>
        )}
      </Box>

      {/* Footer */}
      <Box sx={{ p: 2, textAlign: 'center' }}>
        <Typography variant="caption" sx={{ color: '#6b7280' }}>
          Phase 3.7.6.1 - Navigation Panel
        </Typography>
      </Box>
    </Box>
  );
}
