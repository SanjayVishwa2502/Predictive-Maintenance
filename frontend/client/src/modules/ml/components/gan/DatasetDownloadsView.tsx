import { useEffect, useMemo, useState } from 'react';
import {
  Alert,
  Box,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Divider,
  LinearProgress,
  MenuItem,
  Paper,
  Stack,
  TextField,
  Typography,
  InputAdornment,
  IconButton,
  CircularProgress,
} from '@mui/material';
import { 
  Download as DownloadIcon, 
  Refresh as RefreshIcon,
  LockOutlined as LockIcon,
  Visibility,
  VisibilityOff,
  Warning as WarningIcon,
} from '@mui/icons-material';
import axios from 'axios';

import { ganApi } from '../../api/ganApi';
import type { MachineDetails, MachineListResponse } from '../../types/gan.types';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

interface DatasetDownloadsViewProps {
  userRole?: 'admin' | 'operator' | 'viewer';
}

export default function DatasetDownloadsView({ userRole }: DatasetDownloadsViewProps) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [machines, setMachines] = useState<MachineDetails[]>([]);
  const [selectedMachineId, setSelectedMachineId] = useState<string>('');

  // Password confirmation dialog state
  const [passwordDialogOpen, setPasswordDialogOpen] = useState(false);
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [verifying, setVerifying] = useState(false);
  const [passwordError, setPasswordError] = useState<string | null>(null);

  const loadMachines = async () => {
    setError(null);
    setLoading(true);
    try {
      const resp: MachineListResponse = await ganApi.getMachines();
      const details = resp.machine_details || [];
      setMachines(details);
      if (!selectedMachineId && details.length > 0) {
        setSelectedMachineId(details[0].machine_id);
      }
    } catch (e: any) {
      setError(e?.response?.data?.detail || e?.message || 'Failed to load machines');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadMachines();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const selected = useMemo(
    () => machines.find((m) => m.machine_id === selectedMachineId) || null,
    [machines, selectedMachineId]
  );

  const downloadCsvUrl = () => ganApi.getDatasetDownloadCsvUrl(selectedMachineId);

  const handleDownloadClick = () => {
    setPasswordDialogOpen(true);
    setPassword('');
    setPasswordError(null);
  };

  const handlePasswordDialogClose = () => {
    if (!verifying) {
      setPasswordDialogOpen(false);
      setPassword('');
      setPasswordError(null);
      setShowPassword(false);
    }
  };

  const verifyAndDownload = async () => {
    if (!password.trim()) {
      setPasswordError('Password is required');
      return;
    }

    setVerifying(true);
    setPasswordError(null);

    try {
      const token = localStorage.getItem('pm_access_token');
      if (!token) {
        throw new Error('Not authenticated');
      }

      const response = await axios.post(
        `${API_BASE_URL}/api/auth/verify-password`,
        { password },
        {
          headers: {
            Authorization: `Bearer ${token}`,
            'Content-Type': 'application/json',
          },
        }
      );

      if (response.data.verified === true) {
        // Password verified, trigger download
        window.open(downloadCsvUrl(), '_blank');
        handlePasswordDialogClose();
      } else {
        setPasswordError('Invalid password');
      }
    } catch (err: any) {
      if (err?.response?.status === 401) {
        setPasswordError('Invalid password');
      } else {
        setPasswordError(err?.response?.data?.detail || err?.message || 'Verification failed');
      }
    } finally {
      setVerifying(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && password.trim() && !verifying) {
      verifyAndDownload();
    }
  };

  const isAdmin = userRole === 'admin';

  return (
    <Box>
      <Paper
        elevation={3}
        sx={{
          p: 3,
          background: 'linear-gradient(135deg, rgba(31, 41, 55, 0.8) 0%, rgba(17, 24, 39, 0.8) 100%)',
          backdropFilter: 'blur(10px)',
          border: '1px solid rgba(255, 255, 255, 0.1)',
        }}
      >
        <Stack direction={{ xs: 'column', sm: 'row' }} justifyContent="space-between" spacing={2}>
          <Box>
            <Typography variant="h5" sx={{ fontWeight: 700, color: '#e5e7eb' }}>
              Downloads
            </Typography>
            <Typography variant="body2" sx={{ color: '#9ca3af' }}>
              Download GAN synthetic parquet splits (train/val/test).
            </Typography>
          </Box>

          <Button variant="outlined" startIcon={<RefreshIcon />} onClick={loadMachines} disabled={loading}>
            Refresh
          </Button>
        </Stack>

        <Divider sx={{ my: 2, bgcolor: 'rgba(255, 255, 255, 0.08)' }} />

        {loading && <LinearProgress />}
        {error && (
          <Alert severity="error" sx={{ mt: 2 }}>
            {error}
          </Alert>
        )}

        <Stack spacing={2} sx={{ mt: 2 }}>
          <TextField
            select
            label="Machine"
            value={selectedMachineId}
            onChange={(e) => setSelectedMachineId(e.target.value)}
            disabled={machines.length === 0}
          >
            {machines.map((m) => (
              <MenuItem key={m.machine_id} value={m.machine_id}>
                {m.machine_id}
              </MenuItem>
            ))}
          </TextField>

          {selected && (
            <Typography variant="body2" sx={{ color: '#9ca3af' }}>
              {selected.machine_type} • {selected.manufacturer} • {selected.model}
            </Typography>
          )}

          {isAdmin ? (
            <Stack direction={{ xs: 'column', sm: 'row' }} spacing={1}>
              <Button
                variant="contained"
                startIcon={<DownloadIcon />}
                disabled={!selectedMachineId}
                onClick={handleDownloadClick}
              >
                Download CSV (Admin)
              </Button>
            </Stack>
          ) : (
            <Alert severity="info" sx={{ mt: 1 }}>
              Only administrators can download datasets. Please contact an admin if you need access.
            </Alert>
          )}

          <Typography variant="caption" sx={{ color: '#6b7280' }}>
            Download merges train/val/test into one CSV with a `split` column.
          </Typography>
        </Stack>
      </Paper>

      {/* Password Confirmation Dialog */}
      <Dialog
        open={passwordDialogOpen}
        onClose={handlePasswordDialogClose}
        maxWidth="sm"
        fullWidth
        PaperProps={{
          sx: {
            background: 'linear-gradient(135deg, rgba(31, 41, 55, 0.95) 0%, rgba(17, 24, 39, 0.95) 100%)',
            border: '1px solid rgba(102, 126, 234, 0.3)',
          },
        }}
      >
        <DialogTitle sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
          <LockIcon sx={{ color: '#667eea' }} />
          <Typography variant="h6" component="span" sx={{ color: '#e5e7eb' }}>
            Confirm Download
          </Typography>
        </DialogTitle>

        <DialogContent>
          <Box sx={{ mb: 2 }}>
            <Typography variant="body2" sx={{ color: '#9ca3af', mb: 1 }}>
              Enter your password to download the dataset.
            </Typography>
            <Typography variant="body1" sx={{ color: '#e5e7eb', fontWeight: 600, mb: 2 }}>
              Machine: <span style={{ color: '#667eea' }}>{selectedMachineId}</span>
            </Typography>
            <Alert severity="warning" icon={<WarningIcon />} sx={{ mb: 2 }}>
              Dataset downloads are logged and restricted to administrators only.
            </Alert>
          </Box>

          <TextField
            fullWidth
            label="Enter your password"
            type={showPassword ? 'text' : 'password'}
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={verifying}
            error={!!passwordError}
            helperText={passwordError}
            autoFocus
            InputProps={{
              startAdornment: (
                <InputAdornment position="start">
                  <LockIcon sx={{ color: '#6b7280' }} />
                </InputAdornment>
              ),
              endAdornment: (
                <InputAdornment position="end">
                  <IconButton
                    onClick={() => setShowPassword(!showPassword)}
                    edge="end"
                    size="small"
                  >
                    {showPassword ? <VisibilityOff /> : <Visibility />}
                  </IconButton>
                </InputAdornment>
              ),
            }}
          />
        </DialogContent>

        <DialogActions sx={{ px: 3, pb: 2 }}>
          <Button onClick={handlePasswordDialogClose} disabled={verifying} variant="outlined">
            Cancel
          </Button>
          <Button
            onClick={verifyAndDownload}
            disabled={!password.trim() || verifying}
            variant="contained"
            startIcon={verifying ? <CircularProgress size={16} color="inherit" /> : <DownloadIcon />}
          >
            {verifying ? 'Verifying...' : 'Download'}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}
