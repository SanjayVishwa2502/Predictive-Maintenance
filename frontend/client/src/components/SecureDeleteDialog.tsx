/**
 * SecureDeleteDialog - Password confirmation dialog for admin delete operations
 * 
 * Usage:
 * <SecureDeleteDialog
 *   open={dialogOpen}
 *   onClose={() => setDialogOpen(false)}
 *   onConfirm={handleDeleteConfirmed}
 *   title="Delete Machine"
 *   description="This will permanently delete all data for this machine."
 *   itemName={machineId}
 * />
 */

import React, { useState } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  TextField,
  Typography,
  Alert,
  CircularProgress,
  Box,
  InputAdornment,
  IconButton,
} from '@mui/material';
import { useTheme, alpha } from '@mui/material/styles';
import {
  Warning as WarningIcon,
  Visibility,
  VisibilityOff,
  LockOutlined as LockIcon,
} from '@mui/icons-material';
import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

interface SecureDeleteDialogProps {
  open: boolean;
  onClose: () => void;
  onConfirm: () => Promise<void>;
  title: string;
  description?: string;
  itemName?: string;
  confirmButtonText?: string;
}

export const SecureDeleteDialog: React.FC<SecureDeleteDialogProps> = ({
  open,
  onClose,
  onConfirm,
  title,
  description,
  itemName,
  confirmButtonText = 'Delete',
}) => {
  const theme = useTheme();
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [verifying, setVerifying] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleClose = () => {
    setPassword('');
    setError(null);
    setShowPassword(false);
    onClose();
  };

  const verifyPassword = async (pwd: string): Promise<boolean> => {
    const token = localStorage.getItem('pm_access_token');
    if (!token) {
      throw new Error('Not authenticated');
    }

    const response = await axios.post(
      `${API_BASE_URL}/api/auth/verify-password`,
      { password: pwd },
      {
        headers: {
          Authorization: `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
      }
    );

    return response.data.verified === true;
  };

  const handleConfirm = async () => {
    if (!password.trim()) {
      setError('Password is required');
      return;
    }

    setVerifying(true);
    setError(null);

    try {
      // First verify password
      const verified = await verifyPassword(password);
      if (!verified) {
        setError('Invalid password');
        setVerifying(false);
        return;
      }

      // Password verified, proceed with delete
      await onConfirm();
      handleClose();
    } catch (err: any) {
      if (err?.response?.status === 401) {
        setError('Invalid password');
      } else {
        setError(err?.response?.data?.detail || err?.message || 'Verification failed');
      }
    } finally {
      setVerifying(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && password.trim() && !verifying) {
      handleConfirm();
    }
  };

  return (
    <Dialog
      open={open}
      onClose={handleClose}
      maxWidth="sm"
      fullWidth
      PaperProps={{
        sx: {
          bgcolor: 'background.paper',
          border: 1,
          borderColor: alpha(theme.palette.error.main, theme.palette.mode === 'dark' ? 0.35 : 0.5),
        },
      }}
    >
      <DialogTitle sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
        <WarningIcon sx={{ color: 'error.main' }} />
        <Typography variant="h6" component="span" sx={{ color: 'text.primary' }}>
          {title}
        </Typography>
      </DialogTitle>

      <DialogContent>
        <Box sx={{ mb: 2 }}>
          {description && (
            <Typography variant="body2" sx={{ color: 'text.secondary', mb: 1 }}>
              {description}
            </Typography>
          )}
          {itemName && (
            <Typography variant="body1" sx={{ color: 'text.primary', fontWeight: 600, mb: 2 }}>
              Target: <span style={{ color: theme.palette.error.main }}>{itemName}</span>
            </Typography>
          )}
          <Alert severity="warning" sx={{ mb: 2 }}>
            This action cannot be undone. Please enter your password to confirm.
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
          error={!!error}
          helperText={error}
          autoFocus
          InputProps={{
            startAdornment: (
              <InputAdornment position="start">
                <LockIcon color="action" />
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
        <Button onClick={handleClose} disabled={verifying} variant="outlined">
          Cancel
        </Button>
        <Button
          onClick={handleConfirm}
          disabled={!password.trim() || verifying}
          variant="contained"
          color="error"
          startIcon={verifying ? <CircularProgress size={16} color="inherit" /> : undefined}
        >
          {verifying ? 'Verifying...' : confirmButtonText}
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default SecureDeleteDialog;
