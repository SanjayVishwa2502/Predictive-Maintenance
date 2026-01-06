/**
 * User Management View - Admin Panel for Operator Approvals
 *
 * Allows administrators to:
 * - View pending operator registrations
 * - Approve or reject operator accounts
 * - View all users (future enhancement)
 */

import { useState, useEffect, useCallback } from 'react';
import {
  Box,
  Paper,
  Typography,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Button,
  Chip,
  Stack,
  Alert,
  CircularProgress,
  IconButton,
  Tooltip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogContentText,
  DialogActions,
  Avatar,
} from '@mui/material';
import { alpha } from '@mui/material/styles';
import {
  CheckCircle as ApproveIcon,
  Cancel as RejectIcon,
  Refresh as RefreshIcon,
  HourglassEmpty as PendingIcon,
} from '@mui/icons-material';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';
const ACCESS_TOKEN_KEY = 'pm_access_token';

function getAccessToken(): string | null {
  try {
    const token = window.localStorage.getItem(ACCESS_TOKEN_KEY);
    return token && token.trim() ? token : null;
  } catch {
    return null;
  }
}

interface PendingUser {
  id: string;
  username: string;
  email: string;
  role: string;
  is_active: boolean;
  is_approved: boolean;
  created_at: string;
  updated_at: string;
}

interface UserManagementViewProps {
  userRole?: 'admin' | 'operator' | 'viewer';
}

export default function UserManagementView({ userRole }: UserManagementViewProps) {
  const [pendingUsers, setPendingUsers] = useState<PendingUser[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);

  // Confirmation dialog state
  const [confirmDialog, setConfirmDialog] = useState<{
    open: boolean;
    userId: string;
    username: string;
    action: 'approve' | 'reject';
  } | null>(null);
  const [actionLoading, setActionLoading] = useState(false);

  const fetchPendingUsers = useCallback(async () => {
    const token = getAccessToken();
    if (!token) {
      setError('Not authenticated');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const resp = await fetch(`${API_BASE_URL}/api/auth/pending-users`, {
        method: 'GET',
        headers: {
          Authorization: `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
      });

      if (!resp.ok) {
        const data = await resp.json().catch(() => ({}));
        throw new Error(data.detail || `Failed to fetch pending users: ${resp.status}`);
      }

      const data = await resp.json();
      setPendingUsers(data.pending_users || []);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch pending users');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (userRole === 'admin') {
      fetchPendingUsers();
    }
  }, [userRole, fetchPendingUsers]);

  const handleApprovalAction = async (userId: string, approve: boolean) => {
    const token = getAccessToken();
    if (!token) {
      setError('Not authenticated');
      return;
    }

    setActionLoading(true);
    setError(null);
    setSuccessMessage(null);

    try {
      const resp = await fetch(`${API_BASE_URL}/api/auth/approve-user`, {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          user_id: userId,
          approve,
        }),
      });

      if (!resp.ok) {
        const data = await resp.json().catch(() => ({}));
        throw new Error(data.detail || `Action failed: ${resp.status}`);
      }

      const data = await resp.json();
      setSuccessMessage(data.message || (approve ? 'User approved successfully' : 'User rejected successfully'));

      // Remove user from pending list
      setPendingUsers((prev) => prev.filter((u) => u.id !== userId));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Action failed');
    } finally {
      setActionLoading(false);
      setConfirmDialog(null);
    }
  };

  const openConfirmDialog = (userId: string, username: string, action: 'approve' | 'reject') => {
    setConfirmDialog({ open: true, userId, username, action });
  };

  const closeConfirmDialog = () => {
    setConfirmDialog(null);
  };

  // Non-admin view
  if (userRole !== 'admin') {
    return (
      <Paper
        elevation={3}
        sx={(theme) => ({
          p: 4,
          textAlign: 'center',
          bgcolor: alpha(theme.palette.background.paper, theme.palette.mode === 'dark' ? 0.75 : 0.95),
          backdropFilter: theme.palette.mode === 'dark' ? 'blur(10px)' : undefined,
          border: 1,
          borderColor: 'divider',
        })}
      >
        <Typography variant="h6" sx={{ color: 'text.primary' }}>
          User Management
        </Typography>
        <Typography variant="body2" sx={{ color: 'text.secondary', mt: 1 }}>
          Only administrators can manage user accounts.
        </Typography>
      </Paper>
    );
  }

  return (
    <Paper
      elevation={3}
      sx={(theme) => ({
        p: 3,
        bgcolor: alpha(theme.palette.background.paper, theme.palette.mode === 'dark' ? 0.75 : 0.95),
        backdropFilter: theme.palette.mode === 'dark' ? 'blur(10px)' : undefined,
        border: 1,
        borderColor: 'divider',
      })}
    >
      {/* Header */}
      <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: 3 }}>
        <Box>
          <Typography variant="h5" sx={{ fontWeight: 600, color: 'text.primary' }}>
            Pending Operator Approvals
          </Typography>
          <Typography variant="body2" sx={{ color: 'text.secondary', mt: 0.5 }}>
            Review and approve operator registrations
          </Typography>
        </Box>
        <Tooltip title="Refresh pending users">
          <IconButton onClick={fetchPendingUsers} disabled={loading}>
            <RefreshIcon sx={{ color: loading ? 'text.disabled' : 'primary.main' }} />
          </IconButton>
        </Tooltip>
      </Stack>

      {/* Alerts */}
      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}
      {successMessage && (
        <Alert severity="success" sx={{ mb: 2 }} onClose={() => setSuccessMessage(null)}>
          {successMessage}
        </Alert>
      )}

      {/* Loading State */}
      {loading && (
        <Box sx={{ display: 'flex', justifyContent: 'center', py: 4 }}>
            <CircularProgress size={40} color="primary" />
        </Box>
      )}

      {/* Empty State */}
      {!loading && pendingUsers.length === 0 && (
        <Box
          sx={{
            py: 6,
            textAlign: 'center',
            background: 'rgba(0, 0, 0, 0.2)',
            borderRadius: 2,
          }}
        >
          <PendingIcon sx={{ fontSize: 48, color: '#6b7280', mb: 2 }} />
          <Typography variant="h6" sx={{ color: '#9ca3af' }}>
            No Pending Approvals
          </Typography>
          <Typography variant="body2" sx={{ color: '#6b7280', mt: 1 }}>
            All operator registrations have been reviewed.
          </Typography>
        </Box>
      )}

      {/* Pending Users Table */}
      {!loading && pendingUsers.length > 0 && (
        <TableContainer>
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell sx={{ color: '#9ca3af', fontWeight: 600 }}>User</TableCell>
                <TableCell sx={{ color: '#9ca3af', fontWeight: 600 }}>Email</TableCell>
                <TableCell sx={{ color: '#9ca3af', fontWeight: 600 }}>Role</TableCell>
                <TableCell sx={{ color: '#9ca3af', fontWeight: 600 }}>Registered</TableCell>
                <TableCell sx={{ color: '#9ca3af', fontWeight: 600 }} align="right">
                  Actions
                </TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {pendingUsers.map((user) => (
                <TableRow
                  key={user.id}
                  sx={{
                    '&:hover': { bgcolor: 'rgba(255, 255, 255, 0.05)' },
                  }}
                >
                  <TableCell>
                    <Stack direction="row" spacing={1.5} alignItems="center">
                      <Avatar
                        sx={{
                          width: 32,
                          height: 32,
                          bgcolor: '#3b82f6',
                          fontSize: '0.75rem',
                        }}
                      >
                        {user.username.slice(0, 2).toUpperCase()}
                      </Avatar>
                      <Typography variant="body2" sx={{ color: '#e5e7eb', fontWeight: 500 }}>
                        {user.username}
                      </Typography>
                    </Stack>
                  </TableCell>
                  <TableCell>
                    <Typography variant="body2" sx={{ color: '#9ca3af' }}>
                      {user.email}
                    </Typography>
                  </TableCell>
                  <TableCell>
                    <Chip
                      label="Operator"
                      size="small"
                      sx={{
                        bgcolor: 'rgba(59, 130, 246, 0.2)',
                        color: '#60a5fa',
                        fontWeight: 500,
                      }}
                    />
                  </TableCell>
                  <TableCell>
                    <Typography variant="body2" sx={{ color: '#6b7280' }}>
                      {new Date(user.created_at).toLocaleDateString()}
                    </Typography>
                  </TableCell>
                  <TableCell align="right">
                    <Stack direction="row" spacing={1} justifyContent="flex-end">
                      <Tooltip title="Approve operator">
                        <Button
                          variant="contained"
                          size="small"
                          color="success"
                          startIcon={<ApproveIcon />}
                          onClick={() => openConfirmDialog(user.id, user.username, 'approve')}
                          sx={{ textTransform: 'none' }}
                        >
                          Approve
                        </Button>
                      </Tooltip>
                      <Tooltip title="Reject registration">
                        <Button
                          variant="outlined"
                          size="small"
                          color="error"
                          startIcon={<RejectIcon />}
                          onClick={() => openConfirmDialog(user.id, user.username, 'reject')}
                          sx={{ textTransform: 'none' }}
                        >
                          Reject
                        </Button>
                      </Tooltip>
                    </Stack>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      )}

      {/* Confirmation Dialog */}
      <Dialog
        open={confirmDialog?.open || false}
        onClose={closeConfirmDialog}
        PaperProps={{
          sx: (theme) => ({
            bgcolor: alpha(theme.palette.background.paper, theme.palette.mode === 'dark' ? 0.98 : 1),
            border: 1,
            borderColor: 'divider',
            borderRadius: 2,
          }),
        }}
      >
        <DialogTitle sx={{ color: 'text.primary' }}>
          {confirmDialog?.action === 'approve' ? 'Approve Operator' : 'Reject Registration'}
        </DialogTitle>
        <DialogContent>
          <DialogContentText sx={{ color: 'text.secondary' }}>
            {confirmDialog?.action === 'approve' ? (
              <>
                Are you sure you want to approve <strong>{confirmDialog?.username}</strong> as an operator?
                They will gain full operator permissions including running predictions and training models.
              </>
            ) : (
              <>
                Are you sure you want to reject <strong>{confirmDialog?.username}</strong>'s registration?
                Their account will be deleted and they will need to register again.
              </>
            )}
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={closeConfirmDialog} disabled={actionLoading} sx={{ color: '#9ca3af' }}>
            Cancel
          </Button>
          <Button
            onClick={() =>
              confirmDialog && handleApprovalAction(confirmDialog.userId, confirmDialog.action === 'approve')
            }
            disabled={actionLoading}
            color={confirmDialog?.action === 'approve' ? 'success' : 'error'}
            variant="contained"
            startIcon={actionLoading ? <CircularProgress size={16} color="inherit" /> : null}
          >
            {confirmDialog?.action === 'approve' ? 'Approve' : 'Reject'}
          </Button>
        </DialogActions>
      </Dialog>
    </Paper>
  );
}
