/**
 * Validation Results Component - Phase 3.7.6.2
 * 
 * Step 4 of Profile Upload:
 * - Display validation errors with highlighting
 * - Categorize errors by type (missing field, invalid value, etc.)
 * - Show line numbers if available
 * - Success message for valid profiles
 */

import {
  Box,
  Paper,
  Typography,
  Alert,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Chip,
  Divider,
  Stack,
} from '@mui/material';
import {
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
  Warning as WarningIcon,
} from '@mui/icons-material';
import type { ValidationError } from '../../types/gan.types';

interface ValidationResultsProps {
  errors: ValidationError[];
  isValid: boolean;
  machineId?: string;
}

export default function ValidationResults({
  errors,
  isValid,
  machineId,
}: ValidationResultsProps) {
  const blockingErrors = errors.filter((e) => (e.severity || 'error') === 'error');
  const warnings = errors.filter((e) => e.severity === 'warning');
  const infos = errors.filter((e) => e.severity === 'info');

  if (isValid && blockingErrors.length === 0) {
    return (
      <Box>
        <Alert
          severity="success"
          icon={<CheckCircleIcon />}
          sx={{
            bgcolor: 'rgba(16, 185, 129, 0.1)',
            border: '1px solid rgba(16, 185, 129, 0.3)',
          }}
        >
          <Typography variant="body1" sx={{ fontWeight: 600, mb: 0.5 }}>
            ✓ Profile Validated Successfully
          </Typography>
          <Typography variant="body2">
            Your machine profile has been validated and is ready to use.
            {machineId && (
              <>
                <br />
                <strong>Machine ID:</strong> {machineId}
              </>
            )}
          </Typography>
        </Alert>

        {(warnings.length > 0 || infos.length > 0) && (
          <Paper
            sx={{
              mt: 2,
              p: 2,
              bgcolor: 'rgba(245, 158, 11, 0.05)',
              border: '1px solid rgba(245, 158, 11, 0.2)',
            }}
          >
            <Stack direction="row" spacing={2} sx={{ mb: 2 }}>
              {warnings.length > 0 && (
                <Chip
                  icon={<WarningIcon />}
                  label={`${warnings.length} Warning${warnings.length !== 1 ? 's' : ''}`}
                  sx={{
                    bgcolor: 'rgba(245, 158, 11, 0.15)',
                    color: '#f59e0b',
                  }}
                  size="small"
                />
              )}
              {infos.length > 0 && (
                <Chip
                  label={`${infos.length} Info${infos.length !== 1 ? 's' : ''}`}
                  sx={{
                    bgcolor: 'rgba(59, 130, 246, 0.12)',
                    color: '#60a5fa',
                  }}
                  size="small"
                />
              )}
            </Stack>

            {warnings.length > 0 && (
              <>
                <Typography variant="subtitle2" sx={{ color: '#f59e0b', mb: 1 }}>
                  Warnings (Recommended):
                </Typography>
                <List dense>
                  {warnings.map((issue, index) => (
                    <ListItem
                      key={`warning-ok-${index}`}
                      sx={{
                        bgcolor: 'rgba(245, 158, 11, 0.08)',
                        borderRadius: 1,
                        mb: 0.5,
                      }}
                    >
                      <ListItemIcon sx={{ minWidth: 36 }}>
                        <WarningIcon sx={{ color: '#f59e0b', fontSize: 20 }} />
                      </ListItemIcon>
                      <ListItemText
                        primary={
                          <Typography variant="body2" sx={{ fontWeight: 600 }}>
                            {issue.field}
                          </Typography>
                        }
                        secondary={
                          <Typography variant="caption" sx={{ color: '#d1d5db' }}>
                            {issue.message}
                          </Typography>
                        }
                      />
                    </ListItem>
                  ))}
                </List>
              </>
            )}

            {infos.length > 0 && (
              <>
                {warnings.length > 0 && <Divider sx={{ my: 2 }} />}
                <Typography variant="subtitle2" sx={{ color: '#60a5fa', mb: 1 }}>
                  Info (Non-blocking):
                </Typography>
                <List dense>
                  {infos.map((issue, index) => (
                    <ListItem
                      key={`info-ok-${index}`}
                      sx={{
                        bgcolor: 'rgba(59, 130, 246, 0.06)',
                        borderRadius: 1,
                        mb: 0.5,
                      }}
                    >
                      <ListItemText
                        primary={
                          <Typography variant="body2" sx={{ fontWeight: 600 }}>
                            {issue.field}
                          </Typography>
                        }
                        secondary={
                          <Typography variant="caption" sx={{ color: '#d1d5db' }}>
                            {issue.message}
                          </Typography>
                        }
                      />
                    </ListItem>
                  ))}
                </List>
              </>
            )}
          </Paper>
        )}
      </Box>
    );
  }

  if (errors.length === 0) {
    return null; // No errors and not validated yet
  }

  // Categorize by backend-provided severity
  const criticalErrors = blockingErrors;

  return (
    <Box>
      <Typography variant="h6" gutterBottom sx={{ fontWeight: 600, color: '#ef4444' }}>
        Validation Issues Found
      </Typography>
      <Typography variant="body2" sx={{ color: '#9ca3af', mb: 2 }}>
        Please fix the following issues before proceeding:
      </Typography>

      <Paper
        sx={{
          p: 2,
          bgcolor: 'rgba(239, 68, 68, 0.05)',
          border: '1px solid rgba(239, 68, 68, 0.3)',
        }}
      >
        {/* Summary */}
        <Stack direction="row" spacing={2} sx={{ mb: 2 }}>
          <Chip
            icon={<ErrorIcon />}
            label={`${criticalErrors.length} Critical Error${
              criticalErrors.length !== 1 ? 's' : ''
            }`}
            color="error"
            size="small"
          />
          {warnings.length > 0 && (
            <Chip
              icon={<WarningIcon />}
              label={`${warnings.length} Warning${warnings.length !== 1 ? 's' : ''}`}
              sx={{
                bgcolor: 'rgba(245, 158, 11, 0.15)',
                color: '#f59e0b',
              }}
              size="small"
            />
          )}
          {infos.length > 0 && (
            <Chip
              label={`${infos.length} Info${infos.length !== 1 ? 's' : ''}`}
              sx={{
                bgcolor: 'rgba(59, 130, 246, 0.12)',
                color: '#60a5fa',
              }}
              size="small"
            />
          )}
        </Stack>

        {/* Critical Errors */}
        {criticalErrors.length > 0 && (
          <>
            <Typography variant="subtitle2" sx={{ color: '#ef4444', mb: 1 }}>
              Critical Errors (Must Fix):
            </Typography>
            <List dense>
              {criticalErrors.map((error, index) => (
                <ListItem
                  key={`critical-${index}`}
                  sx={{
                    bgcolor: 'rgba(239, 68, 68, 0.1)',
                    borderRadius: 1,
                    mb: 0.5,
                  }}
                >
                  <ListItemIcon sx={{ minWidth: 36 }}>
                    <ErrorIcon sx={{ color: '#ef4444', fontSize: 20 }} />
                  </ListItemIcon>
                  <ListItemText
                    primary={
                      <Typography variant="body2" sx={{ fontWeight: 600 }}>
                        {error.field}
                        {error.line && (
                          <Chip
                            label={`Line ${error.line}`}
                            size="small"
                            sx={{
                              ml: 1,
                              height: 18,
                              fontSize: '0.7rem',
                              bgcolor: 'rgba(239, 68, 68, 0.2)',
                            }}
                          />
                        )}
                      </Typography>
                    }
                    secondary={
                      <Typography variant="caption" sx={{ color: '#d1d5db' }}>
                        {error.message}
                      </Typography>
                    }
                  />
                </ListItem>
              ))}
            </List>
          </>
        )}

        {/* Warnings */}
        {warnings.length > 0 && (
          <>
            {criticalErrors.length > 0 && <Divider sx={{ my: 2 }} />}
            <Typography variant="subtitle2" sx={{ color: '#f59e0b', mb: 1 }}>
              Warnings (Recommended):
            </Typography>
            <List dense>
              {warnings.map((error, index) => (
                <ListItem
                  key={`warning-${index}`}
                  sx={{
                    bgcolor: 'rgba(245, 158, 11, 0.1)',
                    borderRadius: 1,
                    mb: 0.5,
                  }}
                >
                  <ListItemIcon sx={{ minWidth: 36 }}>
                    <WarningIcon sx={{ color: '#f59e0b', fontSize: 20 }} />
                  </ListItemIcon>
                  <ListItemText
                    primary={
                      <Typography variant="body2" sx={{ fontWeight: 600 }}>
                        {error.field}
                        {error.line && (
                          <Chip
                            label={`Line ${error.line}`}
                            size="small"
                            sx={{
                              ml: 1,
                              height: 18,
                              fontSize: '0.7rem',
                              bgcolor: 'rgba(245, 158, 11, 0.2)',
                            }}
                          />
                        )}
                      </Typography>
                    }
                    secondary={
                      <Typography variant="caption" sx={{ color: '#d1d5db' }}>
                        {error.message}
                      </Typography>
                    }
                  />
                </ListItem>
              ))}
            </List>
          </>
        )}

        {/* Info */}
        {infos.length > 0 && (
          <>
            {(criticalErrors.length > 0 || warnings.length > 0) && <Divider sx={{ my: 2 }} />}
            <Typography variant="subtitle2" sx={{ color: '#60a5fa', mb: 1 }}>
              Info (Non-blocking):
            </Typography>
            <List dense>
              {infos.map((issue, index) => (
                <ListItem
                  key={`info-${index}`}
                  sx={{
                    bgcolor: 'rgba(59, 130, 246, 0.06)',
                    borderRadius: 1,
                    mb: 0.5,
                  }}
                >
                  <ListItemText
                    primary={
                      <Typography variant="body2" sx={{ fontWeight: 600 }}>
                        {issue.field}
                        {issue.line && (
                          <Chip
                            label={`Line ${issue.line}`}
                            size="small"
                            sx={{
                              ml: 1,
                              height: 18,
                              fontSize: '0.7rem',
                              bgcolor: 'rgba(59, 130, 246, 0.18)',
                            }}
                          />
                        )}
                      </Typography>
                    }
                    secondary={
                      <Typography variant="caption" sx={{ color: '#d1d5db' }}>
                        {issue.message}
                      </Typography>
                    }
                  />
                </ListItem>
              ))}
            </List>
          </>
        )}
      </Paper>
    </Box>
  );
}
