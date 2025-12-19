import { Box, Chip, Divider, LinearProgress, Paper, Stack, Typography } from '@mui/material';

import { useTaskSession } from '../context/TaskSessionContext';

export default function TaskSessionView() {
  const { runningTasks, completedTasks } = useTaskSession();

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
              Tasks
            </Typography>
            <Typography variant="body2" sx={{ color: '#9ca3af' }}>
              Running and completed tasks for this session.
            </Typography>
          </Box>

          <Stack direction="row" spacing={1} alignItems="center">
            <Chip size="small" label={`Running: ${runningTasks.length}`} />
            <Chip size="small" label={`Done: ${completedTasks.length}`} />
          </Stack>
        </Stack>

        <Divider sx={{ my: 2, bgcolor: 'rgba(255, 255, 255, 0.08)' }} />

        {runningTasks.length === 0 && completedTasks.length === 0 && (
          <Typography variant="body2" sx={{ color: '#6b7280' }}>
            No tasks started yet.
          </Typography>
        )}

        {runningTasks.length > 0 && (
          <Box sx={{ mb: 2 }}>
            <Typography variant="subtitle2" sx={{ color: '#d1d5db', mb: 1 }}>
              Running
            </Typography>
            <Stack spacing={1.5}>
              {runningTasks.map((t) => (
                <Paper
                  key={t.task_id}
                  sx={{
                    p: 2,
                    bgcolor: 'rgba(255, 255, 255, 0.02)',
                    border: '1px solid rgba(255, 255, 255, 0.08)',
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
                    <Typography variant="body2" sx={{ color: '#9ca3af' }}>
                      {typeof t.progress_percent === 'number' ? `${Math.round(t.progress_percent)}%` : '…'}
                    </Typography>
                  </Stack>

                  <Box sx={{ mt: 1 }}>
                    <LinearProgress
                      variant={typeof t.progress_percent === 'number' ? 'determinate' : 'indeterminate'}
                      value={typeof t.progress_percent === 'number' ? Math.max(0, Math.min(100, t.progress_percent)) : 0}
                    />
                    {t.message && (
                      <Typography variant="caption" sx={{ color: '#6b7280' }}>
                        {t.message}
                      </Typography>
                    )}
                  </Box>
                </Paper>
              ))}
            </Stack>
          </Box>
        )}

        {completedTasks.length > 0 && (
          <Box>
            <Typography variant="subtitle2" sx={{ color: '#d1d5db', mb: 1 }}>
              Completed
            </Typography>
            <Stack spacing={1}>
              {completedTasks.map((t) => (
                <Paper
                  key={t.task_id}
                  sx={{
                    p: 2,
                    bgcolor: 'rgba(255, 255, 255, 0.02)',
                    border: '1px solid rgba(255, 255, 255, 0.08)',
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
                    <Typography
                      variant="body2"
                      sx={{ color: t.status === 'SUCCESS' ? '#86efac' : '#fca5a5', fontWeight: 600 }}
                    >
                      {t.status}
                    </Typography>
                  </Stack>
                </Paper>
              ))}
            </Stack>
          </Box>
        )}
      </Paper>
    </Box>
  );
}
