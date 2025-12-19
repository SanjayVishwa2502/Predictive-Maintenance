import { Alert, Box, Button, Stack, Typography } from '@mui/material';

import { useTaskSession } from '../context/TaskSessionContext';

export default function GlobalTaskBanner({ onViewTasks }: { onViewTasks: () => void }) {
  const { runningTasks } = useTaskSession();

  if (runningTasks.length === 0) return null;

  return (
    <Box sx={{ mb: 2 }}>
      <Alert
        severity="info"
        action={
          <Button color="inherit" size="small" onClick={onViewTasks}>
            View Tasks
          </Button>
        }
        sx={{
          bgcolor: 'rgba(103, 126, 234, 0.12)',
          border: '1px solid rgba(103, 126, 234, 0.25)',
        }}
      >
        <Stack direction="row" spacing={1} alignItems="center" sx={{ flexWrap: 'wrap' }}>
          <Typography variant="body2" sx={{ fontWeight: 600 }}>
            Background tasks running
          </Typography>
          <Typography variant="body2">• {runningTasks.length} active</Typography>
        </Stack>
      </Alert>
    </Box>
  );
}
