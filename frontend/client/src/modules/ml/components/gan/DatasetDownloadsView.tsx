import { useEffect, useMemo, useState } from 'react';
import {
  Alert,
  Box,
  Button,
  Divider,
  LinearProgress,
  MenuItem,
  Paper,
  Stack,
  TextField,
  Typography,
} from '@mui/material';
import { Download as DownloadIcon, Refresh as RefreshIcon } from '@mui/icons-material';

import { ganApi } from '../../api/ganApi';
import type { MachineDetails, MachineListResponse } from '../../types/gan.types';

export default function DatasetDownloadsView() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [machines, setMachines] = useState<MachineDetails[]>([]);
  const [selectedMachineId, setSelectedMachineId] = useState<string>('');

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

          <Stack direction={{ xs: 'column', sm: 'row' }} spacing={1}>
            <Button
              component="a"
              href={selectedMachineId ? downloadCsvUrl() : undefined}
              target="_blank"
              rel="noreferrer"
              variant="contained"
              startIcon={<DownloadIcon />}
              disabled={!selectedMachineId}
            >
              Download CSV
            </Button>
          </Stack>

          <Typography variant="caption" sx={{ color: '#6b7280' }}>
            Download merges train/val/test into one CSV with a `split` column.
          </Typography>
        </Stack>
      </Paper>
    </Box>
  );
}
