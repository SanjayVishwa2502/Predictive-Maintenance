import { useCallback, useEffect, useState } from 'react';
import {
  Alert,
  Box,
  Button,
  Paper,
  Stack,
  TextField,
  Typography,
} from '@mui/material';

const STORAGE_KEY = 'pm_vlm_endpoint';

function normalizeEndpoint(raw: string): string {
  const trimmed = raw.trim();
  if (!trimmed) return '';
  // Allow user to paste host:port; default to http.
  if (!/^https?:\/\//i.test(trimmed)) return `http://${trimmed}`;
  return trimmed;
}

export default function VLMIntegrationView() {
  const [endpointInput, setEndpointInput] = useState<string>('');
  const [savedEndpoint, setSavedEndpoint] = useState<string>('');
  const [savedMsg, setSavedMsg] = useState<string | null>(null);

  useEffect(() => {
    try {
      const existing = window.localStorage.getItem(STORAGE_KEY);
      if (existing && existing.trim()) {
        setEndpointInput(existing);
        setSavedEndpoint(existing);
      }
    } catch {
      // ignore
    }
  }, []);

  const handleSave = useCallback(() => {
    const normalized = normalizeEndpoint(endpointInput);
    setSavedMsg(null);

    if (!normalized) {
      setSavedMsg('Enter a Jetson VLM endpoint (IP:port or full URL).');
      return;
    }

    try {
      window.localStorage.setItem(STORAGE_KEY, normalized);
    } catch {
      // ignore
    }

    setSavedEndpoint(normalized);
    setSavedMsg('Saved. The VLM backend will be integrated later.');
  }, [endpointInput]);

  return (
    <Box>
      <Paper
        elevation={3}
        sx={{
          p: 4,
          background: 'linear-gradient(135deg, rgba(31, 41, 55, 0.8) 0%, rgba(17, 24, 39, 0.8) 100%)',
          backdropFilter: 'blur(10px)',
          border: '1px solid rgba(255, 255, 255, 0.1)',
        }}
      >
        <Stack spacing={2}>
          <Box>
            <Typography variant="h4" sx={{ color: '#9ca3af', mb: 0.5, fontWeight: 700 }}>
              VLM Integration
            </Typography>
            <Typography variant="body1" sx={{ color: '#6b7280' }}>
              Configure the Jetson Orin Nano endpoint. Inference runs on the Nano; this dashboard will only consume the feed/results.
            </Typography>
          </Box>

          <TextField
            label="Jetson VLM Endpoint"
            value={endpointInput}
            onChange={(e) => setEndpointInput(e.target.value)}
            placeholder="192.168.1.50:8080"
            fullWidth
            helperText="Example: 192.168.1.50:8080 (we will add health-check + auth later)"
          />

          <Box sx={{ display: 'flex', gap: 1, alignItems: 'center', flexWrap: 'wrap' }}>
            <Button variant="contained" onClick={handleSave}>
              Save Connection
            </Button>
            <Typography variant="body2" sx={{ color: '#9ca3af' }}>
              Current: {savedEndpoint ? savedEndpoint : 'Not configured'}
            </Typography>
          </Box>

          {savedMsg && <Alert severity="info">{savedMsg}</Alert>}

          <Alert severity="warning">
            VLM service is being built in a separate system. This page is a placeholder connection setup only.
          </Alert>
        </Stack>
      </Paper>
    </Box>
  );
}
