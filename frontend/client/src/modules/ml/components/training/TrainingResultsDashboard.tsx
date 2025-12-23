import { useEffect, useMemo, useState } from 'react';
import {
  Alert,
  Box,
  Button,
  ButtonGroup,
  Card,
  CardContent,
  CardHeader,
  Chip,
  Grid,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableRow,
  Tooltip,
  Typography,
} from '@mui/material';
import DownloadIcon from '@mui/icons-material/Download';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';

import { useDashboard } from '../../context/DashboardContext';
import { mlTrainingApi } from '../../api/mlTrainingApi';
import type { TrainingResults, TrainingModelResult } from '../../api/mlTrainingApi';

export interface ResultsProps {
  taskId: string;
  machineId?: string;
  modelType?: string;
}

function scoreForRanking(m: TrainingModelResult): number | null {
  const pm = m.primary_metric;
  if (!pm) return null;
  // Convert to a "higher is better" score for consistent ranking.
  return pm.higher_is_better ? pm.value : -pm.value;
}

function formatMetric(m: TrainingModelResult): string {
  const pm = m.primary_metric;
  if (!pm) return '—';
  return `${pm.label}: ${pm.value.toFixed(4)}`;
}

export function TrainingResultsDashboard({ taskId, machineId }: ResultsProps) {
  const { setSelectedView } = useDashboard();
  const [results, setResults] = useState<TrainingResults | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const r = await mlTrainingApi.getTrainingResults(taskId);
        if (!cancelled) {
          setResults(r);
          setError(null);
        }
      } catch (e: any) {
        if (!cancelled) {
          setError(e?.message || 'Failed to load training results');
        }
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [taskId]);

  const effectiveMachineId = results?.machine_id || machineId || '';

  const ranked = useMemo(() => {
    if (!results) return [];
    return [...results.models]
      .map((m) => ({ m, s: scoreForRanking(m) }))
      .sort((a, b) => (b.s ?? -Infinity) - (a.s ?? -Infinity))
      .map((x) => x.m);
  }, [results]);

  const downloadSupported = false;

  if (error) {
    return <Alert severity="warning">{error}</Alert>;
  }

  if (!results) {
    return <Alert severity="info">Loading training results…</Alert>;
  }

  return (
    <Box sx={{ mt: 3 }}>
      <Paper sx={{ p: 2 }}>
        <Typography variant="h6" sx={{ mb: 1 }}>
          Training Results
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Machine: {effectiveMachineId || '—'}
        </Typography>

        {results.complete_system && (
          <Alert severity="success" sx={{ mt: 2 }}>
            Complete ML system trained (all 4 model types).
          </Alert>
        )}

        <Grid container spacing={2} sx={{ mt: 1 }}>
          {ranked.slice(0, 4).map((m) => (
            <Grid item xs={12} sm={6} md={3} key={m.model_type}>
              <Card>
                <CardContent>
                  <Typography variant="overline" color="text.secondary">
                    {m.model_type}
                  </Typography>
                  <Typography variant="h5">{m.primary_metric ? m.primary_metric.value.toFixed(4) : '—'}</Typography>
                  <Typography color="text.secondary">{m.primary_metric ? m.primary_metric.label : 'Metric'}</Typography>
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>

        <Grid container spacing={2} sx={{ mt: 1 }}>
          <Grid item xs={12} md={8}>
            <Card>
              <CardHeader title="Model Performance Ranking" />
              <Table size="small">
                <TableBody>
                  {ranked.length === 0 && (
                    <TableRow>
                      <TableCell colSpan={4}>No model results available.</TableCell>
                    </TableRow>
                  )}
                  {ranked.map((m, idx) => (
                    <TableRow key={m.model_type}>
                      <TableCell width={32}>{idx + 1}</TableCell>
                      <TableCell>{m.model_type}</TableCell>
                      <TableCell>{formatMetric(m)}</TableCell>
                      <TableCell align="right">{idx === 0 ? <Chip label="Best" color="success" size="small" /> : null}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </Card>
          </Grid>

          <Grid item xs={12} md={4}>
            <Card>
              <CardHeader title="Actions" />
              <CardContent>
                <ButtonGroup fullWidth orientation="vertical">
                  <Tooltip
                    title={
                      downloadSupported
                        ? 'Download trained model artifacts'
                        : 'Model download is not available yet (no backend download endpoint)'
                    }
                  >
                    <span>
                      <Button startIcon={<DownloadIcon />} disabled={!downloadSupported}>
                        Download Model
                      </Button>
                    </span>
                  </Tooltip>
                  <Button startIcon={<PlayArrowIcon />} onClick={() => setSelectedView('predictions')}>
                    Test Predictions
                  </Button>
                </ButtonGroup>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </Paper>
    </Box>
  );
}
