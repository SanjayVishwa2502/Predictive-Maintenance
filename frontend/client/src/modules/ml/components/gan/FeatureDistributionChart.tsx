import { useMemo, useRef } from 'react';
import { Box, Button, Stack, Typography } from '@mui/material';
import { useTheme } from '@mui/material/styles';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { DataGrid, type GridColDef } from '@mui/x-data-grid';

import { exportFirstSvgInContainerAsPng } from './chartExport';

export type FeatureStats = {
  mean: number;
  std: number;
  min: number;
  max: number;
};

export type FeatureDistribution = {
  feature: string;
  bin_edges: number[];
  seed_counts: number[];
  synthetic_counts: number[];
  seed_stats: FeatureStats;
  synthetic_stats: FeatureStats;
};

interface FeatureDistributionChartProps {
  title?: string;
  distributions: FeatureDistribution[];
  maxFeaturesToRender?: number;
  heightPerFeature?: number;
  filename?: string;
}

function makeBinLabels(binEdges: number[]) {
  // bin_edges length is N+1; create N labels using the bin centers.
  const labels: string[] = [];
  for (let i = 0; i < Math.max(0, binEdges.length - 1); i++) {
    const a = binEdges[i];
    const b = binEdges[i + 1];
    const mid = (a + b) / 2;
    labels.push(Number.isFinite(mid) ? mid.toFixed(2) : String(i));
  }
  return labels;
}

export default function FeatureDistributionChart({
  title = 'Feature Distribution Comparison',
  distributions,
  maxFeaturesToRender = 4,
  heightPerFeature = 220,
  filename = 'feature-distributions.png',
}: FeatureDistributionChartProps) {
  const theme = useTheme();
  const containerRef = useRef<HTMLDivElement>(null);

  const tableRows = useMemo(() => {
    return distributions.map((d) => ({
      id: d.feature,
      feature: d.feature,
      seed_mean: d.seed_stats.mean,
      seed_std: d.seed_stats.std,
      seed_min: d.seed_stats.min,
      seed_max: d.seed_stats.max,
      syn_mean: d.synthetic_stats.mean,
      syn_std: d.synthetic_stats.std,
      syn_min: d.synthetic_stats.min,
      syn_max: d.synthetic_stats.max,
    }));
  }, [distributions]);

  const columns: GridColDef[] = useMemo(
    () => [
      { field: 'feature', headerName: 'Feature', flex: 1, minWidth: 160 },
      { field: 'seed_mean', headerName: 'Seed mean', type: 'number', width: 120 },
      { field: 'seed_std', headerName: 'Seed std', type: 'number', width: 120 },
      { field: 'seed_min', headerName: 'Seed min', type: 'number', width: 120 },
      { field: 'seed_max', headerName: 'Seed max', type: 'number', width: 120 },
      { field: 'syn_mean', headerName: 'Syn mean', type: 'number', width: 120 },
      { field: 'syn_std', headerName: 'Syn std', type: 'number', width: 120 },
      { field: 'syn_min', headerName: 'Syn min', type: 'number', width: 120 },
      { field: 'syn_max', headerName: 'Syn max', type: 'number', width: 120 },
    ],
    []
  );

  const visible = useMemo(() => distributions.slice(0, Math.max(0, maxFeaturesToRender)), [distributions, maxFeaturesToRender]);

  return (
    <Box ref={containerRef}>
      <Stack direction="row" alignItems="center" justifyContent="space-between" sx={{ mb: 1 }}>
        <Typography variant="subtitle2">{title}</Typography>
        <Button
          size="small"
          variant="outlined"
          onClick={() =>
            exportFirstSvgInContainerAsPng(containerRef, filename, {
              backgroundColor: theme.palette.background.paper,
            })
          }
        >
          Export PNG
        </Button>
      </Stack>

      <Stack spacing={2}>
        {visible.map((d) => {
          const labels = makeBinLabels(d.bin_edges);
          const points = labels.map((label, i) => ({
            bin: label,
            seed: d.seed_counts[i] ?? 0,
            synthetic: d.synthetic_counts[i] ?? 0,
          }));

          return (
            <Box key={d.feature} sx={{ height: heightPerFeature }}>
              <Typography variant="caption" sx={{ color: theme.palette.text.secondary }}>
                {d.feature}
              </Typography>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={points} margin={{ top: 10, right: 10, bottom: 30, left: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
                  <XAxis dataKey="bin" interval="preserveStartEnd" angle={-15} textAnchor="end" height={50} />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="seed" name="Seed" fill={theme.palette.primary.main} opacity={0.8} />
                  <Bar dataKey="synthetic" name="Synthetic" fill={theme.palette.success.main} opacity={0.8} />
                </BarChart>
              </ResponsiveContainer>
            </Box>
          );
        })}

        <Box sx={{ height: 320 }}>
          <Typography variant="subtitle2" sx={{ mb: 1 }}>
            Statistical metrics
          </Typography>
          <DataGrid
            rows={tableRows}
            columns={columns}
            disableRowSelectionOnClick
            density="compact"
            pageSizeOptions={[5, 10, 25]}
            initialState={{
              pagination: { paginationModel: { pageSize: 5, page: 0 } },
            }}
          />
        </Box>
      </Stack>
    </Box>
  );
}
