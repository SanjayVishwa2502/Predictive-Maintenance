import { useMemo, useRef } from 'react';
import { Box, Button, Stack, Typography } from '@mui/material';
import { useTheme } from '@mui/material/styles';
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

import { exportFirstSvgInContainerAsPng } from './chartExport';

export type SeedSeriesPoint = {
  timestamp: string;
  rul: number;
  [key: string]: number | string;
};

interface SeedDataChartProps {
  title?: string;
  data: SeedSeriesPoint[];
  sensorKeys: string[];
  height?: number;
  filename?: string;
}

function formatTimestamp(ts: string) {
  const d = new Date(ts);
  if (Number.isNaN(d.getTime())) return ts;
  return d.toLocaleString();
}

export default function SeedDataChart({
  title = 'Seed Data Time Series',
  data,
  sensorKeys,
  height = 260,
  filename = 'seed-data-time-series.png',
}: SeedDataChartProps) {
  const theme = useTheme();
  const containerRef = useRef<HTMLDivElement>(null);

  const series = useMemo(() => {
    const keys = sensorKeys.filter((k) => k && k !== 'timestamp' && k !== 'rul');
    return keys.slice(0, 8);
  }, [sensorKeys]);

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

      <Box sx={{ height }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
            <XAxis dataKey="timestamp" tickFormatter={(v) => formatTimestamp(String(v))} minTickGap={32} />
            <YAxis yAxisId="left" />
            <YAxis yAxisId="right" orientation="right" label={{ value: 'RUL', angle: -90, position: 'insideRight' }} />
            <Tooltip labelFormatter={(v) => formatTimestamp(String(v))} />
            <Legend />

            {/* RUL on right axis */}
            <Line yAxisId="right" type="monotone" dataKey="rul" stroke={theme.palette.primary.main} dot={false} />

            {/* Sensors on left axis */}
            {series.map((k, idx) => (
              <Line
                key={k}
                yAxisId="left"
                type="monotone"
                dataKey={k}
                stroke={idx % 2 === 0 ? theme.palette.success.main : theme.palette.warning.main}
                dot={false}
                opacity={0.9}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </Box>
    </Box>
  );
}
