import { useMemo, useRef } from 'react';
import { Box, Button, Stack, Typography } from '@mui/material';
import { useTheme } from '@mui/material/styles';
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceDot,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

import { exportFirstSvgInContainerAsPng } from './chartExport';

export type LossPoint = { epoch: number; loss: number };

interface TrainingProgressChartProps {
  title?: string;
  data: LossPoint[];
  height?: number;
  smoothingWindow?: number;
  filename?: string;
}

function movingAverage(values: number[], windowSize: number) {
  const w = Math.max(1, Math.floor(windowSize));
  if (w <= 1) return values;

  const out: number[] = [];
  let sum = 0;
  for (let i = 0; i < values.length; i++) {
    sum += values[i];
    if (i >= w) sum -= values[i - w];
    const denom = Math.min(w, i + 1);
    out.push(sum / denom);
  }
  return out;
}

export default function TrainingProgressChart({
  title = 'Training Loss Curve',
  data,
  height = 260,
  smoothingWindow = 5,
  filename = 'training-loss-curve.png',
}: TrainingProgressChartProps) {
  const theme = useTheme();
  const containerRef = useRef<HTMLDivElement>(null);

  const chartData = useMemo(() => {
    const losses = data.map((d) => d.loss);
    const smoothed = movingAverage(losses, smoothingWindow);
    return data.map((d, i) => ({ ...d, loss_smooth: smoothed[i] }));
  }, [data, smoothingWindow]);

  const best = useMemo(() => {
    if (!chartData.length) return null;
    let bestIdx = 0;
    for (let i = 1; i < chartData.length; i++) {
      if (chartData[i].loss_smooth < chartData[bestIdx].loss_smooth) bestIdx = i;
    }
    return chartData[bestIdx];
  }, [chartData]);

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
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
            <XAxis dataKey="epoch" />
            <YAxis />
            <Tooltip />

            <Line
              type="monotone"
              dataKey="loss_smooth"
              name="Loss (smoothed)"
              stroke={theme.palette.success.main}
              dot={false}
            />

            {best && (
              <ReferenceDot
                x={best.epoch}
                y={best.loss_smooth}
                r={5}
                fill={theme.palette.warning.main}
                stroke={theme.palette.warning.main}
                label={{ value: `Best @ ${best.epoch}`, position: 'top' }}
              />
            )}
          </LineChart>
        </ResponsiveContainer>
      </Box>
    </Box>
  );
}
