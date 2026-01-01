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

function normalizeLossPoints(points: LossPoint[], maxPoints = 5000): LossPoint[] {
  const filtered = points.filter(
    (p) =>
      typeof p?.epoch === 'number' &&
      typeof p?.loss === 'number' &&
      Number.isFinite(p.epoch) &&
      Number.isFinite(p.loss)
  );

  if (!filtered.length) return [];

  // Ensure stable X order
  filtered.sort((a, b) => a.epoch - b.epoch);

  // Deduplicate epochs (keep last value for an epoch)
  const out: LossPoint[] = [];
  for (const p of filtered) {
    const last = out[out.length - 1];
    if (last && last.epoch === p.epoch) {
      out[out.length - 1] = p;
    } else {
      out.push(p);
    }
  }

  return out.length > maxPoints ? out.slice(out.length - maxPoints) : out;
}

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

const LossTooltip = ({ active, payload, label }: any) => {
  if (!active || !payload?.length) return null;
  const p = payload[0];
  const value = typeof p?.value === 'number' && Number.isFinite(p.value) ? p.value : null;
  const epoch = typeof label === 'number' && Number.isFinite(label) ? label : null;

  return (
    <Box
      sx={{
        px: 1.25,
        py: 0.75,
        borderRadius: 1,
        bgcolor: 'background.paper',
        border: '1px solid',
        borderColor: 'divider',
        boxShadow: 3,
        pointerEvents: 'none',
        minWidth: 140,
      }}
    >
      <Typography variant="caption" sx={{ display: 'block', color: 'text.secondary' }}>
        Epoch: {epoch ?? '—'}
      </Typography>
      <Typography variant="caption" sx={{ display: 'block', color: 'text.primary' }}>
        Loss: {value !== null ? value.toFixed(6) : '—'}
      </Typography>
    </Box>
  );
};

export default function TrainingProgressChart({
  title = 'Training Loss Curve',
  data,
  height = 260,
  smoothingWindow = 5,
  filename = 'training-loss-curve.png',
}: TrainingProgressChartProps) {
  const theme = useTheme();
  const containerRef = useRef<HTMLDivElement>(null);

  const normalized = useMemo(() => normalizeLossPoints(data), [data]);

  const chartData = useMemo(() => {
    const losses = normalized.map((d) => d.loss);
    const smoothed = movingAverage(losses, smoothingWindow);
    return normalized.map((d, i) => ({ ...d, loss_smooth: smoothed[i] }));
  }, [normalized, smoothingWindow]);

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
            <XAxis dataKey="epoch" type="number" domain={['dataMin', 'dataMax']} allowDecimals={false} />
            <YAxis />
            <Tooltip
              isAnimationActive={false}
              animationDuration={0}
              content={<LossTooltip />}
              wrapperStyle={{ outline: 'none' }}
            />

            <Line
              type="monotone"
              dataKey="loss_smooth"
              name="Loss (smoothed)"
              stroke={theme.palette.success.main}
              dot={false}
              isAnimationActive={false}
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
