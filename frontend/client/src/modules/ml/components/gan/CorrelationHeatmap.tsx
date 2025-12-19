import { useMemo, useRef } from 'react';
import { Box, Button, Stack, Typography } from '@mui/material';
import { alpha, useTheme } from '@mui/material/styles';

import { exportFirstSvgInContainerAsPng } from './chartExport';

interface CorrelationHeatmapProps {
  title?: string;
  features: string[];
  matrix: number[][];
  size?: number;
  cellSize?: number;
  filename?: string;
}

function clamp01(v: number) {
  if (!Number.isFinite(v)) return 0;
  return Math.max(0, Math.min(1, v));
}

export default function CorrelationHeatmap({
  title = 'Correlation Matrix Heatmap',
  features,
  matrix,
  size = 420,
  cellSize = 22,
  filename = 'correlation-heatmap.png',
}: CorrelationHeatmapProps) {
  const theme = useTheme();
  const containerRef = useRef<HTMLDivElement>(null);

  const usableFeatures = useMemo(() => features.slice(0, Math.min(features.length, Math.floor(size / cellSize))), [features, size, cellSize]);
  const n = usableFeatures.length;

  const svgSize = n * cellSize + 120; // room for labels
  const offset = 100;

  const colorForCorr = (corr: number) => {
    // Map [-1,1] -> [0,1]
    const t = clamp01((corr + 1) / 2);
    // Use theme colors; higher correlation -> stronger primary, lower -> stronger error.
    const hi = alpha(theme.palette.primary.main, 0.15 + 0.75 * t);
    const lo = alpha(theme.palette.error.main, 0.15 + 0.75 * (1 - t));
    // Choose whichever has higher alpha contribution via t.
    return t >= 0.5 ? hi : lo;
  };

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

      <Box sx={{ overflowX: 'auto' }}>
        <svg width={svgSize} height={svgSize}>
          {/* axis labels */}
          {usableFeatures.map((f, i) => (
            <text
              key={`x-${f}`}
              x={offset + i * cellSize + cellSize / 2}
              y={offset - 10}
              fontSize={10}
              textAnchor="end"
              transform={`rotate(-45 ${offset + i * cellSize + cellSize / 2} ${offset - 10})`}
              fill={theme.palette.text.secondary}
            >
              {f}
            </text>
          ))}

          {usableFeatures.map((f, i) => (
            <text
              key={`y-${f}`}
              x={offset - 8}
              y={offset + i * cellSize + cellSize / 2 + 3}
              fontSize={10}
              textAnchor="end"
              fill={theme.palette.text.secondary}
            >
              {f}
            </text>
          ))}

          {/* cells */}
          {usableFeatures.map((_row, r) =>
            usableFeatures.map((_col, c) => {
              const corr = matrix?.[r]?.[c];
              const corrNum = typeof corr === 'number' ? corr : 0;
              const x = offset + c * cellSize;
              const y = offset + r * cellSize;
              const fill = colorForCorr(corrNum);
              return (
                <g key={`cell-${r}-${c}`}>
                  <rect x={x} y={y} width={cellSize} height={cellSize} fill={fill} stroke={alpha(theme.palette.divider, 0.4)} />
                  <title>
                    {usableFeatures[r]} × {usableFeatures[c]}: {corrNum.toFixed(3)}
                  </title>
                </g>
              );
            })
          )}
        </svg>
      </Box>
    </Box>
  );
}
