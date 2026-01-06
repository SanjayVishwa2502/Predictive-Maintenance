/**
 * PredictionCard Component
 * 
 * Displays ML prediction results for a single selected machine including:
 * - Health status with confidence score
 * - Remaining Useful Life (RUL) prediction
 * - Failure type probability distribution
 * - Action buttons for prediction, explanation, and history
 * 
 * Design: Professional dark theme with color-coded status indicators
 * Status Colors: Green (Healthy), Yellow (Degrading), Orange (Warning), Red (Critical)
 */

import {
  Box,
  Card,
  CardContent,
  CardHeader,
  Button,
  Typography,
  LinearProgress,
  Skeleton,
  Grid,
  Chip,
  Divider,
  Stack,
  CircularProgress,
  Tooltip,
} from '@mui/material';
import { alpha, useTheme } from '@mui/material/styles';
import {
  PlayArrow as PlayArrowIcon,
  Psychology as PsychologyIcon,
  History as HistoryIcon,
  CheckCircle as CheckCircleIcon,
  Warning as WarningIcon,
  Error as ErrorIcon,
  AccessTime as AccessTimeIcon,
  Refresh as RefreshIcon,
  AutoAwesome as AutoAwesomeIcon,
} from '@mui/icons-material';
import ContentCopyIcon from '@mui/icons-material/ContentCopy';
import { useState, useEffect, useMemo } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

// ============================================================================
// TYPESCRIPT INTERFACES
// ============================================================================

export interface PredictionCardProps {
  machineId: string;
  prediction: PredictionResult | null;
  loading: boolean;
  onRunPrediction: () => void;
  onExplain: () => void;
  onViewHistory?: () => void;
  autoRefresh?: boolean;
  refreshInterval?: number; // seconds

  // Role-based access control
  userRole?: 'admin' | 'operator' | 'viewer';
  isApproved?: boolean; // For operators pending approval

  // Optional: text-output mode (LLM explanation summary)
  // If provided, the card will render this text output panel instead of the full prediction/status UI.
  textOutput?: string;
  textOutputLoading?: boolean;
  textOutputError?: string | null;
  textOutputRefreshSeconds?: number;
}

export interface PredictionResult {
  classification: {
    failure_type: string;
    confidence: number;
    failure_probability: number;
    all_probabilities: Record<string, number>;
  };
  rul?: {
    rul_hours: number;
    rul_days: number;
    urgency: 'low' | 'medium' | 'high' | 'critical';
    maintenance_window: string;
  };
  timestamp: string;
}

interface StatusConfig {
  label: string;
  color: string;
  icon: React.ReactElement;
  bgColor: string;
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/**
 * Get status configuration based on failure probability
 * Status Mapping:
 * - Healthy: failure_prob < 0.15 (Green)
 * - Degrading: 0.15 ≤ failure_prob < 0.40 (Yellow)
 * - Warning: 0.40 ≤ failure_prob < 0.70 (Orange)
 * - Critical: failure_prob ≥ 0.70 (Red)
 */
const getStatusConfig = (failureProbability: number): StatusConfig => {
  if (failureProbability < 0.15) {
    return {
      label: 'Healthy',
      color: '#10b981',
      icon: <CheckCircleIcon sx={{ fontSize: 40 }} />,
      bgColor: 'rgba(16, 185, 129, 0.1)',
    };
  } else if (failureProbability < 0.40) {
    return {
      label: 'Degrading',
      color: '#fbbf24',
      icon: <WarningIcon sx={{ fontSize: 40 }} />,
      bgColor: 'rgba(251, 191, 36, 0.1)',
    };
  } else if (failureProbability < 0.70) {
    return {
      label: 'Warning',
      color: '#f97316',
      icon: <ErrorIcon sx={{ fontSize: 40 }} />,
      bgColor: 'rgba(249, 115, 22, 0.1)',
    };
  } else {
    return {
      label: 'Critical',
      color: '#ef4444',
      icon: <ErrorIcon sx={{ fontSize: 40 }} />,
      bgColor: 'rgba(239, 68, 68, 0.1)',
    };
  }
};

/**
 * Get urgency configuration for RUL prediction
 */
const getUrgencyConfig = (urgency: string) => {
  switch (urgency) {
    case 'low':
      return {
        color: '#10b981',
        label: 'Low',
        description: 'Schedule within 1 week',
      };
    case 'medium':
      return {
        color: '#fbbf24',
        label: 'Medium',
        description: 'Schedule within 3 days',
      };
    case 'high':
      return {
        color: '#f97316',
        label: 'High',
        description: 'Schedule within 24 hours',
      };
    case 'critical':
      return {
        color: '#ef4444',
        label: 'Critical',
        description: 'IMMEDIATE ACTION REQUIRED',
      };
    default:
      return {
        color: '#6b7280',
        label: 'Unknown',
        description: 'Assessment pending',
      };
  }
};

/**
 * Format relative time (e.g., "2 minutes ago")
 */
const getRelativeTime = (timestamp: string): string => {
  const now = new Date();
  const past = new Date(timestamp);
  const diffMs = now.getTime() - past.getTime();
  const diffMins = Math.floor(diffMs / 60000);
  const diffHours = Math.floor(diffMs / 3600000);
  const diffDays = Math.floor(diffMs / 86400000);

  if (diffMins < 1) return 'Just now';
  if (diffMins < 60) return `${diffMins} minute${diffMins > 1 ? 's' : ''} ago`;
  if (diffHours < 24) return `${diffHours} hour${diffHours > 1 ? 's' : ''} ago`;
  return `${diffDays} day${diffDays > 1 ? 's' : ''} ago`;
};

/**
 * Format failure type for display (convert snake_case to Title Case)
 */
const formatFailureType = (type: string): string => {
  return type
    .split('_')
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
};

// ============================================================================
// MAIN COMPONENT
// ============================================================================

export default function PredictionCard({
  machineId,
  prediction,
  loading,
  onRunPrediction,
  onExplain,
  onViewHistory,
  autoRefresh = false,
  refreshInterval = 30,
  userRole,
  isApproved = true,
  textOutput,
  textOutputLoading,
  textOutputError,
  textOutputRefreshSeconds,
}: PredictionCardProps) {
  const theme = useTheme();
  const [countdown, setCountdown] = useState(refreshInterval);

  // Determine if user can run predictions (admin/approved operator only)
  const canRunPrediction = userRole === 'admin' || (userRole === 'operator' && isApproved);
  const predictionDisabledReason = !canRunPrediction
    ? userRole === 'viewer'
      ? 'Viewers cannot run predictions. Contact an administrator to upgrade your role.'
      : 'Your operator account is pending approval. You will be able to run predictions once approved.'
    : undefined;

  const isTextMode =
    typeof textOutput !== 'undefined' ||
    typeof textOutputLoading !== 'undefined' ||
    typeof textOutputError !== 'undefined' ||
    typeof textOutputRefreshSeconds !== 'undefined';

  if (isTextMode) {
    const displayText = (textOutput || '').trim();
    const isBusy = Boolean(textOutputLoading);

    type ParsedLlmSections = {
      overall?: string;
      topCauses: string[];
      immediateActions: string[];
      next7Days: string[];
      safety?: string;
      other: string[];
    };

    const parseRunDetailsText = (raw: string) => {
      const lines = String(raw || '')
        .replaceAll('\r\n', '\n')
        .split('\n')
        .map((l) => l.trimEnd());

      const meta: Record<string, string> = {};
      const sensorLines: string[] = [];
      let i = 0;
      let inSensorBlock = false;
      for (; i < lines.length; i++) {
        const line = lines[i].trim();
        if (!line) {
          if (inSensorBlock) inSensorBlock = false;
          continue;
        }
        if (line === '[LLM]' || line === '[Predictions]' || line === 'Notes') break;
        
        // Parse metadata lines
        const m = line.match(/^(Stamp|Machine|LLM):\s*(.*)$/i);
        if (m) {
          meta[m[1].toLowerCase()] = (m[2] || '').trim();
          continue;
        }
        
        // Parse sensor block
        if (line.startsWith('Sensors at run time:')) {
          inSensorBlock = true;
          continue;
        }
        if (inSensorBlock && line.startsWith('  ')) {
          sensorLines.push(line.trim());
          continue;
        }
      }

      // Collect LLM block + predictions block
      let llmBlock: string[] = [];
      let predictionsBlock: string[] = [];
      let mode: 'none' | 'llm' | 'pred' = 'none';

      for (; i < lines.length; i++) {
        const line = lines[i].trim();
        if (line === '[LLM]' || line === 'Notes') {
          mode = 'llm';
          continue;
        }
        if (line === '[Predictions]') {
          mode = 'pred';
          continue;
        }
        if (!line && mode === 'none') continue;
        if (mode === 'llm') llmBlock.push(lines[i]);
        if (mode === 'pred') predictionsBlock.push(lines[i]);
      }

      const llmRaw = llmBlock.join('\n').trim();
      const predLines = predictionsBlock
        .map((l) => l.trim())
        .filter((l) => Boolean(l));

      const sections: ParsedLlmSections = {
        topCauses: [],
        immediateActions: [],
        next7Days: [],
        other: [],
      };

      let current: keyof Pick<ParsedLlmSections, 'topCauses' | 'immediateActions' | 'next7Days'> | null = null;
      const llmLines = llmRaw.split('\n').map((l) => l.trim());
      for (const line of llmLines) {
        const l = line.trim();
        if (!l) continue;

        const overall = l.match(/^Overall:\s*(.*)$/i);
        if (overall) {
          sections.overall = (overall[1] || '').trim();
          current = null;
          continue;
        }
        if (/^Top causes:\s*$/i.test(l)) {
          current = 'topCauses';
          continue;
        }
        if (/^Immediate actions:\s*$/i.test(l)) {
          current = 'immediateActions';
          continue;
        }
        if (/^Next 7 days:\s*$/i.test(l)) {
          current = 'next7Days';
          continue;
        }
        const safety = l.match(/^Safety:\s*(.*)$/i);
        if (safety) {
          sections.safety = (safety[1] || '').trim();
          current = null;
          continue;
        }

        const bullet = l.match(/^[-*]\s+(.*)$/);
        if (bullet && current) {
          (sections[current] as string[]).push((bullet[1] || '').trim());
          continue;
        }

        // Unstructured line
        sections.other.push(l);
      }

      const overallText = (sections.overall || '').toLowerCase();
      const lowRisk = overallText.includes('low risk') || overallText.includes('operating normally') || overallText.includes('normal');
      const noCauses = sections.topCauses.some((c) => c.toLowerCase().includes('none detected')) || sections.topCauses.length === 0;
      const actionsLabel = lowRisk && noCauses ? 'Preventive actions' : 'Immediate actions';

      return {
        meta: {
          machine: meta['machine'] || '',
          dataStamp: meta['stamp'] || meta['data stamp'] || '',
          compute: meta['llm'] || meta['llm compute'] || '',
        },
        sensors: sensorLines,
        llm: sections,
        actionsLabel,
        lowRisk,
        predictionLines: predLines,
        rawLlm: llmRaw,
      };
    };

    const parsed = useMemo(() => parseRunDetailsText(displayText), [displayText]);

    // Detect if this is a placeholder message (no actual LLM output)
    const isPlaceholderMessage = useMemo(() => {
      const txt = (parsed?.rawLlm || '').trim().toLowerCase();
      return (
        txt.includes('prediction-only run') ||
        txt.includes('pending') ||
        txt.includes('not been requested') ||
        txt.includes('not available') ||
        txt.startsWith('[') // Square bracket messages are placeholders
      );
    }, [parsed?.rawLlm]);

    const renderLlmAsMarkdown = useMemo(() => {
      const txt = (parsed?.rawLlm || '').trim();
      if (!txt || isPlaceholderMessage) return false;
      // Render as markdown when we see typical markdown structures.
      // (We do NOT enable raw HTML.)
      return (
        /(^|\n)\s*#{1,6}\s+/.test(txt) ||
        /(^|\n)\s*---\s*$/.test(txt) ||
        /\|\s*:?-{3,}:?\s*\|/.test(txt) ||
        /(^|\n)\s*\*\*.+\*\*\s*:/.test(txt)
      );
    }, [parsed?.rawLlm, isPlaceholderMessage]);

    const riskChip = (() => {
      if (!parsed?.llm?.overall) return null;
      const t = parsed.llm.overall.toLowerCase();
      if (t.includes('critical')) return { label: 'CRITICAL', color: 'error' as const };
      if (t.includes('high risk')) return { label: 'HIGH RISK', color: 'warning' as const };
      if (t.includes('medium risk')) return { label: 'MEDIUM RISK', color: 'info' as const };
      if (t.includes('low risk') || t.includes('operating normally') || t.includes('normal')) return { label: 'LOW RISK', color: 'success' as const };
      return { label: 'STATUS', color: 'default' as const };
    })();

    type PredictionSectionKey = 'classification' | 'rul' | 'anomaly' | 'forecast' | 'other';
    type PredictionSection = { key: PredictionSectionKey; title: string; lines: string[] };

    const predictionSections = useMemo<PredictionSection[]>(() => {
      const lines = (parsed?.predictionLines || []).map((l) => String(l || '').trim()).filter(Boolean);
      if (!lines.length) return [];

      const titleToKey: Record<string, { key: PredictionSectionKey; title: string }> = {
        classification: { key: 'classification', title: 'Classification' },
        rul: { key: 'rul', title: 'RUL' },
        anomaly: { key: 'anomaly', title: 'Anomaly' },
        forecast: { key: 'forecast', title: 'Forecast' },
      };

      const sections: PredictionSection[] = [];
      let current: PredictionSection | null = null;

      for (const line of lines) {
        const token = line.toLowerCase();
        const hit = titleToKey[token];
        if (hit) {
          current = { key: hit.key, title: hit.title, lines: [] };
          sections.push(current);
          continue;
        }
        if (!current) {
          current = { key: 'other', title: 'Predictions', lines: [] };
          sections.push(current);
        }
        current.lines.push(line);
      }

      // Keep a stable, formal order when present
      const order: PredictionSectionKey[] = ['classification', 'rul', 'anomaly', 'forecast', 'other'];
      return sections.sort((a, b) => order.indexOf(a.key) - order.indexOf(b.key));
    }, [parsed?.predictionLines]);

    const renderPredictionSectionHeader = (section: PredictionSection) => {
      const first = (section.lines[0] || '').toLowerCase();

      if (section.key === 'classification') {
        const status = (section.lines[0] || '').split('|')[0]?.trim() || '';
        const statusLower = status.toLowerCase();
        const chipColor =
          statusLower.includes('normal') || statusLower.includes('healthy')
            ? ('success' as const)
            : statusLower.includes('critical')
              ? ('error' as const)
              : ('warning' as const);
        return status ? <Chip size="small" label={status.toUpperCase()} color={chipColor} /> : null;
      }

      if (section.key === 'anomaly') {
        const isAnomaly = first.includes('is_anomaly=true') || first.includes('is_anomaly=1');
        return <Chip size="small" label={isAnomaly ? 'ANOMALY' : 'NORMAL'} color={isAnomaly ? 'error' : 'success'} />;
      }

      if (section.key === 'rul') {
        const na = first.includes('n/a') || first.includes('no rul model');
        return <Chip size="small" label={na ? 'N/A' : 'AVAILABLE'} color={na ? 'default' : 'info'} />;
      }

      return null;
    };

    const handleCopy = async () => {
      const txt = displayText;
      if (!txt) return;
      try {
        await navigator.clipboard.writeText(txt);
      } catch {
        try {
          const ta = document.createElement('textarea');
          ta.value = txt;
          ta.style.position = 'fixed';
          ta.style.left = '-10000px';
          document.body.appendChild(ta);
          ta.select();
          document.execCommand('copy');
          document.body.removeChild(ta);
        } catch {
          // ignore
        }
      }
    };

    return (
      <Card
        sx={(theme) => ({
          bgcolor: alpha(theme.palette.background.paper, theme.palette.mode === 'dark' ? 0.75 : 0.95),
          backdropFilter: theme.palette.mode === 'dark' ? 'blur(10px)' : undefined,
          border: 1,
          borderColor: 'divider',
          borderRadius: 2,
          mb: 3,
        })}
      >
        <CardHeader
          title={
            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 2 }}>
              <Typography variant="h6" sx={{ color: 'text.primary', fontWeight: 600 }}>
                Run Details
              </Typography>
              <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
                <Button
                  size="small"
                  variant="outlined"
                  startIcon={<ContentCopyIcon />}
                  onClick={handleCopy}
                  disabled={!displayText}
                  sx={{ textTransform: 'none' }}
                >
                  Copy
                </Button>
                {isBusy ? (
                  <Button
                    size="small"
                    variant="contained"
                    color="primary"
                    disabled
                    startIcon={
                      <CircularProgress
                        size={16}
                        thickness={4}
                        color="inherit"
                      />
                    }
                    sx={{ textTransform: 'none' }}
                  >
                    Generating…
                  </Button>
                ) : !Boolean(loading) ? (
                  <Tooltip title={predictionDisabledReason || ''} arrow>
                    <span>
                      <Button
                        size="small"
                        variant="contained"
                        color="primary"
                        startIcon={<PlayArrowIcon />}
                        onClick={onRunPrediction}
                        disabled={!canRunPrediction}
                        sx={{ textTransform: 'none' }}
                      >
                        Prediction
                      </Button>
                    </span>
                  </Tooltip>
                ) : null}
              </Box>
            </Box>
          }
          subheader={
            <Stack direction="row" spacing={1} sx={{ mt: 0.5, flexWrap: 'wrap', alignItems: 'center' }}>
              <Chip
                size="small"
                label={`Machine: ${parsed?.meta?.machine || machineId}`}
                sx={(theme) => ({ bgcolor: alpha(theme.palette.action.selected, 0.1), color: 'text.secondary' })}
              />
              {isBusy ? (
                <Chip
                  size="small"
                  icon={
                    <CircularProgress
                      size={12}
                      thickness={4}
                      sx={{ color: 'primary.light', ml: 0.5 }}
                    />
                  }
                  label="AI Generating"
                  color="primary"
                  sx={{
                    fontWeight: 600,
                    animation: 'pulse 2s ease-in-out infinite',
                    '@keyframes pulse': {
                      '0%, 100%': { opacity: 1 },
                      '50%': { opacity: 0.7 },
                    },
                  }}
                />
              ) : null}
              {parsed?.meta?.dataStamp ? (
                <Chip
                  size="small"
                  label={`Stamp: ${parsed.meta.dataStamp}`}
                  sx={(theme) => ({ bgcolor: alpha(theme.palette.action.selected, 0.1), color: 'text.secondary' })}
                />
              ) : null}
              {parsed?.meta?.compute ? (
                <Chip
                  size="small"
                  label={`LLM: ${parsed.meta.compute}`}
                  sx={(theme) => ({ bgcolor: alpha(theme.palette.action.selected, 0.1), color: 'text.secondary' })}
                />
              ) : null}
              {riskChip ? (
                <Chip
                  size="small"
                  label={riskChip.label}
                  color={riskChip.color}
                  sx={{ fontWeight: 700 }}
                />
              ) : null}
              {parsed?.sensors?.length > 0 ? (
                parsed.sensors.map((sensor, idx) => (
                  <Chip
                    key={idx}
                    size="small"
                    label={sensor}
                    sx={{ bgcolor: 'rgba(59, 130, 246, 0.15)', color: '#93c5fd', fontFamily: 'monospace', fontSize: '0.75rem' }}
                  />
                ))
              ) : null}
            </Stack>
          }
          sx={{
            '& .MuiCardHeader-content': { overflow: 'hidden' },
            pb: 0,
          }}
        />
        <CardContent>
          {textOutputError && (
            <Typography variant="body2" sx={{ color: '#fca5a5', mb: 1 }}>
              {textOutputError}
            </Typography>
          )}

          <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: '1.15fr 0.85fr' }, gap: 2 }}>
            <Box
              sx={{
                minHeight: 220,
                p: 2,
                borderRadius: 2,
                bgcolor: 'action.hover',
                border: 1,
                borderColor: 'divider',
                overflow: 'auto',
                color: 'text.primary',
                fontSize: 14,
                lineHeight: 1.6,
              }}
            >
              {isBusy ? (
                <Box
                  sx={{
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    justifyContent: 'center',
                    minHeight: 180,
                    gap: 2,
                  }}
                >
                  <Box
                    sx={{
                      position: 'relative',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                    }}
                  >
                    <CircularProgress
                      size={56}
                      thickness={3}
                      color="primary"
                      sx={{
                        animation: 'pulse 2s ease-in-out infinite',
                        '@keyframes pulse': {
                          '0%, 100%': { opacity: 1 },
                          '50%': { opacity: 0.6 },
                        },
                      }}
                    />
                    <AutoAwesomeIcon
                      sx={{
                        position: 'absolute',
                        fontSize: 24,
                        color: 'primary.light',
                        animation: 'sparkle 1.5s ease-in-out infinite',
                        '@keyframes sparkle': {
                          '0%, 100%': { transform: 'scale(1)', opacity: 1 },
                          '50%': { transform: 'scale(1.2)', opacity: 0.7 },
                        },
                      }}
                    />
                  </Box>
                  <Box sx={{ textAlign: 'center' }}>
                    <Typography
                      variant="body1"
                      sx={{
                        color: 'text.primary',
                        fontWeight: 600,
                        mb: 0.5,
                      }}
                    >
                      Generating AI Explanation
                    </Typography>
                    <Typography
                      variant="caption"
                      sx={{
                        color: 'text.secondary',
                        display: 'block',
                      }}
                    >
                      The LLM is analyzing your sensor data…
                    </Typography>
                  </Box>
                  <LinearProgress
                    sx={{
                      width: '60%',
                      height: 4,
                      borderRadius: 2,
                      backgroundColor: 'rgba(102, 126, 234, 0.2)',
                      '& .MuiLinearProgress-bar': {
                        backgroundColor: '#667eea',
                        borderRadius: 2,
                      },
                    }}
                  />
                </Box>
              ) : parsed?.rawLlm && !isPlaceholderMessage ? (
                renderLlmAsMarkdown ? (
                  <Box
                    sx={{
                      '& h1, & h2, & h3': { margin: '0.25rem 0 0.25rem', fontWeight: 700 },
                      '& p': { margin: '0.35rem 0' },
                      '& hr': { border: 0, borderTop: 1, borderColor: 'divider', margin: '0.75rem 0' },
                      '& ul': { margin: 0, paddingLeft: '1.25rem' },
                      '& li': { margin: '0.15rem 0' },
                      '& table': {
                        width: '100%',
                        borderCollapse: 'collapse',
                        marginTop: '0.5rem',
                        marginBottom: '0.75rem',
                        fontSize: '0.875rem',
                      },
                      '& th, & td': {
                        border: 1,
                        borderColor: 'divider',
                        padding: '6px 8px',
                        verticalAlign: 'top',
                      },
                      '& th': { backgroundColor: 'action.hover', color: 'text.primary', textAlign: 'left' },
                      '& code': { fontFamily: 'monospace', fontSize: '0.85em' },
                    }}
                  >
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>{parsed.rawLlm}</ReactMarkdown>
                  </Box>
                ) : (
                  <Stack spacing={2}>
                    {/* OVERALL */}
                    {parsed.llm.overall ? (
                      <Box>
                        <Typography variant="overline" sx={{ color: 'text.secondary' }}>
                          Overall
                        </Typography>
                        <Typography variant="body1" sx={{ color: 'text.primary', fontWeight: 600, mt: 0.5 }}>
                          {parsed.llm.overall}
                        </Typography>
                      </Box>
                    ) : null}

                  {/* TOP CAUSES */}
                  {parsed.llm.topCauses.length > 0 ? (
                    <Box>
                      <Typography variant="overline" sx={{ color: 'text.secondary' }}>
                        Top causes
                      </Typography>
                      <Box component="ul" sx={{ m: 0, mt: 0.75, pl: 2.5 }}>
                        {parsed.llm.topCauses.map((c, idx) => (
                          <Typography
                            key={idx}
                            component="li"
                            variant="body2"
                            sx={{ color: 'text.primary', mb: 0.5, '&::marker': { color: 'text.secondary' } }}
                          >
                            {c}
                          </Typography>
                        ))}
                      </Box>
                    </Box>
                  ) : null}

                  {/* ACTIONS */}
                  {parsed.llm.immediateActions.length > 0 ? (
                    <Box>
                      <Typography variant="overline" sx={{ color: 'text.secondary' }}>
                        {parsed.actionsLabel}
                      </Typography>
                      <Box component="ul" sx={{ m: 0, mt: 0.75, pl: 2.5 }}>
                        {parsed.llm.immediateActions.map((a, idx) => (
                          <Typography
                            key={idx}
                            component="li"
                            variant="body2"
                            sx={{ color: 'text.primary', mb: 0.5, '&::marker': { color: 'text.secondary' } }}
                          >
                            {a}
                          </Typography>
                        ))}
                      </Box>
                    </Box>
                  ) : null}

                  {/* NEXT 7 DAYS */}
                  {parsed.llm.next7Days.length > 0 ? (
                    <Box>
                      <Typography variant="overline" sx={{ color: 'text.secondary' }}>
                        Next 7 days
                      </Typography>
                      <Box component="ul" sx={{ m: 0, mt: 0.75, pl: 2.5 }}>
                        {parsed.llm.next7Days.map((a, idx) => (
                          <Typography
                            key={idx}
                            component="li"
                            variant="body2"
                            sx={{ color: 'text.primary', mb: 0.5, '&::marker': { color: 'text.secondary' } }}
                          >
                            {a}
                          </Typography>
                        ))}
                      </Box>
                    </Box>
                  ) : null}

                  {/* SAFETY */}
                  {parsed.llm.safety ? (
                    <Box>
                      <Typography variant="overline" sx={{ color: 'text.secondary' }}>
                        Safety
                      </Typography>
                      <Typography variant="body2" sx={{ color: 'text.primary', mt: 0.75 }}>
                        {parsed.llm.safety}
                      </Typography>
                    </Box>
                  ) : null}

                    {/* NOTES */}
                    {parsed.llm.other.length > 0 ? (
                      <Box>
                        <Typography variant="overline" sx={{ color: 'text.secondary' }}>
                          Notes
                        </Typography>
                        <Typography variant="body2" sx={{ color: 'text.primary', mt: 0.75, whiteSpace: 'pre-wrap' }}>
                          {parsed.llm.other.join('\n')}
                        </Typography>
                      </Box>
                    ) : null}
                  </Stack>
                )
              ) : (
                <Box
                  sx={{
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    justifyContent: 'center',
                    minHeight: 180,
                    gap: 2,
                  }}
                >
                  {/* Check if this is a prediction-only run or no data at all */}
                  {parsed?.rawLlm?.toLowerCase().includes('prediction-only') ? (
                    <>
                      <Box
                        sx={{
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          width: 64,
                          height: 64,
                          borderRadius: '50%',
                          bgcolor: 'rgba(59, 130, 246, 0.1)',
                          border: '2px solid rgba(59, 130, 246, 0.3)',
                        }}
                      >
                        <CheckCircleIcon sx={{ fontSize: 32, color: 'info.main' }} />
                      </Box>
                      <Box sx={{ textAlign: 'center' }}>
                        <Typography variant="body1" sx={{ color: 'text.primary', fontWeight: 600 }}>
                          Prediction Complete
                        </Typography>
                        <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mt: 0.5 }}>
                          ML models analyzed successfully. Click "Prediction" to generate AI explanation.
                        </Typography>
                      </Box>
                    </>
                  ) : (
                    <>
                      <PsychologyIcon sx={{ fontSize: 48, color: 'text.disabled', opacity: 0.7 }} />
                      <Box sx={{ textAlign: 'center' }}>
                        <Typography variant="body1" sx={{ color: 'text.secondary', fontWeight: 500 }}>
                          No Explanation Available
                        </Typography>
                        <Typography variant="caption" sx={{ color: 'text.disabled', display: 'block', mt: 0.5 }}>
                          Click "Prediction" to run analysis and generate an AI explanation
                        </Typography>
                      </Box>
                    </>
                  )}
                </Box>
              )}
            </Box>

            <Box
              sx={{
                minHeight: 220,
                p: 2,
                borderRadius: 2,
                bgcolor: alpha(theme.palette.background.paper, 0.06),
                border: `1px solid ${alpha(theme.palette.divider, 0.25)}`,
                overflow: 'auto',
                color: 'text.primary',
              }}
            >
              <Typography variant="subtitle2" sx={{ color: 'text.secondary', mb: 1 }}>
                Predictions
              </Typography>
              {predictionSections.length ? (
                <Stack spacing={1.25}>
                  {predictionSections.map((section, sectionIdx) => (
                    <Box key={section.key}>
                      <Stack direction="row" alignItems="center" justifyContent="space-between" spacing={1}>
                        <Typography variant="subtitle2" sx={{ color: 'text.primary', fontWeight: 700 }}>
                          {section.title}
                        </Typography>
                        {renderPredictionSectionHeader(section)}
                      </Stack>
                      <Stack spacing={0.75} sx={{ mt: 0.75 }}>
                        {section.lines.length ? (
                          section.lines.map((line, idx) => (
                            <Typography
                              key={idx}
                              variant="body2"
                              sx={{
                                color: 'text.primary',
                                whiteSpace: 'pre-wrap',
                                fontFamily:
                                  'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace',
                                fontSize: '0.85rem',
                              }}
                            >
                              {line}
                            </Typography>
                          ))
                        ) : (
                          <Typography variant="body2" sx={{ color: 'text.secondary' }}>
                            No data.
                          </Typography>
                        )}
                      </Stack>
                      {sectionIdx < predictionSections.length - 1 ? (
                        <Divider sx={{ my: 1.25, borderColor: alpha(theme.palette.divider, 0.25) }} />
                      ) : null}
                    </Box>
                  ))}
                </Stack>
              ) : (
                <Typography variant="body2" sx={{ color: 'text.secondary' }}>
                  No predictions available.
                </Typography>
              )}
            </Box>
          </Box>
        </CardContent>
      </Card>
    );
  }

  // Auto-refresh countdown timer
  useEffect(() => {
    if (!autoRefresh || !prediction) return;

    const timer = setInterval(() => {
      setCountdown((prev) => {
        if (prev <= 1) {
          onRunPrediction();
          return refreshInterval;
        }
        return prev - 1;
      });
    }, 1000);

    return () => clearInterval(timer);
  }, [autoRefresh, prediction, onRunPrediction, refreshInterval]);

  // Reset countdown when prediction changes
  useEffect(() => {
    setCountdown(refreshInterval);
  }, [prediction, refreshInterval]);

  // Compute status configuration
  const statusConfig = useMemo(() => {
    if (!prediction) return null;
    return getStatusConfig(prediction.classification.failure_probability);
  }, [prediction]);

  // Compute urgency configuration
  const urgencyConfig = useMemo(() => {
    if (!prediction?.rul) return null;
    return getUrgencyConfig(prediction.rul.urgency);
  }, [prediction]);

  // Sort probabilities by value (descending)
  const sortedProbabilities = useMemo(() => {
    if (!prediction) return [];
    return Object.entries(prediction.classification.all_probabilities)
      .sort(([, a], [, b]) => b - a)
      .map(([type, probability]) => ({
        type: formatFailureType(type),
        probability,
      }));
  }, [prediction]);

  // ============================================================================
  // RENDER: EMPTY STATE
  // ============================================================================

  if (!prediction && !loading) {
    return (
      <Card
        sx={(theme) => ({
          bgcolor: alpha(theme.palette.background.paper, theme.palette.mode === 'dark' ? 0.75 : 0.95),
          backdropFilter: theme.palette.mode === 'dark' ? 'blur(10px)' : undefined,
          border: 1,
          borderColor: 'divider',
          borderRadius: 2,
        })}
      >
        <CardHeader
          title={
            <Typography variant="h6" sx={{ fontWeight: 600, color: 'text.primary' }}>
              MACHINE HEALTH PREDICTION
            </Typography>
          }
        />
        <Divider sx={{ borderColor: 'divider' }} />
        <CardContent>
          <Box
            sx={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              py: 8,
              gap: 3,
            }}
          >
            <PsychologyIcon sx={{ fontSize: 80, color: 'text.disabled' }} />
            <Typography variant="h6" sx={{ color: 'text.secondary', textAlign: 'center' }}>
              No prediction yet
            </Typography>
            <Typography variant="body2" sx={{ color: 'text.secondary', textAlign: 'center', maxWidth: 400 }}>
              {canRunPrediction
                ? 'Click the button below to run a health prediction for this machine using the latest sensor data.'
                : predictionDisabledReason}
            </Typography>
            <Tooltip title={predictionDisabledReason || ''} arrow>
              <span>
                <Button
                  variant="contained"
                  color="primary"
                  size="large"
                  startIcon={<PlayArrowIcon />}
                  onClick={onRunPrediction}
                  disabled={!canRunPrediction}
                  sx={{ mt: 2 }}
                >
                  Run Prediction
                </Button>
              </span>
            </Tooltip>
          </Box>
        </CardContent>
      </Card>
    );
  }

  // ============================================================================
  // RENDER: LOADING STATE
  // ============================================================================

  if (loading) {
    return (
      <Card
        sx={(theme) => ({
          bgcolor: alpha(theme.palette.background.paper, theme.palette.mode === 'dark' ? 0.75 : 0.95),
          backdropFilter: theme.palette.mode === 'dark' ? 'blur(10px)' : undefined,
          border: 1,
          borderColor: 'divider',
          borderRadius: 2,
        })}
      >
        <CardHeader
          title={
            <Typography variant="h6" sx={{ fontWeight: 600, color: 'text.primary' }}>
              MACHINE HEALTH PREDICTION
            </Typography>
          }
          action={
            <CircularProgress size={24} color="primary" />
          }
        />
        <Divider sx={{ borderColor: 'divider' }} />
        <CardContent>
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Skeleton variant="rectangular" height={200} sx={{ bgcolor: 'action.hover', borderRadius: 2 }} />
            </Grid>
            <Grid item xs={12} md={6}>
              <Skeleton variant="rectangular" height={200} sx={{ bgcolor: 'action.hover', borderRadius: 2 }} />
            </Grid>
            <Grid item xs={12}>
              <Skeleton variant="rectangular" height={150} sx={{ bgcolor: 'action.hover', borderRadius: 2 }} />
            </Grid>
          </Grid>
        </CardContent>
      </Card>
    );
  }

  // ============================================================================
  // RENDER: PREDICTION RESULT
  // ============================================================================

  return (
    <Card
      sx={(theme) => ({
        bgcolor: alpha(theme.palette.background.paper, theme.palette.mode === 'dark' ? 0.75 : 0.95),
        backdropFilter: theme.palette.mode === 'dark' ? 'blur(10px)' : undefined,
        border: 1,
        borderColor: 'divider',
        borderRadius: 2,
      })}
    >
      {/* HEADER */}
      <CardHeader
        title={
          <Typography variant="h6" sx={{ fontWeight: 600, color: 'text.primary' }}>
            MACHINE HEALTH PREDICTION
          </Typography>
        }
        subheader={
          <Stack direction="row" spacing={2} alignItems="center" sx={{ mt: 1 }}>
            <Typography variant="body2" sx={{ color: 'text.secondary' }}>
              Last prediction: {getRelativeTime(prediction!.timestamp)}
            </Typography>
            {autoRefresh && (
              <Chip
                size="small"
                icon={<RefreshIcon />}
                label={`Auto-refresh in ${countdown}s`}
                sx={{
                  bgcolor: 'rgba(102, 126, 234, 0.2)',
                  color: '#667eea',
                  fontSize: '0.75rem',
                }}
              />
            )}
          </Stack>
        }
        action={
          <Tooltip title={predictionDisabledReason || ''} arrow>
            <span>
              <Button
                variant="contained"
                color="primary"
                size="small"
                startIcon={<PlayArrowIcon />}
                onClick={onRunPrediction}
                disabled={!canRunPrediction}
                sx={{ whiteSpace: 'nowrap' }}
                >
                Run Prediction
              </Button>
            </span>
          </Tooltip>
        }
      />
      <Divider sx={{ borderColor: 'divider' }} />

      <CardContent>
        {/* HEALTH STATUS & RUL SECTION */}
        <Grid container spacing={3} sx={{ mb: 3 }}>
          {/* HEALTH STATUS */}
          <Grid item xs={12} md={6}>
            <Box
              sx={{
                p: 3,
                borderRadius: 2,
                background: statusConfig!.bgColor,
                border: `1px solid ${statusConfig!.color}40`,
                height: '100%',
                display: 'flex',
                flexDirection: 'column',
                justifyContent: 'center',
                alignItems: 'center',
                gap: 2,
                position: 'relative',
                overflow: 'hidden',
                ...(statusConfig!.label === 'Critical' && {
                  animation: 'pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
                  '@keyframes pulse': {
                    '0%, 100%': { opacity: 1 },
                    '50%': { opacity: 0.7 },
                  },
                }),
              }}
            >
              <Typography variant="overline" sx={{ color: 'text.secondary', fontWeight: 600 }}>
                HEALTH STATUS
              </Typography>

              <Box sx={{ color: statusConfig!.color }}>
                {statusConfig!.icon}
              </Box>

              <Typography variant="h4" sx={{ color: statusConfig!.color, fontWeight: 700 }}>
                {statusConfig!.label}
              </Typography>

              {/* Confidence Score with Progress Ring */}
              <Box sx={{ position: 'relative', display: 'inline-flex' }}>
                <CircularProgress
                  variant="determinate"
                  value={prediction!.classification.confidence * 100}
                  size={80}
                  thickness={4}
                  sx={{
                    color: statusConfig!.color,
                    '& .MuiCircularProgress-circle': {
                      strokeLinecap: 'round',
                    },
                  }}
                />
                <Box
                  sx={{
                    top: 0,
                    left: 0,
                    bottom: 0,
                    right: 0,
                    position: 'absolute',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                  }}
                >
                  <Typography variant="h6" sx={{ color: 'text.primary', fontWeight: 700 }}>
                    {Math.round(prediction!.classification.confidence * 100)}%
                  </Typography>
                </Box>
              </Box>

              <Typography variant="body2" sx={{ color: 'text.secondary' }}>
                Confidence Score
              </Typography>

              <Typography
                variant="body1"
                sx={{
                  color: 'text.primary',
                  fontWeight: 500,
                  textAlign: 'center',
                  mt: 1,
                }}
              >
                {formatFailureType(prediction!.classification.failure_type)}
              </Typography>
            </Box>
          </Grid>

          {/* REMAINING USEFUL LIFE */}
          {prediction!.rul && (
            <Grid item xs={12} md={6}>
              <Box
                sx={{
                  p: 3,
                  borderRadius: 2,
                  background: `${urgencyConfig!.color}10`,
                  border: `1px solid ${urgencyConfig!.color}40`,
                  height: '100%',
                  display: 'flex',
                  flexDirection: 'column',
                  justifyContent: 'center',
                  alignItems: 'center',
                  gap: 2,
                }}
              >
                <Typography variant="overline" sx={{ color: 'text.secondary', fontWeight: 600 }}>
                  REMAINING USEFUL LIFE
                </Typography>

                <AccessTimeIcon sx={{ fontSize: 40, color: urgencyConfig!.color }} />

                <Typography variant="h3" sx={{ color: urgencyConfig!.color, fontWeight: 700 }}>
                  {Math.round(prediction!.rul.rul_hours)} hours
                </Typography>

                <Typography variant="h6" sx={{ color: 'text.secondary' }}>
                  ({prediction!.rul.rul_days.toFixed(1)} days)
                </Typography>

                <Chip
                  label={urgencyConfig!.label}
                  sx={{
                    bgcolor: urgencyConfig!.color,
                    color: '#fff',
                    fontWeight: 600,
                    fontSize: '0.875rem',
                  }}
                />

                <Typography
                  variant="body2"
                  sx={{
                    color: 'text.secondary',
                    textAlign: 'center',
                    mt: 1,
                  }}
                >
                  {urgencyConfig!.description}
                </Typography>

                <Typography variant="body2" sx={{ color: 'text.disabled' }}>
                  {prediction!.rul.maintenance_window}
                </Typography>
              </Box>
            </Grid>
          )}
        </Grid>

        {/* FAILURE TYPE PROBABILITIES */}
        <Box
          sx={(theme) => ({
            p: 3,
            borderRadius: 2,
            bgcolor: alpha(theme.palette.background.paper, theme.palette.mode === 'dark' ? 0.5 : 0.8),
            border: 1,
            borderColor: 'divider',
          })}
        >
          <Typography
            variant="subtitle1"
            sx={{ color: 'text.primary', fontWeight: 600, mb: 2 }}
          >
            FAILURE TYPE PROBABILITIES
          </Typography>

          <Stack spacing={2}>
            {sortedProbabilities.map(({ type, probability }) => (
              <Box key={type}>
                <Box
                  sx={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    mb: 0.5,
                  }}
                >
                  <Typography variant="body2" sx={{ color: 'text.secondary', fontWeight: 500 }}>
                    {type}
                  </Typography>
                  <Typography variant="body2" sx={{ color: 'text.primary', fontWeight: 600 }}>
                    {Math.round(probability * 100)}%
                  </Typography>
                </Box>
                <LinearProgress
                  variant="determinate"
                  value={probability * 100}
                  sx={(theme) => ({
                    height: 8,
                    borderRadius: 4,
                    bgcolor: alpha(theme.palette.text.disabled, 0.15),
                    '& .MuiLinearProgress-bar': {
                      borderRadius: 4,
                      background:
                        probability >= 0.5
                          ? 'linear-gradient(90deg, #ef4444 0%, #dc2626 100%)'
                          : probability >= 0.25
                          ? 'linear-gradient(90deg, #f97316 0%, #ea580c 100%)'
                          : probability >= 0.10
                          ? 'linear-gradient(90deg, #fbbf24 0%, #f59e0b 100%)'
                          : 'linear-gradient(90deg, #10b981 0%, #059669 100%)',
                    },
                  })}
                />
              </Box>
            ))}
          </Stack>
        </Box>

        {/* ACTION BUTTONS */}
        <Stack
          direction={{ xs: 'column', sm: 'row' }}
          spacing={2}
          sx={{ mt: 3 }}
        >
          <Button
            variant="outlined"
            size="large"
            startIcon={<PsychologyIcon />}
            onClick={onExplain}
            fullWidth
            sx={{
              borderColor: '#667eea',
              color: '#667eea',
              '&:hover': {
                borderColor: '#5568d3',
                bgcolor: 'rgba(102, 126, 234, 0.1)',
              },
            }}
          >
            Get AI Explanation
          </Button>

          {onViewHistory && (
            <Button
              variant="outlined"
              size="large"
              startIcon={<HistoryIcon />}
              onClick={onViewHistory}
              fullWidth
              sx={{
                borderColor: '#764ba2',
                color: '#764ba2',
                '&:hover': {
                  borderColor: '#63408b',
                  bgcolor: 'rgba(118, 75, 162, 0.1)',
                },
              }}
            >
              View History
            </Button>
          )}
        </Stack>
      </CardContent>
    </Card>
  );
}
