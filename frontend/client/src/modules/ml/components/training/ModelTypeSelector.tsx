import { useCallback, useEffect, useMemo } from 'react';
import type { ReactNode } from 'react';
import {
  Alert,
  AlertTitle,
  Box,
  Button,
  Card,
  CardContent,
  Checkbox,
  Chip,
  Grid,
  Typography,
} from '@mui/material';
import CategoryIcon from '@mui/icons-material/Category';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import WarningIcon from '@mui/icons-material/Warning';
import ShowChartIcon from '@mui/icons-material/ShowChart';
import type { ModelType } from '../../api/mlTrainingApi';

const MODEL_DEFS: Array<{
  type: ModelType;
  label: string;
  description: string;
  purpose: string;
  estimatedTime: string;
  required: boolean;
  icon: ReactNode;
}> = [
  {
    type: 'classification',
    label: 'Classification (Failure Prediction)',
    icon: <CategoryIcon />,
    description: 'Binary classification: normal vs. failure',
    purpose: 'Answers: Will this machine fail?',
    estimatedTime: '~1 minute',
    required: true,
  },
  {
    type: 'regression',
    label: 'Regression (RUL Estimation)',
    icon: <TrendingUpIcon />,
    description: 'Predict remaining useful life in hours',
    purpose: 'Answers: How much time until failure?',
    estimatedTime: '~3 minutes',
    required: true,
  },
  {
    type: 'anomaly',
    label: 'Anomaly Detection',
    icon: <WarningIcon />,
    description: 'Ensemble anomaly detectors',
    purpose: 'Answers: Is current behavior abnormal?',
    estimatedTime: '~10 seconds',
    required: true,
  },
  {
    type: 'timeseries',
    label: 'Time-Series Forecast',
    icon: <ShowChartIcon />,
    description: '24-hour ahead sensor predictions',
    purpose: 'Answers: What will sensors look like tomorrow?',
    estimatedTime: '~10 seconds',
    required: true,
  },
];

const ALL_MODELS: ReadonlyArray<ModelType> = ['classification', 'regression', 'anomaly', 'timeseries'];

export function ModelTypeSelector({
  selected,
  onChange,
}: {
  selected: Set<ModelType>;
  onChange: (next: Set<ModelType>) => void;
}) {
  const allSelected = useMemo(() => selected.size === ALL_MODELS.length, [selected]);

  // All 4 models should be selected by default
  useEffect(() => {
    if (selected.size === 0) {
      onChange(new Set(ALL_MODELS));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const setAll = useCallback(() => {
    onChange(new Set(ALL_MODELS));
  }, [onChange]);

  const toggleModel = useCallback(
    (modelType: ModelType) => {
      const next = new Set(selected);
      if (next.has(modelType)) next.delete(modelType);
      else next.add(modelType);
      onChange(next);
    },
    [selected, onChange]
  );

  return (
    <Box sx={{ mt: 3 }}>
      <Alert severity="info" sx={{ mb: 2 }}>
        <AlertTitle>Recommended: Train All 4 Models</AlertTitle>
        Each model serves a distinct purpose. Training all 4 provides complete predictive maintenance coverage.
      </Alert>

      <Button variant="contained" fullWidth sx={{ mb: 2 }} onClick={setAll}>
        Select All 4 Models (Recommended)
      </Button>

      <Grid container spacing={2}>
        {MODEL_DEFS.map((model) => {
          const checked = selected.has(model.type);
          return (
            <Grid item xs={12} md={6} key={model.type}>
              <Card
                variant={checked ? 'outlined' : undefined}
                onClick={() => toggleModel(model.type)}
                sx={{
                  cursor: 'pointer',
                  ...(checked
                    ? {
                        borderWidth: 2,
                      }
                    : null),
                }}
              >
                <CardContent>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                    <Checkbox checked={checked} color="primary" />
                    {model.icon}
                    <Typography variant="h6" sx={{ flex: 1 }}>
                      {model.label}
                    </Typography>
                    {model.required && <Chip label="REQUIRED" size="small" color="primary" />}
                  </Box>

                  <Typography variant="body2" color="text.secondary">
                    {model.description}
                  </Typography>

                  <Typography variant="body2" color="primary" sx={{ mt: 1, fontWeight: 600 }}>
                    {model.purpose}
                  </Typography>

                  <Chip label={model.estimatedTime} size="small" sx={{ mt: 1 }} />
                </CardContent>
              </Card>
            </Grid>
          );
        })}
      </Grid>

      {!allSelected && (
        <Alert severity="warning" sx={{ mt: 2 }}>
          Incomplete ML system: {ALL_MODELS.length - selected.size} model(s) not selected. For full predictive
          maintenance capability, train all 4 models.
        </Alert>
      )}
    </Box>
  );
}

export { ALL_MODELS };
