import { useMemo } from 'react';
import {
  Accordion,
  AccordionDetails,
  AccordionSummary,
  Box,
  FormControl,
  InputLabel,
  MenuItem,
  Select,
  Typography,
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';

export type TimeLimitSeconds = 900 | 1800 | 3600;

export interface TrainingConfig {
  timeLimitSeconds: TimeLimitSeconds;
}

export const DEFAULT_TRAINING_CONFIG: TrainingConfig = {
  timeLimitSeconds: 900,
};

export function TrainingConfigForm({
  config,
  onChange,
}: {
  config: TrainingConfig;
  onChange: (next: TrainingConfig) => void;
}) {
  const timeLimitValue = useMemo(() => config.timeLimitSeconds, [config.timeLimitSeconds]);

  return (
    <Box sx={{ mt: 3 }}>
      <FormControl fullWidth>
        <InputLabel id="ml-training-time-limit">Time Limit</InputLabel>
        <Select
          labelId="ml-training-time-limit"
          label="Time Limit"
          value={timeLimitValue}
          onChange={(e) => onChange({ ...config, timeLimitSeconds: Number(e.target.value) as TimeLimitSeconds })}
        >
          <MenuItem value={900}>15 minutes (Fast - Pi)</MenuItem>
          <MenuItem value={1800}>30 minutes (Standard)</MenuItem>
          <MenuItem value={3600}>60 minutes (High Quality)</MenuItem>
        </Select>
      </FormControl>

      <Accordion sx={{ mt: 2 }}>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>Advanced Options</AccordionSummary>
        <AccordionDetails>
          <Typography variant="body2" color="text.secondary">
            Advanced training options will be added in later phases.
          </Typography>
        </AccordionDetails>
      </Accordion>
    </Box>
  );
}
