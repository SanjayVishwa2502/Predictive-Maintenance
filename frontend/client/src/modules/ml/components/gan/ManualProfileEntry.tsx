/**
 * Manual Profile Entry Component - Phase 3.7.6.2 Enhancement
 * 
 * Form-based manual entry for machine profiles
 * Alternative to file upload for users without templates
 */

import { useState } from 'react';
import {
  Box,
  TextField,
  Button,
  Grid,
  Typography,
  Paper,
  Stack,
  Alert,
  Divider,
  IconButton,
} from '@mui/material';
import {
  Add as AddIcon,
  Remove as RemoveIcon,
  Save as SaveIcon,
} from '@mui/icons-material';
import type { MachineProfile } from '../../types/gan.types';
import { validateMachineProfile } from '../../utils/profileValidation';
import { ganApi } from '../../api/ganApi';

interface ManualProfileEntryProps {
  onProfileCreated: (profile: Partial<MachineProfile>) => void;
}

interface SensorField {
  name: string;
  min: string;
  typical: string;
  max: string;
  unit: string;
}

export default function ManualProfileEntry({ onProfileCreated }: ManualProfileEntryProps) {
  // Basic fields (REQUIRED based on GAN workflow analysis)
  const [machineId, setMachineId] = useState('');
  const [manufacturer, setManufacturer] = useState('');
  const [model, setModel] = useState('');
  const [category, setCategory] = useState('');

  // Simple flat sensor list (validation will auto-group them)
  const [sensors, setSensors] = useState<SensorField[]>([
    { name: 'bearing_temp_C', min: '40', typical: '55', max: '70', unit: '°C' },
    { name: 'overall_rms_mm_s', min: '0.7', typical: '1.2', max: '1.8', unit: 'mm/s' },
    { name: 'current_A', min: '2.0', typical: '11.6', max: '13.0', unit: 'A' },
  ]);

  const [error, setError] = useState<string | null>(null);
  const [validationWarnings, setValidationWarnings] = useState<string[]>([]);

  const addSensor = () => {
    const newSensor: SensorField = { name: '', min: '', typical: '', max: '', unit: '' };
    setSensors([...sensors, newSensor]);
  };

  const removeSensor = (index: number) => {
    setSensors(sensors.filter((_, i) => i !== index));
  };

  const updateSensor = (
    index: number,
    field: keyof SensorField,
    value: string
  ) => {
    const updated = [...sensors];
    updated[index][field] = value;
    setSensors(updated);
  };

  const handleSaveProfile = () => {
    setError(null);
    setValidationWarnings([]);

    if (!category.trim()) {
      setError('category: Please select/enter a machine type (category)');
      return;
    }

    // Build flat profile object - validation will auto-group sensors
    const baseline_normal_operation: any = {};

    // Add all sensors as flat structure
    sensors.forEach(sensor => {
      if (sensor.name) {
        baseline_normal_operation[sensor.name] = {
          min: parseFloat(sensor.min) || 0,
          typical: parseFloat(sensor.typical) || 0,
          max: parseFloat(sensor.max) || 0,
          unit: sensor.unit,
        };
      }
    });

    const profile = {
      machine_id: machineId,
      manufacturer,
      model,
      category,
      baseline_normal_operation,
    };

    // Validate locally first (fast UX)
    const validationResult = validateMachineProfile(profile);

    if (!validationResult.valid) {
      setError(validationResult.errors.map(e => `${e.field}: ${e.message}`).join('\n'));
      return;
    }

    if (validationResult.warnings.length > 0) {
      setValidationWarnings(validationResult.warnings.map(w => `${w.field}: ${w.message}`));
    }

    // Authoritative backend validation (duplicates/TVAE readiness)
    const normalized = validationResult.normalized || profile;
    ganApi.validateProfile(normalized)
      .then((backendResult) => {
        if (!backendResult.valid) {
          setError(backendResult.errors.map(e => `${e.field}: ${e.message}`).join('\n'));
          return;
        }

        onProfileCreated(normalized);
      })
      .catch((err: any) => {
        setError(err?.response?.data?.detail || err?.message || 'Backend validation failed.');
      });
  };

  const renderSensorFields = () => (
    <Paper sx={{ p: 3, bgcolor: 'rgba(255, 255, 255, 0.02)', border: '1px solid rgba(255, 255, 255, 0.1)' }}>
      <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: 2 }}>
        <Typography variant="h6" sx={{ fontWeight: 600 }}>
          Sensor Data
        </Typography>
        <Button
          size="small"
          startIcon={<AddIcon />}
          onClick={addSensor}
          sx={{ color: '#667eea' }}
        >
          Add Sensor
        </Button>
      </Stack>

      <Typography variant="caption" sx={{ color: '#9ca3af', display: 'block', mb: 2 }}>
        Add sensor specifications. Validation will automatically group them by type.
      </Typography>

      <Stack spacing={2}>
        {sensors.map((sensor, index) => (
          <Box key={index}>
            <Grid container spacing={2} alignItems="center">
              <Grid item xs={12} sm={4}>
                <TextField
                  fullWidth
                  size="small"
                  label="Sensor Name"
                  placeholder="e.g., bearing_temp_C, vibration_mm_s"
                  value={sensor.name}
                  onChange={(e) => updateSensor(index, 'name', e.target.value)}
                />
              </Grid>
              <Grid item xs={6} sm={1.5}>
                <TextField
                  fullWidth
                  size="small"
                  label="Min"
                  type="number"
                  value={sensor.min}
                  onChange={(e) => updateSensor(index, 'min', e.target.value)}
                />
              </Grid>
              <Grid item xs={6} sm={1.5}>
                <TextField
                  fullWidth
                  size="small"
                  label="Typical"
                  type="number"
                  value={sensor.typical}
                  onChange={(e) => updateSensor(index, 'typical', e.target.value)}
                />
              </Grid>
              <Grid item xs={6} sm={1.5}>
                <TextField
                  fullWidth
                  size="small"
                  label="Max"
                  type="number"
                  value={sensor.max}
                  onChange={(e) => updateSensor(index, 'max', e.target.value)}
                />
              </Grid>
              <Grid item xs={6} sm={2}>
                <TextField
                  fullWidth
                  size="small"
                  label="Unit"
                  placeholder="e.g., °C, mm/s, A"
                  value={sensor.unit}
                  onChange={(e) => updateSensor(index, 'unit', e.target.value)}
                />
              </Grid>
              <Grid item xs={12} sm={1.5}>
                <IconButton
                  color="error"
                  onClick={() => removeSensor(index)}
                  disabled={sensors.length === 1}
                >
                  <RemoveIcon />
                </IconButton>
              </Grid>
            </Grid>
            {index < sensors.length - 1 && <Divider sx={{ mt: 2, bgcolor: 'rgba(255, 255, 255, 0.05)' }} />}
          </Box>
        ))}
      </Stack>
    </Paper>
  );

  return (
    <Box>
      <Typography variant="h6" gutterBottom sx={{ fontWeight: 600, mb: 3 }}>
        Manual Profile Entry
      </Typography>

      {error && (
        <Alert severity="error" sx={{ mb: 3 }} onClose={() => setError(null)}>
          {error.split('\n').map((line, i) => (
            <Typography key={i} variant="body2">{line}</Typography>
          ))}
        </Alert>
      )}

      {validationWarnings.length > 0 && (
        <Alert severity="info" sx={{ mb: 3 }}>
          <Typography variant="body2" sx={{ fontWeight: 600, mb: 0.5 }}>Validation Notes:</Typography>
          {validationWarnings.map((warning, i) => (
            <Typography key={i} variant="body2">{warning}</Typography>
          ))}
        </Alert>
      )}

      <Stack spacing={3}>
        {/* Basic Information */}
        <Paper sx={{ p: 3, bgcolor: 'rgba(255, 255, 255, 0.02)', border: '1px solid rgba(255, 255, 255, 0.1)' }}>
          <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>
            Basic Information
          </Typography>
          <Alert severity="info" sx={{ mb: 2 }}>
            <Typography variant="caption">
              <strong>Required fields (*)</strong> are essential for GAN workflow (seed generation → training → synthesis)
            </Typography>
          </Alert>
          <Grid container spacing={2}>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                required
                label="Machine ID *"
                placeholder="motor_siemens_1la7_001"
                value={machineId}
                onChange={(e) => setMachineId(e.target.value)}
                helperText="Format: <type>_<manufacturer>_<model>_<id>"
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                label="Category"
                placeholder="motor, pump, cnc, compressor"
                value={category}
                onChange={(e) => setCategory(e.target.value)}
                helperText="Auto-inferred from machine_id if missing"
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                label="Manufacturer"
                placeholder="Siemens, DMG MORI, Grundfos"
                value={manufacturer}
                onChange={(e) => setManufacturer(e.target.value)}
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                label="Model"
                placeholder="1LA7 113-4AA60, GA 30, CR3-19"
                value={model}
                onChange={(e) => setModel(e.target.value)}
              />
            </Grid>
          </Grid>
        </Paper>

        {/* Sensor Data */}
        {renderSensorFields()}

        {/* Save Button */}
        <Button
          variant="contained"
          size="large"
          fullWidth
          startIcon={<SaveIcon />}
          onClick={handleSaveProfile}
          sx={{
            background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
            py: 1.5,
            fontSize: '1rem',
            fontWeight: 600,
            '&:hover': {
              background: 'linear-gradient(135deg, #5568d3 0%, #6a4291 100%)',
            },
          }}
        >
          Save Profile
        </Button>
      </Stack>
    </Box>
  );
}
