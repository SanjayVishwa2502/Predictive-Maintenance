/**
 * Template Selector Component - Phase 3.7.6.2
 * 
 * Step 1 of Profile Upload:
 * - Dropdown to select machine type (11 options)
 * - Download template button
 * - Download example profile button
 */

import { useState } from 'react';
import {
  Box,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Button,
  Typography,
  Alert,
  Stack,
} from '@mui/material';
import type { SelectChangeEvent } from '@mui/material';
import {
  Download as DownloadIcon,
  Description as DescriptionIcon,
} from '@mui/icons-material';
import { MACHINE_TYPES } from '../../types/gan.types';
import { downloadTemplateAsFile, ganApi } from '../../api/ganApi';

interface TemplateSelectorProps {
  onTemplateDownloaded?: (machineType: string) => void;
  selectedType?: string;
  onSelectedTypeChange?: (machineType: string) => void;
}

export default function TemplateSelector({
  onTemplateDownloaded,
  selectedType: selectedTypeProp,
  onSelectedTypeChange,
}: TemplateSelectorProps) {
  const [selectedTypeState, setSelectedTypeState] = useState<string>('');
  const selectedType = selectedTypeProp ?? selectedTypeState;
  const [downloading, setDownloading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleTypeChange = (event: SelectChangeEvent) => {
    const next = event.target.value;
    setSelectedTypeState(next);
    onSelectedTypeChange?.(next);
    setError(null);
  };

  const handleDownloadTemplate = async () => {
    if (!selectedType) {
      setError('Please select a machine type first');
      return;
    }

    setDownloading(true);
    setError(null);

    try {
      await downloadTemplateAsFile(selectedType);
      onTemplateDownloaded?.(selectedType);
    } catch (err) {
      console.error('Template download failed:', err);
      setError('Failed to download template. Please try again.');
    } finally {
      setDownloading(false);
    }
  };

  const handleDownloadExample = async () => {
    if (!selectedType) {
      setError('Please select a machine type first');
      return;
    }

    setDownloading(true);
    setError(null);

    try {
      const example = await ganApi.getExampleProfile(selectedType);
      const blob = new Blob([JSON.stringify(example, null, 2)], {
        type: 'application/json',
      });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `${selectedType}_example.json`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
    } catch (err) {
      console.error('Example download failed:', err);
      setError('Failed to download example. Please try again.');
    } finally {
      setDownloading(false);
    }
  };

  return (
    <Box>
      <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
        Step 1: Download Machine Profile Template
      </Typography>
      <Typography variant="body2" sx={{ color: '#9ca3af', mb: 3 }}>
        Select a machine type and download a JSON template to fill out with your machine's
        specifications
      </Typography>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      <Stack spacing={2}>
        {/* Machine Type Selector */}
        <FormControl fullWidth>
          <InputLabel id="machine-type-label">Machine Type</InputLabel>
          <Select
            labelId="machine-type-label"
            id="machine-type-select"
            value={selectedType}
            label="Machine Type"
            onChange={handleTypeChange}
          >
            {MACHINE_TYPES.map((type) => (
              <MenuItem key={type.value} value={type.value}>
                {type.label}
              </MenuItem>
            ))}
          </Select>
        </FormControl>

        {/* Action Buttons */}
        <Stack direction="row" spacing={2}>
          <Button
            variant="contained"
            startIcon={<DownloadIcon />}
            onClick={handleDownloadTemplate}
            disabled={!selectedType || downloading}
            sx={{
              flex: 1,
              background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
              '&:hover': {
                background: 'linear-gradient(135deg, #5568d3 0%, #6a4291 100%)',
              },
            }}
          >
            {downloading ? 'Downloading...' : 'Download Template'}
          </Button>

          <Button
            variant="outlined"
            startIcon={<DescriptionIcon />}
            onClick={handleDownloadExample}
            disabled={!selectedType || downloading}
            sx={{
              flex: 1,
              borderColor: '#667eea',
              color: '#667eea',
              '&:hover': {
                borderColor: '#5568d3',
                bgcolor: 'rgba(102, 126, 234, 0.1)',
              },
            }}
          >
            {downloading ? 'Downloading...' : 'Download Example'}
          </Button>
        </Stack>

        {/* Instructions */}
        {selectedType && (
          <Alert severity="info" icon={<DescriptionIcon />}>
            <Typography variant="body2" sx={{ fontWeight: 600, mb: 0.5 }}>
              Next Steps:
            </Typography>
            <Typography variant="body2" component="ol" sx={{ pl: 2, m: 0 }}>
              <li>Open the downloaded JSON file in a text editor</li>
              <li>Fill in your machine's sensor names, operational parameters, and RUL config</li>
              <li>Save the file and upload it using the drag-and-drop zone below</li>
            </Typography>
          </Alert>
        )}
      </Stack>
    </Box>
  );
}
