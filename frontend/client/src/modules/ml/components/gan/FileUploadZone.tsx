/**
 * File Upload Zone Component - Phase 3.7.6.2
 * 
 * Step 3 of Profile Upload:
 * - Drag-and-drop zone for JSON/YAML/Excel files
 * - File browser fallback
 * - File type validation
 * - Immediate upload and validation
 */

import { useState, useCallback } from 'react';
import {
  Box,
  Paper,
  Typography,
  Button,
  CircularProgress,
  Alert,
  Chip,
  Stack,
} from '@mui/material';
import {
  CloudUpload as CloudUploadIcon,
  CheckCircle as CheckCircleIcon,
  InsertDriveFile as FileIcon,
} from '@mui/icons-material';
import { ganApi, parseFileContent } from '../../api/ganApi';
import type { ProfileUploadResponse } from '../../types/gan.types';

interface FileUploadZoneProps {
  onUploadSuccess: (response: ProfileUploadResponse, fileContent: string) => void;
  onUploadError: (error: string) => void;
  machineType?: string;
  onMachineTypeDetected?: (machineType: string) => void;
}

export default function FileUploadZone({
  onUploadSuccess,
  onUploadError,
  machineType,
  onMachineTypeDetected,
}: FileUploadZoneProps) {
  const [dragActive, setDragActive] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);

  // Accepted file types
  const acceptedTypes = ['.json', '.yaml', '.yml', '.xlsx', '.xls'];
  const acceptedMimeTypes = [
    'application/json',
    'application/x-yaml',
    'text/yaml',
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    'application/vnd.ms-excel',
  ];

  const validateFileType = (file: File): boolean => {
    const extension = file.name.substring(file.name.lastIndexOf('.')).toLowerCase();
    return acceptedTypes.includes(extension) || acceptedMimeTypes.includes(file.type);
  };

  const handleFileSelect = (file: File) => {
    if (!validateFileType(file)) {
      const msg = 'Invalid file type. Please upload a JSON, YAML, or Excel file.';
      setUploadError(msg);
      onUploadError(msg);
      return;
    }
    setSelectedFile(file);
    setUploadError(null);
  };

  const handleUploadClick = async () => {
    if (!selectedFile) {
      const msg = 'Please select a file first';
      setUploadError(msg);
      onUploadError(msg);
      return;
    }

    setUploading(true);
    setUploadError(null);

    try {
      // Read file content for preview/editing
      const content = await parseFileContent(selectedFile);

      // Auto-detect machine type from profile when present (prevents stale selection mismatches)
      try {
        const parsed = JSON.parse(content);
        const detected = String((parsed as any)?.machine_type || (parsed as any)?.category || '').trim();
        if (detected) onMachineTypeDetected?.(detected);
      } catch {
        // Ignore detection errors here; uploadProfile will handle parse errors.
      }

      // Upload to backend for validation
      const response = await ganApi.uploadProfile(selectedFile, (machineType || '').trim() || undefined);

      if (response.status === 'validated' && response.validation_errors.length === 0) {
        onUploadSuccess(response, content);
      } else {
        onUploadSuccess(response, content);
      }
    } catch (error: any) {
      console.error('Upload failed:', error);
      const msg =
        error?.message ||
        error?.response?.data?.detail ||
        error?.response?.data?.message ||
        'Failed to upload file. Please try again.';
      setUploadError(msg);
      onUploadError(msg);
    } finally {
      setUploading(false);
    }
  };

  // Drag and drop handlers
  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      setDragActive(false);

      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        handleFileSelect(e.dataTransfer.files[0]);
      }
    },
    []
  );

  // File input handler
  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      handleFileSelect(e.target.files[0]);
    }
  };

  return (
    <Box>
      <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
        Step 2: Upload Machine Profile
      </Typography>
      <Typography variant="body2" sx={{ color: '#9ca3af', mb: 3 }}>
        Drag and drop your filled profile file, or click to browse
      </Typography>

      <Paper
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        sx={{
          p: 4,
          textAlign: 'center',
          bgcolor: dragActive
            ? 'rgba(102, 126, 234, 0.1)'
            : 'rgba(255, 255, 255, 0.03)',
          border: dragActive
            ? '2px dashed #667eea'
            : '2px dashed rgba(255, 255, 255, 0.2)',
          borderRadius: 2,
          cursor: uploading ? 'not-allowed' : 'pointer',
          transition: 'all 0.3s ease',
          position: 'relative',
          '&:hover': {
            bgcolor: 'rgba(102, 126, 234, 0.05)',
            borderColor: '#667eea',
          },
        }}
      >
        {uploading ? (
          <Box>
            <CircularProgress size={48} sx={{ color: '#667eea', mb: 2 }} />
            <Typography variant="body1" sx={{ color: '#d1d5db' }}>
              Uploading and validating...
            </Typography>
            {selectedFile && (
              <Typography variant="body2" sx={{ color: '#9ca3af', mt: 1 }}>
                {selectedFile.name} ({Math.round(selectedFile.size / 1024)} KB)
              </Typography>
            )}
          </Box>
        ) : (
          <Box>
            <CloudUploadIcon
              sx={{ fontSize: 64, color: '#667eea', mb: 2, opacity: 0.8 }}
            />
            <Typography variant="h6" sx={{ color: '#e5e7eb', mb: 1 }}>
              {dragActive ? 'Drop file here' : 'Drag & drop your profile file'}
            </Typography>
            <Typography variant="body2" sx={{ color: '#9ca3af', mb: 3 }}>
              or
            </Typography>

            <Button
              component="label"
              variant="contained"
              startIcon={<FileIcon />}
              disabled={uploading}
              sx={{
                background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                '&:hover': {
                  background: 'linear-gradient(135deg, #5568d3 0%, #6a4291 100%)',
                },
              }}
            >
              Browse Files
              <input
                type="file"
                hidden
                accept={acceptedTypes.join(',')}
                onChange={handleFileInput}
                disabled={uploading}
              />
            </Button>

            <Stack
              direction="row"
              spacing={1}
              justifyContent="center"
              sx={{ mt: 3 }}
            >
              <Typography variant="caption" sx={{ color: '#6b7280' }}>
                Accepted formats:
              </Typography>
              {acceptedTypes.slice(0, 3).map((type) => (
                <Chip
                  key={type}
                  label={type.replace('.', '').toUpperCase()}
                  size="small"
                  sx={{
                    bgcolor: 'rgba(102, 126, 234, 0.15)',
                    color: '#667eea',
                    fontSize: '0.7rem',
                  }}
                />
              ))}
            </Stack>
          </Box>
        )}
      </Paper>

      {/* File Info and Upload Button */}
      {selectedFile && !uploading && (
        <Box sx={{ mt: 3 }}>
          {uploadError && (
            <Alert severity="error" sx={{ mb: 2 }}>
              {uploadError}
            </Alert>
          )}
          <Alert
            severity="info"
            icon={<CheckCircleIcon />}
            sx={{ mb: 2 }}
          >
            <Typography variant="body2">
              <strong>File selected:</strong> {selectedFile.name} (
              {Math.round(selectedFile.size / 1024)} KB)
            </Typography>
          </Alert>

          <Button
            variant="contained"
            fullWidth
            size="large"
            startIcon={<CloudUploadIcon />}
            onClick={handleUploadClick}
            sx={{
              background: 'linear-gradient(135deg, #10b981 0%, #059669 100%)',
              py: 1.5,
              fontSize: '1rem',
              fontWeight: 600,
              '&:hover': {
                background: 'linear-gradient(135deg, #059669 0%, #047857 100%)',
              },
            }}
          >
            Upload and Validate Profile
          </Button>
        </Box>
      )}
    </Box>
  );
}
