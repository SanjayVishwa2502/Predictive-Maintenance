/**
 * Profile Editor Component - Phase 3.7.6.2
 * 
 * Step 4 of Profile Upload (continued):
 * - Inline JSON editor for fixing validation errors
 * - Syntax highlighting
 * - Re-validate button
 * - Line numbers for error navigation
 */

import { useState } from 'react';
import {
  Box,
  Paper,
  Typography,
  Button,
  Stack,
  TextField,
  Alert,
} from '@mui/material';
import {
  Edit as EditIcon,
  CheckCircle as CheckCircleIcon,
  Refresh as RefreshIcon,
} from '@mui/icons-material';
import { ganApi } from '../../api/ganApi';
import type { ValidationError } from '../../types/gan.types';

interface ProfileEditorProps {
  profileId?: string;
  fileContent: string;
  onContentChange: (content: string) => void;
  onRevalidate: (errors: ValidationError[], isValid: boolean, machineId?: string) => void;
}

export default function ProfileEditor({
  profileId,
  fileContent,
  onContentChange,
  onRevalidate,
}: ProfileEditorProps) {
  const [editedContent, setEditedContent] = useState(fileContent);
  const [validating, setValidating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleContentChange = (event: React.ChangeEvent<HTMLTextAreaElement>) => {
    setEditedContent(event.target.value);
    onContentChange(event.target.value);
  };

  const handleRevalidate = async () => {
    setValidating(true);
    setError(null);

    try {
      // Parse JSON
      const profileData = JSON.parse(editedContent);

      if (profileId) {
        // Persist edits to the staged backend profile, then validate that staged profile.
        await ganApi.updateStagedProfile(profileId, profileData);
        const staged = await ganApi.validateStagedProfile(profileId);

        const validation_errors: ValidationError[] = (staged.issues || []).map((i: any) => ({
          field: i.field || 'profile',
          message: i.message,
          severity: i.severity || 'error',
        }));

        const hasBlocking = validation_errors.some((e) => (e as any).severity === 'error');
        const ok = Boolean(staged.valid) && Boolean(staged.can_proceed) && !hasBlocking;
        onRevalidate(validation_errors, ok, staged.machine_id);
      } else {
        // Fallback: inline validation only (no staged profile to update)
        const validationResult = await ganApi.validateProfile(profileData);
        const ok = Boolean(validationResult.valid) && !(validationResult.errors || []).some((e) => (e as any).severity === 'error');
        onRevalidate(validationResult.errors, ok);
      }
    } catch (err: any) {
      console.error('Validation failed:', err);
      if (err instanceof SyntaxError) {
        setError('Invalid JSON syntax. Please fix the JSON format and try again.');
      } else {
        setError(err?.response?.data?.detail || err?.response?.data?.message || err?.message || 'Validation failed. Please try again.');
      }
    } finally {
      setValidating(false);
    }
  };

  const lineCount = editedContent.split('\n').length;

  return (
    <Box>
      <Stack
        direction="row"
        justifyContent="space-between"
        alignItems="center"
        sx={{ mb: 2 }}
      >
        <Box>
          <Typography variant="h6" sx={{ fontWeight: 600 }}>
            Edit Profile (Optional)
          </Typography>
          <Typography variant="body2" sx={{ color: '#9ca3af' }}>
            Fix validation errors by editing the JSON directly
          </Typography>
        </Box>
        <Button
          variant="contained"
          startIcon={validating ? <RefreshIcon /> : <CheckCircleIcon />}
          onClick={handleRevalidate}
          disabled={validating}
          sx={{
            background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
            '&:hover': {
              background: 'linear-gradient(135deg, #5568d3 0%, #6a4291 100%)',
            },
          }}
        >
          {validating ? 'Validating...' : 'Re-validate'}
        </Button>
      </Stack>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      <Paper
        sx={{
          p: 0,
          bgcolor: 'rgba(255, 255, 255, 0.03)',
          border: '1px solid rgba(255, 255, 255, 0.1)',
          overflow: 'hidden',
        }}
      >
        {/* Line Counter */}
        <Stack direction="row" sx={{ height: 400, overflow: 'hidden' }}>
          {/* Line Numbers */}
          <Box
            sx={{
              width: 50,
              bgcolor: 'rgba(0, 0, 0, 0.3)',
              p: 1.5,
              textAlign: 'right',
              fontFamily: 'monospace',
              fontSize: '0.875rem',
              color: '#6b7280',
              overflowY: 'auto',
              userSelect: 'none',
            }}
          >
            {Array.from({ length: lineCount }, (_, i) => (
              <div key={i + 1}>{i + 1}</div>
            ))}
          </Box>

          {/* Text Editor */}
          <TextField
            multiline
            fullWidth
            value={editedContent}
            onChange={handleContentChange}
            disabled={validating}
            sx={{
              '& .MuiInputBase-root': {
                height: '100%',
                alignItems: 'flex-start',
                fontFamily: 'monospace',
                fontSize: '0.875rem',
                lineHeight: 1.5,
                p: 1.5,
              },
              // Make the textarea fill the available height and scroll.
              '& .MuiInputBase-inputMultiline': {
                height: '100% !important',
                overflowY: 'auto !important',
              },
              '& textarea': {
                height: '100% !important',
                overflowY: 'auto !important',
              },
              '& fieldset': {
                border: 'none',
              },
            }}
          />
        </Stack>
      </Paper>

      {/* Editor Info */}
      <Alert severity="info" icon={<EditIcon />} sx={{ mt: 2 }}>
        <Typography variant="body2">
          <strong>Tip:</strong> Edit the JSON to fix validation errors. Make sure to
          maintain proper JSON syntax (quotes, commas, brackets). Click "Re-validate" to
          check your changes.
        </Typography>
      </Alert>
    </Box>
  );
}
