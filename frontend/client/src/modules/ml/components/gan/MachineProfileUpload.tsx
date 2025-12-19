/**
 * Machine Profile Upload Component - Phase 3.7.6.2
 * 
 * Main component integrating the 5-step profile upload workflow:
 * 1. Download template
 * 2. Fill profile externally
 * 3. Upload file
 * 4. Review & fix errors
 * 5. Confirm & save
 * 
 * Sub-components:
 * - TemplateSelector
 * - FileUploadZone
 * - ValidationResults
 * - ProfileEditor
 */

import { useState } from 'react';
import {
  Box,
  Paper,
  Stepper,
  Step,
  StepLabel,
  Button,
  Typography,
  Divider,
  Stack,
  Alert,
  LinearProgress,
} from '@mui/material';
import {
  NavigateNext as NextIcon,
  CheckCircle as CheckCircleIcon,
  Refresh as RefreshIcon,
} from '@mui/icons-material';
import TemplateSelector from './TemplateSelector';
import FileUploadZone from './FileUploadZone';
import ValidationResults from './ValidationResults';
import ProfileEditor from './ProfileEditor';import ManualProfileEntry from './ManualProfileEntry';import type { ProfileUploadResponse, ValidationError } from '../../types/gan.types';
import { ganApi } from '../../api/ganApi';

interface MachineProfileUploadProps {
  onProfileCreated: (machineId: string) => void;
}

export default function MachineProfileUpload({
  onProfileCreated,
}: MachineProfileUploadProps) {
  const [currentStep, setCurrentStep] = useState(0);
  const [machineType, setMachineType] = useState<string>('');
  const [uploadResponse, setUploadResponse] = useState<ProfileUploadResponse | null>(
    null
  );
  const [fileContent, setFileContent] = useState<string>('');
  const [validationErrors, setValidationErrors] = useState<ValidationError[]>([]);
  const [isValid, setIsValid] = useState(false);
  const [creatingMachine, setCreatingMachine] = useState(false);
  const [createMachineError, setCreateMachineError] = useState<string | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);

  const steps = [
    'Choose Method',
    'Enter/Upload Data',
    'Review Validation',
    'Create Machine',
  ];

  const [manualEntryMode, setManualEntryMode] = useState(false);

  const handleTemplateDownloaded = () => {
    // Optionally auto-advance to next step
    // setCurrentStep(1);
  };

  const handleMachineTypeSelected = (type: string) => {
    setMachineType(type);
    setUploadError(null);
  };

  const handleManualProfileCreated = (profile: any) => {
    // Simulate upload response for manual entry
    const manualUploadResponse: ProfileUploadResponse = {
      profile_id: (globalThis.crypto?.randomUUID ? globalThis.crypto.randomUUID() : 'manual'),
      machine_id: profile.machine_id || 'manual_entry_001',
      status: 'validated',
      validation_errors: [],
      warnings: [],
      profile: profile,
      next_step: 'Proceed to Create Machine',
    };

    setUploadResponse(manualUploadResponse);
    setFileContent(JSON.stringify(profile, null, 2));
    setValidationErrors([]);
    setIsValid(true);
    setCurrentStep(2); // Move to validation step
  };

  const handleUploadSuccess = (response: ProfileUploadResponse, content: string) => {
    setUploadError(null);
    setUploadResponse(response);
    setFileContent(content);
    setValidationErrors(response.validation_errors || []);
    const hasBlocking = (response.validation_errors || []).some((e) => (e as any).severity === 'error');
    setIsValid(response.status === 'validated' && !hasBlocking);
    setCurrentStep(2); // Move to validation step
  };

  const handleUploadError = (error: string) => {
    console.error('Upload error:', error);
    setUploadError(error || 'Failed to upload file.');
  };

  const handleContentChange = (content: string) => {
    setFileContent(content);
  };

  const handleRevalidate = (errors: ValidationError[], valid: boolean, machineId?: string) => {
    setValidationErrors(errors);
    const hasBlocking = (errors || []).some((e) => (e as any).severity === 'error');
    setIsValid(Boolean(valid) && !hasBlocking);

    if (machineId && uploadResponse && machineId !== uploadResponse.machine_id) {
      setUploadResponse({
        ...uploadResponse,
        machine_id: machineId,
      });
    }

    if (valid && !hasBlocking) {
      // Update upload response
      if (uploadResponse) {
        setUploadResponse({
          ...uploadResponse,
          status: 'validated',
          validation_errors: errors || [],
        });
      }
    }
  };

  const handleCreateMachine = () => {
    if (uploadResponse && isValid) {
      onProfileCreated(uploadResponse.machine_id);
    }
  };

  const handleConfirmAndCreateMachine = async () => {
    if (!uploadResponse || !isValid) return;
    setCreateMachineError(null);
    setCreatingMachine(true);
    try {
      await ganApi.createMachineFromProfile(uploadResponse.profile_id);
      setCurrentStep(3);
    } catch (e: any) {
      setCreateMachineError(e?.response?.data?.detail || e?.message || 'Failed to create machine');
    } finally {
      setCreatingMachine(false);
    }
  };

  const handleReset = () => {
    setCurrentStep(0);
    setMachineType('');
    setUploadResponse(null);
    setFileContent('');
    setValidationErrors([]);
    setIsValid(false);
    setManualEntryMode(false);
    setUploadError(null);
  };

  return (
    <Box>
      {/* Header */}
      <Typography
        variant="h4"
        sx={{
          fontWeight: 700,
          mb: 1,
          background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent',
        }}
      >
        New Machine Profile
      </Typography>
      <Typography variant="body1" sx={{ color: '#9ca3af', mb: 4 }}>
        Create a new machine profile to generate synthetic training data
      </Typography>

      {/* Progress Stepper */}
      <Stepper activeStep={currentStep} sx={{ mb: 4 }}>
        {steps.map((label) => (
          <Step key={label}>
            <StepLabel>{label}</StepLabel>
          </Step>
        ))}
      </Stepper>

      <Divider sx={{ mb: 4, bgcolor: 'rgba(255, 255, 255, 0.1)' }} />

      {/* Step Content */}
      <Paper
        sx={{
          p: 4,
          bgcolor: 'rgba(255, 255, 255, 0.02)',
          border: '1px solid rgba(255, 255, 255, 0.1)',
        }}
      >
        {uploadError && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {uploadError}
          </Alert>
        )}

        {/* Step 0: Choose Method */}
        {currentStep === 0 && (
          <Stack spacing={3}>
            <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>
              Choose Profile Creation Method
            </Typography>

            <Stack direction="row" spacing={3}>
              <Paper
                onClick={() => {
                  setManualEntryMode(false);
                  setCurrentStep(1);
                }}
                sx={{
                  flex: 1,
                  p: 4,
                  textAlign: 'center',
                  cursor: 'pointer',
                  bgcolor: 'rgba(102, 126, 234, 0.1)',
                  border: '2px solid rgba(102, 126, 234, 0.3)',
                  transition: 'all 0.3s',
                  '&:hover': {
                    border: '2px solid #667eea',
                    transform: 'translateY(-4px)',
                    boxShadow: '0 8px 24px rgba(102, 126, 234, 0.2)',
                  },
                }}
              >
                <Typography variant="h6" sx={{ fontWeight: 600, mb: 1 }}>
                  📄 File Upload
                </Typography>
                <Typography variant="body2" sx={{ color: '#9ca3af' }}>
                  Download template, fill it externally, and upload
                </Typography>
              </Paper>

              <Paper
                onClick={() => {
                  setManualEntryMode(true);
                  setCurrentStep(1);
                }}
                sx={{
                  flex: 1,
                  p: 4,
                  textAlign: 'center',
                  cursor: 'pointer',
                  bgcolor: 'rgba(118, 75, 162, 0.1)',
                  border: '2px solid rgba(118, 75, 162, 0.3)',
                  transition: 'all 0.3s',
                  '&:hover': {
                    border: '2px solid #764ba2',
                    transform: 'translateY(-4px)',
                    boxShadow: '0 8px 24px rgba(118, 75, 162, 0.2)',
                  },
                }}
              >
                <Typography variant="h6" sx={{ fontWeight: 600, mb: 1 }}>
                  ✏️ Manual Entry
                </Typography>
                <Typography variant="body2" sx={{ color: '#9ca3af' }}>
                  Enter profile data directly using a form
                </Typography>
              </Paper>
            </Stack>

            <Alert severity="info">
              <strong>File Upload:</strong> Best when you have existing JSON profiles or prefer working in external editors.
              <br />
              <strong>Manual Entry:</strong> Best when creating profiles from scratch or when no template is available.
            </Alert>

            {/* Template Download Section */}
            {!manualEntryMode && (
              <Box sx={{ mt: 3, pt: 3, borderTop: '1px solid rgba(255, 255, 255, 0.1)' }}>
                <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 2 }}>
                  Select Machine Type (Optional for Upload)
                </Typography>
                <TemplateSelector
                  selectedType={machineType}
                  onSelectedTypeChange={handleMachineTypeSelected}
                  onTemplateDownloaded={handleTemplateDownloaded}
                />
              </Box>
            )}
          </Stack>
        )}

        {/* Step 1: Data Entry */}
        {currentStep === 1 && (
          <>
            {manualEntryMode ? (
              <ManualProfileEntry onProfileCreated={handleManualProfileCreated} />
            ) : (
              <FileUploadZone
                onUploadSuccess={handleUploadSuccess}
                onUploadError={handleUploadError}
                machineType={machineType}
                onMachineTypeDetected={handleMachineTypeSelected}
              />
            )}
          </>
        )}

        {/* Step 2: Validation Results + Editor */}
        {currentStep === 2 && (
          <Stack spacing={4}>
            <ValidationResults
              errors={validationErrors}
              isValid={isValid}
              machineId={uploadResponse?.machine_id}
            />

            {/* Manual edit is always available for JSON profiles */}
            <ProfileEditor
              profileId={uploadResponse?.profile_id}
              fileContent={fileContent}
              onContentChange={handleContentChange}
              onRevalidate={handleRevalidate}
            />
          </Stack>
        )}

        {/* Step 3: Confirmation */}
        {currentStep === 3 && (
          <Box sx={{ textAlign: 'center' }}>
            <CheckCircleIcon
              sx={{ fontSize: 80, color: '#10b981', mb: 2, opacity: 0.9 }}
            />
            <Typography variant="h5" sx={{ fontWeight: 600, mb: 1 }}>
              Machine Profile Created!
            </Typography>
            <Typography variant="body1" sx={{ color: '#9ca3af', mb: 3 }}>
              Machine ID: <strong>{uploadResponse?.machine_id}</strong>
            </Typography>
            <Alert severity="success" sx={{ mb: 3 }}>
              Your machine profile has been successfully created and validated. You can now
              proceed to generate seed data using the GAN workflow.
            </Alert>
          </Box>
        )}
      </Paper>

      {/* Navigation Buttons */}
      <Stack direction="row" spacing={2} sx={{ mt: 4 }} justifyContent="flex-end">
        {currentStep > 0 && currentStep < 3 && (
          <Button variant="outlined" onClick={() => setCurrentStep(currentStep - 1)}>
            Back
          </Button>
        )}

        {currentStep === 0 && (
          <Alert severity="info" sx={{ flex: 1 }}>
            Select your preferred method to continue
          </Alert>
        )}

        {currentStep === 1 && (
          <Alert severity="info" sx={{ flex: 1 }}>
            Upload your filled profile file to continue
          </Alert>
        )}

        {currentStep === 2 && !isValid && (
          <Button
            variant="outlined"
            startIcon={<RefreshIcon />}
            onClick={() => setCurrentStep(1)}
          >
            Upload New File
          </Button>
        )}

        {currentStep === 2 && isValid && (
          <Button
            variant="contained"
            endIcon={<NextIcon />}
            onClick={handleConfirmAndCreateMachine}
            disabled={creatingMachine}
            sx={{
              background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
              '&:hover': {
                background: 'linear-gradient(135deg, #5568d3 0%, #6a4291 100%)',
              },
            }}
          >
            Confirm & Create Machine
          </Button>
        )}

        {currentStep === 3 && (
          <>
            <Button variant="outlined" onClick={handleReset} startIcon={<RefreshIcon />}>
              Create Another Machine
            </Button>
            <Button
              variant="contained"
              endIcon={<NextIcon />}
              onClick={handleCreateMachine}
              sx={{
                background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                '&:hover': {
                  background: 'linear-gradient(135deg, #5568d3 0%, #6a4291 100%)',
                },
              }}
            >
              Start GAN Workflow
            </Button>
          </>
        )}
      </Stack>

      {creatingMachine && <LinearProgress sx={{ mt: 2 }} />}
      {createMachineError && (
        <Alert severity="error" sx={{ mt: 2 }}>
          {createMachineError}
        </Alert>
      )}
    </Box>
  );
}
