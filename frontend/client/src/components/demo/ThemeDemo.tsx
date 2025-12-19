/**
 * Theme Demo Component
 * Phase 3.7.3 Day 16.1: Theme Testing
 * 
 * Showcases all theme colors, typography, and components
 * Use this to verify theme is applied correctly
 */

import React from 'react';
import {
  Box,
  Container,
  Typography,
  Button,
  Card,
  CardContent,
  Chip,
  Alert,
  Grid,
  TextField,
  Divider,
  Paper,
} from '@mui/material';

export const ThemeDemo: React.FC = () => {
  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      {/* Header */}
      <Box mb={4}>
        <Typography variant="h1" gutterBottom>
          ML Dashboard Theme Demo
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Professional dark theme with Inter font family
        </Typography>
      </Box>

      {/* Typography Examples */}
      <Card sx={{ mb: 4 }}>
        <CardContent>
          <Typography variant="h2" gutterBottom>
            Typography Scale
          </Typography>
          <Divider sx={{ my: 2 }} />
          
          <Typography variant="h1" gutterBottom>Heading 1 - 32px Bold</Typography>
          <Typography variant="h2" gutterBottom>Heading 2 - 24px Semibold</Typography>
          <Typography variant="h3" gutterBottom>Heading 3 - 20px Semibold</Typography>
          <Typography variant="h4" gutterBottom>Heading 4 - 18px Medium</Typography>
          <Typography variant="h5" gutterBottom>Heading 5 - 16px Medium</Typography>
          <Typography variant="h6" gutterBottom>Heading 6 - 14px Medium</Typography>
          
          <Typography variant="body1" gutterBottom sx={{ mt: 2 }}>
            Body 1 - Regular paragraph text (16px)
          </Typography>
          <Typography variant="body2" gutterBottom>
            Body 2 - Smaller paragraph text (14px)
          </Typography>
          
          <Typography variant="caption" display="block" gutterBottom sx={{ mt: 2 }}>
            Caption text - 12px for supplementary info
          </Typography>
          <Typography variant="overline" display="block">
            Overline - 12px uppercase
          </Typography>
        </CardContent>
      </Card>

      {/* Color Palette */}
      <Card sx={{ mb: 4 }}>
        <CardContent>
          <Typography variant="h2" gutterBottom>
            Color Palette
          </Typography>
          <Divider sx={{ my: 2 }} />
          
          <Grid container spacing={2}>
            <Grid item xs={12} sm={6} md={4}>
              <Paper sx={{ p: 2, bgcolor: 'primary.main' }}>
                <Typography variant="body2">Primary</Typography>
                <Typography variant="caption">#667eea</Typography>
              </Paper>
            </Grid>
            <Grid item xs={12} sm={6} md={4}>
              <Paper sx={{ p: 2, bgcolor: 'secondary.main' }}>
                <Typography variant="body2">Secondary</Typography>
                <Typography variant="caption">#764ba2</Typography>
              </Paper>
            </Grid>
            <Grid item xs={12} sm={6} md={4}>
              <Paper sx={{ p: 2, bgcolor: 'success.main' }}>
                <Typography variant="body2">Success</Typography>
                <Typography variant="caption">#10b981</Typography>
              </Paper>
            </Grid>
            <Grid item xs={12} sm={6} md={4}>
              <Paper sx={{ p: 2, bgcolor: 'warning.main', color: 'warning.contrastText' }}>
                <Typography variant="body2">Warning</Typography>
                <Typography variant="caption">#fbbf24</Typography>
              </Paper>
            </Grid>
            <Grid item xs={12} sm={6} md={4}>
              <Paper sx={{ p: 2, bgcolor: 'error.main' }}>
                <Typography variant="body2">Error</Typography>
                <Typography variant="caption">#ef4444</Typography>
              </Paper>
            </Grid>
            <Grid item xs={12} sm={6} md={4}>
              <Paper sx={{ p: 2, bgcolor: 'info.main' }}>
                <Typography variant="body2">Info</Typography>
                <Typography variant="caption">#3b82f6</Typography>
              </Paper>
            </Grid>
          </Grid>
        </CardContent>
      </Card>

      {/* Buttons */}
      <Card sx={{ mb: 4 }}>
        <CardContent>
          <Typography variant="h2" gutterBottom>
            Button Variants
          </Typography>
          <Divider sx={{ my: 2 }} />
          
          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 2, mb: 3 }}>
            <Button variant="contained" color="primary">Primary Contained</Button>
            <Button variant="contained" color="secondary">Secondary Contained</Button>
            <Button variant="contained" color="success">Success Contained</Button>
            <Button variant="contained" color="error">Error Contained</Button>
          </Box>
          
          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 2, mb: 3 }}>
            <Button variant="outlined" color="primary">Primary Outlined</Button>
            <Button variant="outlined" color="secondary">Secondary Outlined</Button>
            <Button variant="outlined" color="success">Success Outlined</Button>
            <Button variant="outlined" color="error">Error Outlined</Button>
          </Box>
          
          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 2 }}>
            <Button variant="text" color="primary">Primary Text</Button>
            <Button variant="text" color="secondary">Secondary Text</Button>
            <Button variant="text" color="success">Success Text</Button>
            <Button variant="text" color="error">Error Text</Button>
          </Box>
        </CardContent>
      </Card>

      {/* Chips */}
      <Card sx={{ mb: 4 }}>
        <CardContent>
          <Typography variant="h2" gutterBottom>
            Status Chips
          </Typography>
          <Divider sx={{ my: 2 }} />
          
          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 2 }}>
            <Chip label="Normal" color="success" />
            <Chip label="Warning" color="warning" />
            <Chip label="Critical" color="error" />
            <Chip label="Running" color="info" />
            <Chip label="Offline" color="default" />
          </Box>
        </CardContent>
      </Card>

      {/* Alerts */}
      <Card sx={{ mb: 4 }}>
        <CardContent>
          <Typography variant="h2" gutterBottom>
            Alert Messages
          </Typography>
          <Divider sx={{ my: 2 }} />
          
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
            <Alert severity="success">
              Machine operating normally - all sensors within range
            </Alert>
            <Alert severity="warning">
              Bearing temperature elevated - schedule inspection
            </Alert>
            <Alert severity="error">
              Critical failure detected - immediate maintenance required
            </Alert>
            <Alert severity="info">
              Prediction model updated - new data available
            </Alert>
          </Box>
        </CardContent>
      </Card>

      {/* Form Elements */}
      <Card sx={{ mb: 4 }}>
        <CardContent>
          <Typography variant="h2" gutterBottom>
            Form Elements
          </Typography>
          <Divider sx={{ my: 2 }} />
          
          <Grid container spacing={2}>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Machine ID"
                placeholder="motor_siemens_1la7_001"
                helperText="Enter machine identifier"
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Sensor Value"
                type="number"
                placeholder="65.2"
                helperText="Temperature in Celsius"
              />
            </Grid>
          </Grid>
        </CardContent>
      </Card>

      {/* Cards with Different Elevations */}
      <Card sx={{ mb: 4 }}>
        <CardContent>
          <Typography variant="h2" gutterBottom>
            Card Elevations
          </Typography>
          <Divider sx={{ my: 2 }} />
          
          <Grid container spacing={2}>
            <Grid item xs={12} md={4}>
              <Paper elevation={1} sx={{ p: 2 }}>
                <Typography variant="h6">Elevation 1</Typography>
                <Typography variant="body2" color="text.secondary">
                  Subtle shadow for background cards
                </Typography>
              </Paper>
            </Grid>
            <Grid item xs={12} md={4}>
              <Paper elevation={2} sx={{ p: 2 }}>
                <Typography variant="h6">Elevation 2</Typography>
                <Typography variant="body2" color="text.secondary">
                  Default card elevation
                </Typography>
              </Paper>
            </Grid>
            <Grid item xs={12} md={4}>
              <Paper elevation={3} sx={{ p: 2 }}>
                <Typography variant="h6">Elevation 3</Typography>
                <Typography variant="body2" color="text.secondary">
                  Prominent card elevation
                </Typography>
              </Paper>
            </Grid>
          </Grid>
        </CardContent>
      </Card>

      {/* Footer */}
      <Box textAlign="center" py={4}>
        <Typography variant="body2" color="text.secondary">
          Theme: Professional Dark Mode | Font: Inter | Breakpoints: 640/1024/1280
        </Typography>
      </Box>
    </Container>
  );
};

export default ThemeDemo;
