/**
 * App Component - Main Application Entry Point
 * Phase 3.7.1.3: Comprehensive Authentication Implementation
 * 
 * Features:
 * - Login with role-based access (admin, operator, viewer)
 * - User registration with role selection
 * - Token refresh for session persistence
 * - User profile display with role badges
 * - Polished UI matching dashboard theme
 */

import { useCallback, useEffect, useState } from 'react';
import {
  Alert,
  Avatar,
  Box,
  Button,
  Chip,
  CircularProgress,
  Container,
  Divider,
  FormControl,
  FormHelperText,
  IconButton,
  InputAdornment,
  InputLabel,
  MenuItem,
  Paper,
  Select,
  Stack,
  Tab,
  Tabs,
  TextField,
  Tooltip,
  Typography,
} from '@mui/material';
import {
  AdminPanelSettings as AdminIcon,
  Engineering as OperatorIcon,
  Visibility as ViewerIcon,
  VisibilityOff,
  Visibility,
  Person as PersonIcon,
  LockOutlined as LockIcon,
  Email as EmailIcon,
  Settings as SettingsIcon,
} from '@mui/icons-material';

import { ThemeProvider } from './theme';
import MLDashboardPage from './pages/MLDashboardPage';

// ============================================================================
// Types
// ============================================================================

type UserRole = 'admin' | 'operator' | 'viewer';

type TokenResponse = {
  access_token: string;
  refresh_token: string;
  token_type: string;
};

type UserInfo = {
  id: string;
  username: string;
  email: string;
  role: UserRole;
  is_active: boolean;
  created_at: string;
  updated_at: string;
};

type AuthState = {
  user: UserInfo | null;
  loading: boolean;
};

// ============================================================================
// Constants
// ============================================================================

const API_BASE_URL: string = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';
const ACCESS_TOKEN_KEY = 'pm_access_token';
const REFRESH_TOKEN_KEY = 'pm_refresh_token';
const USER_INFO_KEY = 'pm_user_info';

// Token refresh 5 minutes before expiry
const REFRESH_BEFORE_EXPIRY_MS = 5 * 60 * 1000;

// ============================================================================
// Token & Storage Helpers
// ============================================================================

function getAccessToken(): string | null {
  try {
    const token = window.localStorage.getItem(ACCESS_TOKEN_KEY);
    return token && token.trim() ? token : null;
  } catch {
    return null;
  }
}

function getRefreshToken(): string | null {
  try {
    const token = window.localStorage.getItem(REFRESH_TOKEN_KEY);
    return token && token.trim() ? token : null;
  } catch {
    return null;
  }
}

function setTokens(tokens: { accessToken: string; refreshToken: string }): void {
  try {
    window.localStorage.setItem(ACCESS_TOKEN_KEY, tokens.accessToken);
    window.localStorage.setItem(REFRESH_TOKEN_KEY, tokens.refreshToken);
  } catch {
    // ignore
  }
}

function clearTokens(): void {
  try {
    window.localStorage.removeItem(ACCESS_TOKEN_KEY);
    window.localStorage.removeItem(REFRESH_TOKEN_KEY);
    window.localStorage.removeItem(USER_INFO_KEY);
  } catch {
    // ignore
  }
}

function getCachedUserInfo(): UserInfo | null {
  try {
    const raw = window.localStorage.getItem(USER_INFO_KEY);
    if (!raw) return null;
    return JSON.parse(raw) as UserInfo;
  } catch {
    return null;
  }
}

function setCachedUserInfo(user: UserInfo): void {
  try {
    window.localStorage.setItem(USER_INFO_KEY, JSON.stringify(user));
  } catch {
    // ignore
  }
}

function parseJwtPayload(token: string): { exp?: number; sub?: string; role?: string } | null {
  try {
    const parts = token.split('.');
    if (parts.length !== 3) return null;
    const payload = JSON.parse(atob(parts[1]));
    return payload;
  } catch {
    return null;
  }
}

// ============================================================================
// API Functions
// ============================================================================

async function apiLogin(username: string, password: string): Promise<TokenResponse> {
  const resp = await fetch(`${API_BASE_URL}/api/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, password }),
  });

  if (!resp.ok) {
    let detail = 'Login failed';
    try {
      const data = await resp.json();
      if (typeof data?.detail === 'string' && data.detail.trim()) detail = data.detail;
    } catch {
      // ignore
    }
    throw new Error(detail);
  }

  return (await resp.json()) as TokenResponse;
}

async function apiRegister(
  username: string,
  email: string,
  password: string,
  role: UserRole
): Promise<UserInfo> {
  const resp = await fetch(`${API_BASE_URL}/api/auth/register`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, email, password, role }),
  });

  if (!resp.ok) {
    let detail = 'Registration failed';
    try {
      const data = await resp.json();
      if (typeof data?.detail === 'string' && data.detail.trim()) detail = data.detail;
    } catch {
      // ignore
    }
    throw new Error(detail);
  }

  return (await resp.json()) as UserInfo;
}

async function apiRefreshToken(refreshToken: string): Promise<TokenResponse> {
  const resp = await fetch(`${API_BASE_URL}/api/auth/refresh`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ refresh_token: refreshToken }),
  });

  if (!resp.ok) {
    throw new Error('Token refresh failed');
  }

  return (await resp.json()) as TokenResponse;
}

async function apiGetMe(accessToken: string): Promise<UserInfo> {
  const resp = await fetch(`${API_BASE_URL}/api/auth/me`, {
    method: 'GET',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${accessToken}`,
    },
  });

  if (!resp.ok) {
    throw new Error('Failed to fetch user info');
  }

  return (await resp.json()) as UserInfo;
}

// ============================================================================
// Role Helpers
// ============================================================================

const ROLE_CONFIG: Record<UserRole, { label: string; color: 'error' | 'primary' | 'success'; icon: React.ReactElement; description: string }> = {
  admin: {
    label: 'Administrator',
    color: 'error',
    icon: <AdminIcon fontSize="small" />,
    description: 'Full system access including user management',
  },
  operator: {
    label: 'Operator',
    color: 'primary',
    icon: <OperatorIcon fontSize="small" />,
    description: 'Run predictions, manage machines, train models',
  },
  viewer: {
    label: 'Viewer',
    color: 'success',
    icon: <ViewerIcon fontSize="small" />,
    description: 'View dashboards and reports (read-only)',
  },
};

function RoleBadge({ role }: { role: UserRole }) {
  const config = ROLE_CONFIG[role] || ROLE_CONFIG.viewer;
  return (
    <Tooltip title={config.description}>
      <Chip
        icon={config.icon}
        label={config.label}
        color={config.color}
        size="small"
        sx={{ fontWeight: 600 }}
      />
    </Tooltip>
  );
}

// ============================================================================
// Auth Context (exported for dashboard to use)
// ============================================================================

export function useLogout(): () => void {
  return useCallback(() => {
    clearTokens();
    window.location.reload();
  }, []);
}

export function useCurrentUser(): UserInfo | null {
  return getCachedUserInfo();
}

// ============================================================================
// Login/Register Form Component
// ============================================================================

function AuthForm({ onLoggedIn }: { onLoggedIn: (user: UserInfo) => void }) {
  const [tab, setTab] = useState<'login' | 'register'>('login');
  const [showPassword, setShowPassword] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  // Login fields
  const [loginUsername, setLoginUsername] = useState('');
  const [loginPassword, setLoginPassword] = useState('');

  // Register fields
  const [regUsername, setRegUsername] = useState('');
  const [regEmail, setRegEmail] = useState('');
  const [regPassword, setRegPassword] = useState('');
  const [regConfirmPassword, setRegConfirmPassword] = useState('');
  const [regRole, setRegRole] = useState<UserRole>('viewer');

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setSubmitting(true);
    try {
      const tokens = await apiLogin(loginUsername.trim(), loginPassword);
      setTokens({ accessToken: tokens.access_token, refreshToken: tokens.refresh_token });
      const user = await apiGetMe(tokens.access_token);
      setCachedUserInfo(user);
      onLoggedIn(user);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Login failed';
      setError(message);
    } finally {
      setSubmitting(false);
    }
  };

  const handleRegister = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setSuccess(null);

    // Validation
    if (regPassword !== regConfirmPassword) {
      setError('Passwords do not match');
      return;
    }
    if (regPassword.length < 8) {
      setError('Password must be at least 8 characters');
      return;
    }

    setSubmitting(true);
    try {
      const username = regUsername.trim();
      const email = regEmail.trim();
      await apiRegister(username, email, regPassword, regRole);

      // Auto-login after registration so the session is immediately persisted.
      const tokens = await apiLogin(username, regPassword);
      setTokens({ accessToken: tokens.access_token, refreshToken: tokens.refresh_token });
      const user = await apiGetMe(tokens.access_token);
      setCachedUserInfo(user);
      onLoggedIn(user);
      setSuccess('Registration successful! You are now signed in.');

      // Keep the login tab selected for consistency if user logs out later.
      setTab('login');
      setLoginUsername(username);
      // Clear registration form
      setRegUsername('');
      setRegEmail('');
      setRegPassword('');
      setRegConfirmPassword('');
      setRegRole('viewer');
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Registration failed';
      setError(message);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <Box
      sx={{
        minHeight: '100vh',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        background: 'linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%)',
        py: 4,
      }}
    >
      <Container maxWidth="sm">
        <Paper
          elevation={24}
          sx={{
            p: 4,
            borderRadius: 3,
            background: 'rgba(26, 26, 46, 0.95)',
            backdropFilter: 'blur(10px)',
            border: '1px solid rgba(255, 255, 255, 0.1)',
          }}
        >
          <Stack spacing={3}>
            {/* Logo & Title */}
            <Box sx={{ textAlign: 'center' }}>
              <Avatar
                sx={{
                  width: 72,
                  height: 72,
                  mx: 'auto',
                  mb: 2,
                  background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                }}
              >
                <SettingsIcon sx={{ fontSize: 40 }} />
              </Avatar>
              <Typography
                variant="h4"
                sx={{
                  fontWeight: 700,
                  background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                  backgroundClip: 'text',
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent',
                }}
              >
                Predictive Maintenance
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                Industrial AI-Powered Equipment Monitoring
              </Typography>
            </Box>

            {/* Tabs */}
            <Tabs
              value={tab}
              onChange={(_, v) => {
                setTab(v);
                setError(null);
                setSuccess(null);
              }}
              variant="fullWidth"
              sx={{
                '& .MuiTab-root': { fontWeight: 600 },
              }}
            >
              <Tab value="login" label="Sign In" />
              <Tab value="register" label="Register" />
            </Tabs>

            {/* Alerts */}
            {error && <Alert severity="error">{error}</Alert>}
            {success && <Alert severity="success">{success}</Alert>}

            {/* Login Form */}
            {tab === 'login' && (
              <Box component="form" onSubmit={handleLogin} noValidate>
                <Stack spacing={2.5}>
                  <TextField
                    label="Username"
                    value={loginUsername}
                    onChange={(e) => setLoginUsername(e.target.value)}
                    autoComplete="username"
                    required
                    fullWidth
                    disabled={submitting}
                    InputProps={{
                      startAdornment: (
                        <InputAdornment position="start">
                          <PersonIcon color="action" />
                        </InputAdornment>
                      ),
                    }}
                  />
                  <TextField
                    label="Password"
                    value={loginPassword}
                    onChange={(e) => setLoginPassword(e.target.value)}
                    type={showPassword ? 'text' : 'password'}
                    autoComplete="current-password"
                    required
                    fullWidth
                    disabled={submitting}
                    InputProps={{
                      startAdornment: (
                        <InputAdornment position="start">
                          <LockIcon color="action" />
                        </InputAdornment>
                      ),
                      endAdornment: (
                        <InputAdornment position="end">
                          <IconButton
                            onClick={() => setShowPassword(!showPassword)}
                            edge="end"
                            size="small"
                          >
                            {showPassword ? <VisibilityOff /> : <Visibility />}
                          </IconButton>
                        </InputAdornment>
                      ),
                    }}
                  />

                  <Button
                    type="submit"
                    variant="contained"
                    size="large"
                    disabled={submitting || !loginUsername.trim() || !loginPassword}
                    sx={{
                      py: 1.5,
                      fontWeight: 600,
                      background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                      '&:hover': {
                        background: 'linear-gradient(135deg, #5a6fd6 0%, #6a4190 100%)',
                      },
                    }}
                  >
                    {submitting ? (
                      <CircularProgress size={24} color="inherit" />
                    ) : (
                      'Sign In'
                    )}
                  </Button>
                </Stack>
              </Box>
            )}

            {/* Register Form */}
            {tab === 'register' && (
              <Box component="form" onSubmit={handleRegister} noValidate>
                <Stack spacing={2.5}>
                  <TextField
                    label="Username"
                    value={regUsername}
                    onChange={(e) => setRegUsername(e.target.value)}
                    autoComplete="username"
                    required
                    fullWidth
                    disabled={submitting}
                    helperText="Min 3 characters"
                    InputProps={{
                      startAdornment: (
                        <InputAdornment position="start">
                          <PersonIcon color="action" />
                        </InputAdornment>
                      ),
                    }}
                  />
                  <TextField
                    label="Email"
                    value={regEmail}
                    onChange={(e) => setRegEmail(e.target.value)}
                    type="email"
                    autoComplete="email"
                    required
                    fullWidth
                    disabled={submitting}
                    InputProps={{
                      startAdornment: (
                        <InputAdornment position="start">
                          <EmailIcon color="action" />
                        </InputAdornment>
                      ),
                    }}
                  />

                  <FormControl fullWidth disabled={submitting}>
                    <InputLabel id="role-label">Role</InputLabel>
                    <Select
                      labelId="role-label"
                      value={regRole}
                      label="Role"
                      onChange={(e) => setRegRole(e.target.value as UserRole)}
                    >
                      <MenuItem value="viewer">
                        <Stack direction="row" spacing={1} alignItems="center">
                          <ViewerIcon color="success" fontSize="small" />
                          <Box>
                            <Typography variant="body2">Viewer</Typography>
                            <Typography variant="caption" color="text.secondary">
                              Read-only access
                            </Typography>
                          </Box>
                        </Stack>
                      </MenuItem>
                      <MenuItem value="operator">
                        <Stack direction="row" spacing={1} alignItems="center">
                          <OperatorIcon color="primary" fontSize="small" />
                          <Box>
                            <Typography variant="body2">Operator</Typography>
                            <Typography variant="caption" color="text.secondary">
                              Run predictions & manage machines
                            </Typography>
                          </Box>
                        </Stack>
                      </MenuItem>
                      <MenuItem value="admin">
                        <Stack direction="row" spacing={1} alignItems="center">
                          <AdminIcon color="error" fontSize="small" />
                          <Box>
                            <Typography variant="body2">Administrator</Typography>
                            <Typography variant="caption" color="text.secondary">
                              Full system access
                            </Typography>
                          </Box>
                        </Stack>
                      </MenuItem>
                    </Select>
                    <FormHelperText>Select your access level</FormHelperText>
                  </FormControl>

                  <TextField
                    label="Password"
                    value={regPassword}
                    onChange={(e) => setRegPassword(e.target.value)}
                    type={showPassword ? 'text' : 'password'}
                    autoComplete="new-password"
                    required
                    fullWidth
                    disabled={submitting}
                    helperText="Min 8 characters"
                    InputProps={{
                      startAdornment: (
                        <InputAdornment position="start">
                          <LockIcon color="action" />
                        </InputAdornment>
                      ),
                      endAdornment: (
                        <InputAdornment position="end">
                          <IconButton
                            onClick={() => setShowPassword(!showPassword)}
                            edge="end"
                            size="small"
                          >
                            {showPassword ? <VisibilityOff /> : <Visibility />}
                          </IconButton>
                        </InputAdornment>
                      ),
                    }}
                  />
                  <TextField
                    label="Confirm Password"
                    value={regConfirmPassword}
                    onChange={(e) => setRegConfirmPassword(e.target.value)}
                    type={showPassword ? 'text' : 'password'}
                    autoComplete="new-password"
                    required
                    fullWidth
                    disabled={submitting}
                    error={regConfirmPassword.length > 0 && regPassword !== regConfirmPassword}
                    helperText={
                      regConfirmPassword.length > 0 && regPassword !== regConfirmPassword
                        ? 'Passwords do not match'
                        : ''
                    }
                    InputProps={{
                      startAdornment: (
                        <InputAdornment position="start">
                          <LockIcon color="action" />
                        </InputAdornment>
                      ),
                    }}
                  />

                  <Button
                    type="submit"
                    variant="contained"
                    size="large"
                    disabled={
                      submitting ||
                      !regUsername.trim() ||
                      !regEmail.trim() ||
                      !regPassword ||
                      regPassword !== regConfirmPassword
                    }
                    sx={{
                      py: 1.5,
                      fontWeight: 600,
                      background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                      '&:hover': {
                        background: 'linear-gradient(135deg, #5a6fd6 0%, #6a4190 100%)',
                      },
                    }}
                  >
                    {submitting ? (
                      <CircularProgress size={24} color="inherit" />
                    ) : (
                      'Create Account'
                    )}
                  </Button>
                </Stack>
              </Box>
            )}

            <Divider sx={{ borderColor: 'rgba(255,255,255,0.1)' }} />

            {/* Footer */}
            <Typography variant="caption" color="text.secondary" sx={{ textAlign: 'center' }}>
              Phase 3.7 Industrial Predictive Maintenance System
              <br />
              Powered by GAN + ML + LLM AI Stack
            </Typography>
          </Stack>
        </Paper>
      </Container>
    </Box>
  );
}

// ============================================================================
// Main App Component
// ============================================================================

/**
 * Main App Component
 * 
 * Implements comprehensive authentication with:
 * - JWT-based login with role-based access
 * - User registration with role selection
 * - Automatic token refresh
 * - Session persistence
 */
function App() {
  const [authState, setAuthState] = useState<AuthState>({ user: null, loading: true });

  // On mount, check for existing token and fetch user info
  useEffect(() => {
    let cancelled = false;

    (async () => {
      const token = getAccessToken();
      if (!token) {
        setAuthState({ user: null, loading: false });
        return;
      }

      // Try to load cached user first for faster UX
      const cached = getCachedUserInfo();
      if (cached && !cancelled) {
        setAuthState({ user: cached, loading: false });
      }

      // Verify token is still valid by calling /me
      try {
        const user = await apiGetMe(token);
        if (!cancelled) {
          setCachedUserInfo(user);
          setAuthState({ user, loading: false });
        }
      } catch {
        // Token expired or invalid, try refresh
        const refreshToken = getRefreshToken();
        if (refreshToken) {
          try {
            const tokens = await apiRefreshToken(refreshToken);
            setTokens({ accessToken: tokens.access_token, refreshToken: tokens.refresh_token });
            const user = await apiGetMe(tokens.access_token);
            if (!cancelled) {
              setCachedUserInfo(user);
              setAuthState({ user, loading: false });
            }
          } catch {
            // Refresh failed, clear everything
            clearTokens();
            if (!cancelled) {
              setAuthState({ user: null, loading: false });
            }
          }
        } else {
          clearTokens();
          if (!cancelled) {
            setAuthState({ user: null, loading: false });
          }
        }
      }
    })();

    return () => {
      cancelled = true;
    };
  }, []);

  // Setup automatic token refresh
  useEffect(() => {
    if (!authState.user) return;

    const token = getAccessToken();
    if (!token) return;

    const payload = parseJwtPayload(token);
    if (!payload?.exp) return;

    const expiresAt = payload.exp * 1000;
    const now = Date.now();
    const refreshAt = expiresAt - REFRESH_BEFORE_EXPIRY_MS;
    const delay = Math.max(0, refreshAt - now);

    const timer = setTimeout(async () => {
      const refreshToken = getRefreshToken();
      if (!refreshToken) return;

      try {
        const tokens = await apiRefreshToken(refreshToken);
        setTokens({ accessToken: tokens.access_token, refreshToken: tokens.refresh_token });
        // Trigger re-render to reset timer
        setAuthState((prev) => ({ ...prev }));
      } catch {
        // Refresh failed, force re-login
        clearTokens();
        window.location.reload();
      }
    }, delay);

    return () => clearTimeout(timer);
  }, [authState.user, authState]);

  const handleLoggedIn = useCallback((user: UserInfo) => {
    setAuthState({ user, loading: false });
  }, []);

  // Loading state
  if (authState.loading) {
    return (
      <ThemeProvider>
        <Box
          sx={{
            minHeight: '100vh',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            background: 'linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%)',
          }}
        >
          <Stack spacing={2} alignItems="center">
            <CircularProgress size={48} sx={{ color: '#667eea' }} />
            <Typography color="text.secondary">Loading...</Typography>
          </Stack>
        </Box>
      </ThemeProvider>
    );
  }

  return (
    <ThemeProvider>
      {authState.user ? (
        <MLDashboardPage />
      ) : (
        <AuthForm onLoggedIn={handleLoggedIn} />
      )}
    </ThemeProvider>
  );
}

export default App;

// Export for dashboard header to use
export { RoleBadge, clearTokens, getCachedUserInfo, ROLE_CONFIG };
export type { UserInfo, UserRole };
