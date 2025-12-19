/**
 * Professional MUI Theme Configuration
 * Phase 3.7.3 Day 16.1: MUI Theme Setup
 * 
 * Features:
 * - Dark mode optimized for ML dashboards
 * - Custom color palette matching design system
 * - Inter font family for clean, modern UI
 * - Component overrides for Cards, Buttons, etc.
 * - Responsive breakpoints
 * - Smooth transitions and animations
 */

import { createTheme, alpha } from '@mui/material/styles';
import type { ThemeOptions } from '@mui/material/styles';

// ============================================================
// Color Palette
// ============================================================

const colors = {
  // Primary gradient (purple-blue)
  primary: {
    main: '#667eea',
    light: '#8b9bff',
    dark: '#4c63d2',
    contrastText: '#ffffff',
  },
  
  // Secondary gradient (purple)
  secondary: {
    main: '#764ba2',
    light: '#9d6bc7',
    dark: '#5a3680',
    contrastText: '#ffffff',
  },
  
  // Status colors
  success: {
    main: '#10b981',
    light: '#34d399',
    dark: '#059669',
    contrastText: '#ffffff',
  },
  warning: {
    main: '#fbbf24',
    light: '#fcd34d',
    dark: '#f59e0b',
    contrastText: '#1f2937',
  },
  error: {
    main: '#ef4444',
    light: '#f87171',
    dark: '#dc2626',
    contrastText: '#ffffff',
  },
  info: {
    main: '#3b82f6',
    light: '#60a5fa',
    dark: '#2563eb',
    contrastText: '#ffffff',
  },
  
  // Background colors (dark theme)
  background: {
    default: '#0f172a',      // Slate 900
    paper: '#1e293b',         // Slate 800
    elevated: '#334155',      // Slate 700 (for elevated cards)
  },
  
  // Text colors
  text: {
    primary: '#f1f5f9',       // Slate 100
    secondary: '#94a3b8',     // Slate 400
    disabled: '#64748b',      // Slate 500
  },
  
  // Divider
  divider: 'rgba(148, 163, 184, 0.12)',  // Slate 400 with opacity
  
  // Action colors
  action: {
    active: '#f1f5f9',
    hover: 'rgba(148, 163, 184, 0.08)',
    selected: 'rgba(102, 126, 234, 0.16)',
    disabled: 'rgba(148, 163, 184, 0.3)',
    disabledBackground: 'rgba(148, 163, 184, 0.12)',
  },
};


// ============================================================
// Typography
// ============================================================

const typography = {
  fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
  
  // Headings
  h1: {
    fontSize: '2rem',        // 32px
    fontWeight: 700,
    lineHeight: 1.2,
    letterSpacing: '-0.02em',
  },
  h2: {
    fontSize: '1.5rem',      // 24px
    fontWeight: 600,
    lineHeight: 1.3,
    letterSpacing: '-0.01em',
  },
  h3: {
    fontSize: '1.25rem',     // 20px
    fontWeight: 600,
    lineHeight: 1.4,
    letterSpacing: '0em',
  },
  h4: {
    fontSize: '1.125rem',    // 18px
    fontWeight: 500,
    lineHeight: 1.4,
    letterSpacing: '0em',
  },
  h5: {
    fontSize: '1rem',        // 16px
    fontWeight: 500,
    lineHeight: 1.5,
    letterSpacing: '0em',
  },
  h6: {
    fontSize: '0.875rem',    // 14px
    fontWeight: 500,
    lineHeight: 1.5,
    letterSpacing: '0.01em',
  },
  
  // Body text
  body1: {
    fontSize: '1rem',        // 16px
    lineHeight: 1.5,
    letterSpacing: '0em',
  },
  body2: {
    fontSize: '0.875rem',    // 14px
    lineHeight: 1.43,
    letterSpacing: '0.01em',
  },
  
  // Button text
  button: {
    fontSize: '0.875rem',    // 14px
    fontWeight: 500,
    lineHeight: 1.75,
    letterSpacing: '0.02em',
    textTransform: 'none' as const,  // Disable uppercase
  },
  
  // Caption and overline
  caption: {
    fontSize: '0.75rem',     // 12px
    lineHeight: 1.66,
    letterSpacing: '0.03em',
  },
  overline: {
    fontSize: '0.75rem',     // 12px
    fontWeight: 600,
    lineHeight: 2.66,
    letterSpacing: '0.08em',
    textTransform: 'uppercase' as const,
  },
  
  // Subtitle
  subtitle1: {
    fontSize: '1rem',        // 16px
    fontWeight: 400,
    lineHeight: 1.75,
    letterSpacing: '0.01em',
  },
  subtitle2: {
    fontSize: '0.875rem',    // 14px
    fontWeight: 500,
    lineHeight: 1.57,
    letterSpacing: '0.01em',
  },
};


// ============================================================
// Breakpoints
// ============================================================

const breakpoints = {
  values: {
    xs: 0,
    sm: 640,      // Mobile
    md: 1024,     // Tablet
    lg: 1280,     // Desktop
    xl: 1536,     // Wide desktop
  },
};


// ============================================================
// Spacing
// ============================================================

const spacing = 8;  // Base spacing unit (8px)


// ============================================================
// Shape (Border Radius)
// ============================================================

const shape = {
  borderRadius: 12,  // Default border radius
};


// ============================================================
// Shadows (Softer, more subtle)
// ============================================================

const shadows = [
  'none',
  '0px 2px 4px rgba(0, 0, 0, 0.2)',
  '0px 4px 8px rgba(0, 0, 0, 0.25)',
  '0px 8px 16px rgba(0, 0, 0, 0.3)',
  '0px 12px 24px rgba(0, 0, 0, 0.35)',
  '0px 16px 32px rgba(0, 0, 0, 0.4)',
  '0px 20px 40px rgba(0, 0, 0, 0.45)',
  '0px 24px 48px rgba(0, 0, 0, 0.5)',
  '0px 2px 4px rgba(0, 0, 0, 0.2)',
  '0px 4px 8px rgba(0, 0, 0, 0.25)',
  '0px 8px 16px rgba(0, 0, 0, 0.3)',
  '0px 12px 24px rgba(0, 0, 0, 0.35)',
  '0px 16px 32px rgba(0, 0, 0, 0.4)',
  '0px 20px 40px rgba(0, 0, 0, 0.45)',
  '0px 24px 48px rgba(0, 0, 0, 0.5)',
  '0px 2px 4px rgba(0, 0, 0, 0.2)',
  '0px 4px 8px rgba(0, 0, 0, 0.25)',
  '0px 8px 16px rgba(0, 0, 0, 0.3)',
  '0px 12px 24px rgba(0, 0, 0, 0.35)',
  '0px 16px 32px rgba(0, 0, 0, 0.4)',
  '0px 20px 40px rgba(0, 0, 0, 0.45)',
  '0px 24px 48px rgba(0, 0, 0, 0.5)',
  '0px 2px 4px rgba(0, 0, 0, 0.2)',
  '0px 4px 8px rgba(0, 0, 0, 0.25)',
  '0px 8px 16px rgba(0, 0, 0, 0.3)',
] as any;


// ============================================================
// Component Overrides
// ============================================================

const components = {
  // Card component
  MuiCard: {
    styleOverrides: {
      root: {
        borderRadius: 12,
        backgroundImage: 'none',  // Remove gradient background
        border: '1px solid rgba(148, 163, 184, 0.12)',
        transition: 'box-shadow 0.3s ease, transform 0.3s ease',
        '&:hover': {
          boxShadow: '0px 8px 24px rgba(0, 0, 0, 0.4)',
        },
      },
    },
  },
  
  // Paper component
  MuiPaper: {
    styleOverrides: {
      root: {
        backgroundImage: 'none',
      },
      elevation1: {
        boxShadow: '0px 2px 4px rgba(0, 0, 0, 0.2)',
      },
      elevation2: {
        boxShadow: '0px 4px 8px rgba(0, 0, 0, 0.25)',
      },
      elevation3: {
        boxShadow: '0px 8px 16px rgba(0, 0, 0, 0.3)',
      },
    },
  },
  
  // Button component
  MuiButton: {
    styleOverrides: {
      root: {
        borderRadius: 8,
        textTransform: 'none',
        fontWeight: 500,
        padding: '8px 16px',
        transition: 'all 0.2s ease',
      },
      contained: {
        boxShadow: 'none',
        '&:hover': {
          boxShadow: '0px 4px 12px rgba(0, 0, 0, 0.3)',
          transform: 'translateY(-1px)',
        },
        '&:active': {
          transform: 'translateY(0)',
        },
      },
      outlined: {
        borderWidth: '1.5px',
        '&:hover': {
          borderWidth: '1.5px',
        },
      },
    },
  },
  
  // Chip component
  MuiChip: {
    styleOverrides: {
      root: {
        borderRadius: 8,
        fontWeight: 500,
      },
    },
  },
  
  // TextField component
  MuiTextField: {
    defaultProps: {
      variant: 'outlined' as const,
    },
  },
  
  MuiOutlinedInput: {
    styleOverrides: {
      root: {
        borderRadius: 8,
        '&:hover .MuiOutlinedInput-notchedOutline': {
          borderColor: alpha(colors.primary.main, 0.5),
        },
        '&.Mui-focused .MuiOutlinedInput-notchedOutline': {
          borderWidth: '2px',
        },
      },
    },
  },
  
  // Tooltip component
  MuiTooltip: {
    styleOverrides: {
      tooltip: {
        backgroundColor: colors.background.elevated,
        color: colors.text.primary,
        fontSize: '0.75rem',
        padding: '8px 12px',
        borderRadius: 6,
        border: `1px solid ${colors.divider}`,
        boxShadow: '0px 4px 12px rgba(0, 0, 0, 0.3)',
      },
      arrow: {
        color: colors.background.elevated,
        '&::before': {
          border: `1px solid ${colors.divider}`,
        },
      },
    },
  },
  
  // Alert component
  MuiAlert: {
    styleOverrides: {
      root: {
        borderRadius: 8,
      },
      standardSuccess: {
        backgroundColor: alpha(colors.success.main, 0.1),
        color: colors.success.light,
      },
      standardWarning: {
        backgroundColor: alpha(colors.warning.main, 0.1),
        color: colors.warning.light,
      },
      standardError: {
        backgroundColor: alpha(colors.error.main, 0.1),
        color: colors.error.light,
      },
      standardInfo: {
        backgroundColor: alpha(colors.info.main, 0.1),
        color: colors.info.light,
      },
    },
  },
  
  // Badge component
  MuiBadge: {
    styleOverrides: {
      badge: {
        fontWeight: 600,
        fontSize: '0.75rem',
      },
    },
  },
  
  // Divider component
  MuiDivider: {
    styleOverrides: {
      root: {
        borderColor: colors.divider,
      },
    },
  },
  
  // AppBar component
  MuiAppBar: {
    styleOverrides: {
      root: {
        backgroundImage: 'none',
        boxShadow: 'none',
        borderBottom: `1px solid ${colors.divider}`,
      },
    },
  },
  
  // Drawer component
  MuiDrawer: {
    styleOverrides: {
      paper: {
        borderRight: `1px solid ${colors.divider}`,
        backgroundImage: 'none',
      },
    },
  },
};


// ============================================================
// Transitions
// ============================================================

const transitions = {
  duration: {
    shortest: 150,
    shorter: 200,
    short: 250,
    standard: 300,
    complex: 375,
    enteringScreen: 225,
    leavingScreen: 195,
  },
  easing: {
    easeInOut: 'cubic-bezier(0.4, 0, 0.2, 1)',
    easeOut: 'cubic-bezier(0.0, 0, 0.2, 1)',
    easeIn: 'cubic-bezier(0.4, 0, 1, 1)',
    sharp: 'cubic-bezier(0.4, 0, 0.6, 1)',
  },
};


// ============================================================
// Create Theme
// ============================================================

export const professionalTheme = createTheme({
  palette: {
    mode: 'dark',
    ...colors,
  },
  typography,
  breakpoints,
  spacing,
  shape,
  shadows,
  components,
  transitions,
} as ThemeOptions);


// ============================================================
// Export theme and types
// ============================================================

export default professionalTheme;

// Type exports for TypeScript support
export type ProfessionalTheme = typeof professionalTheme;
