/**
 * NotificationProvider Component
 * 
 * Provides toast notification display for the application.
 * Listens for 'pm-notification' events and displays them as snackbars.
 */

import React, { useState, useEffect, useCallback } from 'react';
import { Snackbar, Alert, AlertTitle, Stack, IconButton, Slide } from '@mui/material';
import type { SlideProps } from '@mui/material';
import {
  Close as CloseIcon,
  CheckCircle as SuccessIcon,
  Error as ErrorIcon,
  Warning as WarningIcon,
  Info as InfoIcon,
  Psychology as MLIcon,
  AutoGraph as PredictionIcon,
  Hub as GANIcon,
} from '@mui/icons-material';

// ============================================================================
// Types
// ============================================================================

interface Notification {
  id: string;
  title: string;
  message: string;
  type: string;
  duration: number;
  timestamp: number;
}

interface NotificationProviderProps {
  children: React.ReactNode;
  maxNotifications?: number;
}

// ============================================================================
// Slide Transition
// ============================================================================

function SlideTransition(props: SlideProps) {
  return <Slide {...props} direction="up" />;
}

// ============================================================================
// Get Icon by Type
// ============================================================================

function getNotificationIcon(type: string): React.ReactNode {
  switch (type) {
    case 'prediction_complete':
      return <PredictionIcon sx={{ color: '#10b981' }} />;
    case 'prediction_critical':
      return <WarningIcon sx={{ color: '#ef4444' }} />;
    case 'ml_trained':
      return <MLIcon sx={{ color: '#667eea' }} />;
    case 'gan_trained':
      return <GANIcon sx={{ color: '#8b5cf6' }} />;
    case 'success':
      return <SuccessIcon sx={{ color: '#10b981' }} />;
    case 'error':
      return <ErrorIcon sx={{ color: '#ef4444' }} />;
    case 'warning':
      return <WarningIcon sx={{ color: '#f59e0b' }} />;
    case 'info':
    default:
      return <InfoIcon sx={{ color: '#3b82f6' }} />;
  }
}

function getAlertSeverity(type: string): 'success' | 'error' | 'warning' | 'info' {
  switch (type) {
    case 'prediction_complete':
    case 'ml_trained':
    case 'gan_trained':
    case 'success':
      return 'success';
    case 'prediction_critical':
    case 'error':
      return 'error';
    case 'warning':
      return 'warning';
    case 'info':
    default:
      return 'info';
  }
}

// ============================================================================
// Component
// ============================================================================

export function NotificationProvider({ children, maxNotifications = 5 }: NotificationProviderProps) {
  const [notifications, setNotifications] = useState<Notification[]>([]);

  // Handle incoming notifications
  const handleNotification = useCallback((event: CustomEvent) => {
    const { title, message, type, duration, timestamp } = event.detail;
    
    const notification: Notification = {
      id: `${timestamp}-${Math.random().toString(36).substr(2, 9)}`,
      title,
      message,
      type,
      duration,
      timestamp,
    };

    setNotifications(prev => {
      // Remove oldest if at max
      const updated = prev.length >= maxNotifications 
        ? [...prev.slice(1), notification]
        : [...prev, notification];
      return updated;
    });
  }, [maxNotifications]);

  // Listen for notification events
  useEffect(() => {
    const listener = (e: Event) => handleNotification(e as CustomEvent);
    window.addEventListener('pm-notification', listener);
    return () => window.removeEventListener('pm-notification', listener);
  }, [handleNotification]);

  // Close a notification
  const handleClose = useCallback((id: string) => {
    setNotifications(prev => prev.filter(n => n.id !== id));
  }, []);

  // Auto-close notifications after duration
  useEffect(() => {
    const timers: ReturnType<typeof setTimeout>[] = [];

    notifications.forEach(notification => {
      if (notification.duration > 0) {
        const timer = setTimeout(() => {
          handleClose(notification.id);
        }, notification.duration);
        timers.push(timer);
      }
    });

    return () => {
      timers.forEach(timer => clearTimeout(timer));
    };
  }, [notifications, handleClose]);

  return (
    <>
      {children}
      
      {/* Notification Stack */}
      <Stack
        spacing={1}
        sx={{
          position: 'fixed',
          bottom: 24,
          right: 24,
          zIndex: 9999,
          maxWidth: 400,
        }}
      >
        {notifications.map((notification, index) => (
          <Snackbar
            key={notification.id}
            open={true}
            TransitionComponent={SlideTransition}
            sx={{
              position: 'relative',
              transform: 'none',
              bottom: 'auto',
              right: 'auto',
              left: 'auto',
            }}
          >
            <Alert
              severity={getAlertSeverity(notification.type)}
              icon={getNotificationIcon(notification.type)}
              variant="filled"
              action={
                <IconButton
                  size="small"
                  color="inherit"
                  onClick={() => handleClose(notification.id)}
                >
                  <CloseIcon fontSize="small" />
                </IconButton>
              }
              sx={{
                width: '100%',
                minWidth: 300,
                boxShadow: '0 4px 20px rgba(0, 0, 0, 0.3)',
                animation: index === notifications.length - 1 ? 'slideIn 0.3s ease-out' : undefined,
                '@keyframes slideIn': {
                  from: {
                    opacity: 0,
                    transform: 'translateX(100%)',
                  },
                  to: {
                    opacity: 1,
                    transform: 'translateX(0)',
                  },
                },
              }}
            >
              <AlertTitle sx={{ fontWeight: 600, mb: 0.5 }}>
                {notification.title}
              </AlertTitle>
              {notification.message}
            </Alert>
          </Snackbar>
        ))}
      </Stack>
    </>
  );
}

export default NotificationProvider;
