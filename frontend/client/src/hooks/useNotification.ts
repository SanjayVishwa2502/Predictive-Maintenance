/**
 * useNotification Hook
 * 
 * Provides notification functionality with sound support for system events:
 * - Prediction completed
 * - ML model trained
 * - GAN training completed
 * - Critical alerts
 * - General success/error notifications
 */

import { useCallback, useRef, useEffect } from 'react';

// ============================================================================
// Types
// ============================================================================

export type NotificationType = 
  | 'prediction_complete'
  | 'prediction_critical'
  | 'ml_trained'
  | 'gan_trained'
  | 'success'
  | 'error'
  | 'warning'
  | 'info';

export interface NotificationOptions {
  title: string;
  message: string;
  type: NotificationType;
  playSound?: boolean;
  duration?: number; // ms, 0 = persistent
}

// ============================================================================
// Settings Keys
// ============================================================================

const SETTINGS_KEYS = {
  NOTIFICATIONS_ENABLED: 'pm_notifications_enabled',
  NOTIFICATION_SOUND: 'pm_notification_sound',
  NOTIFICATION_CRITICAL_ONLY: 'pm_notification_critical_only',
};

// ============================================================================
// Sound Generation using Web Audio API
// ============================================================================

// Sound frequencies for different notification types
const SOUND_CONFIGS: Record<NotificationType, { frequencies: number[]; durations: number[]; type: OscillatorType }> = {
  prediction_complete: {
    frequencies: [523.25, 659.25, 783.99], // C5, E5, G5 - pleasant major chord arpeggio
    durations: [100, 100, 200],
    type: 'sine',
  },
  prediction_critical: {
    frequencies: [440, 440, 440, 880], // A4 repeated, then A5 - urgent
    durations: [150, 100, 150, 300],
    type: 'square',
  },
  ml_trained: {
    frequencies: [392, 493.88, 587.33, 783.99], // G4, B4, D5, G5 - triumphant
    durations: [150, 150, 150, 400],
    type: 'sine',
  },
  gan_trained: {
    frequencies: [329.63, 415.30, 523.25, 659.25], // E4, G#4, C5, E5 - bright ascending
    durations: [120, 120, 120, 350],
    type: 'triangle',
  },
  success: {
    frequencies: [523.25, 659.25], // C5, E5 - simple success
    durations: [100, 200],
    type: 'sine',
  },
  error: {
    frequencies: [220, 196], // A3, G3 - descending, concerning
    durations: [200, 300],
    type: 'sawtooth',
  },
  warning: {
    frequencies: [440, 349.23], // A4, F4 - attention
    durations: [150, 200],
    type: 'triangle',
  },
  info: {
    frequencies: [587.33], // D5 - simple ping
    durations: [150],
    type: 'sine',
  },
};

function getStoredValue<T>(key: string, defaultValue: T): T {
  try {
    const stored = localStorage.getItem(key);
    if (stored === null) return defaultValue;
    if (typeof defaultValue === 'boolean') {
      return (stored === 'true') as unknown as T;
    }
    return stored as unknown as T;
  } catch {
    return defaultValue;
  }
}

// ============================================================================
// Hook Implementation
// ============================================================================

export function useNotification() {
  const audioContextRef = useRef<AudioContext | null>(null);

  // Initialize AudioContext on first user interaction
  useEffect(() => {
    const initAudioContext = () => {
      if (!audioContextRef.current) {
        audioContextRef.current = new (window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext)();
      }
    };

    // Initialize on any user interaction
    const events = ['click', 'keydown', 'touchstart'];
    events.forEach(event => {
      document.addEventListener(event, initAudioContext, { once: true });
    });

    return () => {
      events.forEach(event => {
        document.removeEventListener(event, initAudioContext);
      });
    };
  }, []);

  /**
   * Play a notification sound using Web Audio API
   */
  const playSound = useCallback(async (type: NotificationType) => {
    // Check if sounds are enabled
    const soundEnabled = getStoredValue(SETTINGS_KEYS.NOTIFICATION_SOUND, true);
    if (!soundEnabled) return;

    // Initialize AudioContext if needed
    if (!audioContextRef.current) {
      audioContextRef.current = new (window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext)();
    }

    const ctx = audioContextRef.current;
    
    // Resume context if suspended (browser autoplay policy)
    if (ctx.state === 'suspended') {
      await ctx.resume();
    }

    const config = SOUND_CONFIGS[type];
    const masterGain = ctx.createGain();
    masterGain.connect(ctx.destination);
    masterGain.gain.value = 0.3; // Master volume

    let startTime = ctx.currentTime;

    config.frequencies.forEach((freq, index) => {
      const oscillator = ctx.createOscillator();
      const gainNode = ctx.createGain();

      oscillator.type = config.type;
      oscillator.frequency.value = freq;

      // ADSR envelope for smoother sound
      const duration = config.durations[index] / 1000;
      const attackTime = 0.01;
      const releaseTime = 0.05;

      gainNode.gain.setValueAtTime(0, startTime);
      gainNode.gain.linearRampToValueAtTime(0.8, startTime + attackTime);
      gainNode.gain.setValueAtTime(0.8, startTime + duration - releaseTime);
      gainNode.gain.linearRampToValueAtTime(0, startTime + duration);

      oscillator.connect(gainNode);
      gainNode.connect(masterGain);

      oscillator.start(startTime);
      oscillator.stop(startTime + duration);

      startTime += duration;
    });
  }, []);

  /**
   * Show a notification with optional sound
   */
  const notify = useCallback((options: NotificationOptions) => {
    // Check if notifications are enabled
    const notificationsEnabled = getStoredValue(SETTINGS_KEYS.NOTIFICATIONS_ENABLED, true);
    if (!notificationsEnabled) return;

    // Check critical-only mode
    const criticalOnly = getStoredValue(SETTINGS_KEYS.NOTIFICATION_CRITICAL_ONLY, false);
    const isCritical = options.type === 'prediction_critical' || options.type === 'error';
    
    if (criticalOnly && !isCritical) {
      // Still play sound for non-critical if sounds enabled, just don't show toast
      if (options.playSound !== false) {
        playSound(options.type);
      }
      return;
    }

    // Play sound if enabled
    if (options.playSound !== false) {
      playSound(options.type);
    }

    // Dispatch custom event for toast display (handled by NotificationProvider)
    const event = new CustomEvent('pm-notification', {
      detail: {
        title: options.title,
        message: options.message,
        type: options.type,
        duration: options.duration ?? 5000,
        timestamp: Date.now(),
      },
    });
    window.dispatchEvent(event);

    // Also try browser notification if permitted (for background tabs)
    if ('Notification' in window && Notification.permission === 'granted') {
      try {
        new Notification(options.title, {
          body: options.message,
          icon: '/favicon.ico',
          tag: `pm-${options.type}-${Date.now()}`,
        });
      } catch {
        // Browser notifications may fail in some contexts
      }
    }
  }, [playSound]);

  /**
   * Request browser notification permission
   */
  const requestPermission = useCallback(async () => {
    if ('Notification' in window && Notification.permission === 'default') {
      const permission = await Notification.requestPermission();
      return permission === 'granted';
    }
    return Notification.permission === 'granted';
  }, []);

  // Convenience methods for common notifications
  const notifyPredictionComplete = useCallback((machineName: string, status: 'healthy' | 'warning' | 'critical') => {
    const type: NotificationType = status === 'critical' ? 'prediction_critical' : 'prediction_complete';
    notify({
      title: 'Prediction Complete',
      message: `${machineName}: ${status.charAt(0).toUpperCase() + status.slice(1)} status`,
      type,
    });
  }, [notify]);

  const notifyMLTrained = useCallback((modelType: string, accuracy?: number) => {
    notify({
      title: 'ML Model Trained',
      message: accuracy 
        ? `${modelType} trained successfully (${(accuracy * 100).toFixed(1)}% accuracy)`
        : `${modelType} model trained successfully`,
      type: 'ml_trained',
    });
  }, [notify]);

  const notifyGANTrained = useCallback((machineName: string, epochs: number) => {
    notify({
      title: 'GAN Training Complete',
      message: `${machineName}: Completed ${epochs} epochs`,
      type: 'gan_trained',
    });
  }, [notify]);

  const notifyError = useCallback((title: string, message: string) => {
    notify({
      title,
      message,
      type: 'error',
      duration: 8000, // Longer duration for errors
    });
  }, [notify]);

  const notifySuccess = useCallback((title: string, message: string) => {
    notify({
      title,
      message,
      type: 'success',
    });
  }, [notify]);

  return {
    notify,
    playSound,
    requestPermission,
    notifyPredictionComplete,
    notifyMLTrained,
    notifyGANTrained,
    notifyError,
    notifySuccess,
  };
}

export default useNotification;
