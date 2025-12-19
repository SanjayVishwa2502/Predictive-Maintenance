/**
 * Performance Tests (Lighthouse)
 * Phase 3.7.5 Day 20: Quality Assurance
 * 
 * Target: Lighthouse score > 90
 * 
 * Metrics:
 * - Performance
 * - Accessibility
 * - Best Practices
 * - SEO
 */

import { test, expect } from '@playwright/test';
import { playAudit } from 'playwright-lighthouse';
import lighthouse from 'lighthouse';
import * as chromeLauncher from 'chrome-launcher';

test.describe('Performance Tests (Lighthouse)', () => {
  test('should achieve Lighthouse score > 90 for performance', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    // Launch Chrome with Lighthouse
    const chrome = await chromeLauncher.launch({ chromeFlags: ['--headless'] });
    const options = {
      logLevel: 'info',
      output: 'json',
      onlyCategories: ['performance', 'accessibility', 'best-practices', 'seo'],
      port: chrome.port,
    };

    try {
      const runnerResult = await lighthouse('http://localhost:5174', options);

      // Get scores
      const scores = {
        performance: runnerResult?.lhr.categories.performance.score! * 100,
        accessibility: runnerResult?.lhr.categories.accessibility.score! * 100,
        bestPractices: runnerResult?.lhr.categories['best-practices'].score! * 100,
        seo: runnerResult?.lhr.categories.seo.score! * 100,
      };

      console.log('Lighthouse Scores:', scores);

      // Assert scores
      expect(scores.performance).toBeGreaterThanOrEqual(90);
      expect(scores.accessibility).toBeGreaterThanOrEqual(90);
      expect(scores.bestPractices).toBeGreaterThanOrEqual(90);
      expect(scores.seo).toBeGreaterThanOrEqual(80);
    } finally {
      await chrome.kill();
    }
  });

  test('should have fast First Contentful Paint (FCP < 1.8s)', async ({ page }) => {
    await page.goto('/');

    const fcpMetric = await page.evaluate(() => {
      return new Promise((resolve) => {
        new PerformanceObserver((entryList) => {
          const entries = entryList.getEntries();
          const fcp = entries.find((entry) => entry.name === 'first-contentful-paint');
          if (fcp) {
            resolve(fcp.startTime);
          }
        }).observe({ type: 'paint', buffered: true });
      });
    });

    console.log('FCP:', fcpMetric, 'ms');
    expect(fcpMetric).toBeLessThan(1800); // < 1.8 seconds
  });

  test('should have fast Largest Contentful Paint (LCP < 2.5s)', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    const lcpMetric = await page.evaluate(() => {
      return new Promise((resolve) => {
        new PerformanceObserver((entryList) => {
          const entries = entryList.getEntries();
          const lastEntry = entries[entries.length - 1];
          resolve(lastEntry.startTime);
        }).observe({ type: 'largest-contentful-paint', buffered: true });

        // Resolve after 5 seconds if no LCP detected
        setTimeout(() => resolve(5000), 5000);
      });
    });

    console.log('LCP:', lcpMetric, 'ms');
    expect(lcpMetric).toBeLessThan(2500); // < 2.5 seconds
  });

  test('should have low Cumulative Layout Shift (CLS < 0.1)', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    const clsMetric = await page.evaluate(() => {
      return new Promise((resolve) => {
        let clsValue = 0;
        new PerformanceObserver((entryList) => {
          for (const entry of entryList.getEntries()) {
            if (!(entry as any).hadRecentInput) {
              clsValue += (entry as any).value;
            }
          }
          resolve(clsValue);
        }).observe({ type: 'layout-shift', buffered: true });

        // Resolve after 3 seconds
        setTimeout(() => resolve(clsValue), 3000);
      });
    });

    console.log('CLS:', clsMetric);
    expect(clsMetric).toBeLessThan(0.1);
  });

  test('should have fast bundle size', async ({ page }) => {
    await page.goto('/');

    // Get all resource sizes
    const resources = await page.evaluate(() => {
      return performance.getEntriesByType('resource').map((r: any) => ({
        name: r.name,
        size: r.transferSize,
        type: r.initiatorType,
      }));
    });

    const jsSize = resources
      .filter((r: any) => r.type === 'script')
      .reduce((sum: number, r: any) => sum + r.size, 0);

    const cssSize = resources
      .filter((r: any) => r.type === 'link' || r.type === 'css')
      .reduce((sum: number, r: any) => sum + r.size, 0);

    console.log('JS bundle size:', (jsSize / 1024 / 1024).toFixed(2), 'MB');
    console.log('CSS bundle size:', (cssSize / 1024).toFixed(2), 'KB');

    // Bundle should be reasonable (< 2MB for JS, < 100KB for CSS)
    expect(jsSize).toBeLessThan(2 * 1024 * 1024); // 2MB
    expect(cssSize).toBeLessThan(100 * 1024); // 100KB
  });
});
