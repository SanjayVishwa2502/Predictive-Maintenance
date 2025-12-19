/**
 * Accessibility Tests (WCAG 2.1 AA)
 * Phase 3.7.5 Day 20: Quality Assurance
 * 
 * Tests:
 * 1. Automated axe-core scans
 * 2. Keyboard navigation
 * 3. Screen reader compatibility
 * 4. Color contrast
 * 5. ARIA labels
 */

import { test, expect } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';

test.describe('Accessibility Tests (WCAG 2.1 AA)', () => {
  test.beforeEach(async ({ page }) => {
    // Mock API to ensure consistent state
    await page.route('**/api/ml/machines', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          machines: [
            {
              machine_id: 'motor_001',
              display_name: 'Motor SIEMENS 001',
              category: 'Motor',
              manufacturer: 'SIEMENS',
              model: 'MODEL_001',
              sensor_count: 10,
              has_classification_model: true,
              has_regression_model: false,
              has_anomaly_model: false,
              has_timeseries_model: false,
            },
          ],
          total: 1,
        }),
      });
    });

    await page.goto('/');
  });

  test('should not have automatically detectable accessibility issues', async ({
    page,
  }) => {
    // Run axe-core accessibility scan
    const accessibilityScanResults = await new AxeBuilder({ page })
      .withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa'])
      .analyze();

    expect(accessibilityScanResults.violations).toEqual([]);
  });

  test('should have proper heading hierarchy', async ({ page }) => {
    // Check for h1
    const h1 = page.locator('h1, [role="heading"][aria-level="1"]');
    await expect(h1.first()).toBeVisible();

    // H1 should contain dashboard title
    await expect(h1.first()).toContainText(/Dashboard/i);
  });

  test('should support keyboard navigation', async ({ page }) => {
    // Tab through interactive elements
    await page.keyboard.press('Tab');
    
    // First focusable element should be focused
    const focused = await page.evaluate(() => document.activeElement?.tagName);
    expect(['INPUT', 'BUTTON', 'A', 'SELECT']).toContain(focused || '');
  });

  test('should have accessible form controls', async ({ page }) => {
    // Check for input labels
    const inputs = page.locator('input[type="text"], select');
    const count = await inputs.count();

    for (let i = 0; i < count; i++) {
      const input = inputs.nth(i);
      
      // Should have aria-label or associated label
      const ariaLabel = await input.getAttribute('aria-label');
      const id = await input.getAttribute('id');
      
      if (!ariaLabel && id) {
        const label = page.locator(`label[for="${id}"]`);
        await expect(label).toBeVisible();
      } else if (!id) {
        expect(ariaLabel).toBeTruthy();
      }
    }
  });

  test('should have sufficient color contrast', async ({ page }) => {
    const accessibilityScanResults = await new AxeBuilder({ page })
      .withTags(['wcag2aa'])
      .options({ rules: { 'color-contrast': { enabled: true } } })
      .analyze();

    const contrastViolations = accessibilityScanResults.violations.filter(
      (v) => v.id === 'color-contrast'
    );

    expect(contrastViolations).toHaveLength(0);
  });

  test('should have proper ARIA roles for interactive elements', async ({
    page,
  }) => {
    // Buttons should have role="button" or be <button> elements
    const buttons = page.locator('button, [role="button"]');
    const buttonCount = await buttons.count();
    expect(buttonCount).toBeGreaterThan(0);

    // Links should have role="link" or be <a> elements
    const links = page.locator('a, [role="link"]');
    const linkCount = await links.count();
    // Links may not be present on dashboard, so just check they're valid if present
    if (linkCount > 0) {
      for (let i = 0; i < linkCount; i++) {
        const href = await links.nth(i).getAttribute('href');
        expect(href).toBeTruthy();
      }
    }
  });

  test('should have accessible images', async ({ page }) => {
    const images = page.locator('img');
    const count = await images.count();

    for (let i = 0; i < count; i++) {
      const img = images.nth(i);
      const alt = await img.getAttribute('alt');
      
      // All images should have alt text (even if empty for decorative images)
      expect(alt !== null).toBeTruthy();
    }
  });

  test('should have proper focus indicators', async ({ page }) => {
    // Tab to first interactive element
    await page.keyboard.press('Tab');
    
    // Check that focused element has visible outline or focus style
    const focusedElement = await page.evaluate(() => {
      const el = document.activeElement;
      if (!el) return null;
      
      const styles = window.getComputedStyle(el);
      return {
        outline: styles.outline,
        outlineWidth: styles.outlineWidth,
        boxShadow: styles.boxShadow,
      };
    });

    // Should have either outline or box-shadow for focus
    expect(
      focusedElement?.outline !== 'none' ||
      focusedElement?.outlineWidth !== '0px' ||
      focusedElement?.boxShadow !== 'none'
    ).toBeTruthy();
  });

  test('should have accessible status messages', async ({ page }) => {
    // Status indicators should have aria-live regions
    const statusChips = page.locator('[role="status"], [aria-live]');
    const count = await statusChips.count();
    
    // Should have at least connection status indicators
    expect(count).toBeGreaterThan(0);
  });

  test('should support screen reader announcements', async ({ page }) => {
    // Check for aria-live regions
    const liveRegions = page.locator('[aria-live="polite"], [aria-live="assertive"]');
    const count = await liveRegions.count();
    
    // Should have live regions for dynamic updates
    expect(count).toBeGreaterThan(0);
  });
});
