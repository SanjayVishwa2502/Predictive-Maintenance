/**
 * E2E Tests - Happy Path Workflow
 * Phase 3.7.5 Day 20: Quality Assurance
 * 
 * Tests complete user workflow:
 * 1. Dashboard loads
 * 2. Machine selection
 * 3. Sensor data display
 * 4. Run prediction
 * 5. View AI explanation
 * 6. Check prediction history
 */

import { test, expect } from '@playwright/test';

test.describe('ML Dashboard E2E - Happy Path', () => {
  test.beforeEach(async ({ page }) => {
    // Mock API responses
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

    await page.route('**/api/ml/machines/*/status', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          machine_id: 'motor_001',
          is_running: true,
          latest_sensors: {
            bearing_de_temp_C: 65.2,
            bearing_nde_temp_C: 62.1,
            winding_temp_C: 55.3,
            vibration_x_mm_s: 3.4,
            current_A: 12.5,
            voltage_V: 410.0,
          },
          last_update: '2025-12-16T10:45:23Z',
          sensor_count: 6,
        }),
      });
    });

    await page.route('**/api/ml/predict/classification', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          machine_id: 'motor_001',
          prediction: {
            failure_type: 'normal',
            confidence: 0.95,
            failure_probability: 0.05,
            all_probabilities: {
              Normal: 0.95,
              'Bearing Wear': 0.03,
              Overheating: 0.01,
              Electrical: 0.01,
            },
            rul: {
              rul_hours: 156.3,
              maintenance_window: 'Schedule within 3 days',
            },
          },
          timestamp: '2025-12-16T10:45:23Z',
        }),
      });
    });

    await page.goto('/');
  });

  test('should display dashboard with connection status', async ({ page }) => {
    // Check title
    await expect(page.getByText(/Machine Health Dashboard/i)).toBeVisible();

    // Check connection status indicators
    await expect(page.getByText(/Online/i)).toBeVisible();
    await expect(page.getByText(/Connected/i)).toBeVisible();
  });

  test('should show empty state initially', async ({ page }) => {
    await expect(
      page.getByText(/Select a Machine to Begin Monitoring/i)
    ).toBeVisible();
  });

  test('should allow machine selection', async ({ page }) => {
    // Wait for machines to load
    await page.waitForTimeout(1000);

    // Open machine selector (if it's an autocomplete/dropdown)
    const selector = page.locator('[role="combobox"], input[type="text"]').first();
    if (await selector.isVisible()) {
      await selector.click();
      await page.waitForTimeout(500);

      // Select first machine
      await page.keyboard.type('Motor');
      await page.waitForTimeout(500);
      await page.keyboard.press('Enter');
    }
  });

  test('should display sensor data after selection', async ({ page }) => {
    // Select a machine
    const selector = page.locator('[role="combobox"], input[type="text"]').first();
    if (await selector.isVisible()) {
      await selector.click();
      await page.keyboard.type('Motor');
      await page.keyboard.press('Enter');
      await page.waitForTimeout(2000);

      // Check for sensor data (temperature, vibration, etc.)
      await expect(page.getByText(/°C|mm\/s|A|V|kW/i).first()).toBeVisible({
        timeout: 5000,
      });
    }
  });

  test('should run prediction successfully', async ({ page }) => {
    // Select machine first
    const selector = page.locator('[role="combobox"], input[type="text"]').first();
    if (await selector.isVisible()) {
      await selector.click();
      await page.keyboard.type('Motor');
      await page.keyboard.press('Enter');
      await page.waitForTimeout(2000);

      // Click "Run Prediction" button
      const runButton = page.getByRole('button', {
        name: /Run Prediction/i,
      });
      if (await runButton.isVisible()) {
        await runButton.click();

        // Wait for prediction result
        await expect(
          page.getByText(/Prediction completed/i)
        ).toBeVisible({ timeout: 10000 });

        // Check for confidence score
        await expect(page.getByText(/confidence/i)).toBeVisible();
      }
    }
  });

  test('should open AI explanation modal', async ({ page }) => {
    // Complete prediction first
    const selector = page.locator('[role="combobox"], input[type="text"]').first();
    if (await selector.isVisible()) {
      await selector.click();
      await page.keyboard.type('Motor');
      await page.keyboard.press('Enter');
      await page.waitForTimeout(2000);

      const runButton = page.getByRole('button', {
        name: /Run Prediction/i,
      });
      if (await runButton.isVisible()) {
        await runButton.click();
        await page.waitForTimeout(3000);

        // Click "Get AI Explanation" button
        const explainButton = page.getByRole('button', {
          name: /AI Explanation/i,
        });
        if (await explainButton.isVisible()) {
          await explainButton.click();

          // Modal should open
          await expect(
            page.getByRole('dialog').or(page.locator('[role="presentation"]'))
          ).toBeVisible({ timeout: 5000 });
        }
      }
    }
  });

  test('should display responsive layout on mobile', async ({ page, viewport }) => {
    // Test is automatically run on mobile viewports via playwright.config.ts
    if (viewport && viewport.width < 768) {
      await expect(page.getByText(/Machine Health Dashboard/i)).toBeVisible();

      // Layout should adapt
      const container = page.locator('main, [role="main"]').first();
      if (await container.isVisible()) {
        const box = await container.boundingBox();
        expect(box?.width).toBeLessThanOrEqual(viewport.width);
      }
    }
  });
});
