/**
 * App Component - Main Application Entry Point
 * Phase 3.7.3 Day 18.2: ML Dashboard Page Integration
 */

import { ThemeProvider } from './theme';
import MLDashboardPage from './pages/MLDashboardPage';

/**
 * Main App Component
 * 
 * Wraps the application with ThemeProvider and renders the integrated
 * ML Dashboard with all components working together.
 */
function App() {
  return (
    <ThemeProvider>
      <MLDashboardPage />
    </ThemeProvider>
  );
}

export default App;
