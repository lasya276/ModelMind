import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { DatasetProvider } from './context/DatasetContext';
import { ExperimentHistoryProvider } from './context/ExperimentHistoryContext';
import AppLayout from './components/layout/AppLayout';
import OverviewPage from './pages/Overview/OverviewPage';
import DatasetsPage from './pages/Datasets/DatasetsPage';
import ExperimentsPage from './pages/Experiments/ExperimentsPage';
import ReportsPage from './pages/Reports/ReportsPage';

export default function App() {
  return (
    <DatasetProvider>
      <ExperimentHistoryProvider>
        <BrowserRouter>
          <Routes>
            <Route element={<AppLayout />}>
              <Route path="/" element={<OverviewPage />} />
              <Route path="/datasets" element={<DatasetsPage />} />
              <Route path="/experiments" element={<ExperimentsPage />} />
              <Route path="/reports" element={<ReportsPage />} />
              <Route path="*" element={<Navigate to="/" replace />} />
            </Route>
          </Routes>
        </BrowserRouter>
      </ExperimentHistoryProvider>
    </DatasetProvider>
  );
}
