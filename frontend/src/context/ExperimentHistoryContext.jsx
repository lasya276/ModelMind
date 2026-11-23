import React, {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useState,
} from 'react';
import { compactRunForHistory } from '../utils/experimentUtils';

const STORAGE_KEY = 'modelmind_experiment_history';
const MAX_HISTORY = 10;
const ExperimentHistoryContext = createContext(null);

function readHistory() {
  try {
    const stored = sessionStorage.getItem(STORAGE_KEY);
    if (!stored) return [];
    const parsed = JSON.parse(stored);
    if (!Array.isArray(parsed)) return [];
    return parsed
      .filter((run) => run?.run_id && run?.response?.data)
      .slice(0, MAX_HISTORY);
  } catch (error) {
    console.warn('Failed to restore experiment history:', error);
    return [];
  }
}

export function ExperimentHistoryProvider({ children }) {
  const [history, setHistory] = useState(readHistory);
  const [selectedRunId, setSelectedRunId] = useState(null);
  const [persistenceError, setPersistenceError] = useState(null);

  const saveHistory = useCallback((nextHistory) => {
    setHistory(nextHistory);
    try {
      sessionStorage.setItem(STORAGE_KEY, JSON.stringify(nextHistory));
      setPersistenceError(null);
    } catch (error) {
      console.warn('Failed to save experiment history:', error);
      setPersistenceError('This run is available until you leave the app, but could not be saved in session history.');
    }
  }, []);

  const addRun = useCallback((run) => {
    const historyRun = compactRunForHistory({
      ...run,
      run_id: `${Date.now()}-${Math.random().toString(36).slice(2)}`,
    });
    saveHistory([historyRun, ...history].slice(0, MAX_HISTORY));
    setSelectedRunId(historyRun.run_id);
    return historyRun;
  }, [history, saveHistory]);

  const openRun = useCallback((runId) => {
    setSelectedRunId(runId);
  }, []);

  const clearHistory = useCallback(() => {
    saveHistory([]);
    setSelectedRunId(null);
  }, [saveHistory]);

  const selectedRun = history.find((run) => run.run_id === selectedRunId) || null;
  const value = useMemo(() => ({
    history,
    selectedRun,
    selectedRunId,
    persistenceError,
    addRun,
    openRun,
    clearHistory,
  }), [
    history,
    selectedRun,
    selectedRunId,
    persistenceError,
    addRun,
    openRun,
    clearHistory,
  ]);

  return (
    <ExperimentHistoryContext.Provider value={value}>
      {children}
    </ExperimentHistoryContext.Provider>
  );
}

export function useExperimentHistory() {
  const context = useContext(ExperimentHistoryContext);
  if (!context) {
    throw new Error('useExperimentHistory must be used within ExperimentHistoryProvider');
  }
  return context;
}
