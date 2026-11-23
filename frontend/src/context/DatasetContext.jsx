import React, { createContext, useContext, useState, useCallback, useMemo } from 'react';

const STORAGE_KEY = 'modelmind_active_dataset';

const DatasetContext = createContext(null);

export function DatasetProvider({ children }) {
  // Initialize state from sessionStorage if available, preserving original upload timestamp
  const [sessionData, setSessionData] = useState(() => {
    try {
      const stored = sessionStorage.getItem(STORAGE_KEY);
      if (stored) {
        const parsed = JSON.parse(stored);
        if (parsed && (parsed.dataset_id || parsed.original_dataset?.dataset_id)) {
          const original = parsed.original_dataset || {
            dataset_id: parsed.dataset_id,
            filename: parsed.filename || 'dataset.csv',
            row_count: parsed.row_count ?? null,
            column_count: parsed.column_count ?? null,
            columns: Array.isArray(parsed.columns) ? parsed.columns : [],
            uploaded_at: parsed.uploaded_at || Date.now(),
          };

          const cleaned = parsed.cleaned_dataset || null;
          const version = (parsed.active_version === 'cleaned' && cleaned) ? 'cleaned' : 'original';

          return {
            originalDataset: original,
            cleanedDataset: cleaned,
            activeVersion: version,
          };
        }
      }
    } catch (e) {
      console.warn('Failed to load dataset from sessionStorage:', e);
    }
    return {
      originalDataset: null,
      cleanedDataset: null,
      activeVersion: 'original',
    };
  });

  const [datasetNotice, setDatasetNotice] = useState(null);

  // Sync to sessionStorage helper
  const syncToStorage = useCallback((data) => {
    if (!data.originalDataset || !data.originalDataset.dataset_id) {
      try {
        sessionStorage.removeItem(STORAGE_KEY);
      } catch (e) {
        /* ignore */
      }
      return;
    }

    const active = data.activeVersion === 'cleaned' && data.cleanedDataset
      ? data.cleanedDataset
      : data.originalDataset;

    const payload = {
      // Top-level properties represent active dataset for backward compatibility
      dataset_id: active.dataset_id,
      filename: active.filename || 'dataset.csv',
      row_count: active.row_count ?? null,
      column_count: active.column_count ?? null,
      columns: Array.isArray(active.columns) ? active.columns : [],
      uploaded_at: data.originalDataset.uploaded_at || Date.now(),
      // Multi-version fields
      original_dataset: data.originalDataset,
      cleaned_dataset: data.cleanedDataset,
      active_version: data.activeVersion,
    };

    try {
      sessionStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
    } catch (e) {
      console.warn('Failed to save dataset to sessionStorage:', e);
    }
  }, []);

  /**
   * Sets a brand new original dataset (e.g. from CsvUploader).
   * Resets any existing cleaned dataset version because a new file is loaded.
   */
  const setActiveDataset = useCallback((dataset) => {
    if (!dataset || !dataset.dataset_id) {
      const emptyState = {
        originalDataset: null,
        cleanedDataset: null,
        activeVersion: 'original',
      };
      setSessionData(emptyState);
      try {
        sessionStorage.removeItem(STORAGE_KEY);
      } catch (e) {
        /* ignore */
      }
      return;
    }

    const originalPayload = {
      dataset_id: dataset.dataset_id,
      filename: dataset.filename || 'dataset.csv',
      row_count: dataset.row_count ?? null,
      column_count: dataset.column_count ?? null,
      columns: Array.isArray(dataset.columns) ? dataset.columns : [],
      uploaded_at: dataset.uploaded_at || Date.now(),
    };

    const nextState = {
      originalDataset: originalPayload,
      cleanedDataset: null,
      activeVersion: 'original',
    };

    setSessionData(nextState);
    setDatasetNotice(null);
    syncToStorage(nextState);
  }, [syncToStorage]);

  /**
   * Records a newly generated cleaned dataset version.
   * Switches active version to 'cleaned' by default.
   */
  const setCleanedDataset = useCallback((cleanedData, makeActive = true) => {
    if (!cleanedData || !cleanedData.dataset_id) return;

    setSessionData((prev) => {
      const original = prev.originalDataset;
      if (!original) return prev;

      const cleanedPayload = {
        dataset_id: cleanedData.dataset_id,
        original_dataset_id: original.dataset_id,
        filename: cleanedData.filename || `cleaned_${original.filename}`,
        row_count: cleanedData.row_count ?? null,
        column_count: cleanedData.column_count ?? null,
        columns: Array.isArray(cleanedData.columns) ? cleanedData.columns : original.columns,
        cleaned_at: cleanedData.cleaned_at || Date.now(),
        download_url: cleanedData.download_url || `/clean/download/${cleanedData.dataset_id}`,
        report: cleanedData.report || null,
      };

      const nextState = {
        ...prev,
        cleanedDataset: cleanedPayload,
        activeVersion: makeActive ? 'cleaned' : prev.activeVersion,
      };

      syncToStorage(nextState);
      return nextState;
    });
  }, [syncToStorage]);

  /**
   * Switches between 'original' and 'cleaned' dataset version.
   */
  const switchDatasetVersion = useCallback((version) => {
    setSessionData((prev) => {
      if (version === 'cleaned' && !prev.cleanedDataset) {
        console.warn('Cannot switch to cleaned version: no cleaned dataset exists.');
        return prev;
      }
      const targetVersion = version === 'cleaned' ? 'cleaned' : 'original';
      const nextState = {
        ...prev,
        activeVersion: targetVersion,
      };
      syncToStorage(nextState);
      return nextState;
    });
  }, [syncToStorage]);

  /**
   * Clears the active dataset reference and removes it from sessionStorage.
   */
  const clearActiveDataset = useCallback(() => {
    const emptyState = {
      originalDataset: null,
      cleanedDataset: null,
      activeVersion: 'original',
    };
    setSessionData(emptyState);
    try {
      sessionStorage.removeItem(STORAGE_KEY);
    } catch (e) {
      /* ignore */
    }
  }, []);

  /**
   * Notifies that the stored dataset was confirmed missing or expired by a dataset-dependent request.
   * Clears the stale session and presents an explicit re-upload guidance message.
   */
  const handleDatasetExpiredOrMissing = useCallback((customMessage) => {
    clearActiveDataset();
    setDatasetNotice({
      type: 'expired',
      title: 'Dataset No Longer Available',
      message:
        customMessage ||
        'This dataset is no longer in temporary server storage (files are kept for up to 48 hours). Please upload your CSV file again to continue.',
    });
  }, [clearActiveDataset]);

  const dismissNotice = useCallback(() => {
    setDatasetNotice(null);
  }, []);

  // Compute currently active dataset based on activeVersion
  const activeDataset = useMemo(() => {
    if (sessionData.activeVersion === 'cleaned' && sessionData.cleanedDataset) {
      return sessionData.cleanedDataset;
    }
    return sessionData.originalDataset;
  }, [sessionData]);

  const value = {
    activeDataset,
    originalDataset: sessionData.originalDataset,
    cleanedDataset: sessionData.cleanedDataset,
    activeVersion: sessionData.activeVersion,
    isCleanedActive: sessionData.activeVersion === 'cleaned',
    hasCleanedDataset: Boolean(sessionData.cleanedDataset?.dataset_id),
    hasActiveDataset: Boolean(activeDataset?.dataset_id),
    setActiveDataset,
    setCleanedDataset,
    switchDatasetVersion,
    clearActiveDataset,
    handleDatasetExpiredOrMissing,
    datasetNotice,
    dismissNotice,
  };

  return <DatasetContext.Provider value={value}>{children}</DatasetContext.Provider>;
}

export function useDataset() {
  const context = useContext(DatasetContext);
  if (!context) {
    throw new Error('useDataset must be used within a DatasetProvider');
  }
  return context;
}
