import React, { useState, useEffect, useCallback, useRef } from 'react';
import {
  Database,
  FileText,
  Sparkles,
  UploadCloud,
  RefreshCw,
  AlertCircle,
  Loader2,
  SlidersHorizontal,
  ArrowRightLeft,
} from 'lucide-react';
import Card from '../../components/common/Card';
import Badge from '../../components/common/Badge';
import Button from '../../components/common/Button';
import CsvUploader from '../../components/dataset/CsvUploader';
import DatasetReviewView from '../../components/dataset/DatasetReviewView';
import DatasetIntelligenceView from '../../components/dataset/DatasetIntelligenceView';
import DatasetCleaningView from '../../components/dataset/DatasetCleaningView';
import {
  getDatasetProfile,
  getDatasetQuality,
  getDatasetIntelligence,
} from '../../api/datasetApi';
import { getErrorMessage, isDatasetNotFoundError } from '../../api/client';
import { useDataset } from '../../context/DatasetContext';

export default function DatasetsPage() {
  const {
    activeDataset,
    originalDataset,
    cleanedDataset,
    isCleanedActive,
    hasCleanedDataset,
    switchDatasetVersion,
    handleDatasetExpiredOrMissing,
  } = useDataset();

  // Active sub-tab within the Datasets hub
  const [activeTab, setActiveTab] = useState('review');
  const [showUploader, setShowUploader] = useState(false);

  // Cache reports specifically keyed to the active dataset_id to avoid data mixing
  const [profileState, setProfileState] = useState({ data: null, loading: false, error: null, datasetId: null });
  const [qualityState, setQualityState] = useState({ data: null, loading: false, error: null, datasetId: null });
  const [intelligenceState, setIntelligenceState] = useState({ data: null, loading: false, error: null, datasetId: null });

  const abortControllersRef = useRef({});

  // Reset or clear report caches if activeDataset changes or is cleared
  const currentDatasetId = activeDataset?.dataset_id || null;
  const profileDataForActive =
    profileState.datasetId === currentDatasetId ? profileState.data : null;
  const qualityDataForActive =
    qualityState.datasetId === currentDatasetId ? qualityState.data : null;
  const intelligenceDataForActive =
    intelligenceState.datasetId === currentDatasetId ? intelligenceState.data : null;

  useEffect(() => {
    // If active dataset changed or became null, cancel active requests and reset caches
    Object.values(abortControllersRef.current).forEach((ctrl) => ctrl?.abort());
    abortControllersRef.current = {};

    if (!currentDatasetId) {
      setProfileState({ data: null, loading: false, error: null, datasetId: null });
      setQualityState({ data: null, loading: false, error: null, datasetId: null });
      setIntelligenceState({ data: null, loading: false, error: null, datasetId: null });
    } else {
      // If dataset changed, reset caches so old dataset data is never displayed
      setProfileState((prev) => (prev.datasetId === currentDatasetId ? prev : { data: null, loading: false, error: null, datasetId: currentDatasetId }));
      setQualityState((prev) => (prev.datasetId === currentDatasetId ? prev : { data: null, loading: false, error: null, datasetId: currentDatasetId }));
      setIntelligenceState((prev) => (prev.datasetId === currentDatasetId ? prev : { data: null, loading: false, error: null, datasetId: currentDatasetId }));
    }
  }, [currentDatasetId]);

  /**
   * Load Profiling data for the current active dataset
   */
  const loadProfile = useCallback(async (force = false) => {
    if (!currentDatasetId) return;
    if (!force && profileState.data && profileState.datasetId === currentDatasetId) return;

    abortControllersRef.current.profile?.abort();
    const ctrl = new AbortController();
    abortControllersRef.current.profile = ctrl;

    setProfileState({ data: null, loading: true, error: null, datasetId: currentDatasetId });

    try {
      const response = await getDatasetProfile(currentDatasetId, ctrl.signal);
      setProfileState({
        data: response.data,
        loading: false,
        error: null,
        datasetId: currentDatasetId,
      });
    } catch (err) {
      if (err.name === 'CanceledError' || err.message === 'canceled') return;
      if (isDatasetNotFoundError(err)) {
        handleDatasetExpiredOrMissing();
        return;
      }
      setProfileState({
        data: null,
        loading: false,
        error: getErrorMessage(err, 'Failed to generate dataset profile.'),
        datasetId: currentDatasetId,
      });
    }
  }, [currentDatasetId, profileState.data, profileState.datasetId, handleDatasetExpiredOrMissing]);

  /**
   * Load Quality data for the current active dataset
   */
  const loadQuality = useCallback(async (force = false) => {
    if (!currentDatasetId) return;
    if (!force && qualityState.data && qualityState.datasetId === currentDatasetId) return;

    abortControllersRef.current.quality?.abort();
    const ctrl = new AbortController();
    abortControllersRef.current.quality = ctrl;

    setQualityState({ data: null, loading: true, error: null, datasetId: currentDatasetId });

    try {
      const response = await getDatasetQuality(currentDatasetId, ctrl.signal);
      setQualityState({
        data: response.data,
        loading: false,
        error: null,
        datasetId: currentDatasetId,
      });
    } catch (err) {
      if (err.name === 'CanceledError' || err.message === 'canceled') return;
      if (isDatasetNotFoundError(err)) {
        handleDatasetExpiredOrMissing();
        return;
      }
      setQualityState({
        data: null,
        loading: false,
        error: getErrorMessage(err, 'Failed to inspect dataset quality.'),
        datasetId: currentDatasetId,
      });
    }
  }, [currentDatasetId, qualityState.data, qualityState.datasetId, handleDatasetExpiredOrMissing]);

  /**
   * Load Intelligence data for the current active dataset
   */
  const loadIntelligence = useCallback(async (force = false) => {
    if (!currentDatasetId) return;
    if (!force && intelligenceState.data && intelligenceState.datasetId === currentDatasetId) return;

    abortControllersRef.current.intelligence?.abort();
    const ctrl = new AbortController();
    abortControllersRef.current.intelligence = ctrl;

    setIntelligenceState({ data: null, loading: true, error: null, datasetId: currentDatasetId });

    try {
      const response = await getDatasetIntelligence(currentDatasetId, ctrl.signal);
      setIntelligenceState({
        data: response.data,
        loading: false,
        error: null,
        datasetId: currentDatasetId,
      });
    } catch (err) {
      if (err.name === 'CanceledError' || err.message === 'canceled') return;
      if (isDatasetNotFoundError(err)) {
        handleDatasetExpiredOrMissing();
        return;
      }
      setIntelligenceState({
        data: null,
        loading: false,
        error: getErrorMessage(err, 'Failed to analyze dataset intelligence.'),
        datasetId: currentDatasetId,
      });
    }
  }, [currentDatasetId, intelligenceState.data, intelligenceState.datasetId, handleDatasetExpiredOrMissing]);

  // Automatically fetch data when a relevant sub-tab is viewed
  useEffect(() => {
    if (!currentDatasetId) return;

    if (activeTab === 'review') {
      if (!profileState.data && !profileState.loading && !profileState.error) {
        loadProfile();
      }
      if (!qualityState.data && !qualityState.loading && !qualityState.error) {
        loadQuality();
      }
    } else if (activeTab === 'clean') {
      if (!profileState.data && !profileState.loading && !profileState.error) {
        loadProfile();
      }
      if (!qualityState.data && !qualityState.loading && !qualityState.error) {
        loadQuality();
      }
      if (!intelligenceState.data && !intelligenceState.loading && !intelligenceState.error) {
        loadIntelligence();
      }
    } else if (activeTab === 'intelligence' && !intelligenceState.data && !intelligenceState.loading && !intelligenceState.error) {
      loadIntelligence();
    }
  }, [activeTab, currentDatasetId, profileState, qualityState, intelligenceState, loadProfile, loadQuality, loadIntelligence]);

  const tabs = [
    { id: 'review', name: 'Dataset Review', icon: FileText },
    { id: 'clean', name: 'Clean & Transform', icon: SlidersHorizontal },
    { id: 'intelligence', name: 'Intelligence & Target Suggestions', icon: Sparkles },
  ];

  return (
    <div className="space-y-8">
      {/* Editorial Header */}
      <div className="border-b border-stone pb-6">
        <Badge variant="teal" className="mb-2">
          Data Workspace
        </Badge>
        <h1 className="font-serif text-3xl sm:text-4xl font-bold tracking-tight text-navy">
          Datasets
        </h1>
        <p className="mt-1 text-sm text-navy/70 max-w-2xl font-sans">
          Check your data for missing values and errors, understand each column, and explore rule-based recommendations.
        </p>
      </div>

      {/* CSV Upload Section (Shown when no dataset or when explicitly opened) */}
      {(!activeDataset || showUploader) && (
        <Card
          title="Upload CSV Dataset"
          subtitle="Drag and drop your file or browse from your computer. Files are checked and stored safely for your session."
          headerAction={
            activeDataset ? (
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setShowUploader(false)}
              >
                Hide Uploader
              </Button>
            ) : null
          }
        >
          <CsvUploader />
        </Card>
      )}

      {/* Active Dataset Status Card */}
      {activeDataset && (
        <Card
          title="Current Working Dataset"
          subtitle={`Active file: ${activeDataset.filename}`}
          headerAction={
            <div className="flex flex-wrap items-center gap-2">
              <Badge variant={isCleanedActive ? 'teal' : 'sage'}>
                {isCleanedActive ? 'Cleaned Version' : 'Original Version'}
              </Badge>

              {hasCleanedDataset && (
                <Button
                  variant="outline"
                  size="sm"
                  icon={ArrowRightLeft}
                  onClick={() => switchDatasetVersion(isCleanedActive ? 'original' : 'cleaned')}
                >
                  Switch to {isCleanedActive ? 'Original' : 'Cleaned'}
                </Button>
              )}

              {!showUploader && (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setShowUploader(true)}
                  icon={UploadCloud}
                >
                  Upload Different CSV
                </Button>
              )}
            </div>
          }
        >
          <div className="space-y-4 font-mono text-xs">
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 font-sans">
              <div className="p-3 bg-stone-light/40 rounded-sm border border-stone/70">
                <span className="text-navy/50 block text-[11px] uppercase font-mono mb-1">FILE NAME</span>
                <span className="font-semibold text-navy truncate block" title={activeDataset.filename}>
                  {activeDataset.filename}
                </span>
              </div>
              <div className="p-3 bg-stone-light/40 rounded-sm border border-stone/70">
                <span className="text-navy/50 block text-[11px] uppercase font-mono mb-1">ROWS</span>
                <span className="font-semibold text-navy text-sm">
                  {activeDataset.row_count ? activeDataset.row_count.toLocaleString() : '—'}
                </span>
              </div>
              <div className="p-3 bg-stone-light/40 rounded-sm border border-stone/70">
                <span className="text-navy/50 block text-[11px] uppercase font-mono mb-1">COLUMNS</span>
                <span className="font-semibold text-navy text-sm">
                  {activeDataset.column_count ?? '—'}
                </span>
              </div>
              <div className="p-3 bg-stone-light/40 rounded-sm border border-stone/70">
                <span className="text-navy/50 block text-[11px] uppercase font-mono mb-1">STORAGE</span>
                <span className="font-semibold text-navy text-sm font-sans">
                  Up to 48 hrs
                </span>
              </div>
            </div>

            <div className="p-3 bg-stone-light/30 rounded-sm border border-stone/70 font-mono text-xs">
              <span className="text-navy/50 font-sans block text-[11px] mb-1">DATASET IDENTIFIER</span>
              <span className="text-navy font-semibold select-all">{activeDataset.dataset_id}</span>
            </div>
          </div>
        </Card>
      )}

      {/* Analysis Sub-Tabs Navigation (When dataset is loaded) */}
      {activeDataset && (
        <div className="space-y-6">
          <div className="flex border-b border-stone gap-2 overflow-x-auto">
            {tabs.map((tab) => {
              const Icon = tab.icon;
              const isActive = activeTab === tab.id;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex items-center gap-2 px-4 py-2.5 text-xs font-semibold tracking-wide border-b-2 transition-all whitespace-nowrap ${
                    isActive
                      ? 'border-teal text-teal-dark bg-white/50'
                      : 'border-transparent text-navy/60 hover:text-navy hover:bg-stone-light/30'
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  <span>{tab.name}</span>
                </button>
              );
            })}
          </div>

          {/* TAB 1: Dataset Review */}
          {activeTab === 'review' && (
            <div className="space-y-4">
              {(profileState.loading || qualityState.loading || !profileDataForActive || !qualityDataForActive) && (
                <div className="flex items-center gap-2 text-xs text-navy/60">
                  <Loader2 className="w-4 h-4 text-teal animate-spin" />
                  Loading dataset profile and quality findings…
                </div>
              )}
              {profileState.error && (
                <div className="p-3 bg-red-50 border border-red-200 text-red-800 flex items-center justify-between gap-3 text-xs">
                  <span>Profile: {profileState.error}</span>
                  <Button variant="outline" size="sm" onClick={() => loadProfile(true)} icon={RefreshCw}>
                    Retry
                  </Button>
                </div>
              )}
              {qualityState.error && (
                <div className="p-3 bg-red-50 border border-red-200 text-red-800 flex items-center justify-between gap-3 text-xs">
                  <span>Quality audit: {qualityState.error}</span>
                  <Button variant="outline" size="sm" onClick={() => loadQuality(true)} icon={RefreshCw}>
                    Retry
                  </Button>
                </div>
              )}
              {(profileDataForActive || qualityDataForActive) && (
                <DatasetReviewView
                  profileData={profileDataForActive}
                  qualityData={qualityDataForActive}
                  intelligenceData={intelligenceDataForActive}
                />
              )}
            </div>
          )}

          {/* TAB 2: Clean & Transform */}
          {activeTab === 'clean' && (
            <div className="space-y-4">
              {(profileState.error || qualityState.error) && (
                <div className="p-3 bg-red-50 border border-red-200 text-red-800 space-y-2 text-xs">
                  {profileState.error && <p>Profile: {profileState.error}</p>}
                  {qualityState.error && <p>Quality audit: {qualityState.error}</p>}
                  <div className="flex gap-2">
                    {profileState.error && (
                      <Button variant="outline" size="sm" onClick={() => loadProfile(true)} icon={RefreshCw}>
                        Retry profile
                      </Button>
                    )}
                    {qualityState.error && (
                      <Button variant="outline" size="sm" onClick={() => loadQuality(true)} icon={RefreshCw}>
                        Retry quality
                      </Button>
                    )}
                  </div>
                </div>
              )}
              {(profileState.loading ||
                qualityState.loading ||
                profileState.datasetId !== originalDataset?.dataset_id ||
                qualityState.datasetId !== originalDataset?.dataset_id) && (
                <div className="flex items-center gap-2 text-xs text-navy/60">
                  <Loader2 className="w-4 h-4 text-teal animate-spin" />
                  Loading cleaning recommendations…
                </div>
              )}
              <DatasetCleaningView
                key={activeDataset?.dataset_id}
                originalDataset={originalDataset}
                qualityData={
                  qualityState.datasetId === originalDataset?.dataset_id ? qualityState.data : null
                }
                profileData={
                  profileState.datasetId === originalDataset?.dataset_id ? profileState.data : null
                }
                intelligenceData={intelligenceDataForActive}
              />
            </div>
          )}

          {/* TAB 3: Intelligence & Targets */}
          {activeTab === 'intelligence' && (
            <div>
              {intelligenceState.loading && (
                <div className="p-12 text-center bg-white border border-stone rounded-sm space-y-3">
                  <Loader2 className="w-6 h-6 text-teal animate-spin mx-auto" />
                  <p className="text-xs text-navy/70 font-sans">
                    Analyzing column roles, identifying candidate targets, and evaluating ML suitability...
                  </p>
                </div>
              )}

              {intelligenceState.error && (
                <div className="p-4 bg-red-50 border border-red-200 rounded-sm text-xs text-red-800 flex items-start justify-between gap-4">
                  <div className="flex items-start gap-2.5">
                    <AlertCircle className="w-4 h-4 text-red-600 shrink-0 mt-0.5" />
                    <div>
                      <p className="font-semibold">Failed to load intelligence report</p>
                      <p className="text-red-700/90 mt-0.5">{intelligenceState.error}</p>
                    </div>
                  </div>
                  <Button variant="outline" size="sm" onClick={() => loadIntelligence(true)} icon={RefreshCw}>
                    Retry
                  </Button>
                </div>
              )}

              {intelligenceState.data && (
                <DatasetIntelligenceView intelligenceData={intelligenceState.data} />
              )}
            </div>
          )}
        </div>
      )}

    </div>
  );
}
