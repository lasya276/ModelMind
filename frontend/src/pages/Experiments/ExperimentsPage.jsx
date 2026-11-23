import React, { useEffect, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import {
  AlertCircle,
  AlertTriangle,
  ArrowRightLeft,
  Database,
  Loader2,
  Network,
  RefreshCw,
  Target,
  TrendingUp,
} from 'lucide-react';
import Badge from '../../components/common/Badge';
import Button from '../../components/common/Button';
import Card from '../../components/common/Card';
import ExperimentConfiguration from '../../components/experiments/ExperimentConfiguration';
import ExperimentResults from '../../components/experiments/ExperimentResults';
import { runExperiment } from '../../api/experimentApi';
import { getDatasetProfile } from '../../api/datasetApi';
import { getErrorMessage, isDatasetNotFoundError } from '../../api/client';
import { useDataset } from '../../context/DatasetContext';
import { useExperimentHistory } from '../../context/ExperimentHistoryContext';
import { EXPERIMENT_TYPES, validateExperimentConfig } from '../../utils/experimentUtils';

const EXPERIMENT_OPTIONS = [
  { id: 'classification', icon: Target },
  { id: 'regression', icon: TrendingUp },
  { id: 'clustering', icon: Network },
  { id: 'anomaly', icon: AlertTriangle },
];

function createDefaultConfigurations() {
  return {
    classification: { target_column: '', feature_columns: [], test_size: 0.2 },
    regression: { target_column: '', feature_columns: [], test_size: 0.2 },
    clustering: { feature_columns: [], n_clusters: '3' },
    anomaly: { feature_columns: [], contamination: 'auto' },
  };
}

function ExperimentHistory({ history, selectedRunId, openRun, clearHistory }) {
  return (
    <Card
      title="This session's runs"
      subtitle={`${history.length} of 10 recent runs`}
      headerAction={history.length > 0 ? (
        <Button variant="ghost" size="sm" onClick={clearHistory}>Clear</Button>
      ) : null}
    >
      {history.length === 0 ? (
        <p className="text-xs text-navy/60 leading-relaxed">
          Completed runs will appear here. History is kept for this browser session.
        </p>
      ) : (
        <div className="space-y-2">
          {history.map((run) => {
            const type = EXPERIMENT_TYPES[run.experiment_type];
            const completed = run.response?.data?.status === 'completed';
            return (
              <button
                type="button"
                key={run.run_id}
                onClick={() => openRun(run.run_id)}
                aria-pressed={selectedRunId === run.run_id}
                className={`w-full rounded-sm border p-3 text-left transition-colors focus:outline-none focus:ring-2 focus:ring-teal/30 ${
                  selectedRunId === run.run_id
                    ? 'border-teal bg-teal-soft/40'
                    : 'border-stone bg-white hover:bg-stone-light/30'
                }`}
              >
                <div className="flex items-start justify-between gap-2">
                  <span className="text-xs font-semibold text-navy">{type?.label || run.experiment_type}</span>
                  <Badge variant={completed ? 'teal' : 'amber'} size="sm">
                    {completed ? 'Completed' : 'Not completed'}
                  </Badge>
                </div>
                <p className="mt-1 truncate text-[11px] text-navy/65" title={run.filename}>{run.filename}</p>
                <div className="mt-1 flex items-center justify-between gap-2 text-[10px] text-navy/50">
                  <span className="capitalize">{run.dataset_version} version</span>
                  <span>{new Date(run.executed_at).toLocaleString()}</span>
                </div>
              </button>
            );
          })}
        </div>
      )}
    </Card>
  );
}

export default function ExperimentsPage() {
  const {
    activeDataset,
    originalDataset,
    cleanedDataset,
    activeVersion,
    switchDatasetVersion,
    handleDatasetExpiredOrMissing,
    datasetNotice,
  } = useDataset();
  const {
    history,
    selectedRun,
    selectedRunId,
    persistenceError,
    addRun,
    openRun,
    clearHistory,
  } = useExperimentHistory();

  const [selectedType, setSelectedType] = useState('classification');
  const [configurations, setConfigurations] = useState(createDefaultConfigurations);
  const [profileState, setProfileState] = useState({ data: null, loading: false, error: null, datasetId: null });
  const [runError, setRunError] = useState(null);
  const [validationError, setValidationError] = useState(null);
  const [isRunning, setIsRunning] = useState(false);
  const [profileAttempt, setProfileAttempt] = useState(0);
  const lastDatasetId = useRef(null);
  const profile = profileState.datasetId === activeDataset?.dataset_id ? profileState.data : null;
  const config = configurations[selectedType];

  useEffect(() => {
    const datasetId = activeDataset?.dataset_id || null;
    if (lastDatasetId.current && lastDatasetId.current !== datasetId) {
      setConfigurations(createDefaultConfigurations());
      setRunError(null);
      setValidationError(null);
      openRun(null);
    }
    lastDatasetId.current = datasetId;
  }, [activeDataset?.dataset_id, openRun]);

  useEffect(() => {
    const datasetId = activeDataset?.dataset_id;
    if (!datasetId || selectedRun) {
      setProfileState({ data: null, loading: false, error: null, datasetId: datasetId || null });
      return undefined;
    }

    const controller = new AbortController();
    setProfileState({ data: null, loading: true, error: null, datasetId });
    getDatasetProfile(datasetId, controller.signal)
      .then((response) => {
        setProfileState({
          data: response.data,
          loading: false,
          error: null,
          datasetId,
        });
      })
      .catch((error) => {
        if (error.name === 'CanceledError' || error.message === 'canceled') return;
        if (isDatasetNotFoundError(error)) {
          handleDatasetExpiredOrMissing();
          return;
        }
        setProfileState({
          data: null,
          loading: false,
          error: getErrorMessage(error, 'Could not load dataset columns.'),
          datasetId,
        });
      });

    return () => controller.abort();
  }, [activeDataset?.dataset_id, selectedRun, handleDatasetExpiredOrMissing, profileAttempt]);

  const updateConfig = (nextConfig) => {
    setConfigurations((current) => ({ ...current, [selectedType]: nextConfig }));
    setRunError(null);
    setValidationError(null);
  };

  const handleRun = async () => {
    if (!activeDataset?.dataset_id) {
      setRunError('Choose an active dataset before running an experiment.');
      return;
    }
    const rowCount = profile?.total_rows ?? activeDataset.row_count;
    const configError = validateExperimentConfig(selectedType, config, rowCount);
    if (configError) {
      setValidationError(configError);
      return;
    }

    setIsRunning(true);
    setRunError(null);
    setValidationError(null);
    const datasetAtRun = activeDataset;
    const versionAtRun = activeVersion;

    try {
      const response = await runExperiment(selectedType, datasetAtRun.dataset_id, config);
      const resultRun = addRun({
        experiment_type: selectedType,
        dataset_id: datasetAtRun.dataset_id,
        dataset_version: versionAtRun,
        filename: datasetAtRun.filename,
        executed_at: new Date().toISOString(),
        configuration: {
          ...config,
          feature_columns: [...config.feature_columns],
        },
        response,
      });
      openRun(resultRun.run_id);
    } catch (error) {
      if (isDatasetNotFoundError(error)) {
        handleDatasetExpiredOrMissing();
      }
      setRunError(getErrorMessage(error, 'The experiment could not be completed. Please try again.'));
    } finally {
      setIsRunning(false);
    }
  };

  const showConfiguration = () => {
    openRun(null);
    setRunError(null);
  };

  const selectExperiment = (type) => {
    setSelectedType(type);
    openRun(null);
    setRunError(null);
    setValidationError(null);
  };

  const switchVersion = (version) => {
    if (isRunning || selectedRun || !activeDataset || version === activeVersion) return;
    switchDatasetVersion(version);
  };

  const rowCount = profile?.total_rows ?? activeDataset?.row_count;
  const columnCount = profile?.total_columns ?? activeDataset?.column_count;

  return (
    <div className="space-y-7">
      <header className="border-b border-stone pb-6">
        <Badge variant="teal" className="mb-2">Machine Learning</Badge>
        <h1 className="font-serif text-3xl sm:text-4xl font-bold tracking-tight text-navy">Experiments</h1>
        <p className="mt-1 max-w-3xl text-sm text-navy/70 leading-relaxed">
          Configure and run classification, regression, clustering, or anomaly detection on your selected dataset version. Results use the existing ModelMind APIs.
        </p>
      </header>

      {datasetNotice && (
        <div role="alert" className="flex flex-wrap items-center justify-between gap-3 border border-amber-300 bg-amber-50 p-3 text-xs text-amber-950">
          <span>{datasetNotice.message}</span>
          <Link to="/datasets"><Button variant="outline" size="sm">Upload a dataset</Button></Link>
        </div>
      )}

      <div className="grid grid-cols-1 xl:grid-cols-[minmax(0,1fr)_20rem] gap-5 items-start">
        <div className="space-y-5 min-w-0">
          <section aria-labelledby="experiment-type-heading">
            <h2 id="experiment-type-heading" className="mb-3 text-xs font-semibold uppercase tracking-wide text-navy/60">
              Choose experiment type
            </h2>
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
              {EXPERIMENT_OPTIONS.map(({ id, icon: Icon }) => (
                <button
                  type="button"
                  key={id}
                  aria-pressed={selectedType === id}
                  disabled={isRunning}
                  onClick={() => selectExperiment(id)}
                  className={`flex min-h-20 items-center gap-3 rounded-sm border bg-white p-3 text-left transition-colors focus:outline-none focus:ring-2 focus:ring-teal/30 disabled:cursor-not-allowed disabled:opacity-60 ${
                    selectedType === id ? 'border-teal bg-teal-soft/30' : 'border-stone hover:border-teal/50'
                  }`}
                >
                  <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-sm bg-sage/60 text-teal">
                    <Icon className="h-4 w-4" aria-hidden="true" />
                  </span>
                  <span className="min-w-0">
                    <span className="block text-xs font-semibold text-navy">{EXPERIMENT_TYPES[id].label}</span>
                    <span className="mt-0.5 block text-[10px] leading-snug text-navy/55">{EXPERIMENT_TYPES[id].description}</span>
                  </span>
                </button>
              ))}
            </div>
          </section>

          <Card
            title="Active dataset"
            subtitle={activeDataset ? activeDataset.filename : 'Select a dataset in the Datasets workspace to begin.'}
            headerAction={activeDataset ? (
              <Badge variant={activeVersion === 'cleaned' ? 'teal' : 'sage'}>
                {activeVersion === 'cleaned' ? 'Cleaned version' : 'Original version'}
              </Badge>
            ) : null}
          >
            {!activeDataset ? (
              <div className="flex flex-wrap items-center justify-between gap-3">
                <p className="text-xs text-navy/70">Experiments require a stored CSV dataset.</p>
                <Link to="/datasets"><Button variant="primary" size="sm" icon={Database}>Open Datasets</Button></Link>
              </div>
            ) : (
              <div className="space-y-4">
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
                  <div><span className="text-[10px] uppercase text-navy/50">Rows</span><p className="mt-0.5 font-semibold text-navy">{rowCount?.toLocaleString() ?? '—'}</p></div>
                  <div><span className="text-[10px] uppercase text-navy/50">Columns</span><p className="mt-0.5 font-semibold text-navy">{columnCount ?? '—'}</p></div>
                  <div className="sm:col-span-2 min-w-0"><span className="text-[10px] uppercase text-navy/50">Dataset ID</span><p className="mt-0.5 break-all font-mono text-[11px] text-navy/75">{activeDataset.dataset_id}</p></div>
                </div>
                <div className="flex flex-wrap items-center gap-2 border-t border-stone pt-3">
                  <span className="mr-1 text-[11px] text-navy/60">Dataset version:</span>
                  <Button
                    variant={activeVersion === 'original' ? 'primary' : 'outline'}
                    size="sm"
                    disabled={isRunning || Boolean(selectedRun) || !originalDataset}
                    aria-pressed={activeVersion === 'original'}
                    onClick={() => switchVersion('original')}
                  >
                    Original
                  </Button>
                  <Button
                    variant={activeVersion === 'cleaned' ? 'primary' : 'outline'}
                    size="sm"
                    disabled={isRunning || Boolean(selectedRun) || !cleanedDataset}
                    aria-pressed={activeVersion === 'cleaned'}
                    onClick={() => switchVersion('cleaned')}
                    icon={ArrowRightLeft}
                  >
                    Cleaned
                  </Button>
                  {!cleanedDataset && (
                    <span className="text-[11px] text-navy/50">Create a cleaned version in Datasets to enable it.</span>
                  )}
                  {isRunning && <span className="text-[11px] text-navy/55">Version is locked while this run is in progress.</span>}
                  {selectedRun && <span className="text-[11px] text-navy/55">Return to configuration before changing the active version.</span>}
                </div>
              </div>
            )}
          </Card>

          {selectedRun ? (
            <ExperimentResults
              key={selectedRun.run_id}
              run={selectedRun}
              onBack={showConfiguration}
              onNewExperiment={showConfiguration}
            />
          ) : activeDataset ? (
            <>
              {profileState.loading && (
                <div role="status" className="flex items-center gap-2 border border-stone bg-white p-4 text-xs text-navy/65">
                  <Loader2 className="h-4 w-4 animate-spin text-teal" />
                  Loading columns and data types…
                </div>
              )}
              {profileState.error && (
                <div role="alert" className="flex flex-wrap items-center justify-between gap-3 border border-red-200 bg-red-50 p-3 text-xs text-red-800">
                  <span>{profileState.error}</span>
                  <Button variant="outline" size="sm" icon={RefreshCw} onClick={() => {
                    setProfileState({ data: null, loading: false, error: null, datasetId: null });
                    setProfileAttempt((attempt) => attempt + 1);
                  }}>
                    Retry
                  </Button>
                </div>
              )}
              {profile ? (
                <ExperimentConfiguration
                  type={selectedType}
                  config={config}
                  onChange={updateConfig}
                  profile={profile}
                  rowCount={rowCount}
                  loading={isRunning}
                  error={validationError}
                  onRun={handleRun}
                />
              ) : null}
            </>
          ) : (
            <Card title="Ready when your dataset is">
              <p className="text-xs text-navy/70 leading-relaxed">
                Upload a CSV in Datasets. Your original and cleaned versions remain separate; this workspace will use only the version explicitly shown above.
              </p>
            </Card>
          )}

          {persistenceError && (
            <p role="status" className="text-[11px] text-amber-900">{persistenceError}</p>
          )}
          {runError && !selectedRun && (
            <div role="alert" className="flex items-start gap-2 border border-red-200 bg-red-50 p-3 text-xs text-red-800">
              <AlertCircle className="h-4 w-4 shrink-0 mt-0.5" />
              <span>{runError}</span>
            </div>
          )}
        </div>

        <aside className="space-y-4">
          <ExperimentHistory
            history={history}
            selectedRunId={selectedRunId}
            openRun={openRun}
            clearHistory={clearHistory}
          />
          <div className="border border-stone bg-stone-light/30 p-3 text-[11px] leading-relaxed text-navy/65">
            Dataset version and settings are recorded with every run. Results are evaluated on a single split and should be interpreted as evidence, not a guarantee of future performance.
          </div>
        </aside>
      </div>
    </div>
  );
}
