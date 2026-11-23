import React from 'react';
import { AlertCircle, Info } from 'lucide-react';
import Button from '../common/Button';
import Card from '../common/Card';
import ExplainerCallout from '../common/ExplainerCallout';

const modelNames = {
  classification: ['Logistic Regression', 'Decision Tree', 'Random Forest'],
  regression: ['Linear Regression', 'Decision Tree Regressor', 'Random Forest Regressor'],
};

function FeaturePicker({
  columns,
  selected,
  onChange,
  disabled,
  label,
}) {
  const toggleFeature = (column) => {
    const next = selected.includes(column)
      ? selected.filter((item) => item !== column)
      : [...selected, column];
    onChange(next);
  };

  return (
    <fieldset className="space-y-2" disabled={disabled}>
      <legend className="text-xs font-semibold text-navy">{label}</legend>
      {columns.length === 0 ? (
        <p className="text-xs text-navy/60">No eligible columns were found in this dataset.</p>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 max-h-56 overflow-y-auto p-3 border border-stone rounded-sm bg-stone-light/20">
          {columns.map((column) => (
            <label key={column} className="flex items-center gap-2 text-xs text-navy/85 min-w-0">
              <input
                type="checkbox"
                checked={selected.includes(column)}
                onChange={() => toggleFeature(column)}
                className="accent-teal shrink-0"
              />
              <span className="truncate" title={column}>{column}</span>
            </label>
          ))}
        </div>
      )}
      <p className="text-[11px] text-navy/55">
        {selected.length} feature{selected.length === 1 ? '' : 's'} selected.
      </p>
    </fieldset>
  );
}

export default function ExperimentConfiguration({
  type,
  config,
  onChange,
  profile,
  rowCount,
  loading,
  error,
  onRun,
}) {
  const columns = profile?.columns || [];
  const numericColumns = Object.keys(profile?.numeric_summary || {});
  const featureCandidates = type === 'clustering' || type === 'anomaly'
    ? numericColumns
    : columns.filter((column) => column !== config.target_column);
  const setField = (field, value) => onChange({ ...config, [field]: value });
  const setFeatures = (value) => setField('feature_columns', value);
  const isSupervised = type === 'classification' || type === 'regression';
  const targetOptions = type === 'regression' ? numericColumns : columns;

  return (
    <Card
      title="Configure experiment"
      subtitle="Choose the inputs explicitly. ModelMind will run only after you confirm."
    >
      {isSupervised && (
        <div className="space-y-5">
          <div className="space-y-1.5">
            <label htmlFor={`target-${type}`} className="text-xs font-semibold text-navy">
              Target column <span className="text-red-700">*</span>
            </label>
            <select
              id={`target-${type}`}
              value={config.target_column}
              onChange={(event) => {
                const target = event.target.value;
                const features = config.feature_columns.filter((column) => column !== target);
                if (features.length === 0 && target) {
                  onChange({
                    ...config,
                    target_column: target,
                    feature_columns: columns.filter((column) => column !== target),
                  });
                } else {
                  onChange({
                    ...config,
                    target_column: target,
                    feature_columns: features,
                  });
                }
              }}
              disabled={loading || columns.length === 0}
              className="w-full border border-stone rounded-sm bg-white px-3 py-2 text-xs text-navy focus:outline-none focus:ring-2 focus:ring-teal/30"
            >
              <option value="">Choose a target column</option>
              {targetOptions.map((column) => (
                <option key={column} value={column}>{column}</option>
              ))}
            </select>
            {type === 'classification' && (
              <p className="text-[11px] text-navy/55">
                Choose a categorical or discrete class column. No target is selected automatically.
              </p>
            )}
          </div>

          <FeaturePicker
            columns={featureCandidates}
            selected={config.feature_columns}
            onChange={setFeatures}
            disabled={loading}
            label="Feature columns"
          />

          <div className="space-y-1.5 max-w-xs">
            <label htmlFor={`test-size-${type}`} className="text-xs font-semibold text-navy">
              Test data
            </label>
            <div className="flex items-center gap-2">
              <input
                id={`test-size-${type}`}
                type="number"
                min="5"
                max="50"
                step="5"
                value={Math.round(Number(config.test_size) * 100)}
                onChange={(event) => setField('test_size', Number(event.target.value) / 100)}
                disabled={loading}
                className="w-24 border border-stone rounded-sm bg-white px-3 py-2 text-xs text-navy focus:outline-none focus:ring-2 focus:ring-teal/30"
              />
              <span className="text-xs text-navy/65">% held out for testing</span>
            </div>
            <p className="text-[11px] text-navy/55">
              The backend trains every listed model using the same split.
            </p>
          </div>

          <div className="rounded-sm border border-stone bg-stone-light/30 p-3">
            <p className="text-xs font-semibold text-navy">Models compared automatically</p>
            <ul className="mt-1 flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-navy/70">
              {modelNames[type].map((name) => <li key={name}>{name}</li>)}
            </ul>
            <p className="mt-1 text-[11px] text-navy/55">
              The API compares all available models; this is not a single-model selector.
            </p>
          </div>
        </div>
      )}

      {type === 'clustering' && (
        <div className="space-y-5">
          <FeaturePicker
            columns={featureCandidates}
            selected={config.feature_columns}
            onChange={setFeatures}
            disabled={loading}
            label="Numeric feature columns"
          />
          <div className="space-y-1.5 max-w-xs">
            <label htmlFor="cluster-count" className="text-xs font-semibold text-navy">
              Number of clusters
            </label>
            <input
              id="cluster-count"
              type="number"
              min="2"
              max={Number.isFinite(rowCount) ? rowCount - 1 : undefined}
              step="1"
              value={config.n_clusters}
              onChange={(event) => setField('n_clusters', event.target.value)}
              disabled={loading}
              className="w-28 border border-stone rounded-sm bg-white px-3 py-2 text-xs text-navy focus:outline-none focus:ring-2 focus:ring-teal/30"
            />
            <p className="text-[11px] text-navy/55">
              Must be at least 2 and fewer than the number of dataset rows.
            </p>
          </div>
          <ExplainerCallout term="What clustering means">
            K-Means groups rows by similarity in the selected features. The group numbers are labels created by the algorithm; they are not predefined categories.
          </ExplainerCallout>
        </div>
      )}

      {type === 'anomaly' && (
        <div className="space-y-5">
          <FeaturePicker
            columns={featureCandidates}
            selected={config.feature_columns}
            onChange={setFeatures}
            disabled={loading}
            label="Numeric feature columns"
          />
          <div className="space-y-2">
            <label htmlFor="contamination-mode" className="text-xs font-semibold text-navy">
              Contamination
            </label>
            <p className="max-w-2xl text-xs text-navy/65 leading-relaxed">
              Contamination sets the expected share of unusual observations used to determine the model's flagging threshold. It is not a probability that a row is anomalous.
            </p>
            <select
              id="contamination-mode"
              value={config.contamination === 'auto' ? 'auto' : 'custom'}
              onChange={(event) => setField('contamination', event.target.value === 'auto' ? 'auto' : 0.1)}
              disabled={loading}
              className="w-full max-w-xs border border-stone rounded-sm bg-white px-3 py-2 text-xs text-navy focus:outline-none focus:ring-2 focus:ring-teal/30"
            >
              <option value="auto">Automatic (backend default)</option>
              <option value="custom">Set expected fraction</option>
            </select>
            {config.contamination !== 'auto' && (
              <div className="flex items-center gap-2">
                <input
                  id="contamination-value"
                  aria-label="Expected anomaly fraction"
                  type="number"
                  min="0.000001"
                  max="0.5"
                  step="any"
                  value={config.contamination}
                  onChange={(event) => setField('contamination', event.target.value)}
                  disabled={loading}
                  className="w-28 border border-stone rounded-sm bg-white px-3 py-2 text-xs text-navy focus:outline-none focus:ring-2 focus:ring-teal/30"
                />
                <span className="text-xs text-navy/65">greater than 0 and no more than 0.5</span>
              </div>
            )}
            <div className="flex items-start gap-2 rounded-sm bg-stone-light/35 p-3 text-[11px] text-navy/70">
              <Info className="w-4 h-4 shrink-0 text-teal" />
              <span>An anomaly is an unusual observation, not automatically an error, defect, or fraud.</span>
            </div>
          </div>
        </div>
      )}

      {error && (
        <div role="alert" className="mt-5 flex items-start gap-2 border border-red-200 bg-red-50 p-3 text-xs text-red-800">
          <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
          <span>{error}</span>
        </div>
      )}

      <div className="mt-6 flex justify-end border-t border-stone pt-4">
        <Button
          variant="primary"
          onClick={onRun}
          loading={loading}
          disabled={loading || !profile}
        >
          Run Experiment
        </Button>
      </div>
    </Card>
  );
}
