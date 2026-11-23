import React, { lazy, Suspense, useMemo, useState } from 'react';
import Badge from '../common/Badge';
import Button from '../common/Button';
import Card from '../common/Card';
import ExplainerCallout from '../common/ExplainerCallout';
import { EXPERIMENT_TYPES } from '../../utils/experimentUtils';

const COLORS = ['#137C78', '#D18B45', '#54708F', '#986C8B', '#6D8B64', '#B55D54'];
const MAX_MATRIX_LABELS = 30;
const ExperimentCharts = lazy(() => import('./ExperimentCharts'));

function MetricValue({ value, digits = 4 }) {
  if (value === null || value === undefined || !Number.isFinite(Number(value))) return '—';
  return Number(value).toFixed(digits);
}

function SummaryStat({ label, value, note }) {
  return (
    <div className="border border-stone bg-stone-light/30 p-3 rounded-sm">
      <p className="text-[11px] text-navy/55 uppercase tracking-wide">{label}</p>
      <p className="mt-1 text-lg font-semibold text-navy">{value}</p>
      {note && <p className="mt-0.5 text-[11px] text-navy/55">{note}</p>}
    </div>
  );
}

function RunMetadata({ run }) {
  const config = run.configuration || {};
  const configText = Object.entries(config)
    .map(([key, value]) => `${key.replaceAll('_', ' ')}: ${Array.isArray(value) ? value.join(', ') : value}`)
    .join(' · ');
  return (
    <Card
      title={`${EXPERIMENT_TYPES[run.experiment_type]?.label || run.experiment_type} results`}
      subtitle={`${run.filename} · ${run.dataset_version === 'cleaned' ? 'Cleaned' : 'Original'} version`}
      headerAction={<Badge variant={run.dataset_version === 'cleaned' ? 'teal' : 'sage'}>{run.dataset_version}</Badge>}
    >
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-xs">
        <div><span className="text-navy/50">Run time</span><p className="mt-0.5 text-navy">{new Date(run.executed_at).toLocaleString()}</p></div>
        <div><span className="text-navy/50">Dataset ID</span><p className="mt-0.5 text-navy font-mono break-all">{run.dataset_id}</p></div>
        <div><span className="text-navy/50">Configuration</span><p className="mt-0.5 text-navy capitalize">{configText || '—'}</p></div>
      </div>
      {run.configuration_omitted_feature_count > 0 && (
        <p className="mt-2 text-[11px] text-navy/55">
          Session history keeps the first {config.feature_columns?.length || 0} feature names; {run.configuration_omitted_feature_count} more were used in the run.
        </p>
      )}
    </Card>
  );
}

function ClassificationResults({ data }) {
  const classEntries = Object.entries(data.class_distribution || {});
  return (
    <div className="space-y-5">
      <Card title="Best model" subtitle="Selected by highest weighted F1-score on this test split.">
        <div className="flex flex-wrap items-center gap-3">
          <p className="font-serif text-xl text-navy">{data.best_model?.model_name || '—'}</p>
          <Badge variant="teal">Weighted F1 <MetricValue value={data.best_model?.f1_score} /></Badge>
        </div>
        <div className="mt-4 grid grid-cols-2 sm:grid-cols-3 gap-3">
          <SummaryStat label="Training rows" value={data.train_rows?.toLocaleString() ?? '—'} />
          <SummaryStat label="Test rows" value={data.test_rows?.toLocaleString() ?? '—'} />
          <SummaryStat label="Target classes" value={data.classification_type || '—'} />
        </div>
      </Card>

      <Card title="Model comparison" subtitle="All three supported models use the same train/test split.">
        <div className="overflow-x-auto">
          <table className="w-full text-xs text-left">
            <thead className="text-[11px] uppercase tracking-wide text-navy/55 border-b border-stone">
              <tr>
                <th className="py-2 pr-4">Model</th>
                <th className="py-2 px-2" title="Correct predictions divided by all predictions.">Accuracy</th>
                <th className="py-2 px-2" title="Weighted average of precision: among predicted classes, how often was each class correct?">Weighted precision</th>
                <th className="py-2 px-2" title="Weighted average of recall: how many actual examples of each class were found?">Weighted recall</th>
                <th className="py-2 pl-2" title="Weighted balance of precision and recall; higher is better.">Weighted F1</th>
              </tr>
            </thead>
            <tbody>
              {(data.model_comparison || []).map((model) => (
                <tr key={model.model_key} className="border-b border-stone/70">
                  <td className="py-2.5 pr-4 font-medium text-navy">{model.model_name}</td>
                  <td className="py-2.5 px-2"><MetricValue value={model.accuracy} /></td>
                  <td className="py-2.5 px-2"><MetricValue value={model.precision} /></td>
                  <td className="py-2.5 px-2"><MetricValue value={model.recall} /></td>
                  <td className="py-2.5 pl-2 font-semibold"><MetricValue value={model.f1_score} /></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="mt-3 text-[11px] text-navy/55">
          Accuracy is overall correctness. Precision reflects correctness among predicted classes; recall reflects how many actual class examples were found. F1 balances precision and recall; all displayed classification averages are weighted by class frequency.
        </p>
      </Card>

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-5">
        <Card title="Confusion matrices" subtitle="Rows are actual classes; columns are predicted classes.">
          <div className="space-y-5">
            {Object.entries(data.confusion_matrices || {}).map(([key, confusion]) => {
              const labels = confusion.labels.slice(0, MAX_MATRIX_LABELS);
              return (
                <div key={key} className="overflow-x-auto">
                  <h4 className="mb-2 text-xs font-semibold text-navy">
                    {data.models?.[key]?.name || key}
                  </h4>
                  <table className="text-[11px] border-collapse">
                    <thead>
                      <tr>
                        <th className="border border-stone bg-stone-light/40 p-1.5">Actual / Predicted</th>
                        {labels.map((label) => <th key={label} className="border border-stone bg-stone-light/40 p-1.5">{label}</th>)}
                      </tr>
                    </thead>
                    <tbody>
                      {labels.map((label, rowIndex) => (
                        <tr key={label}>
                          <th className="border border-stone bg-stone-light/40 p-1.5 text-left">{label}</th>
                          {labels.map((column, columnIndex) => (
                            <td key={column} className="border border-stone p-1.5 text-center">
                              {confusion.matrix[rowIndex]?.[columnIndex] ?? 0}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  {(confusion.labels.length > MAX_MATRIX_LABELS || confusion.truncated) && (
                    <p className="mt-1 text-[11px] text-navy/55">
                      Showing the first {confusion.labels.length} of {confusion.total_classes || confusion.labels.length} classes.
                    </p>
                  )}
                </div>
              );
            })}
          </div>
        </Card>
        <Card title="Class distribution" subtitle="Counts of non-missing target labels used for classification.">
          <div className="max-h-80 overflow-y-auto divide-y divide-stone/70">
            {classEntries.slice(0, 100).map(([label, count]) => (
              <div key={label} className="flex items-center justify-between gap-3 py-2 text-xs">
                <span className="truncate text-navy" title={label}>{label}</span>
                <span className="font-mono text-navy/70">{Number(count).toLocaleString()}</span>
              </div>
            ))}
          </div>
          {(classEntries.length > 100 || data.class_distribution_total_classes) && (
            <p className="mt-2 text-[11px] text-navy/55">
              Showing {Math.min(classEntries.length, 100)} of {data.class_distribution_total_classes || classEntries.length} classes.
            </p>
          )}
        </Card>
      </div>

      <ExplainerCallout term="How to read these results">
        The best model is the one with the highest weighted F1 on this evaluation split. The confusion matrix shows correct predictions on its diagonal; off-diagonal counts show which classes were mixed up. Results describe this split and do not guarantee future performance.
      </ExplainerCallout>
      <p className="text-xs text-navy/70 leading-relaxed">{data.explanation}</p>
    </div>
  );
}

function RegressionResults({ data }) {
  const modelKeys = Object.keys(data.predictions_preview || {});
  const [selectedModel, setSelectedModel] = useState(data.best_model?.model_key || modelKeys[0] || '');
  const prediction = data.predictions_preview?.[selectedModel];
  const scatterData = useMemo(() => (
    (prediction?.actual_values || []).map((actual, index) => ({
      actual,
      predicted: prediction.predicted_values[index],
      rowIndex: prediction.row_indices[index],
    })).filter((point) => point.actual !== null && point.predicted !== null &&
      Number.isFinite(point.actual) && Number.isFinite(point.predicted))
  ), [prediction]);
  const targetSummary = data.target_summary || {};
  const rawBounds = scatterData.length
    ? [
      Math.min(...scatterData.flatMap((point) => [point.actual, point.predicted])),
      Math.max(...scatterData.flatMap((point) => [point.actual, point.predicted])),
    ]
    : [0, 1];
  const chartBounds = rawBounds[0] === rawBounds[1]
    ? [rawBounds[0] - 1, rawBounds[1] + 1]
    : rawBounds;

  return (
    <div className="space-y-5">
      <Card title="Best model" subtitle="Selected by lowest RMSE on the test split.">
        <div className="flex flex-wrap items-center gap-3">
          <p className="font-serif text-xl text-navy">{data.best_model?.model_name || '—'}</p>
          <Badge variant="teal">RMSE <MetricValue value={data.best_model?.best_rmse} /></Badge>
        </div>
        <div className="mt-4 grid grid-cols-2 sm:grid-cols-3 gap-3">
          <SummaryStat label="Training rows" value={data.train_rows?.toLocaleString() ?? '—'} />
          <SummaryStat label="Test rows" value={data.test_rows?.toLocaleString() ?? '—'} />
          <SummaryStat label="Target" value={data.target_column || '—'} />
        </div>
      </Card>

      <Card title="Model comparison" subtitle="Lower MAE, MSE and RMSE are better. R² can be negative.">
        <div className="overflow-x-auto">
          <table className="w-full text-xs text-left">
            <thead className="text-[11px] uppercase tracking-wide text-navy/55 border-b border-stone">
              <tr>
                <th className="py-2 pr-4">Model</th>
                <th className="py-2 px-2" title="Average absolute error in target units.">MAE</th>
                <th className="py-2 px-2" title="Average squared error; larger errors count more.">MSE</th>
                <th className="py-2 px-2" title="Square root of MSE, in target units.">RMSE</th>
                <th className="py-2 pl-2" title="Variance explained relative to predicting the mean; can be negative.">R²</th>
              </tr>
            </thead>
            <tbody>
              {(data.model_comparison || []).map((model) => (
                <tr key={model.model_key} className="border-b border-stone/70">
                  <td className="py-2.5 pr-4 font-medium text-navy">{model.model_name}</td>
                  <td className="py-2.5 px-2"><MetricValue value={model.mae} /></td>
                  <td className="py-2.5 px-2"><MetricValue value={model.mse} /></td>
                  <td className="py-2.5 px-2"><MetricValue value={model.rmse} /></td>
                  <td className="py-2.5 pl-2"><MetricValue value={model.r2_score} /></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="mt-3 text-[11px] text-navy/55">
          MAE is the average absolute error; MSE squares errors and penalizes large misses more; RMSE is in target units. R² compares the model with predicting the target mean and may be negative.
        </p>
      </Card>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        {Object.entries(targetSummary).map(([key, value]) => (
          <SummaryStat key={key} label={`Target ${key}`} value={<MetricValue value={value} />} />
        ))}
      </div>

      <Card
        title="Actual vs. predicted"
        subtitle="Points nearer the diagonal represent closer predictions."
        headerAction={
          <label className="flex items-center gap-2 text-[11px] text-navy/70">
            Model
            <select
              value={selectedModel}
              onChange={(event) => setSelectedModel(event.target.value)}
              className="border border-stone rounded-sm bg-white px-2 py-1 text-xs text-navy"
            >
              {modelKeys.map((key) => <option key={key} value={key}>{data.models?.[key]?.name || key}</option>)}
            </select>
          </label>
        }
      >
        {scatterData.length > 0 ? (
          <Suspense fallback={<div className="h-72 flex items-center justify-center text-xs text-navy/55">Loading chart…</div>}>
            <ExperimentCharts type="regression" points={scatterData} bounds={chartBounds} />
          </Suspense>
        ) : (
          <p className="text-xs text-navy/60">Prediction preview is not available for this run.</p>
        )}
        {prediction?.truncated && (
          <p className="mt-2 text-[11px] text-navy/55">
            Chart uses {prediction.returned_rows} of {prediction.total_rows} test observations.
          </p>
        )}
      </Card>
      <ExplainerCallout term="Evaluation limitation">
        Metrics and the chart describe only the held-out test split. They are not a guarantee of how the model will perform on future data.
      </ExplainerCallout>
      <p className="text-xs text-navy/70 leading-relaxed">{data.explanation}</p>
    </div>
  );
}

function ClusteringResults({ data }) {
  const visualization = data.visualization_data || { rows: [], feature_columns: [] };
  const featureColumns = visualization.feature_columns || [];
  const [xFeature, setXFeature] = useState(featureColumns[0] || '');
  const [yFeature, setYFeature] = useState(featureColumns[1] || featureColumns[0] || '');
  const rows = visualization.rows || [];
  const clusterIds = Object.keys(data.cluster_sizes || {});
  const centroids = data.centroids || {};

  return (
    <div className="space-y-5">
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <SummaryStat label="Clusters" value={data.n_clusters ?? '—'} />
        <SummaryStat label="Observations" value={data.total_rows?.toLocaleString() ?? '—'} />
        <SummaryStat label="Inertia" value={<MetricValue value={data.inertia} />} />
        <SummaryStat label="Silhouette" value={<MetricValue value={data.silhouette_score} />} note={data.silhouette_score === null ? 'Unavailable' : 'Higher indicates better separation'} />
      </div>
      <Card title="Cluster sizes">
        <div className="flex flex-wrap gap-3">
          {Object.entries(data.cluster_sizes || {}).map(([cluster, size], index) => (
            <div key={cluster} className="min-w-28 border border-stone bg-stone-light/25 p-3 rounded-sm">
              <Badge variant="teal">Cluster {cluster}</Badge>
              <p className="mt-2 text-lg font-semibold text-navy">{Number(size).toLocaleString()}</p>
              <p className="text-[11px] text-navy/55">observations</p>
              <span className="mt-2 block h-1 rounded" style={{ backgroundColor: COLORS[index % COLORS.length] }} />
            </div>
          ))}
        </div>
      </Card>
      <Card title="Cluster visualization" subtitle="Each point is a sampled observation. Missing or infinite values use the same median imputation as K-Means.">
        <div className="mb-3 flex flex-wrap gap-3">
          <label className="flex items-center gap-2 text-[11px] text-navy/70">
            X axis
            <select value={xFeature} onChange={(event) => setXFeature(event.target.value)} className="border border-stone rounded-sm bg-white px-2 py-1 text-xs text-navy">
              {featureColumns.map((feature) => <option key={feature} value={feature}>{feature}</option>)}
            </select>
          </label>
          {featureColumns.length > 1 && (
            <label className="flex items-center gap-2 text-[11px] text-navy/70">
              Y axis
              <select value={yFeature} onChange={(event) => setYFeature(event.target.value)} className="border border-stone rounded-sm bg-white px-2 py-1 text-xs text-navy">
                {featureColumns.map((feature) => <option key={feature} value={feature}>{feature}</option>)}
              </select>
            </label>
          )}
        </div>
        <Suspense fallback={<div className="h-72 flex items-center justify-center text-xs text-navy/55">Loading chart…</div>}>
          <ExperimentCharts
            type="clustering"
            rows={rows}
            featureColumns={featureColumns}
            clusterIds={clusterIds}
            xFeature={xFeature}
            yFeature={yFeature}
          />
        </Suspense>
        {visualization.truncated && (
          <p className="mt-2 text-[11px] text-navy/55">
            Visualization shows {visualization.returned_rows} of {visualization.total_rows} rows.
          </p>
        )}
      </Card>
      <Card title="Cluster centroids" subtitle="Average cluster positions in the original feature units.">
        <div className="overflow-x-auto">
          <table className="w-full text-xs text-left">
            <thead><tr className="border-b border-stone text-[11px] uppercase text-navy/55"><th className="py-2 pr-3">Cluster</th>{data.features_used?.map((feature) => <th key={feature} className="py-2 px-2">{feature}</th>)}</tr></thead>
            <tbody>{Object.entries(centroids).map(([cluster, values]) => <tr key={cluster} className="border-b border-stone/70"><th className="py-2 pr-3">Cluster {cluster}</th>{data.features_used?.map((feature) => <td key={feature} className="py-2 px-2"><MetricValue value={values[feature]} /></td>)}</tr>)}</tbody>
          </table>
        </div>
      </Card>
      {(visualization.omitted_feature_count > 0 || data.cluster_sizes_omitted_count || data.features_used_omitted_count) && (
        <p className="text-[11px] text-navy/55">
          {visualization.feature_columns.length < (data.features_used?.length || 0) + (data.features_used_omitted_count || 0) && (
            <>Visualization uses the first {visualization.feature_columns.length} of {(data.features_used?.length || 0) + (data.features_used_omitted_count || 0)} selected features. </>
          )}
          {data.cluster_sizes_omitted_count ? `${data.cluster_sizes_omitted_count} additional clusters are omitted from this session history view.` : ''}
        </p>
      )}
      <ExplainerCallout term="Interpretation">
        Clustering discovers groups based on the selected features. Cluster numbers are algorithm-generated identifiers, not known or predefined categories. The chosen cluster count is user-specified and is not automatically optimal.
      </ExplainerCallout>
      <p className="text-xs text-navy/70 leading-relaxed">{data.explanation}</p>
    </div>
  );
}

function AnomalyResults({ data }) {
  if (data.status !== 'completed') {
    const explanation = data.explanation || 'Anomaly detection did not complete for this dataset.';
    return (
      <Card title="Anomaly detection not completed">
        <div role="status" className="space-y-2 text-xs text-navy/75">
          <Badge variant="amber">{data.status?.replaceAll('_', ' ') || 'Unavailable'}</Badge>
          <p>{explanation}</p>
          <p>No completed anomaly result is shown for this run.</p>
        </div>
      </Card>
    );
  }

  const visualization = data.visualization_data || { rows: [] };
  const rows = visualization.rows || [];
  const flagged = data.flagged_observations?.rows || [];
  const featureColumns = data.flagged_observations?.feature_columns || visualization.feature_columns || [];
  const normalRows = rows.filter((row) => !row.is_anomaly);
  const anomalyRows = rows.filter((row) => row.is_anomaly);

  return (
    <div className="space-y-5">
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <SummaryStat label="Analyzed" value={Number(data.total_rows_analyzed || 0).toLocaleString()} />
        <SummaryStat label="Normal" value={Number(data.normal_count || 0).toLocaleString()} />
        <SummaryStat label="Flagged" value={Number(data.anomaly_count || 0).toLocaleString()} />
        <SummaryStat label="Flagged share" value={`${data.anomaly_percentage ?? 0}%`} />
      </div>
      <Card title="Anomaly score overview" subtitle="Lower Isolation Forest decision-function scores are more anomalous; scores are not probabilities.">
        <p className="mb-3 text-[11px] text-navy/60">
          Selected features: {(data.features_used || []).join(', ') || 'None'}
          {data.features_used_omitted_count ? ` and ${data.features_used_omitted_count} more` : ''}
        </p>
        <Suspense fallback={<div className="h-72 flex items-center justify-center text-xs text-navy/55">Loading chart…</div>}>
          <ExperimentCharts type="anomaly" normalRows={normalRows} anomalyRows={anomalyRows} />
        </Suspense>
        {visualization.truncated && (
          <p className="mt-2 text-[11px] text-navy/55">
            Score chart shows {visualization.returned_rows} of {visualization.total_rows} rows.
          </p>
        )}
      </Card>
      <Card
        title="Flagged observations"
        subtitle={`Showing the most anomalous observations first (${data.flagged_observations?.returned_count ?? flagged.length} of ${data.flagged_observations?.total_count ?? data.anomaly_count}).`}
      >
        {flagged.length === 0 ? (
          <p className="text-xs text-navy/60">No observations were flagged in the returned result.</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-xs text-left">
              <thead><tr className="border-b border-stone text-[11px] uppercase text-navy/55"><th className="py-2 pr-3">Dataset row</th><th className="py-2 px-2">Score</th>{featureColumns.map((feature) => <th key={feature} className="py-2 px-2">{feature}</th>)}</tr></thead>
              <tbody>
                {flagged.map((row) => (
                  <tr key={row.row_index} className="border-b border-stone/70">
                    <td className="py-2 pr-3 font-mono">{row.row_index}</td>
                    <td className="py-2 px-2 font-mono">{row.score}</td>
                    {featureColumns.map((feature) => <td key={feature} className="py-2 px-2">{row.features?.[feature] ?? '—'}</td>)}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        {data.flagged_observations?.truncated && (
          <p className="mt-2 text-[11px] text-navy/55">
            The bounded table includes the 100 lowest-scoring flagged observations.
          </p>
        )}
      </Card>
      {visualization.feature_columns?.length < (data.features_used?.length || 0) + (data.features_used_omitted_count || 0) && (
        <p className="text-[11px] text-navy/55">
          Observation previews include the first {visualization.feature_columns?.length || 0} of {(data.features_used?.length || 0) + (data.features_used_omitted_count || 0)} selected features.
        </p>
      )}
      <ExplainerCallout term="Interpretation">
        A flagged observation is unusual relative to the selected numeric features. It is not automatically an error, defect, or fraud. Review it in context before drawing conclusions.
      </ExplainerCallout>
      <p className="text-xs text-navy/70 leading-relaxed">{data.explanation}</p>
    </div>
  );
}

export default function ExperimentResults({ run, onBack, onNewExperiment }) {
  const data = run.response?.data || {};
  let content;
  if (run.experiment_type === 'classification') content = <ClassificationResults key={run.run_id} data={data} />;
  else if (run.experiment_type === 'regression') content = <RegressionResults key={run.run_id} data={data} />;
  else if (run.experiment_type === 'clustering') content = <ClusteringResults key={run.run_id} data={data} />;
  else content = <AnomalyResults key={run.run_id} data={data} />;

  return (
    <div className="space-y-5">
      <RunMetadata run={run} />
      {content}
      <div className="flex flex-wrap justify-between gap-3 border-t border-stone pt-4">
        <Button variant="outline" onClick={onBack}>Back to configuration</Button>
        <Button variant="primary" onClick={onNewExperiment}>Start another experiment</Button>
      </div>
    </div>
  );
}
