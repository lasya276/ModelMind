import React from 'react';
import { useNavigate } from 'react-router-dom';
import { Download, FileText } from 'lucide-react';
import Badge from '../../components/common/Badge';
import Button from '../../components/common/Button';
import Card from '../../components/common/Card';
import ExplainerCallout from '../../components/common/ExplainerCallout';
import RetentionNotice from '../../components/common/RetentionNotice';
import { useExperimentHistory } from '../../context/ExperimentHistoryContext';
import { buildReportSummary, EXPERIMENT_TYPES } from '../../utils/experimentUtils';

function formatMetric(value) {
  if (value === null || value === undefined || !Number.isFinite(Number(value))) return 'Not available';
  return Number(value).toFixed(4);
}

function getRunHighlights(run) {
  const data = run.response.data;
  if (run.experiment_type === 'classification') {
    const best = data.best_model || {};
    const metrics = data.models?.[best.model_key] || {};
    return [
      ['Best model', best.model_name || 'Not available'],
      ['Accuracy', formatMetric(metrics.accuracy)],
      ['Weighted F1', formatMetric(metrics.f1_score)],
    ];
  }
  if (run.experiment_type === 'regression') {
    const best = data.best_model || {};
    const metrics = data.models?.[best.model_key] || {};
    return [
      ['Best model', best.model_name || 'Not available'],
      ['RMSE', formatMetric(metrics.rmse)],
      ['R²', formatMetric(metrics.r2_score)],
    ];
  }
  if (run.experiment_type === 'clustering') {
    return [
      ['Clusters', data.n_clusters ?? 'Not available'],
      ['Inertia', formatMetric(data.inertia)],
      ['Silhouette', formatMetric(data.silhouette_score)],
    ];
  }
  return [
    ['Analyzed', Number(data.total_rows_analyzed || 0).toLocaleString()],
    ['Flagged', Number(data.anomaly_count || 0).toLocaleString()],
    ['Flagged share', `${data.anomaly_percentage ?? 0}%`],
  ];
}

function downloadReport(run) {
  const report = buildReportSummary(run);
  const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
  const url = window.URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  const safeFilename = (run.filename || 'dataset').replace(/[^a-z0-9._-]+/gi, '_');
  anchor.href = url;
  anchor.download = `modelmind_${run.experiment_type}_${safeFilename}_summary.json`;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  window.URL.revokeObjectURL(url);
}

export default function ReportsPage() {
  const navigate = useNavigate();
  const { history, openRun } = useExperimentHistory();
  const completedRuns = history.filter((run) => run.response?.data?.status === 'completed');
  const incompleteCount = history.length - completedRuns.length;

  const viewRun = (runId) => {
    openRun(runId);
    navigate('/experiments');
  };

  return (
    <div className="space-y-7">
      <header className="border-b border-stone pb-6">
        <Badge variant="teal" className="mb-2">Summary &amp; Export</Badge>
        <h1 className="font-serif text-3xl sm:text-4xl font-bold tracking-tight text-navy">Reports</h1>
        <p className="mt-1 max-w-3xl text-sm text-navy/70 leading-relaxed">
          Review and export summaries from completed machine learning experiments in this browser session.
        </p>
      </header>

      <Card
        title="Completed experiments"
        subtitle="Summaries contain only metrics and explanations returned by the experiment APIs."
      >
        {completedRuns.length === 0 ? (
          <div className="py-5 text-center">
            <FileText className="mx-auto mb-2 h-7 w-7 text-navy/30" />
            <p className="font-serif text-base font-semibold text-navy">No completed experiment reports yet</p>
            <p className="mx-auto mt-1 max-w-lg text-xs leading-relaxed text-navy/60">
              Run an experiment to create a report-ready summary. No sample or placeholder metrics are generated.
            </p>
            <Button className="mt-4" onClick={() => navigate('/experiments')}>Go to Experiments</Button>
          </div>
        ) : (
          <div className="space-y-4">
            {completedRuns.map((run) => (
              <article key={run.run_id} className="rounded-sm border border-stone bg-white p-4">
                <div className="flex flex-wrap items-start justify-between gap-3">
                  <div className="min-w-0">
                    <div className="flex flex-wrap items-center gap-2">
                      <h2 className="font-serif text-base font-semibold text-navy">
                        {EXPERIMENT_TYPES[run.experiment_type]?.label || run.experiment_type}
                      </h2>
                      <Badge variant={run.dataset_version === 'cleaned' ? 'teal' : 'sage'} size="sm">
                        {run.dataset_version} version
                      </Badge>
                    </div>
                    <p className="mt-1 truncate text-xs text-navy/70" title={run.filename}>{run.filename}</p>
                    <p className="mt-0.5 text-[11px] text-navy/50">{new Date(run.executed_at).toLocaleString()}</p>
                  </div>
                  <div className="flex flex-wrap gap-2">
                    <Button variant="outline" size="sm" onClick={() => viewRun(run.run_id)}>View results</Button>
                    <Button variant="primary" size="sm" icon={Download} onClick={() => downloadReport(run)}>
                      Export JSON
                    </Button>
                  </div>
                </div>
                <dl className="mt-4 grid grid-cols-1 sm:grid-cols-3 gap-3">
                  {getRunHighlights(run).map(([label, value]) => (
                    <div key={label} className="border border-stone bg-stone-light/30 p-3 rounded-sm">
                      <dt className="text-[10px] uppercase tracking-wide text-navy/50">{label}</dt>
                      <dd className="mt-1 text-sm font-semibold text-navy break-words">{value}</dd>
                    </div>
                  ))}
                </dl>
                <p className="mt-3 text-xs leading-relaxed text-navy/70">
                  {run.response.data.explanation}
                </p>
              </article>
            ))}
          </div>
        )}
        {incompleteCount > 0 && (
          <p className="mt-4 border-t border-stone pt-3 text-[11px] text-navy/55">
            {incompleteCount} run{incompleteCount === 1 ? '' : 's'} did not complete and are not presented as reports.
          </p>
        )}
      </Card>

      <ExplainerCallout term="Session history">
        Experiment history is stored locally for this browser session only. Dataset files remain in temporary server storage; export important summaries before closing the session.
      </ExplainerCallout>
      <RetentionNotice />
    </div>
  );
}
