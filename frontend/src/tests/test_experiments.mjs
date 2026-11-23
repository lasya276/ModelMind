import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { MemoryRouter } from 'react-router-dom';
import reactPlugin from '@vitejs/plugin-react';
import { createServer } from 'vite';
import {
  buildExperimentRequest,
  buildReportSummary,
  compactRunForHistory,
  EXPERIMENT_TYPES,
  validateExperimentConfig,
} from '../utils/experimentUtils.js';

const configs = {
  classification: { target_column: 'label', feature_columns: ['x'], test_size: 0.2 },
  regression: { target_column: 'value', feature_columns: ['x'], test_size: 0.25 },
  clustering: { feature_columns: ['x', 'y'], n_clusters: '3' },
  anomaly: { feature_columns: ['x'], contamination: 'auto' },
};

for (const [type, config] of Object.entries(configs)) {
  const request = buildExperimentRequest(type, 'dataset-version-id', config);
  assert.equal(request.endpoint, EXPERIMENT_TYPES[type].endpoint);
  assert.equal(request.payload.dataset_id, 'dataset-version-id');
  assert.deepEqual(request.payload.feature_columns, config.feature_columns);
}
assert.equal(buildExperimentRequest('classification', 'id', configs.classification).payload.test_size, 0.2);
assert.equal(buildExperimentRequest('clustering', 'id', configs.clustering).payload.n_clusters, 3);
assert.equal(buildExperimentRequest('anomaly', 'id', configs.anomaly).payload.contamination, 'auto');

assert.match(validateExperimentConfig('classification', { ...configs.classification, target_column: '' }, 20), /target column/i);
assert.match(validateExperimentConfig('regression', { ...configs.regression, feature_columns: [] }, 20), /feature column/i);
assert.match(validateExperimentConfig('regression', { ...configs.regression, test_size: 0.9 }, 20), /test size/i);
assert.match(validateExperimentConfig('clustering', { ...configs.clustering, n_clusters: '20' }, 20), /less than/i);
assert.match(validateExperimentConfig('anomaly', { ...configs.anomaly, contamination: 0.8 }, 20), /contamination/i);
assert.equal(validateExperimentConfig('classification', configs.classification, 20), null);

const longRegressionPreview = {
  model: {
    row_indices: Array.from({ length: 501 }, (_, i) => i),
    actual_values: Array.from({ length: 501 }, (_, i) => i),
    predicted_values: Array.from({ length: 501 }, (_, i) => i + 0.5),
    total_rows: 501,
    returned_rows: 501,
    truncated: false,
  },
};
const compactRegression = compactRunForHistory({
  experiment_type: 'regression',
  configuration: { feature_columns: ['x'] },
  response: { data: { predictions_preview: longRegressionPreview } },
});
assert.equal(compactRegression.response.data.predictions_preview.model.row_indices.length, 100);
assert.equal(compactRegression.response.data.predictions_preview.model.truncated, true);

const compactAnomaly = compactRunForHistory({
  experiment_type: 'anomaly',
  configuration: { feature_columns: ['x'] },
  response: {
    data: {
      features_used: ['x'],
      anomaly_labels: Array(500).fill(1),
      anomaly_scores: Array(500).fill(0.1),
      is_anomaly: Array(500).fill(false),
      visualization_data: {
        rows: Array.from({ length: 500 }, (_, row_index) => ({
          row_index,
          features: { x: row_index },
          label: 1,
          score: 0.1,
          is_anomaly: false,
        })),
        returned_rows: 500,
      },
      flagged_observations: { rows: [], returned_count: 0, total_count: 0 },
    },
  },
});
assert.equal(compactAnomaly.response.data.anomaly_scores, undefined);
assert.equal(compactAnomaly.response.data.visualization_data.rows.length, 100);

const report = buildReportSummary({
  experiment_type: 'regression',
  filename: 'sales.csv',
  dataset_id: 'dataset-id',
  dataset_version: 'cleaned',
  executed_at: '2026-01-01T00:00:00.000Z',
  configuration: configs.regression,
  response: {
    data: {
      status: 'completed',
      target_column: 'value',
      train_rows: 80,
      test_rows: 20,
      target_summary: { mean: 10 },
      best_model: { model_name: 'Linear Regression' },
      model_comparison: [{ model_name: 'Linear Regression', rmse: 1.2 }],
      predictions_preview: { linear_regression: { actual_values: [1], predicted_values: [1] } },
      explanation: 'Actual API explanation.',
    },
  },
});
assert.equal(report.dataset.version, 'cleaned');
assert.equal(report.result.model_comparison[0].rmse, 1.2);
assert.equal(report.result.predictions_preview, undefined);
assert.equal(report.result.explanation, 'Actual API explanation.');

const frontendRoot = process.cwd();
const vite = await createServer({
  configFile: false,
  root: frontendRoot,
  plugins: [reactPlugin()],
  server: { middlewareMode: true },
  appType: 'custom',
});

try {
  const { default: ExperimentResults } = await vite.ssrLoadModule('/src/components/experiments/ExperimentResults.jsx');
  const { default: ExperimentConfiguration } = await vite.ssrLoadModule('/src/components/experiments/ExperimentConfiguration.jsx');
  const { default: ExperimentsPage } = await vite.ssrLoadModule('/src/pages/Experiments/ExperimentsPage.jsx');
  const { default: ReportsPage } = await vite.ssrLoadModule('/src/pages/Reports/ReportsPage.jsx');
  const { DatasetProvider } = await vite.ssrLoadModule('/src/context/DatasetContext.jsx');
  const { ExperimentHistoryProvider } = await vite.ssrLoadModule('/src/context/ExperimentHistoryContext.jsx');
  const { default: apiClient } = await vite.ssrLoadModule('/src/api/client.js');
  const { runExperiment } = await vite.ssrLoadModule('/src/api/experimentApi.js');

  const originalAdapter = apiClient.defaults.adapter;
  const requests = [];
  apiClient.defaults.adapter = async (request) => {
    requests.push(request);
    return {
      data: { status: 'success' },
      status: 200,
      statusText: 'OK',
      headers: {},
      config: request,
    };
  };
  try {
    for (const [type, config] of Object.entries(configs)) {
      await runExperiment(type, 'chosen-version-id', config);
    }
  } finally {
    apiClient.defaults.adapter = originalAdapter;
  }
  assert.equal(requests.length, 4);
  for (const [index, type] of Object.keys(configs).entries()) {
    assert.equal(requests[index].url, EXPERIMENT_TYPES[type].endpoint);
    assert.equal(JSON.parse(requests[index].data).dataset_id, 'chosen-version-id');
  }

  const fixtures = [
    {
      experiment_type: 'classification',
      response: { data: { status: 'completed', best_model: { model_name: 'Random Forest', f1_score: 0.9 }, model_comparison: [], models: {}, confusion_matrices: {}, class_distribution: {}, train_rows: 8, test_rows: 2, explanation: 'Classification explanation.' } },
      expected: 'Confusion matrices',
    },
    {
      experiment_type: 'regression',
      response: { data: { status: 'completed', best_model: { model_name: 'Linear Regression', best_rmse: 1 }, model_comparison: [], models: {}, predictions_preview: {}, target_summary: {}, train_rows: 8, test_rows: 2, explanation: 'Regression explanation.' } },
      expected: 'Actual vs. predicted',
    },
    {
      experiment_type: 'clustering',
      response: { data: { status: 'completed', n_clusters: 2, total_rows: 4, cluster_sizes: { 0: 2, 1: 2 }, features_used: ['x', 'y'], visualization_data: { rows: [], feature_columns: ['x', 'y'], returned_rows: 0, total_rows: 4 }, centroids: {}, inertia: 2, silhouette_score: 0.5, explanation: 'Cluster explanation.' } },
      expected: 'Cluster visualization',
    },
    {
      experiment_type: 'anomaly',
      response: { data: { status: 'completed', total_rows_analyzed: 4, normal_count: 3, anomaly_count: 1, anomaly_percentage: 25, features_used: ['x'], visualization_data: { rows: [] }, flagged_observations: { rows: [], total_count: 1, returned_count: 0 }, explanation: 'Anomaly explanation.' } },
      expected: 'Anomaly score overview',
    },
  ];

  for (const [index, fixture] of fixtures.entries()) {
    const markup = renderToStaticMarkup(React.createElement(ExperimentResults, {
      run: {
        run_id: `run-${index}`,
        ...fixture,
        filename: 'data.csv',
        dataset_id: 'dataset-id',
        dataset_version: 'original',
        executed_at: '2026-01-01T00:00:00.000Z',
        configuration: {},
      },
      onBack() {},
      onNewExperiment() {},
    }));
    assert.ok(markup.includes(fixture.expected), `${fixture.experiment_type} results should render`);
  }

  const incompleteMarkup = renderToStaticMarkup(React.createElement(ExperimentResults, {
    run: {
      run_id: 'incomplete',
      experiment_type: 'anomaly',
      filename: 'small.csv',
      dataset_id: 'dataset-id',
      dataset_version: 'original',
      executed_at: '2026-01-01T00:00:00.000Z',
      configuration: {},
      response: { data: { status: 'insufficient_data', explanation: 'Too few rows.' } },
    },
    onBack() {},
    onNewExperiment() {},
  }));
  assert.ok(incompleteMarkup.includes('Anomaly detection not completed'));
  assert.ok(!incompleteMarkup.includes('Anomaly score overview'));

  const configMarkup = renderToStaticMarkup(React.createElement(ExperimentConfiguration, {
    type: 'classification',
    config: configs.classification,
    onChange() {},
    profile: {
      columns: ['label', 'x'],
      numeric_summary: { x: { mean: 1 } },
    },
    rowCount: 20,
    loading: false,
    error: null,
    onRun() {},
  }));
  assert.ok(configMarkup.includes('Choose a target column'));
  assert.ok(configMarkup.includes('Logistic Regression'));
  assert.ok(configMarkup.includes('Random Forest'));
  assert.ok(configMarkup.includes('Test data'));

  const storage = new Map();
  globalThis.sessionStorage = {
    getItem(key) { return storage.get(key) || null; },
    setItem(key, value) { storage.set(key, value); },
    removeItem(key) { storage.delete(key); },
  };
  storage.set('modelmind_active_dataset', JSON.stringify({
    dataset_id: 'cleaned-id',
    filename: 'sales_cleaned.csv',
    row_count: 18,
    column_count: 2,
    original_dataset: {
      dataset_id: 'original-id',
      filename: 'sales.csv',
      row_count: 20,
      column_count: 2,
      columns: ['x', 'label'],
    },
    cleaned_dataset: {
      dataset_id: 'cleaned-id',
      filename: 'sales_cleaned.csv',
      row_count: 18,
      column_count: 2,
      columns: ['x', 'label'],
    },
    active_version: 'cleaned',
  }));
  const pageMarkup = renderToStaticMarkup(
    React.createElement(MemoryRouter, null,
      React.createElement(DatasetProvider, null,
        React.createElement(ExperimentHistoryProvider, null,
          React.createElement(ExperimentsPage)
        )
      )
    )
  );
  assert.ok(pageMarkup.includes('Classification'));
  assert.ok(pageMarkup.includes('Regression'));
  assert.ok(pageMarkup.includes('Clustering'));
  assert.ok(pageMarkup.includes('Anomaly Detection'));
  assert.ok(pageMarkup.includes('Cleaned version'));
  assert.ok(pageMarkup.includes('cleaned-id'));
  assert.ok(pageMarkup.includes('Original'));

  storage.set('modelmind_experiment_history', JSON.stringify([{
    run_id: 'saved-run',
    experiment_type: 'classification',
    dataset_id: 'cleaned-id',
    dataset_version: 'cleaned',
    filename: 'sales_cleaned.csv',
    executed_at: '2026-01-01T00:00:00.000Z',
    configuration: configs.classification,
    response: {
      data: {
        status: 'completed',
        best_model: { model_key: 'random_forest', model_name: 'Random Forest', f1_score: 0.9 },
        models: { random_forest: { accuracy: 0.9, f1_score: 0.9 } },
        model_comparison: [{ model_name: 'Random Forest', accuracy: 0.9, f1_score: 0.9 }],
        explanation: 'Saved completed result.',
      },
    },
  }]));
  const reportsMarkup = renderToStaticMarkup(
    React.createElement(MemoryRouter, null,
      React.createElement(DatasetProvider, null,
        React.createElement(ExperimentHistoryProvider, null,
          React.createElement(ReportsPage)
        )
      )
    )
  );
  assert.ok(reportsMarkup.includes('Completed experiments'));
  assert.ok(reportsMarkup.includes('Saved completed result.'));
  assert.ok(reportsMarkup.includes('Export JSON'));
} finally {
  await vite.close();
}

const pageSource = fs.readFileSync(
  path.join(frontendRoot, 'src/pages/Experiments/ExperimentsPage.jsx'),
  'utf8'
);
assert.ok(pageSource.includes('switchDatasetVersion'));
assert.ok(pageSource.includes('EXPERIMENT_OPTIONS'));
assert.ok(pageSource.includes('openRun'));

console.log('Experiment frontend contract, history, report, and SSR result tests passed.');
