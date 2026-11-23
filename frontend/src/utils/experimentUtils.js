export const EXPERIMENT_TYPES = {
  classification: {
    label: 'Classification',
    description: 'Predict a category or class label.',
    endpoint: '/classify',
  },
  regression: {
    label: 'Regression',
    description: 'Estimate a continuous numeric value.',
    endpoint: '/regression',
  },
  clustering: {
    label: 'Clustering',
    description: 'Group similar observations without target labels.',
    endpoint: '/cluster',
  },
  anomaly: {
    label: 'Anomaly Detection',
    description: 'Find observations that are unusual in numeric feature space.',
    endpoint: '/anomaly',
  },
};

export function buildExperimentRequest(type, datasetId, config) {
  const experiment = EXPERIMENT_TYPES[type];
  if (!experiment) {
    throw new Error(`Unknown experiment type: ${type}`);
  }
  if (!datasetId) {
    throw new Error('Choose an active dataset before running an experiment.');
  }

  const payload = {
    dataset_id: datasetId,
    feature_columns: config.feature_columns,
  };

  if (type === 'classification' || type === 'regression') {
    payload.target_column = config.target_column;
    payload.test_size = Number(config.test_size);
  } else if (type === 'clustering') {
    payload.n_clusters = Number(config.n_clusters);
  } else {
    payload.contamination = config.contamination === 'auto'
      ? 'auto'
      : Number(config.contamination);
  }

  return { endpoint: experiment.endpoint, payload };
}

export function validateExperimentConfig(type, config, rowCount) {
  if (!EXPERIMENT_TYPES[type]) return 'Choose a valid experiment type.';
  if (!Array.isArray(config.feature_columns) || config.feature_columns.length === 0) {
    return 'Select at least one feature column.';
  }

  if (type === 'classification' || type === 'regression') {
    if (!config.target_column) return 'Select a target column.';
    const testSize = Number(config.test_size);
    if (!Number.isFinite(testSize) || testSize < 0.05 || testSize > 0.5) {
      return 'Test size must be between 0.05 and 0.5.';
    }
    if (config.feature_columns.includes(config.target_column)) {
      return 'The target column cannot also be used as a feature.';
    }
  }

  if (type === 'clustering') {
    const clusters = Number(config.n_clusters);
    if (!Number.isInteger(clusters) || clusters < 2) {
      return 'Choose a whole number of clusters of at least 2.';
    }
    if (Number.isFinite(rowCount) && clusters >= rowCount) {
      return 'The number of clusters must be less than the number of dataset rows.';
    }
  }

  if (type === 'anomaly' && config.contamination !== 'auto') {
    const contamination = Number(config.contamination);
    if (!Number.isFinite(contamination) || contamination <= 0 || contamination > 0.5) {
      return 'Contamination must be greater than 0 and no more than 0.5.';
    }
  }

  return null;
}

function limitRows(container, limit) {
  if (!container || !Array.isArray(container.rows)) return container;
  return {
    ...container,
    rows: container.rows.slice(0, limit),
    returned_rows: Math.min(container.rows.length, limit),
    truncated: container.truncated || container.rows.length > limit,
  };
}

export function compactRunForHistory(run) {
  const compact = {
    ...run,
    configuration: { ...run.configuration },
    response: {
      ...run.response,
      data: { ...run.response.data },
    },
  };
  const data = compact.response.data;
  const configFeatures = compact.configuration.feature_columns;
  if (Array.isArray(configFeatures) && configFeatures.length > 20) {
    compact.configuration.feature_columns = configFeatures.slice(0, 20);
    compact.configuration_omitted_feature_count = configFeatures.length - 20;
  }

  if (run.experiment_type === 'regression' && data.predictions_preview) {
    for (const key of ['selected_feature_columns', 'numerical_features', 'categorical_features']) {
      if (Array.isArray(data[key]) && data[key].length > 20) {
        data[`${key}_omitted_count`] = data[key].length - 20;
        data[key] = data[key].slice(0, 20);
      }
    }
    data.predictions_preview = Object.fromEntries(
      Object.entries(data.predictions_preview).map(([key, preview]) => [
        key,
        {
          ...preview,
          row_indices: preview.row_indices.slice(0, 100),
          actual_values: preview.actual_values.slice(0, 100),
          predicted_values: preview.predicted_values.slice(0, 100),
          returned_rows: Math.min(preview.returned_rows, 100),
          truncated: preview.truncated || preview.returned_rows > 100,
        },
      ])
    );
  }

  if (run.experiment_type === 'clustering') {
    delete data.cluster_assignments;
    data.visualization_data = limitRows(data.visualization_data, 100);
    const features = data.features_used || [];
    if (features.length > 5) {
      data.features_used_omitted_count = features.length - 5;
      data.features_used = features.slice(0, 5);
    }
    if (data.visualization_data) {
      data.visualization_data.omitted_feature_count = Math.max(
        data.visualization_data.omitted_feature_count || 0,
        features.length - 5
      );
    }
    for (const key of ['cluster_sizes', 'centroids']) {
      const entries = Object.entries(data[key] || {});
      if (entries.length > 50) {
        if (key === 'cluster_sizes') {
          data.cluster_sizes_omitted_count = entries.length - 50;
        } else {
          data.centroids_omitted_count = entries.length - 50;
        }
        data[key] = Object.fromEntries(entries.slice(0, 50));
      }
    }
    if (data.centroids) {
      for (const centroid of Object.values(data.centroids)) {
        for (const feature of Object.keys(centroid)) {
          if (!data.features_used.includes(feature)) delete centroid[feature];
        }
      }
    }
    if (data.visualization_data?.rows) {
      data.visualization_data = {
        ...data.visualization_data,
        feature_columns: data.visualization_data.feature_columns.slice(0, 5),
        rows: data.visualization_data.rows.map((row) => ({
          ...row,
          features: Object.fromEntries(Object.entries(row.features).slice(0, 5)),
        })),
      };
    }
  }

  if (run.experiment_type === 'anomaly') {
    delete data.anomaly_labels;
    delete data.anomaly_scores;
    delete data.is_anomaly;
    const originalFeatureCount = data.features_used.length;
    if (data.features_used.length > 5) {
      data.features_used_omitted_count = data.features_used.length - 5;
      data.features_used = data.features_used.slice(0, 5);
    }
    data.visualization_data = limitRows(data.visualization_data, 100);
    data.flagged_observations = limitRows(data.flagged_observations, 100);
    if (data.visualization_data) {
      data.visualization_data.omitted_feature_count = Math.max(
        data.visualization_data.omitted_feature_count || 0,
        originalFeatureCount - 5
      );
    }
    for (const key of ['visualization_data', 'flagged_observations']) {
      const preview = data[key];
      if (preview?.rows) {
        preview.feature_columns = data.features_used.slice(0, 5);
        preview.rows = preview.rows.map((row) => ({
          ...row,
          features: Object.fromEntries(Object.entries(row.features).slice(0, 5)),
        }));
      }
    }
  }

  if (run.experiment_type === 'classification' && data.confusion_matrices) {
    const distributions = Object.entries(data.class_distribution || {});
    if (distributions.length > 100) {
      data.class_distribution_total_classes = distributions.length;
      data.class_distribution = Object.fromEntries(distributions.slice(0, 100));
    }
    for (const matrix of Object.values(data.confusion_matrices)) {
      if (matrix.labels.length > 30) {
        matrix.total_classes = matrix.labels.length;
        matrix.labels = matrix.labels.slice(0, 30);
        matrix.matrix = matrix.matrix.slice(0, 30).map((row) => row.slice(0, 30));
        matrix.truncated = true;
      }
    }
  }

  return compact;
}

export function buildReportSummary(run) {
  const data = run.response.data;
  const common = {
    experiment_type: run.experiment_type,
    dataset: {
      filename: run.filename,
      version: run.dataset_version,
      dataset_id: run.dataset_id,
    },
    executed_at: run.executed_at,
    configuration: run.configuration,
  };

  if (run.experiment_type === 'classification') {
    return {
      ...common,
      result: {
        status: data.status,
        target_column: data.target_column,
        classification_type: data.classification_type,
        train_rows: data.train_rows,
        test_rows: data.test_rows,
        best_model: data.best_model,
        model_comparison: data.model_comparison,
        class_distribution: Object.fromEntries(Object.entries(data.class_distribution || {}).slice(0, 30)),
        explanation: data.explanation,
      },
    };
  }

  if (run.experiment_type === 'regression') {
    return {
      ...common,
      result: {
        status: data.status,
        target_column: data.target_column,
        train_rows: data.train_rows,
        test_rows: data.test_rows,
        target_summary: data.target_summary,
        best_model: data.best_model,
        model_comparison: data.model_comparison,
        explanation: data.explanation,
      },
    };
  }

  if (run.experiment_type === 'clustering') {
    return {
      ...common,
      result: {
        status: data.status,
        features_used: data.features_used,
        n_clusters: data.n_clusters,
        cluster_sizes: data.cluster_sizes,
        centroids: data.centroids,
        inertia: data.inertia,
        silhouette_score: data.silhouette_score,
        explanation: data.explanation,
      },
    };
  }

  return {
    ...common,
    result: {
      status: data.status,
      features_used: data.features_used,
      total_rows_analyzed: data.total_rows_analyzed,
      normal_count: data.normal_count,
      anomaly_count: data.anomaly_count,
      anomaly_percentage: data.anomaly_percentage,
      explanation: data.explanation,
    },
  };
}
