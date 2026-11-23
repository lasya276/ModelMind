import React, { useState } from 'react';
import { AlertTriangle, ArrowRight, CheckCircle2, Download, Eye, Plus, Trash2, X } from 'lucide-react';
import Badge from '../common/Badge';
import Button from '../common/Button';
import { formatNumber } from '../../utils/formatters';

function strategyLabel(strategy, value) {
  if (strategy === 'constant') return `Custom Constant${value !== '' ? `: ${value}` : ''}`;
  if (strategy === 'mean') return 'Mean';
  if (strategy === 'median') return 'Median';
  if (strategy === 'mode') return 'Mode';
  return 'Leave Unchanged';
}

function recommendationReason(recommendation) {
  if (recommendation.isEmpty) return '100% missing; no observed values are available.';
  if (recommendation.isIdentifier) return 'Identifier values should remain unchanged.';
  if (recommendation.strategy === 'mean') return 'The observed distribution supports mean imputation.';
  if (recommendation.strategy === 'median') return 'The median is more robust to skew or limited statistics.';
  if (recommendation.strategy === 'mode') return 'A discrete or dominant value is best preserved.';
  if (recommendation.strategy === 'constant') return 'A neutral value avoids introducing a dominant category.';
  return recommendation.reason;
}

function selectedActionItems(operations) {
  const items = [];
  Object.entries(operations.missing_values || {}).forEach(([column, strategy]) => {
    const label =
      typeof strategy === 'object'
        ? `Custom Constant: ${strategy.value}`
        : strategyLabel(strategy);
    items.push({ key: `missing-${column}`, text: `${column}: ${label}` });
  });
  Object.entries(operations.invalid_numerical_values || {}).forEach(([column, config]) => {
    if (config.infinity_strategy) {
      const strategy =
        config.infinity_strategy === 'constant'
          ? `Custom Constant${config.infinity_constant !== undefined ? `: ${config.infinity_constant}` : ''}`
          : config.infinity_strategy === 'leave_unchanged'
            ? 'Leave Unchanged'
            : config.infinity_strategy.charAt(0).toUpperCase() + config.infinity_strategy.slice(1);
      items.push({
        key: `infinities-${column}`,
        text: `${column}: replace infinities using ${strategy}`,
      });
    }
    if (config.action && config.action !== 'none') {
      items.push({
        key: `invalid-${column}`,
        text: `${column}: ${config.action === 'replace_with_nan' ? 'Replace domain-bound violations with missing values' : 'Remove rows with domain-bound violations'}`,
      });
    }
  });
  Object.entries(operations.value_mapping || {}).forEach(([column, mapping]) => {
    Object.entries(mapping).forEach(([from, to]) => {
      items.push({ key: `mapping-${column}-${from}`, text: `${column}: "${from}" → "${to}"` });
    });
  });
  if (operations.remove_duplicates) {
    items.push({ key: 'duplicates', text: 'Remove duplicate rows' });
  }
  if (operations.trim_whitespace) {
    items.push({ key: 'trim', text: 'Trim leading and trailing whitespace' });
  }
  Object.entries(operations.casing || {}).forEach(([column, casing]) => {
    items.push({ key: `casing-${column}`, text: `${column}: convert to ${casing}case` });
  });
  if (operations.drop_empty_columns) {
    items.push({ key: 'empty-columns', text: 'Drop completely empty columns' });
  }
  if (operations.drop_constant_columns) {
    items.push({ key: 'constant-columns', text: 'Drop constant columns' });
  }
  return items;
}

export default function DatasetCleaningPlanView({
  originalDataset,
  cleanedDataset,
  isCleanedActive,
  switchDatasetVersion,
  profileData,
  qualityData,
  columnsList,
  columnTypes,
  columnsWithMissingValues,
  columnsWithInfinities,
  isNumericColumn,
  isTextColumn,
  getRecommendation,
  missingConfigs,
  setMissingConfigs,
  invalidNumConfigs,
  setInvalidNumConfigs,
  valueMappings,
  setValueMappings,
  removeDuplicates,
  setRemoveDuplicates,
  useDuplicateSubset,
  setUseDuplicateSubset,
  duplicateSubset,
  setDuplicateSubset,
  trimAllWhitespace,
  setTrimAllWhitespace,
  hasWhitespaceVariations,
  casingConfigs,
  setCasingConfigs,
  dropEmptyColumns,
  setDropEmptyColumns,
  dropConstantColumns,
  setDropConstantColumns,
  compiledOperations,
  hasSelectedOperations,
  willDeleteRows,
  dataReady,
  cleaningError,
  downloadError,
  cleaningResult,
  isCleaning,
  isDownloading,
  cleanedPreview,
  isPreviewLoading,
  previewError,
  showApprovalModal,
  setShowApprovalModal,
  onExecuteClean,
  onDownload,
  onViewCleanedData,
  onClosePreview,
}) {
  const [customMappingColumn, setCustomMappingColumn] = useState('');
  const textColumns = columnsList.filter((column) => isTextColumn(column));
  const selectedCustomMappingColumn = textColumns.includes(customMappingColumn)
    ? customMappingColumn
    : textColumns[0] || '';
  const missingColumns = columnsList.filter((column) => columnsWithMissingValues.includes(column));
  const mappingColumns = Object.entries(valueMappings).filter(
    ([, mappings]) => Array.isArray(mappings) && mappings.length > 0
  );
  const emptyColumns = (qualityData?.empty_columns || []).map((item) =>
    typeof item === 'string' ? item : item.column
  );
  const constantColumns = (qualityData?.constant_columns || []).map((item) =>
    typeof item === 'string' ? item : item.column
  );
  const duplicateCount = qualityData?.duplicate_rows?.duplicate_count || 0;
  const invalidRangeColumns = columnsWithInfinities.filter(({ column }) => {
    const config = invalidNumConfigs[column] || {};
    return (
      config.min_value !== '' &&
      config.min_value !== undefined &&
      config.max_value !== '' &&
      config.max_value !== undefined &&
      Number(config.min_value) > Number(config.max_value)
    );
  });
  const invalidDuplicateSubset = removeDuplicates && useDuplicateSubset && duplicateSubset.length === 0;
  const actions = selectedActionItems(compiledOperations);
  const noRecommendations =
    missingColumns.length === 0 &&
    columnsWithInfinities.length === 0 &&
    mappingColumns.length === 0 &&
    duplicateCount === 0 &&
    emptyColumns.length === 0 &&
    constantColumns.length === 0;

  const updateMapping = (column, index, updates) => {
    setValueMappings((previous) => {
      const mappings = [...(previous[column] || [])];
      mappings[index] = { ...mappings[index], ...updates };
      return { ...previous, [column]: mappings };
    });
  };

  const addCustomMapping = (column) => {
    if (!column) return;
    setValueMappings((previous) => ({
      ...previous,
      [column]: [...(previous[column] || []), { from: '', to: '', approved: true, isCustom: true }],
    }));
  };

  const renderImputationOptions = (isEmpty, isIdentifier, numeric) => (
    <>
      <option value="none">Leave Unchanged</option>
      {!isEmpty && numeric && !isIdentifier && <option value="mean">Mean</option>}
      {!isEmpty && numeric && !isIdentifier && <option value="median">Median</option>}
      {!isEmpty && !isIdentifier && <option value="mode">Mode</option>}
      {!isEmpty && <option value="constant">Custom Constant</option>}
    </>
  );

  return (
    <div className="space-y-7">
      {cleaningResult && (
        <section className="border border-teal/30 bg-sage/20 px-4 py-4">
          <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-4">
            <div>
              <div className="flex items-center gap-2 text-teal-dark">
                <CheckCircle2 className="w-4 h-4" />
                <h2 className="font-serif text-lg font-semibold">Cleaned CSV ready</h2>
              </div>
              <p className="mt-1 text-xs text-navy/70">
                A separate cleaned dataset was created. Your original remains unchanged.
              </p>
            </div>
            <div className="flex flex-wrap gap-2">
              <Button
                variant="outline"
                size="sm"
                icon={Eye}
                onClick={onViewCleanedData}
                loading={isPreviewLoading}
              >
                View Cleaned Data
              </Button>
              <Button variant="primary" size="sm" icon={Download} onClick={onDownload} loading={isDownloading}>
                Download cleaned CSV
              </Button>
              {cleanedDataset && (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => switchDatasetVersion(isCleanedActive ? 'original' : 'cleaned')}
                >
                  Switch to {isCleanedActive ? 'original' : 'cleaned'}
                </Button>
              )}
            </div>
          </div>
          <div className="mt-4 grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
            <div>
              <span className="block text-navy/55">Original rows / columns</span>
              <span className="font-mono font-medium text-navy">
                {formatNumber(cleaningResult.data?.initial_shape?.rows ?? originalDataset?.row_count)} / {formatNumber(cleaningResult.data?.initial_shape?.columns ?? originalDataset?.column_count)}
              </span>
            </div>
            <div>
              <span className="block text-navy/55">Cleaned rows / columns</span>
              <span className="font-mono font-medium text-navy">
                {formatNumber(cleaningResult.data?.cleaned_shape?.rows)} / {formatNumber(cleaningResult.data?.cleaned_shape?.columns)}
              </span>
            </div>
          </div>
          {downloadError && <p className="mt-3 text-xs text-red-700">{downloadError}</p>}
          {previewError && (
            <p role="alert" className="mt-3 text-xs text-red-700">
              {previewError}
            </p>
          )}
          {cleanedPreview && (
            <section className="mt-4 border border-stone bg-white" aria-label="Cleaned dataset preview">
              <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-3 border-b border-stone px-3 py-3">
                <div>
                  <h3 className="font-serif text-base font-semibold text-navy">
                    Cleaned dataset preview
                  </h3>
                  <p className="mt-1 text-xs text-navy/60">
                    Showing {formatNumber(cleanedPreview.rows.length)} of {formatNumber(cleanedPreview.total_rows)} rows.
                  </p>
                </div>
                <Button variant="outline" size="sm" icon={X} onClick={onClosePreview}>
                  Close preview
                </Button>
              </div>
              <div className="grid grid-cols-2 gap-3 border-b border-stone px-3 py-2 text-xs">
                <span className="text-navy/60">
                  Total rows: <strong className="font-mono text-navy">{formatNumber(cleanedPreview.total_rows)}</strong>
                </span>
                <span className="text-navy/60">
                  Total columns: <strong className="font-mono text-navy">{formatNumber(cleanedPreview.total_columns)}</strong>
                </span>
              </div>
              {cleanedPreview.columns.length === 0 ? (
                <p className="px-3 py-5 text-xs text-navy/60">This cleaned dataset is empty.</p>
              ) : (
                <div className="overflow-x-auto">
                  <table className="w-full min-w-[640px] text-left text-xs text-navy">
                    <thead className="border-b border-stone bg-stone-light/50 text-navy/70">
                      <tr>
                        {cleanedPreview.columns.map((column, index) => (
                          <th key={`${column}-${index}`} className="px-3 py-2 font-medium break-words">
                            {column}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-stone">
                      {cleanedPreview.rows.length === 0 ? (
                        <tr>
                          <td colSpan={cleanedPreview.columns.length} className="px-3 py-4 text-center text-navy/55">
                            No rows to preview.
                          </td>
                        </tr>
                      ) : (
                        cleanedPreview.rows.map((row, rowIndex) => (
                          <tr key={rowIndex} className="hover:bg-stone-light/25">
                            {row.map((value, columnIndex) => (
                              <td key={columnIndex} className="max-w-xs px-3 py-2 align-top break-words">
                                {value === null || value === undefined ? '—' : String(value)}
                              </td>
                            ))}
                          </tr>
                        ))
                      )}
                    </tbody>
                  </table>
                </div>
              )}
            </section>
          )}
        </section>
      )}

      {isCleanedActive ? (
        <p className="border-y border-stone py-5 text-sm text-navy/65">
          Switch to the original dataset to prepare another cleaning plan.
        </p>
      ) : !dataReady ? (
        <p className="border-y border-stone py-5 text-sm text-navy/65">
          Loading the original dataset profile and quality findings…
        </p>
      ) : (
      <section>
        <div className="flex flex-col sm:flex-row sm:items-end sm:justify-between gap-2 mb-3">
          <div>
            <h2 className="font-serif text-xl font-semibold text-navy">Recommended Cleaning Actions</h2>
            <p className="text-xs text-navy/60 mt-1">
              Recommended choices are selected for review only. Cleaning runs only after confirmation.
            </p>
          </div>
          {hasSelectedOperations && (
            <span className="text-[11px] text-navy/60 font-mono">
              {actions.length} selected action{actions.length === 1 ? '' : 's'}
            </span>
          )}
        </div>

        {noRecommendations && (
          <p className="border-y border-stone py-5 text-sm text-navy/60">
            No cleaning recommendations were detected for this dataset.
          </p>
        )}
        <div className="border-y border-stone bg-white divide-y divide-stone">
            {missingColumns.length > 0 && (
              <div className="px-3 py-3">
                <h3 className="text-[11px] uppercase tracking-wide text-navy/55 font-mono mb-2">
                  Missing values
                </h3>
                <div className="divide-y divide-stone/70">
                  {missingColumns.map((column) => {
                    const recommendation = getRecommendation(column);
                    const isEmpty = Boolean(recommendation.isEmpty);
                    const isIdentifier = Boolean(recommendation.isIdentifier);
                    const numeric = isNumericColumn(column);
                    const config = missingConfigs[column] || { strategy: 'none', value: '' };
                    const selectedConfig = isEmpty ? { strategy: 'none', value: '' } : config;
                    return (
                      <div
                        key={column}
                        className="grid grid-cols-1 lg:grid-cols-[minmax(130px,0.7fr)_minmax(180px,1.2fr)_minmax(180px,1fr)_minmax(180px,1fr)] items-start gap-2 lg:gap-4 py-3"
                      >
                        <div>
                          <span className="block font-mono font-medium text-navy">{column}</span>
                          <span className="text-[11px] text-navy/55">
                            {formatNumber(
                              qualityData?.missing_values?.columns?.find((item) => item.column === column)?.missing_count ??
                                profileData?.missing_values?.[column]?.count
                            )} missing
                          </span>
                        </div>
                        <div>
                          <span className="block text-[10px] uppercase text-navy/50">Recommended</span>
                          <span className="text-xs font-medium text-teal-dark">
                            {recommendation.isEmpty
                              ? 'Drop Column'
                              : strategyLabel(
                                  recommendation.strategy,
                                  recommendation.defaultConstant || ''
                                )}
                          </span>
                          <p className="mt-0.5 text-[11px] text-navy/60">
                            {recommendationReason(recommendation)}
                          </p>
                        </div>
                        <label className="text-[10px] uppercase text-navy/50">
                          Current selection
                          <select
                            value={selectedConfig.strategy || 'none'}
                            onChange={(event) => {
                              const strategy = event.target.value;
                              setMissingConfigs((previous) => ({
                                ...previous,
                                [column]: { ...previous[column], strategy },
                              }));
                            }}
                            className="mt-1 block w-full px-2 py-1.5 border border-stone bg-white text-xs text-navy normal-case"
                          >
                            {renderImputationOptions(isEmpty, isIdentifier, numeric)}
                          </select>
                        </label>
                        {selectedConfig.strategy === 'constant' ? (
                          <label className="text-[10px] uppercase text-navy/50">
                            Constant value
                            <input
                              type={numeric ? 'number' : 'text'}
                              value={selectedConfig.value ?? ''}
                              onChange={(event) => {
                                const value = event.target.value;
                                setMissingConfigs((previous) => ({
                                  ...previous,
                                  [column]: {
                                    ...previous[column],
                                    value: numeric && value !== '' ? Number(value) : value,
                                  },
                                }));
                              }}
                              className="mt-1 block w-full px-2 py-1.5 border border-stone text-xs text-navy normal-case"
                            />
                          </label>
                        ) : (
                          <div className="text-[10px] uppercase text-navy/50">
                            Current strategy
                            <span className="mt-1 block text-xs font-medium text-navy normal-case">
                              {strategyLabel(selectedConfig.strategy, selectedConfig.value)}
                            </span>
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            {columnsWithInfinities.length > 0 && (
              <div className="px-3 py-3">
                <h3 className="text-[11px] uppercase tracking-wide text-navy/55 font-mono mb-2">
                  Invalid numerical values
                </h3>
                <div className="divide-y divide-stone/70">
                  {columnsWithInfinities.map((item) => {
                    const config = invalidNumConfigs[item.column] || {
                      action: 'none',
                      infinity_strategy: item.mode_available
                        ? 'mode'
                        : item.finite_value_count > 0
                          ? 'median'
                          : 'leave_unchanged',
                      min_value: '',
                      max_value: '',
                      handle_infinities: true,
                    };
                    const rangeInvalid =
                      config.min_value !== '' &&
                      config.max_value !== '' &&
                      Number(config.min_value) > Number(config.max_value);
                    return (
                      <div
                        key={item.column}
                        className="grid grid-cols-1 md:grid-cols-[minmax(150px,0.7fr)_minmax(180px,1fr)_minmax(220px,1.3fr)] items-start gap-2 md:gap-4 py-3"
                      >
                        <div>
                          <span className="block font-mono font-medium text-navy">{item.column}</span>
                          <span className="text-[11px] text-navy/55">{formatNumber(item.total)} infinite value(s)</span>
                        </div>
                        <label className="text-[10px] uppercase text-navy/50">
                          Infinity replacement strategy
                          <select
                            value={config.infinity_strategy || 'leave_unchanged'}
                            onChange={(event) =>
                              setInvalidNumConfigs((previous) => ({
                                ...previous,
                                [item.column]: {
                                  ...previous[item.column],
                                  infinity_strategy: event.target.value,
                                },
                              }))
                            }
                            className="mt-1 block w-full px-2 py-1.5 border border-stone bg-white text-xs text-navy normal-case"
                          >
                            <option value="mode" disabled={!item.mode_available}>
                              Mode{item.mode_available ? '' : ' (unavailable)'}
                            </option>
                            <option value="mean" disabled={item.finite_value_count === 0}>Mean</option>
                            <option value="median" disabled={item.finite_value_count === 0}>Median</option>
                            <option value="constant">Custom Constant</option>
                            <option value="leave_unchanged">Leave Unchanged</option>
                          </select>
                          {!item.mode_available && item.mode_unavailable_reason && (
                            <span className="mt-1 block text-[11px] normal-case text-navy/60">
                              Mode unavailable: {item.mode_unavailable_reason}
                            </span>
                          )}
                          {config.infinity_strategy === 'constant' && (
                            <input
                              type="number"
                              value={config.infinity_constant ?? ''}
                              aria-label={`Custom infinity replacement constant for ${item.column}`}
                              onChange={(event) =>
                                setInvalidNumConfigs((previous) => ({
                                  ...previous,
                                  [item.column]: {
                                    ...previous[item.column],
                                    infinity_constant: event.target.value,
                                  },
                                }))
                              }
                              className="mt-2 block w-full px-2 py-1.5 border border-stone bg-white text-xs text-navy normal-case"
                              placeholder="Enter a finite number"
                            />
                          )}
                        </label>
                        <div>
                          <label className="block text-[10px] uppercase text-navy/50">
                            Action for domain-bound violations
                            <select
                              value={config.action || 'none'}
                              onChange={(event) =>
                                setInvalidNumConfigs((previous) => ({
                                  ...previous,
                                  [item.column]: {
                                    ...previous[item.column],
                                    action: event.target.value,
                                  },
                                }))
                              }
                              className="mt-1 block w-full px-2 py-1.5 border border-stone bg-white text-xs text-navy normal-case"
                            >
                              <option value="none">Leave unchanged</option>
                              <option value="replace_with_nan">Replace with missing value</option>
                              <option value="drop_rows">Remove rows with invalid values</option>
                            </select>
                          </label>
                          {config.action && config.action !== 'none' && (
                            <div className="grid grid-cols-2 gap-2">
                              <label className="text-[10px] uppercase text-navy/50">
                                Minimum
                                <input
                                  type="number"
                                  value={config.min_value ?? ''}
                                  onChange={(event) =>
                                    setInvalidNumConfigs((previous) => ({
                                      ...previous,
                                      [item.column]: { ...previous[item.column], min_value: event.target.value },
                                    }))
                                  }
                                  className="mt-1 block w-full px-2 py-1.5 border border-stone text-xs text-navy"
                                />
                              </label>
                              <label className="text-[10px] uppercase text-navy/50">
                                Maximum
                                <input
                                  type="number"
                                  value={config.max_value ?? ''}
                                  onChange={(event) =>
                                    setInvalidNumConfigs((previous) => ({
                                      ...previous,
                                      [item.column]: { ...previous[item.column], max_value: event.target.value },
                                    }))
                                  }
                                  className="mt-1 block w-full px-2 py-1.5 border border-stone text-xs text-navy"
                                />
                              </label>
                              {rangeInvalid && (
                                <span className="col-span-2 text-[11px] text-red-700">
                                  Minimum must not exceed maximum.
                                </span>
                              )}
                            </div>
                          )}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            <div className="border-y border-stone bg-ivory/60 px-3 py-2">
              <h3 className="font-serif text-base font-semibold text-navy">Customization</h3>
              <p className="text-[11px] text-navy/60">
                Review individual category mappings and opt into optional structural or text changes.
              </p>
            </div>

            <div className="px-3 py-3">
              <div className="flex flex-wrap items-center justify-between gap-3 mb-2">
                <h3 className="text-[11px] uppercase tracking-wide text-navy/55 font-mono">
                  Categorical normalization
                </h3>
                {textColumns.length > 0 && (
                  <div className="flex items-center gap-2">
                    <select
                      aria-label="Column for custom mapping"
                      value={selectedCustomMappingColumn}
                      onChange={(event) => setCustomMappingColumn(event.target.value)}
                      className="px-2 py-1 border border-stone bg-white text-xs text-navy"
                    >
                      {textColumns.map((column) => <option key={column} value={column}>{column}</option>)}
                    </select>
                    <button
                      type="button"
                      onClick={() => addCustomMapping(selectedCustomMappingColumn)}
                      className="inline-flex items-center gap-1 text-xs font-medium text-teal-dark hover:underline"
                    >
                      <Plus className="w-3.5 h-3.5" />
                      Add Custom Mapping
                    </button>
                  </div>
                )}
              </div>
              {mappingColumns.length === 0 ? (
                <p className="py-2 text-xs text-navy/60">
                  {qualityData ? 'No inconsistent categorical values were detected.' : 'Categorical findings are not available yet.'}
                </p>
              ) : (
                <div className="space-y-4">
                  {mappingColumns.map(([column, mappings]) => (
                    <div key={column}>
                      <div className="flex items-center justify-between gap-3 mb-1.5">
                        <span className="font-mono text-xs font-medium text-navy">{column}</span>
                        <span className="text-[11px] text-navy/55 font-mono">
                          {mappings.filter((item) => item.approved).length} / {mappings.length} selected for review
                        </span>
                      </div>
                      <p className="mb-1.5 text-[11px] text-navy/55">
                        Recommended canonicalization for detected casing or whitespace variants.
                      </p>
                      <div className="divide-y divide-stone/70 border-y border-stone/70">
                        {mappings.map((mapping, index) => (
                          <div
                            key={`${column}-${index}`}
                            className="grid grid-cols-[auto_minmax(0,1fr)_auto_minmax(0,1fr)_auto] items-center gap-2 py-2"
                          >
                            <input
                              type="checkbox"
                              aria-label={`Select mapping ${mapping.from} to ${mapping.to}`}
                              checked={Boolean(mapping.approved)}
                              onChange={(event) => updateMapping(column, index, { approved: event.target.checked })}
                              className="rounded border-stone text-teal focus:ring-teal"
                            />
                            <input
                              type="text"
                              aria-label={`Original value for ${column} mapping ${index + 1}`}
                              value={mapping.from}
                              onChange={(event) => updateMapping(column, index, { from: event.target.value })}
                              className="min-w-0 px-2 py-1 border border-stone bg-white text-xs text-navy font-mono"
                            />
                            <ArrowRight className="w-3.5 h-3.5 text-navy/45" />
                            <input
                              type="text"
                              aria-label={`Canonical value for ${column} mapping ${index + 1}`}
                              value={mapping.to}
                              onChange={(event) => updateMapping(column, index, { to: event.target.value })}
                              className="min-w-0 px-2 py-1 border border-stone bg-white text-xs text-navy font-mono"
                            />
                            <button
                              type="button"
                              aria-label={`Remove ${column} mapping ${index + 1}`}
                              onClick={() =>
                                setValueMappings((previous) => ({
                                  ...previous,
                                  [column]: previous[column].filter((_, itemIndex) => itemIndex !== index),
                                }))
                              }
                              className="p-1 text-navy/40 hover:text-red-700"
                            >
                              <Trash2 className="w-3.5 h-3.5" />
                            </button>
                          </div>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {(duplicateCount > 0 || emptyColumns.length > 0 || constantColumns.length > 0) && (
              <div className="px-3 py-3">
                <h3 className="text-[11px] uppercase tracking-wide text-navy/55 font-mono mb-2">
                  Optional structural changes
                </h3>
                <div className="space-y-2">
                  {duplicateCount > 0 && (
                    <label className="flex items-start gap-2 text-xs text-navy">
                      <input
                        type="checkbox"
                        checked={removeDuplicates}
                        onChange={(event) => setRemoveDuplicates(event.target.checked)}
                        className="mt-0.5 rounded border-stone text-teal"
                      />
                      <span>
                        Remove {formatNumber(duplicateCount)} duplicate row(s)
                        <span className="block text-[11px] text-navy/55">Optional; unchecked until selected.</span>
                        <span className="block text-[11px] text-navy/55">Exact duplicate records can overweight repeated observations.</span>
                      </span>
                    </label>
                  )}
                  {removeDuplicates && (
                    <details className="ml-6 text-xs text-navy/70">
                      <summary className="cursor-pointer">Choose duplicate matching scope</summary>
                      <label className="mt-2 flex items-center gap-2">
                        <input
                          type="checkbox"
                          checked={useDuplicateSubset}
                          onChange={(event) => setUseDuplicateSubset(event.target.checked)}
                        />
                        Match only selected key columns
                      </label>
                      {useDuplicateSubset && (
                        <div className="mt-2 flex flex-wrap gap-2">
                          {columnsList.map((column) => {
                            const selected = duplicateSubset.includes(column);
                            return (
                              <button
                                key={column}
                                type="button"
                                onClick={() =>
                                  setDuplicateSubset((previous) =>
                                    selected ? previous.filter((item) => item !== column) : [...previous, column]
                                  )
                                }
                                className={`px-2 py-1 border text-[11px] font-mono ${selected ? 'bg-teal text-white border-teal' : 'bg-white border-stone text-navy'}`}
                              >
                                {column}
                              </button>
                            );
                          })}
                        </div>
                      )}
                    </details>
                  )}
                  {emptyColumns.length > 0 && (
                    <label className="flex items-start gap-2 text-xs text-navy">
                      <input
                        type="checkbox"
                        checked={dropEmptyColumns}
                        onChange={(event) => setDropEmptyColumns(event.target.checked)}
                        className="mt-0.5 rounded border-stone text-teal"
                      />
                      <span>
                        Drop completely empty columns: <span className="font-mono">{emptyColumns.join(', ')}</span>
                        <span className="block text-[11px] text-navy/55">Optional; unchecked until selected.</span>
                        <span className="block text-[11px] text-navy/55">These columns contain no observed values.</span>
                      </span>
                    </label>
                  )}
                  {constantColumns.length > 0 && (
                    <label className="flex items-start gap-2 text-xs text-navy">
                      <input
                        type="checkbox"
                        checked={dropConstantColumns}
                        onChange={(event) => setDropConstantColumns(event.target.checked)}
                        className="mt-0.5 rounded border-stone text-teal"
                      />
                      <span>
                        Drop constant columns: <span className="font-mono">{constantColumns.join(', ')}</span>
                        <span className="block text-[11px] text-navy/55">Optional; unchecked until selected.</span>
                        <span className="block text-[11px] text-navy/55">These columns contain no variation.</span>
                      </span>
                    </label>
                  )}
                </div>
              </div>
            )}

            {(hasWhitespaceVariations || textColumns.length > 0) && (
              <details className="px-3 py-3">
                <summary className="cursor-pointer text-[11px] uppercase tracking-wide text-navy/55 font-mono">
                  Optional text formatting
                </summary>
                <div className="mt-3 space-y-3">
                  {hasWhitespaceVariations && (
                    <label className="flex items-center gap-2 text-xs text-navy">
                      <input
                        type="checkbox"
                        checked={trimAllWhitespace}
                        onChange={(event) => setTrimAllWhitespace(event.target.checked)}
                        className="rounded border-stone text-teal"
                      />
                      Trim leading and trailing whitespace on text columns
                    </label>
                  )}
                  {textColumns.length > 0 && (
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                      {textColumns.map((column) => (
                        <label key={column} className="flex items-center justify-between gap-3 text-xs">
                          <span className="font-mono text-navy">{column}</span>
                          <select
                            value={casingConfigs[column] || 'none'}
                            onChange={(event) =>
                              setCasingConfigs((previous) => ({
                                ...previous,
                                [column]: event.target.value,
                              }))
                            }
                            className="px-2 py-1 border border-stone bg-white text-xs text-navy"
                          >
                            <option value="none">Keep casing</option>
                            <option value="lower">Lowercase</option>
                            <option value="upper">Uppercase</option>
                          </select>
                        </label>
                      ))}
                    </div>
                  )}
                </div>
              </details>
            )}
        </div>
      </section>
      )}

      {cleaningError && (
        <p role="alert" className="border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-800">
          Cleaning failed: {cleaningError}
        </p>
      )}
      {invalidRangeColumns.length > 0 && (
        <p role="alert" className="text-xs text-red-700">
          Minimum values must not exceed maximum values for {invalidRangeColumns.map((item) => item.column).join(', ')}.
        </p>
      )}
      {invalidDuplicateSubset && (
        <p role="alert" className="text-xs text-red-700">
          Select at least one key column or choose matching across all columns.
        </p>
      )}
      {willDeleteRows && (
        <p className="flex items-start gap-2 text-xs text-amber-800">
          <AlertTriangle className="w-4 h-4 shrink-0" />
          The selected plan may remove rows.
        </p>
      )}

      <div className="flex justify-end border-t border-stone pt-4">
        <Button
          variant="primary"
          size="lg"
          disabled={isCleanedActive || !dataReady || !hasSelectedOperations || invalidRangeColumns.length > 0 || invalidDuplicateSubset || isCleaning}
          loading={isCleaning}
          onClick={() => setShowApprovalModal(true)}
        >
          Clean CSV
        </Button>
      </div>

      {showApprovalModal && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-navy/60 p-4"
          role="presentation"
        >
          <div
            role="dialog"
            aria-modal="true"
            aria-labelledby="clean-confirmation-title"
            className="max-h-[calc(100vh-2rem)] w-full max-w-lg overflow-y-auto border border-stone bg-white p-5 shadow-xl"
          >
            <h2 id="clean-confirmation-title" className="font-serif text-xl font-semibold text-navy">
              Confirm cleaning actions
            </h2>
            <p className="mt-1 text-xs text-navy/65">
              The original dataset will remain unchanged. A separate cleaned CSV will be created.
            </p>
            <ul className="mt-4 max-h-60 space-y-2 overflow-y-auto border-y border-stone py-3 text-xs text-navy">
              {actions.map((action) => (
                <li key={action.key} className="font-mono">{action.text}</li>
              ))}
            </ul>
            {willDeleteRows && (
              <p className="mt-3 text-xs text-amber-800">
                This selection may remove rows from the cleaned dataset.
              </p>
            )}
            <div className="mt-5 flex justify-end gap-2">
              <Button variant="outline" size="sm" onClick={() => setShowApprovalModal(false)} disabled={isCleaning}>
                Go back and customize
              </Button>
              <Button variant="primary" size="sm" onClick={onExecuteClean} loading={isCleaning}>
                Confirm & Clean CSV
              </Button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
