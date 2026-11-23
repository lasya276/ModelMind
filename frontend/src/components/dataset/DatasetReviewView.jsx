import React from 'react';
import Badge from '../common/Badge';
import { formatNumber } from '../../utils/formatters';

function severityBadge(severity) {
  const value = String(severity || 'info').toLowerCase();
  const variant = value === 'high' ? 'red' : value === 'medium' ? 'amber' : 'stone';
  const label = value === 'info' || value === 'informational' ? 'Informational' : value;
  return <Badge variant={variant} size="sm">{label}</Badge>;
}

function columnsLabel(columns) {
  return columns.length > 0 ? columns.join(', ') : 'All columns';
}

function getIssueSummaries(qualityData) {
  if (!qualityData) return [];

  const missingColumns = qualityData.missing_values?.columns || [];
  const duplicateRows = qualityData.duplicate_rows || {};
  const emptyColumns = qualityData.empty_columns || [];
  const constantColumns = qualityData.constant_columns || [];
  const identifierColumns = qualityData.potential_id_columns || [];
  const categoricalColumns = qualityData.inconsistent_categorical_values?.columns || [];
  const invalidColumns = qualityData.invalid_numerical_values?.columns || [];
  const missingCount = missingColumns.reduce((sum, item) => sum + (item.missing_count || 0), 0);
  const variationCount = categoricalColumns.reduce(
    (sum, item) =>
      sum +
      (item.inconsistencies || []).reduce(
        (columnSum, inconsistency) =>
          columnSum + (inconsistency.variation_count || inconsistency.variations?.length || 0),
        0
      ),
    0
  );
  const invalidCount = invalidColumns.reduce(
    (sum, item) => sum + (item.total_invalid_count || 0),
    0
  );

  return [
    missingCount > 0 && {
      name: 'Missing values',
      count: `${formatNumber(missingCount)} values`,
      columns: missingColumns.map((item) => item.column),
      explanation: 'Some records have empty entries in these columns.',
      severity: missingColumns.some((item) => item.severity === 'high')
        ? 'high'
        : missingColumns.some((item) => item.severity === 'medium')
          ? 'medium'
          : 'low',
    },
    duplicateRows.duplicate_count > 0 && {
      name: 'Duplicate rows',
      count: `${formatNumber(duplicateRows.duplicate_count)} rows`,
      columns: [],
      explanation: duplicateRows.explanation || 'Identical rows were detected.',
      severity: duplicateRows.severity,
    },
    emptyColumns.length > 0 && {
      name: 'Completely empty columns',
      count: `${formatNumber(emptyColumns.length)} columns`,
      columns: emptyColumns.map((item) => item.column),
      explanation: 'These columns contain no non-missing values.',
      severity: 'high',
    },
    constantColumns.length > 0 && {
      name: 'Constant columns',
      count: `${formatNumber(constantColumns.length)} columns`,
      columns: constantColumns.map((item) => item.column),
      explanation: 'These columns contain only one distinct value.',
      severity: 'medium',
    },
    identifierColumns.length > 0 && {
      name: 'Potential identifier columns',
      count: `${formatNumber(identifierColumns.length)} columns`,
      columns: identifierColumns.map((item) => item.column),
      explanation: 'These columns may identify records rather than provide predictive information.',
      severity: 'info',
    },
    categoricalColumns.length > 0 && {
      name: 'Inconsistent categorical values',
      count: `${formatNumber(variationCount)} variations`,
      columns: categoricalColumns.map((item) => item.column),
      explanation: 'Some categories differ in casing or surrounding whitespace.',
      severity: categoricalColumns.some((item) => item.severity === 'high') ? 'high' : 'medium',
    },
    invalidCount > 0 && {
      name: 'Invalid numerical values',
      count: `${formatNumber(invalidCount)} values`,
      columns: invalidColumns.map((item) => item.column),
      explanation: 'Infinite values cannot be handled by many numerical calculations.',
      severity: 'high',
    },
  ].filter(Boolean);
}

export default function DatasetReviewView({ profileData, qualityData, intelligenceData }) {
  const columns = profileData?.columns || [];
  const missingValues = profileData?.missing_values || {};
  const missingCount = Object.values(missingValues).reduce(
    (sum, item) => sum + (item?.count || 0),
    0
  );
  const qualityMissingCount = (qualityData?.missing_values?.columns || []).reduce(
    (sum, item) => sum + (item.missing_count || 0),
    0
  );
  const totalMissingCount = profileData ? missingCount : qualityMissingCount;
  const totalRows = profileData?.total_rows ?? qualityData?.total_rows;
  const totalColumns = profileData?.total_columns ?? qualityData?.total_columns ?? columns.length;
  const duplicateCount =
    qualityData?.duplicate_rows?.duplicate_count ?? profileData?.duplicate_rows ?? 0;
  const issues = getIssueSummaries(qualityData);
  const potentialIds = new Set(
    (qualityData?.potential_id_columns || []).map((item) => item.column)
  );

  const getRole = (column, dtype) => {
    const detectedRole = intelligenceData?.column_roles?.[column]?.role;
    if (detectedRole) return detectedRole.replace(/_/g, ' ');
    if (potentialIds.has(column)) return 'Potential identifier';
    const normalizedType = String(dtype || '').toLowerCase();
    if (normalizedType.includes('bool')) return 'Boolean';
    if (normalizedType.includes('date') || normalizedType.includes('time')) return 'Datetime';
    if (
      normalizedType.includes('int') ||
      normalizedType.includes('float') ||
      normalizedType.includes('double') ||
      normalizedType.includes('number')
    ) {
      return 'Numeric';
    }
    return dtype ? 'Categorical' : '—';
  };

  return (
    <div className="space-y-6 text-xs leading-5">
      <section>
        <h2 className="px-3 font-serif text-lg leading-6 font-semibold text-navy mb-3">Dataset Overview</h2>
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 border-y border-stone divide-x divide-y sm:divide-y-0 divide-stone bg-white">
          {[
            ['Total rows', totalRows],
            ['Total columns', totalColumns],
            ['Missing values', totalMissingCount],
            ['Duplicate rows', duplicateCount],
            ['Quality issue types', qualityData ? issues.length : '—'],
          ].map(([label, value]) => (
            <div key={label} className="px-3 py-3">
              <span className="block text-xs leading-5 uppercase tracking-wide text-navy/55">
                {label}
              </span>
              <span className="block mt-1 text-lg leading-6 font-semibold text-navy font-mono">
                {formatNumber(value)}
              </span>
            </div>
          ))}
        </div>
      </section>

      <section>
        <div className="flex items-baseline justify-between gap-3 mb-3">
          <h2 className="px-3 font-serif text-lg leading-6 font-semibold text-navy">Column Overview</h2>
          <span className="px-3 text-xs leading-5 text-navy/55">{formatNumber(columns.length)} columns</span>
        </div>
        <div className="overflow-x-auto border-y border-stone bg-white">
          <table className="w-full min-w-[640px] text-left text-xs leading-5 text-navy">
            <thead className="bg-stone-light/50 border-b border-stone text-xs uppercase tracking-wide text-navy/60">
              <tr>
                <th className="px-3 py-2.5 font-medium">Column</th>
                <th className="px-3 py-2.5 font-medium">Data type</th>
                <th className="px-3 py-2.5 font-medium">Missing</th>
                <th className="px-3 py-2.5 font-medium">Unique</th>
                <th className="px-3 py-2.5 font-medium">Detected role</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-stone font-sans">
              {columns.map((column) => {
                const dtype = profileData?.column_types?.[column] || 'Unknown';
                return (
                  <tr key={column} className="hover:bg-stone-light/25">
                    <td className="px-3 py-2.5 font-mono font-medium break-words">{column}</td>
                    <td className="px-3 py-2.5">{dtype}</td>
                    <td className="px-3 py-2.5 font-mono">
                      {formatNumber(missingValues[column]?.count || 0)}
                    </td>
                    <td className="px-3 py-2.5 font-mono">
                      {formatNumber(profileData?.unique_values?.[column])}
                    </td>
                    <td className="px-3 py-2.5 capitalize">{getRole(column, dtype)}</td>
                  </tr>
                );
              })}
              {columns.length === 0 && (
                <tr>
                  <td colSpan="5" className="px-3 py-6 text-center text-navy/55">
                    Column details are not available yet.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </section>

      <section>
        <div className="flex items-baseline justify-between gap-3 mb-3">
          <h2 className="px-3 font-serif text-lg leading-6 font-semibold text-navy">Data Quality Issues</h2>
          {qualityData && (
            <span className="px-3 text-xs leading-5 text-navy/55">
              {issues.length} issue type{issues.length === 1 ? '' : 's'}
            </span>
          )}
        </div>
        {!qualityData ? (
          <p className="border-y border-stone px-3 py-5 text-xs leading-5 text-navy/60">
            Quality findings are not available yet.
          </p>
        ) : issues.length === 0 ? (
          <p className="border-y border-stone px-3 py-5 text-xs leading-5 text-navy/60">
            No quality issues were detected.
          </p>
        ) : (
          <div className="border-y border-stone bg-white divide-y divide-stone">
            {issues.map((issue) => (
              <div
                key={issue.name}
                className="grid min-w-0 grid-cols-1 md:grid-cols-[minmax(0,0.8fr)_minmax(0,0.4fr)_minmax(0,1fr)_minmax(0,1.5fr)_auto] items-start gap-2 md:gap-4 px-3 py-3 text-xs leading-5"
              >
                <span className="min-w-0 font-medium text-navy break-words">{issue.name}</span>
                <span className="font-mono text-navy/75">{issue.count}</span>
                <span className="min-w-0 font-mono text-navy/70 break-words">
                  {columnsLabel(issue.columns)}
                </span>
                <span className="min-w-0 text-navy/65 break-words">{issue.explanation}</span>
                {severityBadge(issue.severity)}
              </div>
            ))}
          </div>
        )}
      </section>
    </div>
  );
}
