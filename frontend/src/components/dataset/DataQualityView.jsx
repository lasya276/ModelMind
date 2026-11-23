import React, { useState } from 'react';
import {
  CheckCircle2,
  AlertTriangle,
  AlertCircle,
  Copy,
  FileQuestion,
  KeyRound,
  CaseSensitive,
  Infinity as InfinityIcon,
  HelpCircle,
  MinusCircle,
} from 'lucide-react';
import Card from '../common/Card';
import Badge from '../common/Badge';

export default function DataQualityView({ qualityData }) {
  if (!qualityData) {
    return null;
  }

  const {
    has_issues = false,
    issue_counts = {},
    missing_values = { total_affected_columns: 0, columns: [] },
    duplicate_rows = { duplicate_count: 0, percentage: 0, severity: 'low', explanation: '' },
    empty_columns = [],
    constant_columns = [],
    potential_id_columns = [],
    inconsistent_categorical_values = { total_affected_columns: 0, columns: [] },
    invalid_numerical_values = { total_affected_columns: 0, columns: [] },
  } = qualityData;

  const severityBadge = (severity) => {
    switch (severity?.toLowerCase()) {
      case 'high':
        return <Badge variant="red" size="sm">High Severity</Badge>;
      case 'medium':
        return <Badge variant="amber" size="sm">Medium</Badge>;
      case 'low':
      default:
        return <Badge variant="stone" size="sm">Informational</Badge>;
    }
  };

  const hasAnyIssues = Boolean(
    issue_counts.missing_value_columns > 0 ||
    issue_counts.duplicate_rows > 0 ||
    issue_counts.empty_columns > 0 ||
    issue_counts.constant_columns > 0 ||
    issue_counts.potential_id_columns > 0 ||
    issue_counts.inconsistent_categorical_columns > 0 ||
    issue_counts.invalid_numerical_columns > 0
  );

  return (
    <div className="space-y-6">
      {/* Integrity Summary Banner */}
      <div
        className={`p-4 rounded-sm border flex items-start gap-3.5 ${
          !hasAnyIssues
            ? 'bg-sage/40 border-sage-dark/40 text-navy'
            : 'bg-amber-50/70 border-amber-200 text-amber-950'
        }`}
      >
        {!hasAnyIssues ? (
          <CheckCircle2 className="w-5 h-5 text-teal shrink-0 mt-0.5" />
        ) : (
          <AlertTriangle className="w-5 h-5 text-amber-700 shrink-0 mt-0.5" />
        )}
        <div className="space-y-1">
          <p className="font-serif text-base font-semibold">
            {!hasAnyIssues
              ? 'No Quality Anomalies Detected'
              : 'Data Quality Observations Detected'}
          </p>
          <p className="text-xs opacity-85 leading-relaxed font-sans">
            {!hasAnyIssues
              ? 'Your dataset has clean structure with zero missing values, no duplicate rows, and no empty or constant fields.'
              : 'Review the individual checks below. Not all findings are errors—some, like identifier columns, are natural structural properties to keep in mind when modeling.'}
          </p>
        </div>
      </div>

      {/* Actual Quality Indicators Grid (No synthetic score) */}
      <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-7 gap-3 text-xs font-mono">
        <div className="p-3 bg-white border border-stone rounded-sm">
          <div className="text-[10px] text-navy/50 font-sans uppercase mb-1">Columns with Missing Values</div>
          <div className="text-lg font-semibold text-navy">
            {issue_counts.missing_value_columns ?? 0}
          </div>
        </div>

        <div className="p-3 bg-white border border-stone rounded-sm">
          <div className="text-[10px] text-navy/50 font-sans uppercase mb-1">Duplicate Rows</div>
          <div className="text-lg font-semibold text-navy">
            {issue_counts.duplicate_rows ?? 0}
          </div>
        </div>

        <div className="p-3 bg-white border border-stone rounded-sm">
          <div className="text-[10px] text-navy/50 font-sans uppercase mb-1">Empty Columns</div>
          <div className="text-lg font-semibold text-navy">
            {issue_counts.empty_columns ?? 0}
          </div>
        </div>

        <div className="p-3 bg-white border border-stone rounded-sm">
          <div className="text-[10px] text-navy/50 font-sans uppercase mb-1">Constant Columns</div>
          <div className="text-lg font-semibold text-navy">
            {issue_counts.constant_columns ?? 0}
          </div>
        </div>

        <div className="p-3 bg-white border border-stone rounded-sm">
          <div className="text-[10px] text-navy/50 font-sans uppercase mb-1">Potential IDs</div>
          <div className="text-lg font-semibold text-navy">
            {issue_counts.potential_id_columns ?? 0}
          </div>
        </div>

        <div className="p-3 bg-white border border-stone rounded-sm">
          <div className="text-[10px] text-navy/50 font-sans uppercase mb-1">Casing Variants</div>
          <div className="text-lg font-semibold text-navy">
            {issue_counts.inconsistent_categorical_columns ?? 0}
          </div>
        </div>

        <div className="p-3 bg-white border border-stone rounded-sm">
          <div className="text-[10px] text-navy/50 font-sans uppercase mb-1">Infinite Values</div>
          <div className="text-lg font-semibold text-navy">
            {issue_counts.invalid_numerical_columns ?? 0}
          </div>
        </div>
      </div>

      {/* Detailed Quality Checks */}
      <div className="space-y-4">
        {/* 1. Missing Values Finding */}
        <Card
          title="Missing Values"
          subtitle="Gaps in your data where records have empty or null entries."
          headerAction={
            missing_values.total_affected_columns > 0 ? (
              <Badge variant="amber">{missing_values.total_affected_columns} Column(s) Affected</Badge>
            ) : (
              <Badge variant="sage">Complete (0 Missing)</Badge>
            )
          }
        >
          {missing_values.columns && missing_values.columns.length > 0 ? (
            <div className="space-y-3">
              <p className="text-xs text-navy/70 font-sans leading-relaxed">
                Why this matters: Missing values can cause algorithms to fail or produce skewed results if not handled via imputation or filtering.
              </p>
              <div className="divide-y divide-stone border border-stone rounded-sm overflow-hidden text-xs">
                {missing_values.columns.map((col) => (
                  <div key={col.column} className="p-3 flex items-center justify-between gap-4 bg-white hover:bg-ivory/40">
                    <div>
                      <span className="font-mono font-semibold text-navy">{col.column}</span>
                      <p className="text-[11px] text-navy/60 font-sans mt-0.5">{col.explanation}</p>
                    </div>
                    <div className="flex items-center gap-3 shrink-0 font-mono">
                      <span>{col.missing_count} missing ({col.percentage}%)</span>
                      {severityBadge(col.severity)}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <p className="text-xs text-navy/60 font-sans">
              All records have complete data. No missing values were found in any column.
            </p>
          )}
        </Card>

        {/* 2. Duplicate Rows Finding */}
        <Card
          title="Duplicate Rows"
          subtitle="Identical rows where every single column value matches an existing record."
          headerAction={
            duplicate_rows.duplicate_count > 0 ? (
              <Badge variant="amber">{duplicate_rows.duplicate_count} Duplicate(s)</Badge>
            ) : (
              <Badge variant="sage">0 Duplicates</Badge>
            )
          }
        >
          {duplicate_rows.duplicate_count > 0 ? (
            <div className="space-y-2 text-xs">
              <p className="text-navy/75 font-sans leading-relaxed">{duplicate_rows.explanation}</p>
              <p className="text-navy/60 font-sans text-[11px]">
                Why this matters: Duplicate records can artificially inflate sample sizes and lead models to over-emphasize repeated data points.
              </p>
              <div className="flex items-center gap-2 pt-1">
                {severityBadge(duplicate_rows.severity)}
                <span className="font-mono text-navy/70">
                  Impact: {duplicate_rows.percentage}% of dataset rows
                </span>
              </div>
            </div>
          ) : (
            <p className="text-xs text-navy/60 font-sans">
              Every row in this dataset is distinct. No exact duplicate rows were detected.
            </p>
          )}
        </Card>

        {/* 3. Empty & Constant Columns */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <Card
            title="Empty Columns"
            subtitle="Columns that contain 100% missing values."
            headerAction={
              empty_columns.length > 0 ? (
                <Badge variant="red">{empty_columns.length} Empty</Badge>
              ) : (
                <Badge variant="sage">0 Empty</Badge>
              )
            }
          >
            {empty_columns.length > 0 ? (
              <div className="space-y-2 text-xs">
                <p className="text-navy/70 font-sans">
                  Why this matters: Empty columns provide zero signal for analysis and should usually be dropped before training models.
                </p>
                <div className="space-y-1.5 font-mono">
                  {empty_columns.map((c) => (
                    <div key={c.column} className="p-2.5 bg-stone-light/30 border border-stone rounded flex justify-between items-center">
                      <span className="font-semibold text-navy">{c.column}</span>
                      {severityBadge(c.severity)}
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <p className="text-xs text-navy/60 font-sans">
                No completely empty columns. Every column contains non-null information.
              </p>
            )}
          </Card>

          <Card
            title="Constant Columns"
            subtitle="Columns where every row has the exact same value."
            headerAction={
              constant_columns.length > 0 ? (
                <Badge variant="amber">{constant_columns.length} Constant</Badge>
              ) : (
                <Badge variant="sage">0 Constant</Badge>
              )
            }
          >
            {constant_columns.length > 0 ? (
              <div className="space-y-2 text-xs">
                <p className="text-navy/70 font-sans">
                  Why this matters: Columns with zero variance cannot differentiate between rows and add no predictive power.
                </p>
                <div className="space-y-1.5 font-mono">
                  {constant_columns.map((c) => (
                    <div key={c.column} className="p-2.5 bg-stone-light/30 border border-stone rounded flex justify-between items-center">
                      <div>
                        <span className="font-semibold text-navy">{c.column}</span>
                        <span className="text-navy/50 text-[11px] block">
                          Value: "{String(c.constant_value ?? c.unique_value)}"
                        </span>
                      </div>
                      {severityBadge(c.severity)}
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <p className="text-xs text-navy/60 font-sans">
                No constant columns found. All columns have variation in their values.
              </p>
            )}
          </Card>
        </div>

        {/* 4. Potential Identifiers & Casing Discrepancies */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <Card
            title="Potential Identifier Columns"
            subtitle="Columns with near 100% uniqueness or ID naming patterns."
            headerAction={
              potential_id_columns.length > 0 ? (
                <Badge variant="stone">{potential_id_columns.length} Candidate(s)</Badge>
              ) : (
                <Badge variant="sage">0 IDs Flagged</Badge>
              )
            }
          >
            {potential_id_columns.length > 0 ? (
              <div className="space-y-2 text-xs">
                <p className="text-navy/70 font-sans">
                  Why this matters: Identifiers (like customer IDs or account numbers) are natural keys, but using them as predictive features in machine learning can cause overfitting.
                </p>
                <div className="space-y-1.5 font-mono">
                  {potential_id_columns.map((idCol) => (
                    <div key={idCol.column} className="p-2.5 bg-stone-light/30 border border-stone rounded flex justify-between items-center">
                      <div>
                        <span className="font-semibold text-navy">{idCol.column}</span>
                        <span className="text-navy/50 text-[11px] block">
                          {idCol.uniqueness_percentage}% unique values
                        </span>
                      </div>
                      <Badge variant="stone" size="sm">ID Heuristic</Badge>
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <p className="text-xs text-navy/60 font-sans">
                No high-uniqueness identifier columns detected.
              </p>
            )}
          </Card>

          <Card
            title="Inconsistent Categorical Text"
            subtitle="Categories with casing or spacing differences (e.g., 'Yes' vs 'yes')."
            headerAction={
              inconsistent_categorical_values.total_affected_columns > 0 ? (
                <Badge variant="amber">
                  {inconsistent_categorical_values.total_affected_columns} Column(s)
                </Badge>
              ) : (
                <Badge variant="sage">Consistent</Badge>
              )
            }
          >
            {inconsistent_categorical_values.columns && inconsistent_categorical_values.columns.length > 0 ? (
              <div className="space-y-2 text-xs">
                <p className="text-navy/70 font-sans">
                  Why this matters: Computers treat 'Active' and 'active' as two completely separate categories unless text is normalized.
                </p>
                <div className="space-y-2 font-mono">
                  {inconsistent_categorical_values.columns.map((col) => (
                    <div key={col.column} className="p-2.5 bg-stone-light/30 border border-stone rounded space-y-1.5">
                      <div className="flex justify-between items-center">
                        <span className="font-semibold text-navy">{col.column}</span>
                        {severityBadge(col.severity)}
                      </div>
                      {Array.isArray(col.discrepancies) && (
                        <div className="text-[11px] text-navy/75 space-y-1">
                          {col.discrepancies.map((d, i) => (
                            <div key={i} className="bg-white px-2 py-1 rounded border border-stone/60">
                              <span className="text-navy/50">Variants:</span>{' '}
                              <span className="text-teal font-semibold">
                                {d.variants ? d.variants.map((v) => `"${v}"`).join(', ') : ''}
                              </span>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <p className="text-xs text-navy/60 font-sans">
                Categorical values show consistent casing and spacing.
              </p>
            )}
          </Card>
        </div>

        {/* 5. Invalid Numerical Values */}
        {invalid_numerical_values.total_affected_columns > 0 && (
          <Card
            title="Invalid Numerical Values"
            subtitle="Infinite numbers (+inf / -inf) that can break calculations."
            headerAction={<Badge variant="red">{invalid_numerical_values.total_affected_columns} Column(s)</Badge>}
          >
            <div className="space-y-2 text-xs">
              <p className="text-navy/70 font-sans">
                Why this matters: Infinity values often result from division by zero and cannot be handled by machine learning estimators directly.
              </p>
              <div className="space-y-1.5 font-mono">
                {invalid_numerical_values.columns.map((c) => (
                  <div key={c.column} className="p-2.5 bg-stone-light/30 border border-stone rounded flex justify-between items-center">
                    <div>
                      <span className="font-semibold text-navy">{c.column}</span>
                      <span className="text-navy/50 text-[11px] block">{c.explanation}</span>
                    </div>
                    {severityBadge(c.severity)}
                  </div>
                ))}
              </div>
            </div>
          </Card>
        )}
      </div>
    </div>
  );
}
