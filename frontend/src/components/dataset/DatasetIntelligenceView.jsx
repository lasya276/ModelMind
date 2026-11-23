import React from 'react';
import {
  Sparkles,
  Target,
  BarChart2,
  AlertTriangle,
  Network,
  TrendingUp,
  CheckCircle2,
  Info,
  ShieldCheck,
  ChevronRight,
} from 'lucide-react';
import Card from '../common/Card';
import Badge from '../common/Badge';

export default function DatasetIntelligenceView({ intelligenceData }) {
  if (!intelligenceData) {
    return null;
  }

  const {
    total_rows = 0,
    total_columns = 0,
    role_counts = {},
    column_roles = {},
    possible_targets = [],
    recommended_analyses = [],
  } = intelligenceData;

  const getRoleBadgeVariant = (role) => {
    switch (role) {
      case 'numeric':
        return 'teal';
      case 'categorical':
        return 'stone';
      case 'boolean':
        return 'sage';
      case 'potential_id':
        return 'amber';
      case 'datetime':
        return 'teal';
      case 'empty':
        return 'red';
      default:
        return 'stone';
    }
  };

  const getAnalysisIcon = (type) => {
    switch (type) {
      case 'eda':
        return BarChart2;
      case 'anomaly_detection':
        return AlertTriangle;
      case 'classification':
        return Target;
      case 'regression':
        return TrendingUp;
      case 'clustering':
        return Network;
      default:
        return Sparkles;
    }
  };

  return (
    <div className="space-y-6">
      {/* Editorial Disclaimer on Target Selection */}
      <div className="p-4 bg-sage/40 border border-sage-dark/40 rounded-sm text-xs text-navy flex items-start gap-3">
        <Info className="w-4 h-4 text-teal mt-0.5 shrink-0" />
        <div className="space-y-1 font-sans">
          <p className="font-semibold text-navy">
            Important Guidance on Machine Learning Targets
          </p>
          <p className="text-navy/80 leading-relaxed">
            Candidate target columns shown below are rule-based suggestions, not decisions.
            ModelMind never automatically assigns a target column or forces supervised learning.
            If your dataset has no natural outcome to predict, unsupervised analyses—such as
            exploratory data analysis, anomaly detection, or clustering—are often the most insightful paths.
          </p>
        </div>
      </div>

      {/* Role Distribution Badges */}
      <Card
        title="Detected Column Roles"
        subtitle="Identified through data types, uniqueness rates, and heuristic naming patterns."
      >
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3 text-xs font-mono mb-6">
          <div className="p-3 bg-stone-light/30 border border-stone rounded-sm">
            <span className="text-navy/50 font-sans block text-[10px] uppercase">Numeric</span>
            <span className="text-lg font-semibold text-teal">{role_counts.numeric ?? 0}</span>
          </div>

          <div className="p-3 bg-stone-light/30 border border-stone rounded-sm">
            <span className="text-navy/50 font-sans block text-[10px] uppercase">Categorical</span>
            <span className="text-lg font-semibold text-navy">{role_counts.categorical ?? 0}</span>
          </div>

          <div className="p-3 bg-stone-light/30 border border-stone rounded-sm">
            <span className="text-navy/50 font-sans block text-[10px] uppercase">Boolean</span>
            <span className="text-lg font-semibold text-teal-dark">{role_counts.boolean ?? 0}</span>
          </div>

          <div className="p-3 bg-stone-light/30 border border-stone rounded-sm">
            <span className="text-navy/50 font-sans block text-[10px] uppercase">Potential ID</span>
            <span className="text-lg font-semibold text-amber-800">{role_counts.potential_id ?? 0}</span>
          </div>

          <div className="p-3 bg-stone-light/30 border border-stone rounded-sm">
            <span className="text-navy/50 font-sans block text-[10px] uppercase">Datetime</span>
            <span className="text-lg font-semibold text-navy">{role_counts.datetime ?? 0}</span>
          </div>

          <div className="p-3 bg-stone-light/30 border border-stone rounded-sm">
            <span className="text-navy/50 font-sans block text-[10px] uppercase">Empty</span>
            <span className="text-lg font-semibold text-red-700">{role_counts.empty ?? 0}</span>
          </div>
        </div>

        {/* Column Role Table */}
        <div className="overflow-x-auto border border-stone rounded-sm">
          <table className="w-full text-left text-xs">
            <thead className="bg-stone-light/40 border-b border-stone font-mono text-[11px] text-navy/60 uppercase">
              <tr>
                <th className="py-2.5 px-4 font-semibold">Column Name</th>
                <th className="py-2.5 px-4 font-semibold">Detected Role</th>
                <th className="py-2.5 px-4 font-semibold">Uniqueness</th>
                <th className="py-2.5 px-4 font-semibold">Heuristic Reason</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-stone font-sans">
              {Object.entries(column_roles).map(([colName, info]) => (
                <tr key={colName} className="hover:bg-ivory/40">
                  <td className="py-2.5 px-4 font-mono font-medium text-navy">{colName}</td>
                  <td className="py-2.5 px-4">
                    <Badge variant={getRoleBadgeVariant(info.role)} size="sm">
                      {info.role}
                    </Badge>
                  </td>
                  <td className="py-2.5 px-4 font-mono text-navy/70">
                    {info.uniqueness_percentage}% ({info.unique_count} distinct)
                  </td>
                  <td className="py-2.5 px-4 text-navy/75 text-[11px] leading-relaxed">
                    {info.reason}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      {/* Suggested Target Columns Section */}
      <Card
        title="Candidate Target Columns (Suggestions Only)"
        subtitle="Columns that could potentially be predicted based on category or numeric distributions."
        headerAction={
          possible_targets.length > 0 ? (
            <Badge variant="teal">{possible_targets.length} Candidate(s)</Badge>
          ) : (
            <Badge variant="stone">No Obvious Target</Badge>
          )
        }
      >
        {possible_targets.length > 0 ? (
          <div className="space-y-3">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {possible_targets.map((tgt) => (
                <div
                  key={tgt.column}
                  className="p-3.5 bg-stone-light/20 border border-stone rounded-sm space-y-2 hover:bg-stone-light/40 transition-colors"
                >
                  <div className="flex items-start justify-between gap-2">
                    <div>
                      <span className="font-mono text-sm font-semibold text-navy">
                        {tgt.column}
                      </span>
                      <span className="text-[11px] text-navy/50 block font-mono">
                        {tgt.unique_values_count} distinct values • {tgt.target_type}
                      </span>
                    </div>
                    <div className="flex items-center gap-1.5 shrink-0">
                      <Badge variant={tgt.suggested_task === 'classification' ? 'teal' : 'navy'} size="sm">
                        {tgt.suggested_task === 'classification' ? 'Classification' : 'Regression'}
                      </Badge>
                      <Badge variant="stone" size="sm">
                        {tgt.confidence} confidence
                      </Badge>
                    </div>
                  </div>
                  <p className="text-xs text-navy/70 font-sans leading-relaxed">
                    {tgt.reason}
                  </p>
                </div>
              ))}
            </div>
          </div>
        ) : (
          <div className="p-4 bg-stone-light/30 border border-stone rounded-sm text-xs text-navy/75 space-y-1 font-sans">
            <p className="font-semibold text-navy">No clear target column detected</p>
            <p>
              This dataset does not have an obvious outcome column with suitable categories or bounded values.
              That is completely normal! Unsupervised workflows—such as exploratory charts, outlier detection,
              or clustering—are designed specifically for datasets without a target column.
            </p>
          </div>
        )}
      </Card>

      {/* Recommended Analysis Options */}
      <div>
        <h3 className="font-serif text-lg font-bold text-navy mb-1">
          Recommended Analytical Paths
        </h3>
        <p className="text-xs text-navy/60 font-sans mb-4">
          Rule-based analysis suggestions based on your dataset's shape, features, and candidate targets.
        </p>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {recommended_analyses.map((rec) => {
            const Icon = getAnalysisIcon(rec.analysis_type);
            const isRec = rec.is_recommended;

            return (
              <Card
                key={rec.analysis_type}
                className={`flex flex-col justify-between ${
                  !isRec ? 'opacity-70 bg-stone-light/10' : ''
                }`}
                title={rec.name}
                headerAction={
                  <Badge variant={isRec ? 'sage' : 'stone'} size="sm">
                    {isRec ? 'Recommended' : 'Not Recommended'}
                  </Badge>
                }
              >
                <div className="space-y-3 text-xs font-sans">
                  <div className="flex items-start gap-2.5">
                    <div className={`p-2 rounded shrink-0 ${isRec ? 'bg-sage text-teal-dark' : 'bg-stone text-navy/40'}`}>
                      <Icon className="w-4 h-4" />
                    </div>
                    <p className="text-navy/75 leading-relaxed font-sans">{rec.description}</p>
                  </div>

                  <div className="space-y-1.5 pt-2 border-t border-stone/70 text-[11px]">
                    <div>
                      <span className="font-semibold text-navy block">Suitability Reason:</span>
                      <span className="text-navy/70 leading-relaxed">{rec.suitability_reason}</span>
                    </div>

                    <div>
                      <span className="font-semibold text-navy block">Prerequisites:</span>
                      <span className="text-navy/60 leading-relaxed">{rec.prerequisites_and_limitations}</span>
                    </div>

                    {rec.requires_target && (
                      <div className="pt-1">
                        <span className="text-[10px] uppercase font-mono text-amber-800 bg-amber-50 px-2 py-0.5 rounded border border-amber-200">
                          Requires Target Selection
                        </span>
                      </div>
                    )}
                  </div>
                </div>
              </Card>
            );
          })}
        </div>
      </div>
    </div>
  );
}
