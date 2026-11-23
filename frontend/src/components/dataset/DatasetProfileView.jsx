import React, { useState, useMemo } from 'react';
import { Table, Search, Hash, Tag, AlertTriangle, Layers, ArrowUpDown } from 'lucide-react';
import Card from '../common/Card';
import Badge from '../common/Badge';
import { formatNumber } from '../../utils/formatters';

export default function DatasetProfileView({ profileData }) {
  const [searchTerm, setSearchTerm] = useState('');
  const [typeFilter, setTypeFilter] = useState('all');

  if (!profileData) {
    return null;
  }

  const {
    total_rows = 0,
    total_columns = 0,
    columns = [],
    column_types = {},
    missing_values = {},
    unique_values = {},
    duplicate_rows = 0,
    numeric_summary = {},
    categorical_summary = {},
  } = profileData;

  // Filter columns by search term and type
  const filteredColumns = useMemo(() => {
    return columns.filter((col) => {
      const matchesSearch = col.toLowerCase().includes(searchTerm.toLowerCase().trim());
      if (!matchesSearch) return false;

      if (typeFilter === 'all') return true;
      const isNumeric = col in numeric_summary;
      if (typeFilter === 'numeric') return isNumeric;
      if (typeFilter === 'categorical') return !isNumeric;
      return true;
    });
  }, [columns, searchTerm, typeFilter, numeric_summary]);

  // Aggregate missing values count across all columns
  const totalMissingCells = useMemo(() => {
    return Object.values(missing_values).reduce((acc, curr) => acc + (curr?.count || 0), 0);
  }, [missing_values]);

  const numericCount = Object.keys(numeric_summary).length;
  const categoricalCount = Object.keys(categorical_summary).length;

  return (
    <div className="space-y-6">
      {/* High-Level Dimension Cards */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 font-mono text-xs">
        <div className="p-4 bg-white border border-stone rounded-sm shadow-xs">
          <div className="text-[11px] text-navy/50 font-sans uppercase mb-1">Total Rows</div>
          <div className="text-xl font-semibold text-navy">{formatNumber(total_rows)}</div>
          <div className="text-[11px] text-navy/60 font-sans mt-0.5">Records in dataset</div>
        </div>

        <div className="p-4 bg-white border border-stone rounded-sm shadow-xs">
          <div className="text-[11px] text-navy/50 font-sans uppercase mb-1">Total Columns</div>
          <div className="text-xl font-semibold text-navy">{formatNumber(total_columns)}</div>
          <div className="text-[11px] text-navy/60 font-sans mt-0.5">
            {numericCount} numeric • {categoricalCount} text/cat
          </div>
        </div>

        <div className="p-4 bg-white border border-stone rounded-sm shadow-xs">
          <div className="text-[11px] text-navy/50 font-sans uppercase mb-1">Duplicate Rows</div>
          <div className="text-xl font-semibold text-navy">
            {formatNumber(duplicate_rows)}
          </div>
          <div className="text-[11px] text-navy/60 font-sans mt-0.5">
            {total_rows > 0 ? `${((duplicate_rows / total_rows) * 100).toFixed(1)}% of total` : '0%'}
          </div>
        </div>

        <div className="p-4 bg-white border border-stone rounded-sm shadow-xs">
          <div className="text-[11px] text-navy/50 font-sans uppercase mb-1">Missing Cells</div>
          <div className="text-xl font-semibold text-navy">{formatNumber(totalMissingCells)}</div>
          <div className="text-[11px] text-navy/60 font-sans mt-0.5">
            {total_rows * total_columns > 0
              ? `${((totalMissingCells / (total_rows * total_columns)) * 100).toFixed(1)}% of cells`
              : '0%'}
          </div>
        </div>
      </div>

      {/* Column Profiles Table Card */}
      <Card
        title="Column Profiles"
        subtitle="Detailed breakdown of detected types, missingness, uniqueness, and summary statistics."
        headerAction={
          <div className="flex flex-wrap items-center gap-2">
            {/* Search Filter */}
            <div className="relative">
              <Search className="w-3.5 h-3.5 text-navy/40 absolute left-2.5 top-1/2 -translate-y-1/2" />
              <input
                type="text"
                placeholder="Search columns..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="pl-8 pr-3 py-1 bg-stone-light/40 border border-stone rounded text-xs text-navy placeholder:text-navy/40 focus:outline-none focus:ring-1 focus:ring-teal"
              />
            </div>

            {/* Type Filter */}
            <select
              value={typeFilter}
              onChange={(e) => setTypeFilter(e.target.value)}
              className="bg-stone-light/40 border border-stone rounded px-2.5 py-1 text-xs text-navy focus:outline-none focus:ring-1 focus:ring-teal"
            >
              <option value="all">All Types ({columns.length})</option>
              <option value="numeric">Numeric ({numericCount})</option>
              <option value="categorical">Text / Categorical ({categoricalCount})</option>
            </select>
          </div>
        }
        noPadding
      >
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs text-navy">
            <thead className="bg-stone-light/40 border-b border-stone font-mono text-[11px] text-navy/60 uppercase">
              <tr>
                <th className="py-3 px-4 font-semibold">Column Name</th>
                <th className="py-3 px-4 font-semibold">Data Type</th>
                <th className="py-3 px-4 font-semibold">Missing Values</th>
                <th className="py-3 px-4 font-semibold">Unique Values</th>
                <th className="py-3 px-4 font-semibold">Summary Statistics</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-stone font-sans">
              {filteredColumns.length === 0 ? (
                <tr>
                  <td colSpan="5" className="py-8 text-center text-xs text-navy/60">
                    No columns match your filter criteria.
                  </td>
                </tr>
              ) : (
                filteredColumns.map((col) => {
                  const dtype = column_types[col] || 'unknown';
                  const missing = missing_values[col] || { count: 0, percentage: 0 };
                  const uniqueCount = unique_values[col] ?? 0;
                  const isNumeric = col in numeric_summary;
                  const numStats = numeric_summary[col];
                  const catStats = categorical_summary[col];

                  return (
                    <tr key={col} className="hover:bg-ivory/50 transition-colors">
                      {/* Name */}
                      <td className="py-3 px-4 font-medium font-mono text-navy">
                        <div className="flex items-center gap-2">
                          <span className="truncate max-w-[180px]" title={col}>
                            {col}
                          </span>
                        </div>
                      </td>

                      {/* Data Type */}
                      <td className="py-3 px-4">
                        <Badge variant={isNumeric ? 'teal' : 'stone'} size="sm">
                          {dtype}
                        </Badge>
                      </td>

                      {/* Missing Values */}
                      <td className="py-3 px-4 font-mono">
                        {missing.count > 0 ? (
                          <div className="space-y-1">
                            <span className="text-amber-800 font-medium">
                              {missing.count.toLocaleString()} ({missing.percentage}%)
                            </span>
                            <div className="w-20 h-1 bg-stone rounded-full overflow-hidden">
                              <div
                                className="h-full bg-amber-500 rounded-full"
                                style={{ width: `${Math.min(missing.percentage, 100)}%` }}
                              />
                            </div>
                          </div>
                        ) : (
                          <span className="text-navy/50">0 (0%)</span>
                        )}
                      </td>

                      {/* Unique Values */}
                      <td className="py-3 px-4 font-mono">
                        <span>{uniqueCount.toLocaleString()}</span>
                        {total_rows > 0 && (
                          <span className="text-navy/50 text-[11px] block">
                            {((uniqueCount / total_rows) * 100).toFixed(1)}% unique
                          </span>
                        )}
                      </td>

                      {/* Summary Stats */}
                      <td className="py-3 px-4 font-mono text-[11px] text-navy/80">
                        {isNumeric && numStats ? (
                          <div className="space-y-0.5">
                            <div className="flex gap-3">
                              <span><strong>Mean:</strong> {numStats.mean !== null ? numStats.mean : '—'}</span>
                              <span><strong>Med:</strong> {numStats.median !== null ? numStats.median : '—'}</span>
                              <span><strong>Std:</strong> {numStats.std !== null ? numStats.std : '—'}</span>
                            </div>
                            <div className="text-navy/60 flex gap-3">
                              <span><strong>Min:</strong> {numStats.min !== null ? numStats.min : '—'}</span>
                              <span><strong>Max:</strong> {numStats.max !== null ? numStats.max : '—'}</span>
                            </div>
                          </div>
                        ) : catStats ? (
                          <div className="space-y-0.5">
                            <div>
                              <strong>Most common:</strong>{' '}
                              <span className="text-navy font-semibold truncate inline-block max-w-[120px] align-bottom" title={catStats.top_value}>
                                {catStats.top_value !== null ? String(catStats.top_value) : '—'}
                              </span>
                            </div>
                            <div className="text-navy/60">
                              Frequency: {catStats.top_frequency.toLocaleString()} occurrences
                            </div>
                          </div>
                        ) : (
                          <span className="text-navy/40">—</span>
                        )}
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}
