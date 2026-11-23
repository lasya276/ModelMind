import React from 'react';
import { NavLink, Link } from 'react-router-dom';
import { Database, Sparkles, FileText, Compass, X, AlertCircle } from 'lucide-react';
import { useDataset } from '../../context/DatasetContext';

export default function Navbar() {
  const { activeDataset, clearActiveDataset, datasetNotice, dismissNotice } = useDataset();

  const navItems = [
    { name: 'Overview', to: '/', icon: Compass },
    { name: 'Datasets', to: '/datasets', icon: Database },
    { name: 'Experiments', to: '/experiments', icon: Sparkles },
    { name: 'Reports', to: '/reports', icon: FileText },
  ];

  return (
    <header className="border-b border-stone bg-white sticky top-0 z-40">
      {/* Notice banner for expired / unavailable dataset */}
      {datasetNotice && (
        <div className="bg-amber-50 border-b border-amber-200 px-6 py-2.5 text-xs text-amber-900 flex items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <AlertCircle className="w-4 h-4 text-amber-700 shrink-0" />
            <span>
              <strong>{datasetNotice.title}:</strong> {datasetNotice.message}
            </span>
          </div>
          <button
            onClick={dismissNotice}
            className="p-1 hover:bg-amber-100 rounded text-amber-800 transition-colors"
            title="Dismiss notice"
          >
            <X className="w-3.5 h-3.5" />
          </button>
        </div>
      )}

      <div className="max-w-7xl mx-auto px-6 py-3.5 flex items-center justify-between gap-6">
        {/* Editorial Masthead Branding */}
        <Link to="/" className="flex items-baseline gap-3 shrink-0 group">
          <span className="font-serif text-2xl font-bold tracking-tight text-navy group-hover:text-teal transition-colors">
            ModelMind
          </span>
          <span className="hidden md:inline-block text-[11px] uppercase tracking-widest text-navy/50 font-sans border-l border-stone pl-3 py-0.5">
            Editorial Data Lab
          </span>
        </Link>

        {/* Primary Navigation Tabs */}
        <nav className="flex items-center gap-1">
          {navItems.map((item) => {
            const Icon = item.icon;
            return (
              <NavLink
                key={item.name}
                to={item.to}
                className={({ isActive }) =>
                  `flex items-center gap-2 px-3.5 py-1.5 rounded-sm text-xs font-medium tracking-wide transition-all duration-150 ${
                    isActive
                      ? 'bg-ivory text-navy border-b-2 border-teal shadow-xs'
                      : 'text-navy/70 hover:text-navy hover:bg-stone-light/50'
                  }`
                }
              >
                <Icon className="w-3.5 h-3.5 opacity-70" />
                <span>{item.name}</span>
              </NavLink>
            );
          })}
        </nav>

        {/* Active Dataset Status Pill */}
        <div className="flex items-center gap-2">
          {activeDataset ? (
            <div className="flex items-center gap-2 bg-sage/60 border border-sage-dark/40 px-3 py-1 rounded-sm text-xs text-navy font-mono">
              <span className="w-2 h-2 rounded-full bg-teal animate-pulse" />
              <span className="max-w-[140px] md:max-w-[180px] truncate font-medium" title={activeDataset.filename}>
                {activeDataset.filename}
              </span>
              {activeDataset.row_count !== null && (
                <span className="text-[11px] text-navy/60 hidden sm:inline">
                  ({activeDataset.row_count.toLocaleString()} rows)
                </span>
              )}
              <button
                onClick={clearActiveDataset}
                title="Clear active dataset"
                className="hover:bg-sage rounded p-0.5 ml-1 text-navy/60 hover:text-navy transition-colors"
              >
                <X className="w-3 h-3" />
              </button>
            </div>
          ) : (
            <Link
              to="/datasets"
              className="text-xs text-navy/60 hover:text-teal font-sans border border-stone/80 bg-stone-light/40 px-2.5 py-1 rounded-sm transition-colors"
            >
              No dataset loaded
            </Link>
          )}
        </div>
      </div>
    </header>
  );
}
