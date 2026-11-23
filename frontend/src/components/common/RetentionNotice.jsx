import React from 'react';
import { Clock, Info } from 'lucide-react';
import { useDataset } from '../../context/DatasetContext';

/**
 * Editorial disclaimer explaining the server's opportunistic 48-hour temporary retention policy.
 * Transparently conveys that backend cleanup is opportunistic rather than guaranteeing an exact millisecond deletion.
 */
export default function RetentionNotice({ compact = false, className = '' }) {
  const { activeDataset } = useDataset();

  if (compact) {
    return (
      <div
        className={`flex items-center gap-2 text-xs text-navy/70 bg-stone-light/60 px-3 py-1.5 rounded-sm border border-stone/80 ${className}`}
      >
        <Clock className="w-3.5 h-3.5 text-teal shrink-0" />
        <span>
          Datasets are stored temporarily for up to <strong>48 hours</strong>.
        </span>
      </div>
    );
  }

  return (
    <div
      className={`bg-stone-light/50 border border-stone rounded-sm p-4 text-xs text-navy/80 flex items-start gap-3 ${className}`}
    >
      <Info className="w-4 h-4 text-teal mt-0.5 shrink-0" />
      <div className="space-y-1">
        <p className="font-semibold text-navy">
          Temporary Storage Notice (Up to 48 Hours)
        </p>
        <p className="leading-relaxed text-navy/75">
          Uploaded files and cleaned versions are stored securely on the server for up to 48 hours so you can run analyses and experiments. Expired files are removed when the server runs its cleanup process. Be sure to download any cleaned CSV files you want to keep permanently.
        </p>
        {activeDataset?.uploaded_at && (
          <p className="text-[11px] text-navy/60 pt-0.5 font-mono">
            Uploaded:{' '}
            {new Date(activeDataset.uploaded_at).toLocaleString(undefined, {
              dateStyle: 'medium',
              timeStyle: 'short',
            })}
          </p>
        )}
      </div>
    </div>
  );
}
