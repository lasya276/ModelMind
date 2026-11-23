import React from 'react';

export default function Footer() {
  return (
    <footer className="border-t border-stone bg-white/70 py-6 text-xs text-navy/60">
      <div className="max-w-7xl mx-auto px-6 flex flex-col sm:flex-row items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <span className="font-serif font-bold text-navy">ModelMind</span>
          <span>—</span>
          <span>From raw data to intelligent decisions.</span>
        </div>
        <div className="text-[11px] font-mono text-navy/50">
          A friendly platform for data intelligence and machine learning experimentation.
        </div>
      </div>
    </footer>
  );
}
