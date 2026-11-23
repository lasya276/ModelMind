import React from 'react';
import { BookOpen } from 'lucide-react';

/**
 * Editorial callout specifically designed to demystify complex data science and ML terms.
 * Styled in Soft Sage with Deep Navy text to provide informative, welcoming guidance.
 */
export default function ExplainerCallout({
  term,
  children,
  className = '',
}) {
  return (
    <div
      className={`bg-sage/40 border border-sage-dark/40 rounded-sm p-4 text-xs text-navy flex items-start gap-3 ${className}`}
    >
      <div className="p-1 rounded-sm bg-sage text-teal-dark shrink-0 mt-0.5">
        <BookOpen className="w-3.5 h-3.5" />
      </div>
      <div className="space-y-0.5 leading-relaxed">
        {term && (
          <h4 className="font-semibold text-navy text-xs uppercase tracking-wider font-sans">
            Concept: {term}
          </h4>
        )}
        <div className="text-navy/85">{children}</div>
      </div>
    </div>
  );
}
