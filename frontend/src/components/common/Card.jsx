import React from 'react';

/**
 * Editorial Data Lab surface card.
 * Pure White background with delicate Soft Stone border and minimal shadow.
 */
export default function Card({
  children,
  className = '',
  title,
  subtitle,
  headerAction,
  footer,
  noPadding = false,
  as: Component = 'div',
  ...props
}) {
  return (
    <Component
      className={`bg-white border border-stone rounded-sm shadow-sm overflow-hidden ${className}`}
      {...props}
    >
      {(title || subtitle || headerAction) && (
        <div className="px-6 py-4 border-b border-stone flex items-start justify-between gap-4">
          <div>
            {title && (
              <h3 className="font-serif text-lg font-semibold text-navy tracking-tight">
                {title}
              </h3>
            )}
            {subtitle && (
              <p className="text-xs text-navy/70 mt-0.5 font-sans leading-relaxed">
                {subtitle}
              </p>
            )}
          </div>
          {headerAction && <div className="shrink-0">{headerAction}</div>}
        </div>
      )}
      <div className={noPadding ? '' : 'p-6'}>{children}</div>
      {footer && (
        <div className="px-6 py-3.5 bg-stone-light/40 border-t border-stone text-xs text-navy/75">
          {footer}
        </div>
      )}
    </Component>
  );
}
