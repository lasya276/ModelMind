import React from 'react';

/**
 * Editorial status and categorization pill.
 */
export default function Badge({
  children,
  variant = 'sage',
  size = 'md',
  className = '',
}) {
  const sizeStyles = {
    sm: 'text-[11px] px-2 py-0.5',
    md: 'text-xs px-2.5 py-1',
  };

  const variantStyles = {
    sage: 'bg-sage text-navy border border-sage-dark/30 font-medium',
    teal: 'bg-teal-soft text-teal-dark border border-teal/20 font-medium',
    navy: 'bg-navy text-ivory font-medium',
    stone: 'bg-stone text-navy/80 border border-stone-dark/30 font-medium',
    amber: 'bg-amber-100 text-amber-900 border border-amber-300 font-medium',
    red: 'bg-red-50 text-red-800 border border-red-200 font-medium',
  };

  return (
    <span
      className={`inline-flex items-center rounded-sm font-sans tracking-wide leading-none select-none ${sizeStyles[size] || sizeStyles.md} ${variantStyles[variant] || variantStyles.sage} ${className}`}
    >
      {children}
    </span>
  );
}
