import React from 'react';

/**
 * Editorial Data Lab button component.
 * Uses Deep Teal for primary emphasis, Soft Stone borders for secondary actions.
 */
export default function Button({
  children,
  variant = 'primary',
  size = 'md',
  disabled = false,
  loading = false,
  onClick,
  type = 'button',
  className = '',
  icon: Icon,
  ...props
}) {
  const baseStyles =
    'inline-flex items-center justify-center font-medium transition-colors duration-150 focus:outline-none focus:ring-2 focus:ring-teal/30 focus:ring-offset-1 select-none';

  const sizeStyles = {
    sm: 'text-xs px-3 py-1.5 rounded-sm gap-1.5',
    md: 'text-sm px-4 py-2 rounded gap-2',
    lg: 'text-base px-6 py-2.5 rounded gap-2.5',
  };

  const variantStyles = {
    primary:
      'bg-teal text-white hover:bg-teal-dark active:bg-teal-dark/90 shadow-sm disabled:bg-teal/50 disabled:cursor-not-allowed',
    outline:
      'bg-white text-navy border border-stone hover:bg-ivory hover:border-stone-dark active:bg-stone-light disabled:opacity-50 disabled:cursor-not-allowed',
    ghost:
      'bg-transparent text-navy hover:bg-stone-light active:bg-stone disabled:opacity-50 disabled:cursor-not-allowed',
    secondary:
      'bg-stone-light text-navy border border-stone hover:bg-stone disabled:opacity-50 disabled:cursor-not-allowed',
    danger:
      'bg-red-700 text-white hover:bg-red-800 active:bg-red-900 disabled:opacity-50 disabled:cursor-not-allowed',
  };

  return (
    <button
      type={type}
      disabled={disabled || loading}
      onClick={onClick}
      className={`${baseStyles} ${sizeStyles[size] || sizeStyles.md} ${variantStyles[variant] || variantStyles.primary} ${className}`}
      {...props}
    >
      {loading ? (
        <svg
          className="animate-spin h-4 w-4 text-current"
          xmlns="http://www.w3.org/2000/svg"
          fill="none"
          viewBox="0 0 24 24"
        >
          <circle
            className="opacity-25"
            cx="12"
            cy="12"
            r="10"
            stroke="currentColor"
            strokeWidth="4"
          />
          <path
            className="opacity-75"
            fill="currentColor"
            d="M4 12a8 8 0 018-8v8H4z"
          />
        </svg>
      ) : Icon ? (
        <Icon className="w-4 h-4" />
      ) : null}
      <span>{children}</span>
    </button>
  );
}
