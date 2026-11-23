/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        ivory: '#F7F6F0',
        navy: {
          DEFAULT: '#172B43',
          light: '#233F5E',
          dark: '#0E1C2C',
        },
        teal: {
          DEFAULT: '#477C75',
          dark: '#38635E',
          light: '#5B938B',
          soft: '#EAF2F0',
        },
        stone: {
          DEFAULT: '#EAE9E1',
          light: '#F3F2EC',
          dark: '#D5D3C7',
        },
        sage: {
          DEFAULT: '#DCEBE5',
          light: '#EBF4F0',
          dark: '#B8D5CB',
        },
      },
      fontFamily: {
        serif: ['"DM Serif Display"', 'Georgia', 'serif'],
        sans: ['Inter', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'SFMono-Regular', 'Menlo', 'monospace'],
      },
    },
  },
  plugins: [],
};
