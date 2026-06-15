/** @type {import('tailwindcss').Config} */
export default {
  content: [
    './index.html',
    './src/**/*.{js,ts,jsx,tsx}',
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'ui-monospace', 'monospace'],
      },
      colors: {
        surface: {
          DEFAULT: '#111827',
          card: '#1a2235',
          muted: '#0f172a',
        },
      },
      keyframes: {
        'intent-flash': {
          '0%':   { opacity: '1', transform: 'scale(1.06)' },
          '40%':  { opacity: '0.80' },
          '100%': { opacity: '1', transform: 'scale(1)' },
        },
        'fade-in': {
          '0%':   { opacity: '0', transform: 'translateY(4px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
      },
      animation: {
        'intent-flash': 'intent-flash 0.55s cubic-bezier(0.4, 0, 0.2, 1)',
        'fade-in':      'fade-in 0.3s ease-out',
      },
    },
  },
  plugins: [],
};
