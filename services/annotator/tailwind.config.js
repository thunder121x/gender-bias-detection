/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx,ts,tsx}'],
  theme: {
    extend: {
      colors: {
        rationale: {
          light: '#fce7f3',
          DEFAULT: '#f472b6'
        },
        trigger: {
          light: '#fecdd3',
          DEFAULT: '#ef4444'
        }
      }
    }
  },
  plugins: []
};
