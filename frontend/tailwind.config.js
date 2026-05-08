/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
    './app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      fontFamily: {
        display: ['Instrument Serif', 'serif'],
        mono: ['IBM Plex Mono', 'monospace'],
        body: ['Epilogue', 'sans-serif'],
      },
    },
  },
  plugins: [],
}
