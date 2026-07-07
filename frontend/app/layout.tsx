import type { Metadata } from 'next'
import { Instrument_Serif, IBM_Plex_Mono, Epilogue } from 'next/font/google'
import './globals.css'

const instrumentSerif = Instrument_Serif({
  subsets: ['latin'],
  weight: '400',
  style: ['normal', 'italic'],
  variable: '--font-serif',
  display: 'swap',
})

const plexMono = IBM_Plex_Mono({
  subsets: ['latin'],
  weight: ['300', '400', '500'],
  variable: '--font-mono',
  display: 'swap',
})

const epilogue = Epilogue({
  subsets: ['latin'],
  weight: ['300', '400', '500', '600'],
  variable: '--font-sans',
  display: 'swap',
})

export const metadata: Metadata = {
  title: 'MindVault',
  description: 'Your personal knowledge retrieval system',
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${instrumentSerif.variable} ${plexMono.variable} ${epilogue.variable}`}>
      <head>
        {/* Applies the last-known theme synchronously, before first paint.
            Without this, the page always starts in the CSS default (light)
            and only switches once the /preferences fetch resolves, which
            shows as a flash to dark for returning dark-theme users. */}
        <script
          dangerouslySetInnerHTML={{
            __html: `(function(){try{var t=localStorage.getItem('mv_theme');if(t)document.documentElement.setAttribute('data-theme',t)}catch(e){}})();`
          }}
        />
      </head>
      <body>{children}</body>
    </html>
  )
}
