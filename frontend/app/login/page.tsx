'use client'

import { useState } from 'react'
import { createClient } from '@supabase/supabase-js'

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!
)

export default function LoginPage() {
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [isSignup, setIsSignup] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [message, setMessage] = useState('')

  const handle = async () => {
    setLoading(true)
    setError('')
    setMessage('')

    try {
      if (isSignup) {
        const { error } = await supabase.auth.signUp({ email, password })
        if (error) throw error
        setMessage('Check your email to confirm your account.')
      } else {
        const { error } = await supabase.auth.signInWithPassword({ email, password })
        if (error) throw error
        window.location.href = '/'
      }
    } catch (e: any) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="flex items-center justify-center h-screen" style={{ background: 'var(--bg)' }}>
      <div style={{
        background: 'var(--surface)',
        border: '1px solid var(--border)',
        borderRadius: 12,
        padding: 40,
        width: '100%',
        maxWidth: 400
      }}>
        <div className="flex items-center gap-3 mb-8">
          <div className="logo-mark">M</div>
          <div>
            <h1 style={{ fontFamily: 'Instrument Serif', fontSize: 22, fontStyle: 'italic', color: 'var(--text)' }}>MindVault</h1>
            <p style={{ fontSize: 10, color: 'var(--text3)', fontFamily: 'IBM Plex Mono' }}>knowledge · retrieved</p>
          </div>
        </div>

        <p style={{ fontSize: 14, color: 'var(--text2)', marginBottom: 24 }}>
          {isSignup ? 'Create your vault' : 'Sign in to your vault'}
        </p>

        <div className="flex flex-col gap-3">
          <input
            type="email"
            placeholder="Email"
            value={email}
            onChange={e => setEmail(e.target.value)}
            className="vault-input"
            style={{ minHeight: 44 }}
          />
          <input
            type="password"
            placeholder="Password"
            value={password}
            onChange={e => setPassword(e.target.value)}
            className="vault-input"
            style={{ minHeight: 44 }}
            onKeyDown={e => e.key === 'Enter' && handle()}
          />

          {error && (
            <p style={{ fontSize: 12, color: '#e87171', fontFamily: 'IBM Plex Mono' }}>{error}</p>
          )}
          {message && (
            <p style={{ fontSize: 12, color: 'var(--accent3)', fontFamily: 'IBM Plex Mono' }}>{message}</p>
          )}

          <button
            className="action-btn"
            onClick={handle}
            disabled={loading || !email || !password}
            style={{ marginTop: 8, padding: '12px 0', fontSize: 13 }}
          >
            {loading ? 'Please wait...' : isSignup ? 'Create Account' : 'Sign In'}
          </button>

          <button
            onClick={() => { setIsSignup(!isSignup); setError(''); setMessage('') }}
            style={{ fontSize: 12, color: 'var(--text3)', fontFamily: 'IBM Plex Mono', background: 'none', border: 'none', cursor: 'pointer', marginTop: 4 }}
          >
            {isSignup ? 'Already have an account? Sign in' : "Don't have an account? Sign up"}
          </button>
        </div>
      </div>
    </div>
  )
}
