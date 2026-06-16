'use client'
export const dynamic = 'force-dynamic'

import { useEffect, useState } from 'react'
import ReactMarkdown from 'react-markdown'

const BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

interface Message { role: string; content: string; timestamp: string }
interface SharedSession { name: string; created_at: string; messages: Message[] }

function timeStr(ts: string) {
  try { return new Date(ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) }
  catch { return '' }
}

export default function SharePage({ params }: { params: { token: string } }) {
  const [session, setSession] = useState<SharedSession | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetch(`${BASE}/share/${params.token}`)
      .then(r => r.ok ? r.json() : Promise.reject(r.status))
      .then(data => setSession(data))
      .catch(e => setError(e === 404 ? 'This link has expired or been revoked.' : 'Could not load session.'))
  }, [params.token])

  return (
    <div style={{ minHeight: '100vh', background: '#0e0e10', color: '#e8e6e0', fontFamily: 'system-ui, sans-serif', padding: '32px 16px' }}>
      <div style={{ maxWidth: 720, margin: '0 auto' }}>

        {/* Header */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 32 }}>
          <div style={{ width: 32, height: 32, borderRadius: 8, background: 'rgba(126,184,164,0.15)', border: '1px solid rgba(126,184,164,0.3)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 16 }}>⬡</div>
          <div>
            <span style={{ fontSize: 13, fontWeight: 700, color: '#7eb8a4', letterSpacing: '0.04em', fontFamily: 'IBM Plex Mono, monospace' }}>MindVault</span>
            <span style={{ fontSize: 11, color: '#666', marginLeft: 8, fontFamily: 'IBM Plex Mono, monospace' }}>shared conversation</span>
          </div>
        </div>

        {error && (
          <div style={{ background: 'rgba(248,113,113,0.08)', border: '1px solid rgba(248,113,113,0.2)', borderRadius: 10, padding: '24px', textAlign: 'center', color: '#f87171' }}>
            <p style={{ fontSize: 20, marginBottom: 8 }}>🔒</p>
            <p>{error}</p>
          </div>
        )}

        {!session && !error && (
          <div style={{ textAlign: 'center', color: '#666', padding: '60px 0' }}>
            <div style={{ width: 32, height: 32, border: '2px solid #7eb8a4', borderTopColor: 'transparent', borderRadius: '50%', margin: '0 auto 16px', animation: 'spin 1s linear infinite' }} />
            <p style={{ fontFamily: 'IBM Plex Mono, monospace', fontSize: 12 }}>Loading conversation…</p>
            <style>{`@keyframes spin { to { transform: rotate(360deg) } }`}</style>
          </div>
        )}

        {session && (
          <>
            <div style={{ marginBottom: 24 }}>
              <h1 style={{ fontSize: 20, fontWeight: 600, color: '#e8e6e0', marginBottom: 4 }}>{session.name}</h1>
              <p style={{ fontSize: 11, color: '#666', fontFamily: 'IBM Plex Mono, monospace' }}>
                {session.messages.length} messages · shared {new Date(session.created_at).toLocaleDateString()}
              </p>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
              {session.messages.map((msg, i) => {
                const isUser = msg.role === 'user'
                return (
                  <div key={i} style={{ display: 'flex', justifyContent: isUser ? 'flex-end' : 'flex-start' }}>
                    <div style={{ maxWidth: '80%' }}>
                      {!isUser && (
                        <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 6 }}>
                          <div style={{ width: 20, height: 20, borderRadius: 5, background: 'rgba(126,184,164,0.15)', border: '1px solid rgba(126,184,164,0.25)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 10 }}>⬡</div>
                          <span style={{ fontSize: 10, color: '#7eb8a4', fontFamily: 'IBM Plex Mono, monospace' }}>MindVault</span>
                        </div>
                      )}
                      <div style={{
                        padding: '10px 14px', borderRadius: isUser ? '16px 16px 4px 16px' : '4px 16px 16px 16px',
                        background: isUser ? 'rgba(126,184,164,0.12)' : '#1a1a1d',
                        border: `1px solid ${isUser ? 'rgba(126,184,164,0.2)' : '#2a2a2e'}`,
                        fontSize: 14, lineHeight: 1.65, color: '#e8e6e0',
                        wordBreak: 'break-word',
                      }}>
                        {isUser
                          ? <p style={{ margin: 0 }}>{msg.content}</p>
                          : <div className="prose" style={{ fontSize: 14 }}><ReactMarkdown>{msg.content}</ReactMarkdown></div>
                        }
                      </div>
                      <p style={{ fontSize: 10, color: '#444', fontFamily: 'IBM Plex Mono, monospace', marginTop: 4, textAlign: isUser ? 'right' : 'left' }}>
                        {timeStr(msg.timestamp)}
                      </p>
                    </div>
                  </div>
                )
              })}
            </div>

            {/* Footer */}
            <div style={{ marginTop: 48, paddingTop: 24, borderTop: '1px solid #1f1f22', textAlign: 'center' }}>
              <p style={{ fontSize: 12, color: '#555', fontFamily: 'IBM Plex Mono, monospace' }}>
                Read-only · Generated by{' '}
                <a href="/" style={{ color: '#7eb8a4', textDecoration: 'none' }}>MindVault</a>
              </p>
            </div>
          </>
        )}
      </div>
    </div>
  )
}