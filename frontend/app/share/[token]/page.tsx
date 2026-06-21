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

function dateStr(ts: string) {
  try { return new Date(ts).toLocaleDateString([], { month: 'short', day: 'numeric', year: 'numeric' }) }
  catch { return '' }
}

export default function SharePage({ params }: { params: { token: string } }) {
  const [session, setSession] = useState<SharedSession | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetch(`${BASE}/share/${params.token}`)
      .then(r => r.ok ? r.json() : Promise.reject(r.status))
      .then(data => setSession(data))
      .catch(e => setError(e === 404 ? "This link has expired or been revoked." : "Couldn't load this conversation."))
  }, [params.token])

  return (
    // Fixed height + its own scroll container -- the main app's globals.css
    // sets `overflow: hidden` on <body> for its fixed sidebar/chat shell.
    // This page isn't part of that shell, so it needs an explicit scrolling
    // region rather than relying on document scroll.
    <div style={{
      height: '100vh', overflowY: 'auto', background: 'var(--bg)', color: 'var(--text)',
    }}>
      <div style={{ maxWidth: 720, margin: '0 auto', padding: '40px 20px 80px' }}>

        {/* Header — same logo mark + wordmark as the main app */}
        <div className="flex items-center gap-3" style={{ marginBottom: 40 }}>
          <div className="logo-mark">M</div>
          <div>
            <h1 style={{ fontFamily: 'Instrument Serif', fontSize: 19, fontStyle: 'italic', color: 'var(--text)', lineHeight: 1 }}>
              MindVault
            </h1>
            <p style={{ fontSize: 10, color: 'var(--text3)', fontFamily: 'IBM Plex Mono', letterSpacing: '0.05em', textTransform: 'uppercase', marginTop: 3 }}>
              Shared conversation
            </p>
          </div>
        </div>

        {error && (
          <div style={{
            border: '1px solid rgba(181, 72, 61, 0.25)', background: 'rgba(181, 72, 61, 0.06)',
            borderRadius: 10, padding: '32px 24px', textAlign: 'center'
          }}>
            <p style={{ fontSize: 28, marginBottom: 10 }}>🔒</p>
            <p style={{ color: 'var(--danger)', fontSize: 14, marginBottom: 4 }}>{error}</p>
            <p style={{ color: 'var(--text3)', fontSize: 12 }}>Ask the person who sent this link to share it again.</p>
          </div>
        )}

        {!session && !error && (
          <div className="flex flex-col items-center justify-center" style={{ padding: '80px 0', gap: 14 }}>
            <div className="spinner" style={{ width: 22, height: 22 }} />
            <p style={{ fontFamily: 'IBM Plex Mono', fontSize: 11, color: 'var(--text3)', letterSpacing: '0.04em' }}>
              Loading conversation…
            </p>
          </div>
        )}

        {session && (
          <>
            <div style={{ marginBottom: 28, paddingBottom: 20, borderBottom: '1px solid var(--border)' }}>
              <h2 style={{ fontSize: 21, fontWeight: 500, color: 'var(--text)', marginBottom: 6 }}>{session.name}</h2>
              <p style={{ fontSize: 11, color: 'var(--text3)', fontFamily: 'IBM Plex Mono' }}>
                {session.messages.length} message{session.messages.length !== 1 ? 's' : ''} · {dateStr(session.created_at)}
              </p>
            </div>

            <div className="flex flex-col" style={{ gap: 20 }}>
              {session.messages.map((msg, i) => {
                const isUser = msg.role === 'user'
                return (
                  <div key={i} className={`flex fade-up ${isUser ? 'justify-end' : 'justify-start'}`} style={{ animationDelay: `${Math.min(i * 0.04, 0.4)}s` }}>
                    <div style={{ maxWidth: '78%' }}>
                      {!isUser && (
                        <div className="flex items-center gap-2" style={{ marginBottom: 7 }}>
                          <div className="logo-mark" style={{ width: 18, height: 18, fontSize: 10, borderRadius: 4 }}>M</div>
                          <span style={{ fontSize: 10, color: 'var(--text3)', fontFamily: 'IBM Plex Mono', letterSpacing: '0.04em' }}>
                            MindVault
                          </span>
                        </div>
                      )}
                      {isUser ? (
                        <div className="msg-user" style={{ maxWidth: '100%' }}>
                          <p style={{ fontSize: 14, lineHeight: 1.65, color: 'var(--text)', whiteSpace: 'pre-wrap', wordBreak: 'break-word', margin: 0 }}>
                            {msg.content}
                          </p>
                        </div>
                      ) : (
                        <div className="md" style={{ fontSize: 14 }}>
                          <ReactMarkdown>{msg.content}</ReactMarkdown>
                        </div>
                      )}
                      <p style={{
                        fontSize: 10, color: 'var(--text3)', fontFamily: 'IBM Plex Mono',
                        marginTop: 5, textAlign: isUser ? 'right' : 'left'
                      }}>
                        {timeStr(msg.timestamp)}
                      </p>
                    </div>
                  </div>
                )
              })}
            </div>

            <div className="divider" style={{ marginTop: 48, marginBottom: 20 }} />
            <p style={{ fontSize: 11, color: 'var(--text3)', fontFamily: 'IBM Plex Mono', textAlign: 'center' }}>
              Read-only ·{' '}
              <a href="/" style={{ color: 'var(--accent3)', textDecoration: 'none' }}>Built with MindVault</a>
            </p>
          </>
        )}
      </div>
    </div>
  )
}