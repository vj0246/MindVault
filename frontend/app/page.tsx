'use client'
export const dynamic = 'force-dynamic'
import { getSupabase, signOut } from '../lib/supabase'
import React, { useState, useEffect, useRef, useCallback } from 'react'
import ReactMarkdown from 'react-markdown'
import GraphPanel from '../components/GraphPanel'
import {
  uploadDocument,
  queryKnowledge,
  queryWithAttachment,
  streamQuery,
  shareSession,
  unshareSession,
  getDocuments,
  exportSession,
  exportSessionPDF,
  getGraphTopic,
  getFullGraph,
  clearSession,
  createSession,
  listSessions,
  renameSession,
  deleteSession,
  getSessionHistory,
} from '../lib/api'

type Mode = 'student' | 'lawyer' | 'developer' | 'default'
type Intent = 'answer' | 'compare' | 'test' | 'summarize'

interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  sources?: string[]
  chunks?: ChunkSource[]
  confidence?: number
  intent?: Intent
  related_concepts?: { id: string; sources: string[] }[]
  timestamp: string
}

interface Doc {
  id: string
  filename: string
  chunk_count: number
  uploaded_at: string
}

interface ChatSession {
  id: string
  name: string
  created_at: string
  last_active: string
  number?: number
}

interface ChunkSource {
  content: string
  similarity: number
  filename: string
  page_number?: number
  chunk_index?: number
}

interface Toast {
  msg: string
  type: 'success' | 'error' | 'info'
}

interface GraphData {
  nodes: { id: string; sources: string[] }[]
  edges: { source: string; target: string; relation: string }[]
}

function genId() {
  return Math.random().toString(36).slice(2) + Date.now().toString(36)
}

function timeStr(ts: string) {
  return new Date(ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

const MODES: { value: Mode; label: string; desc: string }[] = [
  { value: 'student',   label: 'Student',   desc: 'Exam-ready bullet points' },
  { value: 'lawyer',    label: 'Legal',     desc: 'Formal & precise' },
  { value: 'developer', label: 'Dev',       desc: 'Technical & implementation' },
  { value: 'default',   label: 'Default',   desc: 'Balanced responses' },
]

function getDynamicSuggestions(docs: Doc[]) {
  if (docs.length === 0) return []
  const names = docs.map(d => d.filename.replace(/\.[^.]+$/, '').slice(0, 30))
  return [
    { label: `Summarize ${names[0]}`, intent: 'summarize' },
    { label: `Generate MCQs from ${names[0]}`, intent: 'test' },
    { label: `What are the key concepts in ${names[0]}?`, intent: 'answer' },
    docs.length > 1
      ? { label: `Compare ${names[0]} and ${names[1]}`, intent: 'compare' }
      : { label: `Explain the main topics in ${names[0]}`, intent: 'answer' },
  ]
}

function TypingIndicator() {
  return (
    <div className="flex items-start gap-3 fade-up px-1">
      <div className="w-5 h-5 rounded-full flex-shrink-0 mt-1 flex items-center justify-center"
        style={{ background: 'rgba(232,197,71,0.12)', border: '1px solid rgba(232,197,71,0.2)' }}>
        <span style={{ fontSize: 8, color: 'var(--accent)' }}>M</span>
      </div>
      <div className="flex items-center gap-1.5 py-3">
        <div className="dot" /><div className="dot" /><div className="dot" />
        <span style={{ fontSize: 11, color: 'var(--text3)', fontFamily: 'IBM Plex Mono', marginLeft: 6 }}>
          retrieving from vault
        </span>
      </div>
    </div>
  )
}

function IntentPill({ intent }: { intent: string }) {
  return <span className={`intent-pill intent-${intent}`}>{intent}</span>
}

function ConfidenceBadge({ score }: { score: number }) {
  const pct = Math.round(score * 100)
  const color = score >= 0.7 ? '#4ade80' : score >= 0.4 ? '#facc15' : '#f87171'
  const label = score >= 0.7 ? 'High' : score >= 0.4 ? 'Medium' : 'Low'
  return (
    <span style={{
      fontSize: 9, fontFamily: 'IBM Plex Mono', padding: '2px 7px',
      borderRadius: 4, border: `1px solid ${color}33`,
      color, background: `${color}11`, marginLeft: 6
    }}>
      {label} {pct}%
    </span>
  )
}

function SourcePanel({ chunks }: { chunks: ChunkSource[] }) {
  const [open, setOpen] = useState(false)
  const [copied, setCopied] = useState<number | null>(null)
  const copyChunk = (text: string, i: number) => {
    navigator.clipboard.writeText(text).then(() => { setCopied(i); setTimeout(() => setCopied(null), 1500) })
  }
  return (
    <div style={{ display: 'inline-block' }}>
      <button
        onClick={() => setOpen(!open)}
        style={{
          fontSize: 10, fontFamily: 'IBM Plex Mono', color: 'var(--text3)',
          background: 'var(--surface2)', border: '1px solid var(--border)',
          borderRadius: 4, padding: '2px 7px', cursor: 'pointer', marginLeft: 4,
          transition: 'color 0.15s'
        }}
        title="View source chunks"
      >
        {open ? '▲ sources' : 'ℹ sources'}
      </button>
      {open && (
        <div style={{
          marginTop: 8, background: 'var(--surface2)', border: '1px solid var(--border)',
          borderRadius: 8, overflow: 'hidden'
        }}>
          {chunks.map((c, i) => (
            <div key={i} style={{
              padding: '10px 12px',
              borderBottom: i < chunks.length - 1 ? '1px solid var(--border)' : 'none'
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                  <span style={{ fontSize: 10, fontFamily: 'IBM Plex Mono', color: 'var(--accent)', fontWeight: 600 }}>
                    📄 {c.filename}
                  </span>
                  {(c.page_number != null && c.page_number > 0) && (
                    <span style={{ fontSize: 9, fontFamily: 'IBM Plex Mono', color: 'var(--text3)', background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 3, padding: '1px 5px' }}>
                      p.{(c.page_number ?? 0) + 1}
                    </span>
                  )}
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                  <span style={{ fontSize: 9, fontFamily: 'IBM Plex Mono', color: 'var(--accent3)', background: 'rgba(126,184,164,0.1)', border: '1px solid rgba(126,184,164,0.2)', borderRadius: 3, padding: '1px 6px' }}>
                    {Math.round(c.similarity * 100)}% match
                  </span>
                  <button onClick={() => copyChunk(c.content, i)} title="Copy chunk text — paste in PDF viewer to find passage"
                    style={{ fontSize: 9, cursor: 'pointer', background: 'none', border: 'none', color: copied === i ? 'var(--accent)' : 'var(--text3)', padding: 0 }}>
                    {copied === i ? '✓' : '⎘'}
                  </button>
                </div>
              </div>
              <p style={{ fontSize: 11, color: 'var(--text3)', lineHeight: 1.55, fontFamily: 'IBM Plex Mono' }}>
                {c.content}{c.content.length >= 200 ? '…' : ''}
              </p>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

// Memoized — without this, every keystroke in the input box re-renders
// the ENTIRE message list (since input/messages live in the same parent
// component). With many messages, that re-render cost compounds into
// visible lag while typing. memo() makes each bubble only re-render when
// its own `msg` prop actually changes (e.g. new streaming tokens for that
// specific message), not on every parent re-render.
const MessageBubble = React.memo(function MessageBubble({ msg, onConceptClick }: {
  msg: Message
  onConceptClick: (c: string) => void
}) {
  if (msg.role === 'user') {
    // handleSend prefixes attachment messages as "📎 filename\nquestion".
    // Split that out so the filename renders as a distinct chip instead of
    // running into the question as plain text on its own line.
    const attachMatch = msg.content.match(/^📎 (.+?)\n([\s\S]*)$/)
    const attachName = attachMatch ? attachMatch[1] : null
    const bodyText = attachMatch ? attachMatch[2] : msg.content

    return (
      <div className="flex justify-end fade-up" style={{ width: '100%' }}>
        <div style={{ maxWidth: '75%', display: 'flex', flexDirection: 'column', alignItems: 'flex-end' }}>
          {attachName && (
            <span style={{
              display: 'inline-flex', alignItems: 'center', gap: 5,
              fontSize: 10, fontFamily: 'IBM Plex Mono', color: 'var(--accent3)',
              background: 'rgba(126,184,164,0.08)', border: '1px solid rgba(126,184,164,0.2)',
              borderRadius: 5, padding: '3px 9px', marginBottom: 6, maxWidth: '100%'
            }}>
              <span style={{ flexShrink: 0 }}>📎</span>
              <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{attachName}</span>
            </span>
          )}
          <div className="msg-user" style={{ textAlign: 'left', display: 'inline-block', maxWidth: '100%' }}>
            <p style={{ fontSize: 14, lineHeight: 1.65, color: 'var(--text)', whiteSpace: 'pre-wrap', wordBreak: 'break-word', margin: 0 }}>{bodyText}</p>
          </div>
          <p style={{ fontSize: 10, color: 'var(--text3)', fontFamily: 'IBM Plex Mono', textAlign: 'right', marginTop: 4 }}>
            {timeStr(msg.timestamp)}
          </p>
        </div>
      </div>
    )
  }

  return (
    <div className="flex items-start gap-3 fade-up">
      <div className="w-5 h-5 rounded-full flex-shrink-0 mt-1 flex items-center justify-center"
        style={{ background: 'rgba(232,197,71,0.12)', border: '1px solid rgba(232,197,71,0.2)' }}>
        <span style={{ fontSize: 8, color: 'var(--accent)', fontFamily: 'Instrument Serif', fontStyle: 'italic' }}>M</span>
      </div>
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 mb-2">
          <span style={{ fontSize: 11, color: 'var(--text3)', fontFamily: 'IBM Plex Mono' }}>MindVault</span>
          {msg.intent && <IntentPill intent={msg.intent} />}
        </div>
        <div className="md"><ReactMarkdown>{msg.content}</ReactMarkdown></div>
        {msg.sources && msg.sources.length > 0 && msg.sources[0] !== 'conversation history' && (
          <div className="mt-3">
            <div className="flex flex-wrap gap-1 items-center">
              {msg.sources.map((s, i) => <span key={i} className="source-chip">📄 {s}</span>)}
              {msg.chunks && msg.chunks.length > 0 && (
                <SourcePanel chunks={msg.chunks} />
              )}
            </div>
          </div>
        )}
        {msg.related_concepts && msg.related_concepts.length > 0 && (
          <div className="mt-3">
            <p style={{ fontSize: 10, color: 'var(--text3)', fontFamily: 'IBM Plex Mono', marginBottom: 6, textTransform: 'uppercase', letterSpacing: '0.06em' }}>
              Related concepts — click to explore graph
            </p>
            <div className="flex flex-wrap gap-1.5">
              {msg.related_concepts.slice(0, 6).map((c, i) => (
                <button key={i} className="concept-tag" onClick={() => onConceptClick(c.id)}>⬡ {c.id}</button>
              ))}
            </div>
          </div>
        )}
        <p style={{ fontSize: 10, color: 'var(--text3)', fontFamily: 'IBM Plex Mono', marginTop: 6 }}>
          {timeStr(msg.timestamp)}
        </p>
      </div>
    </div>
  )
})

function Sidebar({ docs, onUpload, uploading, uploadStatus, sessionId, msgCount, onExport, onExportPDF, onNewSession, onClearSession, open, onClose, selectedDocs, onToggleDoc, sessions, onSelectSession, onDeleteSession, onShare, sharingId, userEmail, onSignOut }: {
  docs: Doc[]; onUpload: (files: File[]) => void; uploading: boolean; uploadStatus: string
  sessionId: string; msgCount: number; onExport: () => void; onExportPDF: () => void; onNewSession: () => void
  onClearSession: () => void; open: boolean; onClose: () => void
  selectedDocs: string[]; onToggleDoc: (id: string) => void
  sessions: ChatSession[]; onSelectSession: (id: string) => void; onDeleteSession: (id: string) => void
  onShare: (id: string) => void; sharingId: string | null
  userEmail: string; onSignOut: () => void
}) {
  const fileRef = useRef<HTMLInputElement>(null)
  const [dragging, setDragging] = useState(false)

  return (
    <>
      {/* Mobile overlay */}
      {open && (
        <div className="fixed inset-0 z-40 bg-black/60 md:hidden" onClick={onClose} />
      )}

      <aside className={`sidebar fixed md:relative z-50 md:z-auto transition-transform duration-300
        ${open ? 'translate-x-0' : '-translate-x-full md:translate-x-0'}`}>
        <div className="flex items-center justify-between gap-3 p-5 pb-4">
          <div className="flex items-center gap-3">
            <div className="logo-mark">M</div>
            <div>
              <h1 style={{ fontFamily: 'Instrument Serif', fontSize: 18, fontStyle: 'italic', color: 'var(--text)', lineHeight: 1 }}>MindVault</h1>
              <p style={{ fontSize: 10, color: 'var(--text3)', fontFamily: 'IBM Plex Mono', marginTop: 2 }}>knowledge · retrieved</p>
            </div>
          </div>
          <button className="md:hidden" onClick={onClose}
            style={{ color: 'var(--text3)', fontSize: 18, padding: 4 }}>✕</button>
        </div>
        <div className="divider" />
        <div className="flex flex-col gap-4 p-4 flex-1 overflow-y-auto">
          <div>
            <p style={{ fontSize: 9, color: 'var(--text3)', fontFamily: 'IBM Plex Mono', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 8 }}>Session</p>
            <div style={{ background: 'var(--surface2)', border: '1px solid var(--border)', borderRadius: 6, padding: '8px 10px' }}>
              <div className="flex items-center gap-2 mb-1">
                <div className="status-dot" />
                <span style={{ fontSize: 10, color: 'var(--accent3)', fontFamily: 'IBM Plex Mono' }}>Active</span>
              </div>
              <p style={{ fontSize: 10, color: 'var(--text3)', fontFamily: 'IBM Plex Mono', wordBreak: 'break-all' }}>
                {sessionId ? sessionId.slice(0, 20) + '...' : 'Loading...'}
              </p>
              <p style={{ fontSize: 10, color: 'var(--text3)', marginTop: 4, fontFamily: 'IBM Plex Mono' }}>{msgCount} messages</p>
            </div>
          </div>

          <div>
            <p style={{ fontSize: 9, color: 'var(--text3)', fontFamily: 'IBM Plex Mono', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 8 }}>Upload Document</p>
            <div
              className={`upload-zone ${dragging ? 'drag-over' : ''}`}
              onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
              onDragLeave={() => setDragging(false)}
              onDrop={(e) => { e.preventDefault(); setDragging(false); const files = Array.from(e.dataTransfer.files); if (files.length) onUpload(files) }}
              onClick={() => !uploading && fileRef.current?.click()}
            >
              {uploading ? (
                <div className="flex flex-col items-center gap-2 py-2">
                  <div className="spinner" />
                  <p style={{ fontSize: 11, color: 'var(--accent)', fontFamily: 'IBM Plex Mono' }}>{uploadStatus}</p>
                </div>
              ) : (
                <div className="flex flex-col items-center gap-1 py-2">
                  <span style={{ fontSize: 22, marginBottom: 2 }}>📄</span>
                  <p style={{ fontSize: 12, color: 'var(--text2)' }}>Drop files or click</p>
                  <p style={{ fontSize: 10, color: 'var(--text3)', fontFamily: 'IBM Plex Mono' }}>PDF · TXT · MD · DOCX · Images</p>
                </div>
              )}
            </div>
            <input ref={fileRef} type="file" accept=".pdf,.txt,.md,.docx,.doc,.jpg,.jpeg,.png,.gif,.webp" multiple className="hidden"
              onChange={(e) => { const files = Array.from(e.target.files || []); if (files.length) onUpload(files) }} />
          </div>


            {/* ── Chat Sessions ─────────────────────────────── */}
            <div style={{ marginBottom: 20 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                <p style={{ fontSize: 9, color: 'var(--text3)', fontFamily: 'IBM Plex Mono', textTransform: 'uppercase', letterSpacing: '0.08em' }}>
                  Chats ({sessions.length})
                </p>
                <button onClick={onNewSession} style={{
                  fontSize: 9, fontFamily: 'IBM Plex Mono', color: 'var(--accent)',
                  background: 'rgba(126,184,164,0.08)', border: '1px solid rgba(126,184,164,0.2)',
                  borderRadius: 4, padding: '2px 8px', cursor: 'pointer'
                }}>+ New</button>
              </div>
              {/* Bounded + independently scrollable -- without this, a long
                  chat history pushes Knowledge Base far below the fold,
                  since both used to share one continuous scroll region. */}
              <div style={{ maxHeight: 220, overflowY: 'auto' }}>
                {sessions.map(s => (
                  <div key={s.id} onClick={() => onSelectSession(s.id)} style={{
                    display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                    padding: '6px 10px', borderRadius: 6, cursor: 'pointer', marginBottom: 2,
                    background: s.id === sessionId ? 'rgba(126,184,164,0.1)' : 'transparent',
                    border: s.id === sessionId ? '1px solid rgba(126,184,164,0.2)' : '1px solid transparent',
                  }}>
                    <span style={{
                      display: 'flex', alignItems: 'center', gap: 6, flex: 1, minWidth: 0,
                      fontSize: 11, color: s.id === sessionId ? 'var(--accent)' : 'var(--text2)',
                      fontFamily: 'IBM Plex Mono'
                    }}>
                      <span style={{ flexShrink: 0, opacity: 0.5, fontSize: 10 }}>#{s.number ?? '–'}</span>
                      <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                        {s.name.slice(0, 24)}{s.name.length > 24 ? '…' : ''}
                      </span>
                    </span>
                    <div style={{ display: 'flex', gap: 2, flexShrink: 0 }}>
                      <button onClick={e => { e.stopPropagation(); onShare(s.id) }}
                        disabled={sharingId === s.id}
                        title="Copy share link"
                        style={{ fontSize: 10, color: 'var(--text3)', background: 'none', border: 'none', cursor: 'pointer', padding: '0 3px' }}>
                        {sharingId === s.id ? '…' : '🔗'}
                      </button>
                      <button onClick={e => { e.stopPropagation(); onDeleteSession(s.id) }}
                        style={{ fontSize: 10, color: 'var(--text3)', background: 'none', border: 'none', cursor: 'pointer', padding: '0 3px' }}>✕</button>
                    </div>
                  </div>
                ))}
                {sessions.length === 0 && (
                  <p style={{ fontSize: 10, color: 'var(--text3)', fontFamily: 'IBM Plex Mono' }}>No chats yet</p>
                )}
              </div>
            </div>

          <div className="flex-1">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
              <p style={{ fontSize: 9, color: 'var(--text3)', fontFamily: 'IBM Plex Mono', textTransform: 'uppercase', letterSpacing: '0.08em' }}>
                Knowledge Base ({docs.length})
              </p>
              <p style={{ fontSize: 9, fontFamily: 'IBM Plex Mono', color: selectedDocs.length > 0 ? 'var(--accent)' : 'var(--text3)' }}>
                {selectedDocs.length > 0 ? `${selectedDocs.length} selected` : 'all'}
              </p>
            </div>
            {docs.length === 0 ? (
              <p style={{ fontSize: 12, color: 'var(--text3)', textAlign: 'center', padding: '16px 0' }}>No documents yet</p>
            ) : (
              <div className="flex flex-col gap-1">
                {docs.map((doc, i) => (
                  <div key={i} className={`doc-item fade-up ${selectedDocs.includes(doc.id) ? 'active' : ''}`} style={{ cursor: 'pointer' }}
                    onClick={() => onToggleDoc(doc.id)}>
                    <input
                      type="checkbox"
                      checked={selectedDocs.includes(doc.id)}
                      onChange={() => onToggleDoc(doc.id)}
                      onClick={e => e.stopPropagation()}
                      style={{ flexShrink: 0, accentColor: 'var(--accent)', cursor: 'pointer' }}
                    />
                    <div className="flex-1 min-w-0">
                      <p style={{ fontSize: 12, color: 'var(--text)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{doc.filename}</p>
                      <p style={{ fontSize: 10, color: 'var(--text3)', fontFamily: 'IBM Plex Mono' }}>
                        {doc.chunk_count} chunks · {new Date(doc.uploaded_at).toLocaleDateString()}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          <div className="flex flex-col gap-2">
            <div className="flex gap-2">
              <button className="action-btn" style={{ flex: 1 }} onClick={onExport}>↓ MD</button>
              <button className="action-btn" style={{ flex: 1 }} onClick={onExportPDF}>↓ PDF</button>
            </div>
            <button className="action-btn" onClick={onNewSession}>+ New Session</button>
            <button className="action-btn danger" onClick={onClearSession}>✕ Clear History</button>
          </div>
        </div>

        {/* Account — fixed footer, outside the scrollable area */}
        <div className="divider" />
        <div className="flex items-center justify-between gap-2 p-4">
          <div className="flex items-center gap-2 min-w-0">
            <div style={{
              width: 26, height: 26, borderRadius: 7, flexShrink: 0,
              background: 'var(--surface2)', border: '1px solid var(--border2)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              fontFamily: 'IBM Plex Mono', fontSize: 11, color: 'var(--accent3)'
            }}>
              {userEmail ? userEmail[0].toUpperCase() : '?'}
            </div>
            <p style={{
              fontSize: 11, color: 'var(--text2)', fontFamily: 'IBM Plex Mono',
              overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap'
            }} title={userEmail}>
              {userEmail || 'Signed in'}
            </p>
          </div>
          <button onClick={onSignOut} title="Sign out"
            style={{
              fontSize: 10, fontFamily: 'IBM Plex Mono', color: 'var(--text3)',
              background: 'none', border: 'none', cursor: 'pointer', padding: '4px 2px',
              flexShrink: 0, letterSpacing: '0.03em'
            }}
            onMouseEnter={e => e.currentTarget.style.color = 'var(--danger)'}
            onMouseLeave={e => e.currentTarget.style.color = 'var(--text3)'}
          >
            Sign out
          </button>
        </div>
      </aside>
    </>
  )
}

function EmptyState({ onSuggest, docs }: { onSuggest: (q: string) => void; docs: Doc[] }) {
  const suggestions = getDynamicSuggestions(docs)

  return (
    <div className="flex flex-col items-center justify-center h-full gap-8 px-6">
      <div className="text-center">
        <p style={{ fontFamily: 'Instrument Serif', fontSize: 36, fontStyle: 'italic', color: 'var(--text)', lineHeight: 1.1, marginBottom: 10 }}>
          What do you want<br />to know?
        </p>
        <p style={{ fontSize: 14, color: 'var(--text3)', lineHeight: 1.6 }}>
          {docs.length === 0
            ? 'Upload a document to get started.'
            : 'Ask anything from your uploaded documents.'}
        </p>
      </div>

      {suggestions.length > 0 && (
        <div className="grid grid-cols-2 gap-2 w-full max-w-lg">
          {suggestions.map((s, i) => (
            <button key={i} className="suggestion fade-up" style={{ animationDelay: `${i * 0.08}s` }} onClick={() => onSuggest(s.label)}>
              <span className={`intent-pill intent-${s.intent}`} style={{ display: 'inline-block', marginBottom: 6 }}>{s.intent}</span>
              <p style={{ fontSize: 12, color: 'var(--text2)', lineHeight: 1.5, marginTop: 4 }}>{s.label}</p>
            </button>
          ))}
        </div>
      )}

      {docs.length === 0 && (
        <div style={{ borderTop: '1px solid var(--border)', paddingTop: 16, width: '100%', maxWidth: 480 }}>
          <p style={{ fontSize: 11, color: 'var(--text3)', fontFamily: 'IBM Plex Mono', textAlign: 'center', lineHeight: 1.6 }}>
            Upload · Ask · Learn
          </p>
        </div>
      )}
    </div>
  )
}

export default function Home() {
  const [sessionId, setSessionId] = useState('')
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [mode, setMode] = useState<Mode>('student')
  const [loading, setLoading] = useState(false)
  const [docs, setDocs] = useState<Doc[]>([])
  const [uploading, setUploading] = useState(false)
  const [uploadStatus, setUploadStatus] = useState('')
  const [toast, setToast] = useState<Toast | null>(null)
  const [graphOpen, setGraphOpen] = useState(false)
  const [graphData, setGraphData] = useState<GraphData | null>(null)
  const [graphTopic, setGraphTopic] = useState('')
  const [graphLoading, setGraphLoading] = useState(false)
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [selectedDocs, setSelectedDocs] = useState<string[]>([])
  const [sessions, setSessions] = useState<ChatSession[]>([])
  const [userId, setUserId] = useState<string>('')
  const [userEmail, setUserEmail] = useState<string>('')
  const [shareUrl, setShareUrl] = useState<string | null>(null)
  const [sharingId, setSharingId] = useState<string | null>(null)
  const bottomRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const attachRef = useRef<HTMLInputElement>(null)
  const [attachedFile, setAttachedFile] = useState<File | null>(null)
  useEffect(() => {
  getSupabase().auth.getSession().then(({ data: { session } }) => {
    if (!session) { window.location.href = '/login'; return }
    const uid = session.user.id
    setUserId(uid)
    setUserEmail(session.user.email || '')
    // Restore last active session from localStorage, or create fresh one
    const stored = localStorage.getItem(`mv_session_${uid}`)
    if (stored) {
      setSessionId(stored)
    } else {
      createSession().then(s => {
        setSessionId(s.session_id)
        localStorage.setItem(`mv_session_${uid}`, s.session_id)
      })
    }
    loadSessions()
  })
  loadDocs()
}, [])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  const showToast = (msg: string, type: Toast['type'] = 'info') => {
    setToast({ msg, type })
    setTimeout(() => setToast(null), 3000)
  }

  const loadSessions = async () => {
    try {
      const data = await listSessions()
      setSessions(data.sessions || [])
    } catch { /* silent */ }
  }

  const loadDocs = async (showCached = true) => {
    try {
      if (showCached) {
        const cached = localStorage.getItem('mindvault_docs')
        if (cached) setDocs(JSON.parse(cached))
      }
      const data = await getDocuments()
      const docs = data.documents || []
      setDocs(docs)
      localStorage.setItem('mindvault_docs', JSON.stringify(docs))
    } catch { /* silent */ }
  }

  // useCallback with [] deps -- every call inside (setState setters,
  // getGraphTopic, showToast) is stable/state-independent, so an empty
  // dependency array is safe. This keeps the function reference stable
  // across renders, which is REQUIRED for MessageBubble's React.memo above
  // to actually skip re-renders -- a memoized component still re-renders
  // if any of its function props get a new reference every render.
  const handleViewGraph = useCallback(async (topic: string) => {
    setGraphOpen(true)
    setGraphTopic(topic)
    setGraphLoading(true)
    setGraphData(null)
    try {
      const data = await getGraphTopic(topic)
      setGraphData(data)
    } catch {
      showToast('Graph fetch failed', 'error')
    } finally {
      setGraphLoading(false)
    }
  }, [])

  const handleViewFullGraph = async () => {
    setGraphOpen(true)
    setGraphTopic('Knowledge Graph')
    setGraphLoading(true)
    setGraphData(null)
    try {
      const data = await getFullGraph()
      if (!data.nodes || data.nodes.length === 0) {
        showToast('No graph data yet — upload documents to build the graph', 'info')
      }
      setGraphData(data)
    } catch {
      showToast('Graph fetch failed', 'error')
    } finally {
      setGraphLoading(false)
    }
  }

  const handleUpload = async (files: File[]) => {
    if (!files.length) return
    setUploading(true)
    let successCount = 0
    for (let i = 0; i < files.length; i++) {
      const file = files[i]
      try {
        setUploadStatus(files.length > 1 ? `Uploading ${i + 1}/${files.length}: ${file.name.slice(0, 20)}...` : 'Embedding chunks...')
        await uploadDocument(file)
        successCount++
      } catch {
        showToast(`Failed: ${file.name}`, 'error')
      }
    }
    localStorage.removeItem('mindvault_docs')
    await loadDocs(false)
    if (successCount > 0) {
      setUploadStatus(`Done — ${successCount} file${successCount > 1 ? 's' : ''} added`)
      showToast(`${successCount} file${successCount > 1 ? 's' : ''} added to vault`, 'success')
    }
    setTimeout(() => { setUploading(false); setUploadStatus('') }, 2500)
  }

  const handleSend = useCallback(async (text?: string) => {
    const question = (text || input).trim()
    if (!question || loading || !sessionId) return

    const fileToSend = attachedFile
    const userContent = fileToSend ? `📎 ${fileToSend.name}\n${question}` : question
    const assistantId = genId()

    setMessages(prev => [
      ...prev,
      { id: genId(), role: 'user', content: userContent, timestamp: new Date().toISOString() },
      { id: assistantId, role: 'assistant', content: '', timestamp: new Date().toISOString() }
    ])
    setInput('')
    setAttachedFile(null)
    setLoading(true)
    if (textareaRef.current) textareaRef.current.style.height = 'auto'

    if (fileToSend) {
      // Attachment: use non-streaming path (multipart/form-data can't stream easily)
      try {
        const result = await queryWithAttachment(question, sessionId, mode, selectedDocs, fileToSend)
        setMessages(prev => prev.map(m => m.id === assistantId ? {
          ...m,
          content: result.answer || 'No answer returned.',
          sources: result.sources || [],
          chunks: result.chunks || [],
          confidence: result.confidence ?? undefined,
          intent: result.intent,
          related_concepts: result.related_concepts || [],
        } : m))
      } catch (err: any) {
        const msg = err?.response?.data?.detail || err?.message || 'Could not reach MindVault. Please try again.'
        setMessages(prev => prev.map(m => m.id === assistantId ? { ...m, content: msg } : m))
        showToast(msg.length > 60 ? 'Request failed' : msg, 'error')
      } finally { setLoading(false) }
      return
    }

    // Normal query: SSE streaming
    const { getSupabase: _sb } = await import('../lib/supabase')
    const { data: { session: _session } } = await _sb().auth.getSession()
    const token = _session?.access_token || ''

    const cancel = streamQuery(
      question, sessionId, mode, selectedDocs, token,
      (meta) => {
        setMessages(prev => prev.map(m => m.id === assistantId ? {
          ...m,
          sources: meta.sources || [],
          chunks: meta.chunks || [],
          confidence: meta.confidence ?? undefined,
          intent: meta.intent,
        } : m))
        // Auto-name session
        const activeSession = sessions.find(s => s.id === sessionId)
        if (!activeSession || activeSession.name === 'New Chat') {
          renameSession(sessionId, question.slice(0, 45)).then(() => loadSessions()).catch(() => {})
        }
      },
      (tokenText) => {
        setMessages(prev => prev.map(m => m.id === assistantId
          ? { ...m, content: (m.content || '') + tokenText }
          : m))
      },
      (done) => {
        setMessages(prev => prev.map(m => m.id === assistantId ? {
          ...m,
          related_concepts: done.related_concepts || [],
        } : m))
        setLoading(false)
      },
      (err) => {
        setMessages(prev => prev.map(m => m.id === assistantId ? { ...m, content: `Error: ${err}` } : m))
        showToast('Stream failed', 'error')
        setLoading(false)
      }
    )

    // Cleanup on unmount (rare but safe)
    return cancel
  }, [input, loading, mode, sessionId, attachedFile, selectedDocs, sessions])

  const handleExport = async () => {
    try {
      const result = await exportSession(sessionId)
      const blob = new Blob([result.report], { type: 'text/markdown' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url; a.download = `mindvault-${sessionId.slice(0, 8)}.md`; a.click()
      URL.revokeObjectURL(url)
      showToast('Session exported', 'success')
    } catch { showToast('Export failed', 'error') }
  }

  const handleExportPDF = async () => {
    try {
      const blob = await exportSessionPDF(sessionId)
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url; a.download = `mindvault-${sessionId.slice(0, 8)}.pdf`; a.click()
      URL.revokeObjectURL(url)
      showToast('PDF exported', 'success')
    } catch (err: any) {
      console.error('PDF export error:', err)
      showToast(err?.message || 'PDF export failed', 'error')
    }
  }

  const handleNewSession = async () => {
    try {
      const s = await createSession()
      setSessionId(s.session_id)
      if (userId) localStorage.setItem(`mv_session_${userId}`, s.session_id)
      setMessages([])
      setGraphOpen(false)
      await loadSessions()
      showToast('New session started', 'info')
    } catch { showToast('Could not start new session', 'error') }
  }

  const handleSelectSession = async (id: string) => {
    setSessionId(id)
    if (userId) localStorage.setItem(`mv_session_${userId}`, id)
    setGraphOpen(false)
    try {
      const data = await getSessionHistory(id)
      const history = data.history || []
      setMessages(history.map((m: any) => ({
        id: genId(),
        role: m.role,
        content: m.content,
        timestamp: m.timestamp,
        sources: [],
      })))
    } catch { setMessages([]) }
  }

  const handleDeleteSession = async (id: string) => {
    try {
      await deleteSession(id)
      setSessions(prev => prev.filter(s => s.id !== id))
      if (id === sessionId) handleNewSession()
    } catch { /* silent */ }
  }

  const handleShare = async (id: string) => {
    try {
      setSharingId(id)
      const data = await shareSession(id)
      setShareUrl(data.share_url)
      await navigator.clipboard.writeText(data.share_url)
      showToast('Share link copied to clipboard!', 'success')
    } catch { showToast('Could not generate share link', 'error') }
    finally { setSharingId(null) }
  }

  const handleUnshare = async (id: string) => {
    try {
      await unshareSession(id)
      setShareUrl(null)
      showToast('Share link revoked', 'info')
    } catch { /* silent */ }
  }

  const handleToggleDoc = (id: string) => {
    setSelectedDocs(prev =>
      prev.includes(id) ? prev.filter(d => d !== id) : [...prev, id]
    )
  }

  const handleClearSession = async () => {
    try { await clearSession(sessionId) } catch { /* silent */ }
    setMessages([]); showToast('History cleared', 'info')
  }

  const handleSignOut = async () => {
    await signOut()
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend() }
  }

  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar
        docs={docs} onUpload={handleUpload} uploading={uploading} uploadStatus={uploadStatus}
        sessionId={sessionId} msgCount={messages.length}
        onExport={handleExport} onExportPDF={handleExportPDF} onNewSession={handleNewSession} onClearSession={handleClearSession}
        open={sidebarOpen} onClose={() => setSidebarOpen(false)}
        selectedDocs={selectedDocs} onToggleDoc={handleToggleDoc}
        sessions={sessions} onSelectSession={handleSelectSession} onDeleteSession={handleDeleteSession}
        onShare={handleShare} sharingId={sharingId}
        userEmail={userEmail} onSignOut={handleSignOut}
      />

      <main className="flex-1 flex flex-col overflow-hidden min-w-0" style={{ background: 'var(--bg)' }}>
        {/* Top bar */}
        <div className="flex items-center justify-between px-4 py-3"
          style={{ borderBottom: '1px solid var(--border)', background: 'var(--bg)' }}>
          <div className="flex items-center gap-2">
            {/* Hamburger — mobile only */}
            <button className="md:hidden mr-2 p-1"
              style={{ color: 'var(--text3)', fontSize: 18 }}
              onClick={() => setSidebarOpen(true)}>☰</button>
            <span style={{ fontSize: 11, color: 'var(--text3)', fontFamily: 'IBM Plex Mono', textTransform: 'uppercase', letterSpacing: '0.06em' }}>Mode</span>
            <div className="flex gap-1 flex-wrap">
              {MODES.map(m => (
                <button key={m.value} className={`mode-tab ${mode === m.value ? 'active' : ''}`}
                  onClick={() => setMode(m.value)} title={m.desc}>{m.label}</button>
              ))}
            </div>
            <button
              className="mode-tab"
              onClick={handleViewFullGraph}
              style={{ marginLeft: 6, color: 'var(--accent3)', borderColor: 'rgba(126,184,164,0.3)' }}
            >
              ⬡ Graph
            </button>
          </div>
          <div className="hidden md:flex items-center gap-4">
            <div className="flex items-center gap-2">
              <div className="status-dot" />
              <span style={{ fontSize: 11, color: 'var(--text3)', fontFamily: 'IBM Plex Mono' }}>
                {docs.length} doc{docs.length !== 1 ? 's' : ''} in vault
              </span>
            </div>
            <div style={{ width: 1, height: 16, background: 'var(--border)' }} />
            <span style={{ fontSize: 11, color: 'var(--text3)', fontFamily: 'IBM Plex Mono' }}>Powered by MindVault</span>
          </div>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto" style={{ padding: '24px 0' }}>
          {messages.length === 0 ? (
            <EmptyState onSuggest={(q) => handleSend(q)} docs={docs} />
          ) : (
            <div className="flex flex-col gap-6 max-w-3xl mx-auto px-4 md:px-6">
              {messages.map(msg => (
                <MessageBubble key={msg.id} msg={msg} onConceptClick={handleViewGraph} />
              ))}
              {loading && <TypingIndicator />}
              <div ref={bottomRef} />
            </div>
          )}
        </div>

        {/* Input */}
        <div style={{ borderTop: '1px solid var(--border)', padding: '12px 16px', background: 'var(--bg)' }}>
          {attachedFile && (
            <div className="max-w-3xl mx-auto" style={{ marginBottom: 8 }}>
              <span style={{
                display: 'inline-flex', alignItems: 'center', gap: 6,
                fontSize: 11, fontFamily: 'IBM Plex Mono', color: 'var(--accent)',
                background: 'rgba(126,184,164,0.08)', border: '1px solid rgba(126,184,164,0.2)',
                borderRadius: 6, padding: '4px 10px'
              }}>
                📎 {attachedFile.name}
                <button onClick={() => setAttachedFile(null)}
                  style={{ background: 'none', border: 'none', color: 'var(--text3)', cursor: 'pointer', fontSize: 12, padding: 0 }}
                  title="Remove attachment">✕</button>
              </span>
            </div>
          )}
          <div className="flex gap-2 items-end max-w-3xl mx-auto">
            <input ref={attachRef} type="file" className="hidden"
              accept=".pdf,.txt,.md,.docx,.doc,.jpg,.jpeg,.png,.gif,.webp,.bmp,.tiff"
              onChange={(e) => {
                const f = e.target.files?.[0]
                if (f) setAttachedFile(f)
                if (attachRef.current) attachRef.current.value = ''
              }} />
            <button
              onClick={() => attachRef.current?.click()}
              disabled={loading || !sessionId}
              title="Attach a file or image for this question"
              style={{
                flexShrink: 0, height: 46, width: 46, borderRadius: 10,
                border: '1px solid var(--border)', background: 'var(--surface2)',
                color: attachedFile ? 'var(--accent)' : 'var(--text3)',
                fontSize: 16, cursor: 'pointer', display: 'flex',
                alignItems: 'center', justifyContent: 'center'
              }}>
              📎
            </button>
            <div className="flex-1">
              <textarea ref={textareaRef} value={input} onChange={e => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder={attachedFile ? "Ask something about this file..." : "Ask your vault anything..."}
                rows={1} className="vault-input" style={{ minHeight: 46, maxHeight: 140 }}
                onInput={(e) => {
                  const t = e.target as HTMLTextAreaElement
                  t.style.height = 'auto'
                  t.style.height = Math.min(t.scrollHeight, 140) + 'px'
                }} />
            </div>
            <button className="send-btn" onClick={() => handleSend()} disabled={!input.trim() || loading || !sessionId}>
              {loading ? <div className="spinner" style={{ width: 14, height: 14 }} /> : '→'}
            </button>
          </div>
          <p style={{ fontSize: 10, color: 'var(--text3)', fontFamily: 'IBM Plex Mono', textAlign: 'center', marginTop: 8, letterSpacing: '0.03em' }}>
            {attachedFile ? 'Attachment used for this question only · Not saved to your vault' : 'Answers grounded in your documents · Encrypted · Private'}
          </p>
        </div>
      </main>

      <GraphPanel
        open={graphOpen}
        topic={graphTopic}
        data={graphData}
        loading={graphLoading}
        onClose={() => setGraphOpen(false)}
        onNodeClick={(id) => { setGraphOpen(false); handleSend(`explain ${id}`) }}
      />

      {toast && <div className={`toast ${toast.type}`}>{toast.msg}</div>}
    </div>
  )
}