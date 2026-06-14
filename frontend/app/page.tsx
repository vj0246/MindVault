'use client'
export const dynamic = 'force-dynamic'
import { getSupabase, signOut } from '../lib/supabase'
import { useState, useEffect, useRef, useCallback } from 'react'
import ReactMarkdown from 'react-markdown'
import GraphPanel from '../components/GraphPanel'
import {
  uploadDocument,
  queryKnowledge,
  getDocuments,
  exportSession,
  getGraphTopic,
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
}

interface ChunkSource {
  content: string
  similarity: number
  filename: string
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
                <span style={{ fontSize: 10, fontFamily: 'IBM Plex Mono', color: 'var(--accent)', fontWeight: 600 }}>
                  📄 {c.filename}
                </span>
                <span style={{
                  fontSize: 9, fontFamily: 'IBM Plex Mono', color: 'var(--accent3)',
                  background: 'rgba(126,184,164,0.1)', border: '1px solid rgba(126,184,164,0.2)',
                  borderRadius: 3, padding: '1px 6px'
                }}>
                  {Math.round(c.similarity * 100)}% match
                </span>
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

function MessageBubble({ msg, onConceptClick }: {
  msg: Message
  onConceptClick: (c: string) => void
}) {
  if (msg.role === 'user') {
    return (
      <div className="flex justify-end fade-up">
        <div>
          <div className="msg-user">
            <p style={{ fontSize: 14, lineHeight: 1.65, color: 'var(--text)' }}>{msg.content}</p>
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
}

function Sidebar({ docs, onUpload, uploading, uploadStatus, sessionId, msgCount, onExport, onNewSession, onClearSession, open, onClose, selectedDocs, onToggleDoc, sessions, onSelectSession, onDeleteSession }: {
  docs: Doc[]; onUpload: (files: File[]) => void; uploading: boolean; uploadStatus: string
  sessionId: string; msgCount: number; onExport: () => void; onNewSession: () => void
  onClearSession: () => void; open: boolean; onClose: () => void
  selectedDocs: string[]; onToggleDoc: (id: string) => void
  sessions: ChatSession[]; onSelectSession: (id: string) => void; onDeleteSession: (id: string) => void
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
              {sessions.map(s => (
                <div key={s.id} onClick={() => onSelectSession(s.id)} style={{
                  display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                  padding: '6px 10px', borderRadius: 6, cursor: 'pointer', marginBottom: 2,
                  background: s.id === sessionId ? 'rgba(126,184,164,0.1)' : 'transparent',
                  border: s.id === sessionId ? '1px solid rgba(126,184,164,0.2)' : '1px solid transparent',
                }}>
                  <span style={{
                    fontSize: 11, color: s.id === sessionId ? 'var(--accent)' : 'var(--text2)',
                    overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', flex: 1,
                    fontFamily: 'IBM Plex Mono'
                  }}>
                    {s.name.slice(0, 32)}{s.name.length > 32 ? '…' : ''}
                  </span>
                  <button onClick={e => { e.stopPropagation(); onDeleteSession(s.id) }}
                    style={{ fontSize: 10, color: 'var(--text3)', background: 'none',
                             border: 'none', cursor: 'pointer', padding: '0 4px', flexShrink: 0 }}>✕</button>
                </div>
              ))}
              {sessions.length === 0 && (
                <p style={{ fontSize: 10, color: 'var(--text3)', fontFamily: 'IBM Plex Mono' }}>No chats yet</p>
              )}
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
            <button className="action-btn" onClick={onExport}>↓ Export Session</button>
            <button className="action-btn" onClick={onNewSession}>+ New Session</button>
            <button className="action-btn danger" onClick={onClearSession}>✕ Clear History</button>
          </div>
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
  const bottomRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  useEffect(() => {
  getSupabase().auth.getSession().then(({ data: { session } }) => {
    if (!session) { window.location.href = '/login'; return }
    const uid = session.user.id
    setUserId(uid)
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

  const handleViewGraph = async (topic: string) => {
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

    setMessages(prev => [...prev, {
      id: genId(), role: 'user', content: question, timestamp: new Date().toISOString()
    }])
    setInput('')
    setLoading(true)
    if (textareaRef.current) textareaRef.current.style.height = 'auto'

    try {
      const result = await queryKnowledge(question, sessionId, mode, selectedDocs)
      // Auto-name session from first message if still "New Chat"
      const activeSession = sessions.find(s => s.id === sessionId)
      if (!activeSession || activeSession.name === 'New Chat') {
        const autoName = question.slice(0, 45)
        renameSession(sessionId, autoName).then(() => loadSessions()).catch(() => {})
      }
      setMessages(prev => [...prev, {
        id: genId(), role: 'assistant',
        content: result.answer || 'No answer returned.',
        sources: result.sources || [],
        chunks: result.chunks || [],
        confidence: result.confidence ?? undefined,
        intent: result.intent,
        related_concepts: result.related_concepts || [],
        timestamp: new Date().toISOString(),
      }])
    } catch {
      setMessages(prev => [...prev, {
        id: genId(), role: 'assistant',
        content: 'Could not reach MindVault. Please try again.',
        timestamp: new Date().toISOString(),
      }])
      showToast('Backend unreachable', 'error')
    } finally {
      setLoading(false)
    }
  }, [input, loading, mode, sessionId])

  const handleExport = async () => {
    try {
      const result = await exportSession(sessionId)
      const blob = new Blob([result.report], { type: 'text/markdown' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url; a.download = `mindvault-${sessionId.slice(0, 8)}.md`; a.click()
      showToast('Session exported', 'success')
    } catch { showToast('Export failed', 'error') }
  }

  const handleNewSession = () => {
    setSessionId(genId()); setMessages([]); setGraphOpen(false)
    showToast('New session started', 'info')
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

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend() }
  }

  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar
        docs={docs} onUpload={handleUpload} uploading={uploading} uploadStatus={uploadStatus}
        sessionId={sessionId} msgCount={messages.length}
        onExport={handleExport} onNewSession={handleNewSession} onClearSession={handleClearSession}
        open={sidebarOpen} onClose={() => setSidebarOpen(false)}
        selectedDocs={selectedDocs} onToggleDoc={handleToggleDoc}
        sessions={sessions} onSelectSession={handleSelectSession} onDeleteSession={handleDeleteSession}
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
              onClick={() => handleViewGraph('process')}
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
          <div className="flex gap-2 items-end max-w-3xl mx-auto">
            <div className="flex-1">
              <textarea ref={textareaRef} value={input} onChange={e => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Ask your vault anything..."
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
            Answers grounded in your documents · Encrypted · Private
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