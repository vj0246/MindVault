'use client'

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
} from '../lib/api'

type Mode = 'student' | 'lawyer' | 'developer' | 'default'
type Intent = 'answer' | 'compare' | 'test' | 'summarize'

interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  sources?: string[]
  intent?: Intent
  related_concepts?: { id: string; sources: string[] }[]
  timestamp: string
}

interface Doc {
  filename: string
  chunk_count: number
  uploaded_at: string
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

const SUGGESTIONS = [
  { label: 'What is deadlock?',                    intent: 'answer' },
  { label: 'Compare mutex and semaphore',           intent: 'compare' },
  { label: 'Summarize scheduling algorithms',       intent: 'summarize' },
  { label: 'Generate 5 MCQs on memory management', intent: 'test' },
]

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
          <div className="flex flex-wrap gap-1 mt-3">
            {msg.sources.map((s, i) => <span key={i} className="source-chip">📄 {s}</span>)}
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

function Sidebar({ docs, onUpload, uploading, uploadStatus, sessionId, msgCount, onExport, onNewSession, onClearSession }: {
  docs: Doc[]; onUpload: (f: File) => void; uploading: boolean; uploadStatus: string
  sessionId: string; msgCount: number; onExport: () => void; onNewSession: () => void; onClearSession: () => void
}) {
  const fileRef = useRef<HTMLInputElement>(null)
  const [dragging, setDragging] = useState(false)

  return (
    <aside className="sidebar">
      <div className="flex items-center gap-3 p-5 pb-4">
        <div className="logo-mark">M</div>
        <div>
          <h1 style={{ fontFamily: 'Instrument Serif', fontSize: 18, fontStyle: 'italic', color: 'var(--text)', lineHeight: 1 }}>MindVault</h1>
          <p style={{ fontSize: 10, color: 'var(--text3)', fontFamily: 'IBM Plex Mono', marginTop: 2 }}>knowledge · retrieved</p>
        </div>
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
            onDrop={(e) => { e.preventDefault(); setDragging(false); const f = e.dataTransfer.files[0]; if (f) onUpload(f) }}
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
                <p style={{ fontSize: 12, color: 'var(--text2)' }}>Drop file or click</p>
                <p style={{ fontSize: 10, color: 'var(--text3)', fontFamily: 'IBM Plex Mono' }}>PDF · TXT · MD</p>
              </div>
            )}
          </div>
          <input ref={fileRef} type="file" accept=".pdf,.txt,.md" className="hidden"
            onChange={(e) => e.target.files?.[0] && onUpload(e.target.files[0])} />
        </div>

        <div className="flex-1">
          <p style={{ fontSize: 9, color: 'var(--text3)', fontFamily: 'IBM Plex Mono', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 8 }}>
            Knowledge Base ({docs.length})
          </p>
          {docs.length === 0 ? (
            <p style={{ fontSize: 12, color: 'var(--text3)', textAlign: 'center', padding: '16px 0' }}>No documents yet</p>
          ) : (
            <div className="flex flex-col gap-1">
              {docs.map((doc, i) => (
                <div key={i} className="doc-item fade-up">
                  <span style={{ fontSize: 13, flexShrink: 0 }}>📄</span>
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
  )
}

function EmptyState({ onSuggest }: { onSuggest: (q: string) => void }) {
  return (
    <div className="flex flex-col items-center justify-center h-full gap-8 px-8">
      <div className="text-center">
        <p style={{ fontFamily: 'Instrument Serif', fontSize: 40, fontStyle: 'italic', color: 'var(--text)', lineHeight: 1.1, marginBottom: 10 }}>
          What do you want<br />to know?
        </p>
        <p style={{ fontSize: 14, color: 'var(--text3)', lineHeight: 1.6 }}>
          Ask anything from your uploaded documents.<br />Your knowledge, instantly retrieved.
        </p>
      </div>
      <div className="grid grid-cols-2 gap-2 w-full max-w-lg">
        {SUGGESTIONS.map((s, i) => (
          <button key={i} className="suggestion fade-up" style={{ animationDelay: `${i * 0.08}s` }} onClick={() => onSuggest(s.label)}>
            <span className={`intent-pill intent-${s.intent}`} style={{ display: 'inline-block', marginBottom: 6 }}>{s.intent}</span>
            <p style={{ fontSize: 12, color: 'var(--text2)', lineHeight: 1.5, marginTop: 4 }}>{s.label}</p>
          </button>
        ))}
      </div>
      <div style={{ borderTop: '1px solid var(--border)', paddingTop: 16, width: '100%', maxWidth: 480 }}>
        <p style={{ fontSize: 11, color: 'var(--text3)', fontFamily: 'IBM Plex Mono', textAlign: 'center', lineHeight: 1.6 }}>
          Upload documents first · Then ask anything · Zero data leaves your machine
        </p>
      </div>
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
  const bottomRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    setSessionId(genId())
    loadDocs()
    setTimeout(() => loadDocs(), 1500)
  }, [])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  const showToast = (msg: string, type: Toast['type'] = 'info') => {
    setToast({ msg, type })
    setTimeout(() => setToast(null), 3000)
  }

  const loadDocs = async () => {
    try {
      const data = await getDocuments()
      setDocs(data.documents || [])
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

  const handleUpload = async (file: File) => {
    setUploading(true)
    setUploadStatus('Reading file...')
    try {
      setUploadStatus('Embedding chunks...')
      const result = await uploadDocument(file)
      setUploadStatus(`Done — ${result.chunks} chunks`)
      await loadDocs()
      showToast(`${file.name} added to vault`, 'success')
      setTimeout(() => { setUploading(false); setUploadStatus('') }, 2500)
    } catch {
      setUploadStatus('Upload failed')
      showToast('Upload failed. Check backend.', 'error')
      setTimeout(() => { setUploading(false); setUploadStatus('') }, 2500)
    }
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
      const result = await queryKnowledge(question, sessionId, mode)
      setMessages(prev => [...prev, {
        id: genId(), role: 'assistant',
        content: result.answer || 'No answer returned.',
        sources: result.sources || [],
        intent: result.intent,
        related_concepts: result.related_concepts || [],
        timestamp: new Date().toISOString(),
      }])
    } catch {
      setMessages(prev => [...prev, {
        id: genId(), role: 'assistant',
        content: 'Failed to reach MindVault backend. Make sure the server is running on port 8000.',
        timestamp: new Date().toISOString(),
      }])
      showToast('Backend unreachable', 'error')
    } finally {
      setLoading(false) }
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
      />

      <main className="flex-1 flex flex-col overflow-hidden" style={{ background: 'var(--bg)' }}>
        {/* Top bar */}
        <div className="flex items-center justify-between px-6 py-3"
          style={{ borderBottom: '1px solid var(--border)', background: 'var(--bg)' }}>
          <div className="flex items-center gap-2">
            <span style={{ fontSize: 11, color: 'var(--text3)', fontFamily: 'IBM Plex Mono', textTransform: 'uppercase', letterSpacing: '0.06em' }}>Mode</span>
            <div className="flex gap-1">
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
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <div className="status-dot" />
              <span style={{ fontSize: 11, color: 'var(--text3)', fontFamily: 'IBM Plex Mono' }}>
                {docs.length} doc{docs.length !== 1 ? 's' : ''} in vault
              </span>
            </div>
            <div style={{ width: 1, height: 16, background: 'var(--border)' }} />
            <span style={{ fontSize: 11, color: 'var(--text3)', fontFamily: 'IBM Plex Mono' }}>Local AI · Zero egress</span>
          </div>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto" style={{ padding: '24px 0' }}>
          {messages.length === 0 ? (
            <EmptyState onSuggest={(q) => handleSend(q)} />
          ) : (
            <div className="flex flex-col gap-6 max-w-3xl mx-auto px-6">
              {messages.map(msg => (
                <MessageBubble key={msg.id} msg={msg} onConceptClick={handleViewGraph} />
              ))}
              {loading && <TypingIndicator />}
              <div ref={bottomRef} />
            </div>
          )}
        </div>

        {/* Input */}
        <div style={{ borderTop: '1px solid var(--border)', padding: '16px 24px', background: 'var(--bg)' }}>
          <div className="flex gap-3 items-end max-w-3xl mx-auto">
            <div className="flex-1">
              <textarea ref={textareaRef} value={input} onChange={e => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Ask your vault anything... (Shift+Enter for new line)"
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
          <p style={{ fontSize: 10, color: 'var(--text3)', fontFamily: 'IBM Plex Mono', textAlign: 'center', marginTop: 10, letterSpacing: '0.03em' }}>
            Answers grounded strictly in your documents · Powered by local AI · 100% private
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
