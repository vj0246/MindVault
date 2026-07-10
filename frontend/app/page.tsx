'use client'
export const dynamic = 'force-dynamic'
import { getSupabase, signOut } from '../lib/supabase'
import { useState, useEffect, useRef, useCallback } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import GraphPanel from '../components/GraphPanel'
import {
  uploadDocument,
  queryKnowledge,
  streamQuery,
  streamQueryWithAttachment,
  shareSession,
  unshareSession,
  getDocuments,
  setDocumentFolder,
  deleteDocument,
  searchMessages,
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
  getPreferences,
  savePreferences,
  listMemoryNotes,
  addMemoryNote,
  deleteMemoryNote,
  type Preferences,
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
  answer_type?: 'grounded' | 'general_knowledge'
  unverified?: boolean
  tokens?: { message: number | null; daily_used: number; daily_pct: number }
}

interface Doc {
  id: string
  filename: string
  chunk_count: number
  uploaded_at: string
  folder?: string | null
}

type PreferencesState = Preferences

const DEFAULT_PREFERENCES: PreferencesState = {
  name: '', tone: 'Neutral', priorities: [], system_prompt: '', theme: 'Light',
}

interface MemoryNote {
  id: string
  content: string
  created_at: string
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
    <div className="flex items-center gap-3 fade-up px-1 msg-ai-spine">
      <div className="flex items-center gap-1.5 py-2">
        <div className="dot" /><div className="dot" /><div className="dot" />
        <span className="eyebrow" style={{ marginLeft: 6 }}>retrieving from vault</span>
      </div>
    </div>
  )
}

function IntentPill({ intent }: { intent: string }) {
  return <span className={`intent-pill intent-${intent}`}>{intent}</span>
}

function ConfidenceBadge({ score }: { score: number }) {
  const pct = Math.round(score * 100)
  const color = score >= 0.7 ? '#0f9d78' : score >= 0.4 ? '#d0870f' : '#dc3545'
  const label = score >= 0.7 ? 'High' : score >= 0.4 ? 'Medium' : 'Low'
  return (
    <span className="chip" style={{ border: `1px solid ${color}33`, color, background: `${color}11`, marginLeft: 6 }}>
      {label} {pct}%
    </span>
  )
}

function SourcePanel({ chunks, jumpTarget }: { chunks: ChunkSource[]; jumpTarget?: { filename: string; nonce: number } | null }) {
  const [open, setOpen] = useState(false)
  const [copied, setCopied] = useState<number | null>(null)
  const [highlighted, setHighlighted] = useState<number | null>(null)
  const rowRefs = useRef<(HTMLDivElement | null)[]>([])
  const copyChunk = (text: string, i: number) => {
    navigator.clipboard.writeText(text).then(() => { setCopied(i); setTimeout(() => setCopied(null), 1500) })
  }

  useEffect(() => {
    if (!jumpTarget) return
    const idx = chunks.findIndex(c => c.filename === jumpTarget.filename)
    if (idx < 0) return
    setOpen(true)
    setHighlighted(idx)
    setTimeout(() => rowRefs.current[idx]?.scrollIntoView({ behavior: 'smooth', block: 'center' }), 60)
    const t = setTimeout(() => setHighlighted(null), 2000)
    return () => clearTimeout(t)
  }, [jumpTarget, chunks])

  return (
    <div style={{ display: 'inline-block' }}>
      <button onClick={() => setOpen(!open)} className="chip tap-target" title="View source chunks"
        style={{ background: 'var(--surface2)', border: '1px solid var(--border)', color: 'var(--text3)', cursor: 'pointer', marginLeft: 4 }}>
        {open ? '▲ sources' : 'ℹ sources'}
      </button>
      {open && (
        <div style={{ marginTop: 8, background: 'var(--surface2)', border: '1px solid var(--border)', borderRadius: 'var(--r-md)', overflow: 'hidden' }}>
          {chunks.map((c, i) => (
            <div key={i} ref={el => { rowRefs.current[i] = el }} style={{
              padding: '10px 12px',
              borderBottom: i < chunks.length - 1 ? '1px solid var(--border)' : 'none',
              background: highlighted === i ? 'var(--glow)' : undefined,
              transition: 'background-color 0.3s var(--ease)'
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                  <span className="chip" style={{ color: 'var(--accent)', fontWeight: 600, padding: 0 }}>📄 {c.filename}</span>
                  {(c.page_number != null && c.page_number > 0) && (
                    <span className="chip" style={{ color: 'var(--text3)', background: 'var(--surface)', border: '1px solid var(--border)' }}>
                      p.{(c.page_number ?? 0) + 1}
                    </span>
                  )}
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                  <span className="chip" style={{ color: 'var(--accent3)', background: 'rgba(61,127,104,0.1)', border: '1px solid rgba(61,127,104,0.2)' }}>
                    {Math.round(c.similarity * 100)}% match
                  </span>
                  <button onClick={() => copyChunk(c.content, i)} title="Copy chunk text — paste in PDF viewer to find passage"
                    className="tap-target"
                    style={{ fontSize: 11, cursor: 'pointer', background: 'none', border: 'none', color: copied === i ? 'var(--accent)' : 'var(--text3)', padding: 0 }}>
                    {copied === i ? '✓' : '⎘'}
                  </button>
                </div>
              </div>
              <p style={{ fontSize: 11, color: 'var(--text3)', lineHeight: 1.55, fontFamily: 'var(--mono)' }}>
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
  const [jumpTarget, setJumpTarget] = useState<{ filename: string; nonce: number } | null>(null)

  if (msg.role === 'user') {
    return (
      <div className="flex justify-end fade-up" style={{ width: '100%' }}>
        <div style={{ maxWidth: '75%', display: 'flex', flexDirection: 'column', alignItems: 'flex-end' }}>
          <div className="msg-user" style={{ textAlign: 'left', display: 'inline-block', maxWidth: '100%' }}>
            <p style={{ fontSize: 14, lineHeight: 1.65, color: 'var(--text)', whiteSpace: 'pre-wrap', wordBreak: 'break-word', margin: 0 }}>{msg.content}</p>
          </div>
          <p className="eyebrow" style={{ textAlign: 'right', marginTop: 4, textTransform: 'none', letterSpacing: 0 }}>
            {timeStr(msg.timestamp)}
          </p>
        </div>
      </div>
    )
  }

  return (
    <div className="fade-up msg-ai-spine">
      <div className="flex items-center gap-2 mb-2">
        <span className="eyebrow" style={{ fontFamily: 'var(--mono)', textTransform: 'none', letterSpacing: 0, color: 'var(--text3)' }}>MindVault</span>
        {msg.intent && <IntentPill intent={msg.intent} />}
        {msg.answer_type === 'general_knowledge' && (
          <span className="eyebrow" style={{ textTransform: 'none', letterSpacing: 0, color: 'var(--accent2)', border: '1px solid var(--accent2)', borderRadius: 999, padding: '1px 8px' }}>
            General knowledge — not from your documents
          </span>
        )}
      </div>
      <div className="md"><ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.content.replace(/^\[General knowledge\]\s*\n*/, '')}</ReactMarkdown></div>
      {msg.unverified && (
        <p className="eyebrow" style={{ marginTop: 6, textTransform: 'none', letterSpacing: 0, color: 'var(--danger)' }}>
          ⚠ This answer may not be fully supported by your uploaded documents.
        </p>
      )}
      {msg.sources && msg.sources.length > 0 && msg.sources[0] !== 'conversation history' && (
        <div className="mt-3 flex flex-wrap gap-1 items-center">
          {msg.sources.map((s, i) => {
            const hasChunk = msg.chunks?.some(c => c.filename === s)
            return (
              <span key={i} className="source-chip"
                onClick={hasChunk ? () => setJumpTarget({ filename: s, nonce: Date.now() }) : undefined}
                title={hasChunk ? 'Click to view the exact retrieved snippet' : undefined}
                style={hasChunk ? { cursor: 'pointer' } : undefined}>
                📄 {s}
              </span>
            )
          })}
          {msg.chunks && msg.chunks.length > 0 && <SourcePanel chunks={msg.chunks} jumpTarget={jumpTarget} />}
        </div>
      )}
      {msg.related_concepts && msg.related_concepts.length > 0 && (
        <div className="mt-3">
          <p className="eyebrow" style={{ marginBottom: 6 }}>Related concepts — click to explore graph</p>
          <div className="flex flex-wrap gap-1.5">
            {msg.related_concepts.slice(0, 6).map((c, i) => (
              <button key={i} className="concept-tag" onClick={() => onConceptClick(c.id)}>⬡ {c.id}</button>
            ))}
          </div>
        </div>
      )}
      <p className="eyebrow" style={{ marginTop: 6, textTransform: 'none', letterSpacing: 0 }}>
        {timeStr(msg.timestamp)}
        {msg.tokens?.message != null && <span> · {msg.tokens.message} tokens</span>}
      </p>
    </div>
  )
}

function Sidebar({ docs, onUpload, uploading, uploadStatus, sessionId, msgCount, onExport, onExportPDF, onNewSession, onClearSession, open, onClose, width, onWidthChange, selectedDocs, onToggleDoc, sessions, onSelectSession, onDeleteSession, onShare, sharingId, onEditPreferences, onOpenMemory, dailyTokenPct, onOpenDocManager }: {
  docs: Doc[]; onUpload: (files: File[]) => void; uploading: boolean; uploadStatus: string
  sessionId: string; msgCount: number; onExport: () => void; onExportPDF: () => void; onNewSession: () => void
  onClearSession: () => void; open: boolean; onClose: () => void
  width: number; onWidthChange: (w: number) => void
  selectedDocs: string[]; onToggleDoc: (id: string) => void
  sessions: ChatSession[]; onSelectSession: (id: string) => void; onDeleteSession: (id: string) => void
  onShare: (id: string) => void; sharingId: string | null; onEditPreferences: () => void; onOpenMemory: () => void
  dailyTokenPct: number | null; onOpenDocManager: () => void
}) {
  const fileRef = useRef<HTMLInputElement>(null)
  const [dragging, setDragging] = useState(false)
  const [resizing, setResizing] = useState(false)
  const resizingRef = useRef(false)
  const [searchQuery, setSearchQuery] = useState('')
  const [searchResults, setSearchResults] = useState<{ session_id: string; role: string; content: string; timestamp: string }[] | null>(null)
  const [searching, setSearching] = useState(false)
  const [docFilter, setDocFilter] = useState('')
  const [sidebarTab, setSidebarTab] = useState<'docs' | 'chats'>('docs')

  const runSearch = async (q: string) => {
    if (!q.trim()) { setSearchResults(null); return }
    setSearching(true)
    try {
      const data = await searchMessages(q.trim())
      setSearchResults(data.results || [])
    } catch {
      setSearchResults([])
    } finally {
      setSearching(false)
    }
  }

  const sessionName = (id: string) => sessions.find(s => s.id === id)?.name || 'Chat'

  const matchesFilter = (d: Doc) => d.filename.toLowerCase().includes(docFilter.trim().toLowerCase())
  const foldersMap = new Map<string, Doc[]>()
  for (const d of docs) {
    if (!matchesFilter(d)) continue
    const key = d.folder || 'Uncategorized'
    if (!foldersMap.has(key)) foldersMap.set(key, [])
    foldersMap.get(key)!.push(d)
  }
  const filteredFolderNames = Array.from(foldersMap.keys()).sort((a, b) =>
    a === 'Uncategorized' ? 1 : b === 'Uncategorized' ? -1 : a.localeCompare(b))

  useEffect(() => {
    const onMove = (e: MouseEvent) => {
      if (!resizingRef.current) return
      onWidthChange(Math.min(420, Math.max(220, e.clientX)))
    }
    const onUp = () => {
      if (!resizingRef.current) return
      resizingRef.current = false
      setResizing(false)
      document.body.style.cursor = ''
      document.body.style.userSelect = ''
    }
    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseup', onUp)
    return () => { window.removeEventListener('mousemove', onMove); window.removeEventListener('mouseup', onUp) }
  }, [onWidthChange])

  const startResize = (e: React.MouseEvent) => {
    e.preventDefault()
    resizingRef.current = true
    setResizing(true)
    document.body.style.cursor = 'col-resize'
    document.body.style.userSelect = 'none'
  }

  return (
    <>
      <aside className="sidebar" style={{
        position: 'relative', flexShrink: 0, height: '100vh',
        width: open ? width : 0, minWidth: open ? width : 0,
        transition: resizing ? 'none' : 'width 0.18s var(--ease), min-width 0.18s var(--ease)',
        overflow: 'hidden',
      }}>
        <div style={{ width, height: '100%', display: 'flex', flexDirection: 'column' }}>
        <div className="flex items-center justify-between gap-3 p-5 pb-4">
          <div className="flex items-center gap-3">
            <div className="logo-mark">M</div>
            <div>
              <h1 style={{ fontFamily: 'var(--serif)', fontSize: 19, fontStyle: 'italic', color: 'var(--text)', lineHeight: 1 }}>MindVault</h1>
              <p className="eyebrow" style={{ marginTop: 2, textTransform: 'none', letterSpacing: 0 }}>knowledge · retrieved</p>
            </div>
          </div>
          <button className="icon-btn" onClick={onClose} style={{ color: 'var(--text3)', fontSize: 18 }}>✕</button>
        </div>
        {dailyTokenPct != null && (
          <div className="px-5" style={{ marginTop: -6, marginBottom: 8 }} title="Share of today's fairness token budget used">
            <div className="flex items-center justify-between" style={{ marginBottom: 3 }}>
              <span className="eyebrow" style={{ textTransform: 'none', letterSpacing: 0 }}>Daily usage</span>
              <span className="eyebrow" style={{ textTransform: 'none', letterSpacing: 0 }}>{dailyTokenPct}%</span>
            </div>
            <div style={{ height: 3, borderRadius: 999, background: 'var(--surface2)', overflow: 'hidden' }}>
              <div style={{
                height: '100%', width: `${Math.min(100, dailyTokenPct)}%`,
                background: dailyTokenPct >= 90 ? 'var(--danger)' : dailyTokenPct >= 60 ? 'var(--accent2)' : 'var(--accent)',
                transition: 'width 0.3s var(--ease)'
              }} />
            </div>
          </div>
        )}
        <div className="divider" />
        <div className="px-4 pt-3 flex-shrink-0">
          <div className="segmented segmented-full">
            <button className={`segmented-item ${sidebarTab === 'docs' ? 'active' : ''}`} onClick={() => setSidebarTab('docs')}>
              Documents ({docs.length})
            </button>
            <button className={`segmented-item ${sidebarTab === 'chats' ? 'active' : ''}`} onClick={() => setSidebarTab('chats')}>
              Chats ({sessions.length})
            </button>
          </div>
        </div>
        <div className="flex flex-col gap-3 p-4 flex-1 min-h-0">
          {sidebarTab === 'chats' ? (
            <>
              <div className="flex-shrink-0" style={{ position: 'relative' }}>
                <input
                  type="text"
                  value={searchQuery}
                  placeholder="Search past conversations…"
                  className="vault-input"
                  style={{ minHeight: 32, fontSize: 12, border: '1px solid var(--border2)', background: 'var(--bg)' }}
                  onChange={e => { setSearchQuery(e.target.value); runSearch(e.target.value) }}
                  onKeyDown={e => { if (e.key === 'Escape') { setSearchQuery(''); setSearchResults(null) } }}
                />
                {searchQuery && (searching || (searchResults && searchResults.length > 0)) && (
                  <div style={{
                    position: 'absolute', top: '100%', left: 0, right: 0, zIndex: 20, marginTop: 4,
                    background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 'var(--r-md)',
                    boxShadow: 'var(--shadow-md)', maxHeight: 240, overflowY: 'auto'
                  }}>
                    {searching && <p style={{ fontSize: 11, color: 'var(--text3)', padding: '8px 10px' }}>Searching…</p>}
                    {!searching && searchResults?.map((r, i) => (
                      <div key={i} className="tap-target"
                        onClick={() => { onSelectSession(r.session_id); setSearchQuery(''); setSearchResults(null) }}
                        style={{ padding: '8px 10px', cursor: 'pointer', borderBottom: i < searchResults.length - 1 ? '1px solid var(--border)' : 'none' }}>
                        <p style={{ fontSize: 11, color: 'var(--accent)', fontFamily: 'var(--mono)' }}>{sessionName(r.session_id)}</p>
                        <p style={{ fontSize: 11, color: 'var(--text2)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                          {r.content.slice(0, 80)}
                        </p>
                      </div>
                    ))}
                    {!searching && searchResults && searchResults.length === 0 && (
                      <p style={{ fontSize: 11, color: 'var(--text3)', padding: '8px 10px' }}>No matches</p>
                    )}
                  </div>
                )}
              </div>

              <div className="flex flex-col flex-1 min-h-0">
                <div className="flex justify-between items-center mb-2 flex-shrink-0">
                  <p className="eyebrow">All chats</p>
                  <button onClick={onNewSession} className="chip" style={{
                    color: 'var(--accent)', background: 'var(--glow)', border: '1px solid rgba(79,70,229,0.2)', cursor: 'pointer'
                  }}>+ New</button>
                </div>
                <div className="flex flex-col gap-0.5 overflow-y-auto">
                  {sessions.map((s) => (
                    <div key={s.id} onClick={() => onSelectSession(s.id)} className={`ledger-row ${s.id === sessionId ? 'active' : ''}`}>
                      <span className="ledger-tick">{String(s.number ?? '·').padStart(2, '0')}</span>
                      <span style={{
                        fontSize: 11, color: s.id === sessionId ? 'var(--accent)' : 'var(--text2)',
                        overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', flex: 1, fontFamily: 'var(--mono)'
                      }}>
                        {s.name.slice(0, 26)}{s.name.length > 26 ? '…' : ''}
                      </span>
                      <div className="flex gap-0.5 flex-shrink-0">
                        <button onClick={e => { e.stopPropagation(); onShare(s.id) }} disabled={sharingId === s.id}
                          title="Copy share link" className="tap-target"
                          style={{ fontSize: 11, color: 'var(--text3)', background: 'none', border: 'none', cursor: 'pointer', padding: '0 4px' }}>
                          {sharingId === s.id ? '…' : '🔗'}
                        </button>
                        <button onClick={e => { e.stopPropagation(); onDeleteSession(s.id) }} className="tap-target"
                          style={{ fontSize: 11, color: 'var(--text3)', background: 'none', border: 'none', cursor: 'pointer', padding: '0 4px' }}>✕</button>
                      </div>
                    </div>
                  ))}
                  {sessions.length === 0 && <p style={{ fontSize: 10, color: 'var(--text3)', fontFamily: 'var(--mono)' }}>No chats yet</p>}
                </div>
              </div>
            </>
          ) : (
          <div className="flex flex-col flex-1 min-h-0 gap-3">
          <div className="flex-shrink-0">
            <p className="eyebrow" style={{ marginBottom: 8 }}>Upload document</p>
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
                  <p style={{ fontSize: 11, color: 'var(--accent)', fontFamily: 'var(--mono)' }}>{uploadStatus}</p>
                </div>
              ) : (
                <div className="flex items-center gap-2 py-1">
                  <span style={{ fontSize: 18 }}>📄</span>
                  <div className="min-w-0">
                    <p style={{ fontSize: 12, color: 'var(--text2)' }}>Drop files or click</p>
                    <p style={{ fontSize: 10, color: 'var(--text3)', fontFamily: 'var(--mono)' }}>PDF · TXT · MD · DOCX · Images</p>
                  </div>
                </div>
              )}
            </div>
            <input ref={fileRef} type="file" accept=".pdf,.txt,.md,.docx,.doc,.jpg,.jpeg,.png,.gif,.webp" multiple className="hidden"
              onChange={(e) => { const files = Array.from(e.target.files || []); if (files.length) onUpload(files) }} />
          </div>

          <div className="flex flex-col flex-1 min-h-0">
            <div className="flex justify-between items-center mb-2 flex-shrink-0">
              <p className="eyebrow">Knowledge base ({docs.length})</p>
              <button className="chip" onClick={onOpenDocManager} style={{
                color: 'var(--accent)', background: 'var(--glow)', border: '1px solid rgba(79,70,229,0.2)', cursor: 'pointer'
              }}>Manage</button>
            </div>
            {docs.length > 4 && (
              <input
                type="text"
                value={docFilter}
                placeholder="Find a document…"
                className="vault-input"
                style={{ minHeight: 30, fontSize: 12, border: '1px solid var(--border2)', background: 'var(--bg)', marginBottom: 8, flexShrink: 0 }}
                onChange={e => setDocFilter(e.target.value)}
              />
            )}
            {docs.length === 0 ? (
              <p style={{ fontSize: 12, color: 'var(--text3)', textAlign: 'center', padding: '16px 0' }}>No documents yet</p>
            ) : filteredFolderNames.length === 0 ? (
              <p style={{ fontSize: 12, color: 'var(--text3)', textAlign: 'center', padding: '16px 0' }}>No matches</p>
            ) : (
              <div className="flex flex-col gap-2 overflow-y-auto">
                {filteredFolderNames.map(folderName => (
                  <div key={folderName}>
                    <p className="eyebrow" style={{ marginBottom: 4, color: 'var(--text3)' }}>
                      {folderName} ({foldersMap.get(folderName)!.length})
                    </p>
                    <div className="flex flex-col gap-1">
                      {foldersMap.get(folderName)!.map((doc, i) => (
                        <div key={doc.id} className={`doc-item tilt-card fade-up ${selectedDocs.includes(doc.id) ? 'active' : ''}`}
                          style={{ animationDelay: `${i * 0.04}s` }}
                          onClick={() => onToggleDoc(doc.id)}
                          title={doc.filename}>
                          <input type="checkbox" checked={selectedDocs.includes(doc.id)} onChange={() => onToggleDoc(doc.id)}
                            onClick={e => e.stopPropagation()} style={{ flexShrink: 0, accentColor: 'var(--accent)', cursor: 'pointer' }} />
                          <div className="flex-1 min-w-0">
                            <p style={{ fontSize: 12.5, color: 'var(--text)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{doc.filename}</p>
                            <p style={{ fontSize: 10, color: 'var(--text3)', fontFamily: 'var(--mono)' }}>
                              {doc.chunk_count === -1
                                ? <span style={{ color: 'var(--accent)' }}>⟳ Processing…</span>
                                : `${doc.chunk_count} chunks`} · {new Date(doc.uploaded_at).toLocaleDateString()}
                            </p>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            )}
            {selectedDocs.length > 0 && (
              <p style={{ fontSize: 10, fontFamily: 'var(--mono)', color: 'var(--accent)', marginTop: 6, flexShrink: 0 }}>
                {selectedDocs.length} selected for next question
              </p>
            )}
          </div>
          </div>
          )}

          <div className="flex flex-col gap-2 flex-shrink-0">
            <div className="flex gap-2">
              <button className="action-btn" style={{ flex: 1 }} onClick={onExport}>↓ MD</button>
              <button className="action-btn" style={{ flex: 1 }} onClick={onExportPDF}>↓ PDF</button>
            </div>
            <button className="action-btn" onClick={onNewSession}>+ New Session</button>
            <div className="flex gap-2">
              <button className="action-btn" style={{ flex: 1 }} onClick={onEditPreferences}>⚙ Preferences</button>
              <button className="action-btn" style={{ flex: 1 }} onClick={onOpenMemory}>◆ Memory</button>
            </div>
            <button className="action-btn danger" onClick={onClearSession}>✕ Clear History</button>
          </div>
        </div>
        </div>
        <div
          onMouseDown={startResize}
          title="Drag to resize"
          style={{ position: 'absolute', top: 0, right: 0, width: 6, height: '100%', cursor: 'col-resize', zIndex: 10 }}
        />
      </aside>
    </>
  )
}

function EmptyState({ onSuggest, docs }: { onSuggest: (q: string) => void; docs: Doc[] }) {
  const suggestions = getDynamicSuggestions(docs)

  return (
    <div className="flex flex-col items-center justify-center h-full gap-8 px-6">
      <div className="text-center">
        <p style={{ fontFamily: 'var(--serif)', fontSize: 38, fontStyle: 'italic', lineHeight: 1.1, marginBottom: 10, color: 'var(--text)' }}>
          What do you want<br />to know?
        </p>
        <p style={{ fontSize: 15, color: 'var(--text2)', lineHeight: 1.6 }}>
          {docs.length === 0 ? 'Upload a document to get started.' : 'Ask anything from your uploaded documents.'}
        </p>
      </div>

      {suggestions.length > 0 && (
        <div className="grid grid-cols-2 gap-2 w-full max-w-lg">
          {suggestions.map((s, i) => (
            <button key={i} className="suggestion tilt-card fade-up" style={{ animationDelay: `${i * 0.06}s` }} onClick={() => onSuggest(s.label)}>
              <span className={`intent-pill intent-${s.intent}`} style={{ display: 'inline-block', marginBottom: 6 }}>{s.intent}</span>
              <p style={{ fontSize: 12, color: 'var(--text2)', lineHeight: 1.5, marginTop: 4 }}>{s.label}</p>
            </button>
          ))}
        </div>
      )}

      {docs.length === 0 && (
        <div style={{ borderTop: '1px solid var(--border)', paddingTop: 16, width: '100%', maxWidth: 480 }}>
          <p className="eyebrow" style={{ textAlign: 'center', letterSpacing: '0.1em' }}>Upload · Ask · Learn</p>
        </div>
      )}
    </div>
  )
}

const PREF_TONES = ['Friendly', 'Neutral', 'Formal'] as const
const PREF_PRIORITIES = ['Accuracy', 'Conciseness', 'Step-by-step', 'Examples', 'Speed'] as const
const PREF_THEMES = ['Light', 'Dark'] as const

function PreferencesModal({ initial, memoryNotes, onSave, onAddMemoryNote, onDeleteMemoryNote, onClose, dismissable }: {
  initial: PreferencesState
  memoryNotes: MemoryNote[]
  onSave: (prefs: PreferencesState) => void
  onAddMemoryNote: (content: string) => void
  onDeleteMemoryNote: (id: string) => void
  onClose: () => void
  dismissable: boolean
}) {
  const [name, setName] = useState(initial.name)
  const [tone, setTone] = useState(initial.tone)
  const [priorities, setPriorities] = useState<string[]>(initial.priorities)
  const [systemPrompt, setSystemPrompt] = useState(initial.system_prompt)
  const [theme, setTheme] = useState(initial.theme)
  const [memoryDraft, setMemoryDraft] = useState('')

  const togglePriority = (opt: string) => {
    setPriorities(prev => prev.includes(opt) ? prev.filter(p => p !== opt) : [...prev, opt])
  }

  const submitMemory = () => {
    if (memoryDraft.trim()) { onAddMemoryNote(memoryDraft.trim()); setMemoryDraft('') }
  }

  const Section = ({ label, hint, children }: { label: string; hint?: string; children: React.ReactNode }) => (
    <div style={{ marginBottom: 20 }}>
      <p className="eyebrow" style={{ marginBottom: 8, fontSize: 10.5 }}>{label}</p>
      {hint && <p style={{ fontSize: 11.5, color: 'var(--text3)', marginBottom: 8, marginTop: -4 }}>{hint}</p>}
      {children}
    </div>
  )

  const Picker = ({ options, value, onPick }: { options: readonly string[]; value: string; onPick: (v: string) => void }) => (
    <div className="flex gap-2 flex-wrap">
      {options.map(opt => (
        <button key={opt} type="button" onClick={() => onPick(opt)}
          className={`pref-chip ${value === opt ? 'active' : ''}`}>{opt}</button>
      ))}
    </div>
  )

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center p-4" style={{ background: 'rgba(23,26,36,0.55)' }}>
      <div style={{
        background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 'var(--r-lg)',
        boxShadow: 'var(--shadow-lift)', padding: 28, width: '100%', maxWidth: 480, maxHeight: '85vh', overflowY: 'auto',
      }}>
        <p style={{ fontFamily: 'var(--serif)', fontSize: 23, fontStyle: 'italic', color: 'var(--text)', marginBottom: 4 }}>
          How should MindVault talk to you?
        </p>
        <p style={{ fontSize: 12.5, color: 'var(--text2)', marginBottom: 22, lineHeight: 1.5 }}>
          A couple of quick picks — change any of this anytime from Preferences in the sidebar.
        </p>

        <Section label="Your name">
          <input className="pref-field" value={name} onChange={e => setName(e.target.value)} placeholder="What should we call you?" />
        </Section>

        <Section label="Style">
          <Picker options={PREF_TONES} value={tone} onPick={setTone} />
        </Section>

        <Section label="Priorities" hint="Pick any that matter to you">
          <div className="flex gap-2 flex-wrap">
            {PREF_PRIORITIES.map(opt => (
              <button key={opt} type="button" onClick={() => togglePriority(opt)}
                className={`pref-chip ${priorities.includes(opt) ? 'active' : ''}`}>{opt}</button>
            ))}
          </div>
        </Section>

        <Section label="Custom instructions" hint="Optional — anything else MindVault should know">
          <textarea className="pref-field" value={systemPrompt} onChange={e => setSystemPrompt(e.target.value)}
            placeholder="e.g. I'm prepping for the bar exam — flag anything that's commonly tested."
            rows={3} style={{ resize: 'vertical' }} />
        </Section>

        <Section label="Theme">
          <Picker options={PREF_THEMES} value={theme} onPick={setTheme} />
        </Section>

        <Section label="Long-term memory" hint="Facts MindVault remembers across every chat — add as many as you like">
          <div className="flex gap-2" style={{ marginBottom: 10 }}>
            <input className="pref-field" value={memoryDraft} onChange={e => setMemoryDraft(e.target.value)}
              onKeyDown={e => { if (e.key === 'Enter') { e.preventDefault(); submitMemory() } }}
              placeholder="e.g. I'm studying for the CFA exam" />
            <button className="action-btn" style={{ width: 'auto', padding: '0 16px', flexShrink: 0 }} onClick={submitMemory}>Add</button>
          </div>
          {memoryNotes.length > 0 && (
            <div className="flex flex-col gap-1.5" style={{ maxHeight: 140, overflowY: 'auto' }}>
              {memoryNotes.map(n => (
                <div key={n.id} className="doc-item" style={{ cursor: 'default' }}>
                  <p style={{ fontSize: 12.5, color: 'var(--text)', flex: 1 }}>{n.content}</p>
                  <button onClick={() => onDeleteMemoryNote(n.id)} className="tap-target"
                    style={{ fontSize: 12, color: 'var(--text3)', background: 'none', border: 'none', cursor: 'pointer', padding: '0 4px' }}>✕</button>
                </div>
              ))}
            </div>
          )}
        </Section>

        <div className="flex gap-2" style={{ marginTop: 8 }}>
          {dismissable && (
            <button className="action-btn" style={{ flex: 1 }} onClick={onClose}>Skip</button>
          )}
          <button className="btn-primary" style={{ flex: 1 }}
            onClick={() => onSave({ name, tone, priorities, system_prompt: systemPrompt, theme })}>Save preferences</button>
        </div>
      </div>
    </div>
  )
}

function MemoryModal({ notes, onAdd, onDelete, onClose }: {
  notes: { id: string; content: string }[]
  onAdd: (content: string) => void
  onDelete: (id: string) => void
  onClose: () => void
}) {
  const [draft, setDraft] = useState('')
  const submit = () => { if (draft.trim()) { onAdd(draft.trim()); setDraft('') } }

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center p-4" style={{ background: 'rgba(23,26,36,0.45)' }} onClick={onClose}>
      <div onClick={e => e.stopPropagation()} style={{
        background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 'var(--r-lg)',
        boxShadow: 'var(--shadow-lift)', padding: 28, width: '100%', maxWidth: 460, maxHeight: '80vh', display: 'flex', flexDirection: 'column',
      }}>
        <p style={{ fontFamily: 'var(--serif)', fontSize: 22, fontStyle: 'italic', color: 'var(--text)', marginBottom: 4 }}>Long-term memory</p>
        <p style={{ fontSize: 12.5, color: 'var(--text3)', marginBottom: 16 }}>
          Facts you add here are remembered across every session and chat.
        </p>

        <div className="flex gap-2" style={{ marginBottom: 14 }}>
          <input className="pref-field" value={draft} onChange={e => setDraft(e.target.value)} onKeyDown={e => { if (e.key === 'Enter') submit() }}
            placeholder="e.g. I'm studying for the CFA exam" style={{ flex: 1 }} />
          <button className="action-btn" style={{ width: 'auto', padding: '0 16px', flexShrink: 0 }} onClick={submit}>Add</button>
        </div>

        <div className="flex flex-col gap-1.5" style={{ overflowY: 'auto', flex: 1 }}>
          {notes.length === 0 && <p style={{ fontSize: 12, color: 'var(--text3)', textAlign: 'center', padding: '16px 0' }}>No memory notes yet.</p>}
          {notes.map(n => (
            <div key={n.id} className="doc-item" style={{ cursor: 'default' }}>
              <p style={{ fontSize: 12.5, color: 'var(--text)', flex: 1 }}>{n.content}</p>
              <button onClick={() => onDelete(n.id)} className="tap-target"
                style={{ fontSize: 12, color: 'var(--text3)', background: 'none', border: 'none', cursor: 'pointer', padding: '0 4px' }}>✕</button>
            </div>
          ))}
        </div>

        <button className="action-btn" style={{ marginTop: 14 }} onClick={onClose}>Close</button>
      </div>
    </div>
  )
}

function DocumentManagerModal({ docs, onSetFolder, onDelete, onClose }: {
  docs: Doc[]
  onSetFolder: (id: string, folder: string | null) => void
  onDelete: (id: string, filename: string) => void
  onClose: () => void
}) {
  const [query, setQuery] = useState('')
  const [editingId, setEditingId] = useState<string | null>(null)
  const [draft, setDraft] = useState('')

  const filtered = docs.filter(d => d.filename.toLowerCase().includes(query.trim().toLowerCase()))
  const sorted = [...filtered].sort((a, b) => a.filename.localeCompare(b.filename))

  const saveFolder = (id: string) => {
    onSetFolder(id, draft.trim() || null)
    setEditingId(null)
  }

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center p-4" style={{ background: 'rgba(23,26,36,0.45)' }} onClick={onClose}>
      <div onClick={e => e.stopPropagation()} style={{
        background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 'var(--r-lg)',
        boxShadow: 'var(--shadow-lift)', padding: 28, width: '100%', maxWidth: 620, maxHeight: '82vh', display: 'flex', flexDirection: 'column',
      }}>
        <p style={{ fontFamily: 'var(--serif)', fontSize: 22, fontStyle: 'italic', color: 'var(--text)', marginBottom: 4 }}>Manage documents</p>
        <p style={{ fontSize: 12.5, color: 'var(--text3)', marginBottom: 16 }}>
          {docs.length} document{docs.length === 1 ? '' : 's'} in your knowledge base. Organize into folders or remove ones you no longer need.
        </p>

        <input
          type="text"
          value={query}
          placeholder="Search by document name…"
          className="pref-field"
          style={{ marginBottom: 14 }}
          onChange={e => setQuery(e.target.value)}
          autoFocus
        />

        <div className="flex flex-col gap-2" style={{ overflowY: 'auto', flex: 1 }}>
          {sorted.length === 0 && (
            <p style={{ fontSize: 12.5, color: 'var(--text3)', textAlign: 'center', padding: '24px 0' }}>
              {docs.length === 0 ? 'No documents uploaded yet.' : 'No documents match your search.'}
            </p>
          )}
          {sorted.map(doc => (
            <div key={doc.id} style={{
              display: 'flex', alignItems: 'center', gap: 12, padding: '12px 14px',
              border: '1px solid var(--border)', borderRadius: 'var(--r-md)', background: 'var(--bg)'
            }}>
              <div className="flex-1 min-w-0">
                <p style={{ fontSize: 13.5, color: 'var(--text)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', marginBottom: 3 }}>
                  {doc.filename}
                </p>
                <p style={{ fontSize: 11, color: 'var(--text3)', fontFamily: 'var(--mono)' }}>
                  {doc.chunk_count === -1
                    ? <span style={{ color: 'var(--accent)' }}>⟳ Processing…</span>
                    : `${doc.chunk_count} chunks`} · {new Date(doc.uploaded_at).toLocaleDateString()}
                  {doc.folder && <> · <span style={{ color: 'var(--text2)' }}>{doc.folder}</span></>}
                </p>
              </div>

              {editingId === doc.id ? (
                <input
                  autoFocus
                  value={draft}
                  onChange={e => setDraft(e.target.value)}
                  onKeyDown={e => {
                    if (e.key === 'Enter') saveFolder(doc.id)
                    if (e.key === 'Escape') setEditingId(null)
                  }}
                  onBlur={() => saveFolder(doc.id)}
                  placeholder="Folder name…"
                  style={{ width: 140, fontSize: 12, background: 'var(--surface)', border: '1px solid var(--border2)', borderRadius: 6, padding: '6px 8px', flexShrink: 0 }}
                />
              ) : (
                <button
                  className="action-btn"
                  style={{ width: 'auto', padding: '6px 12px', fontSize: 11.5, flexShrink: 0 }}
                  onClick={() => { setEditingId(doc.id); setDraft(doc.folder || '') }}
                >Folder</button>
              )}

              <button
                className="action-btn"
                style={{ width: 'auto', padding: '6px 12px', fontSize: 11.5, flexShrink: 0, color: 'var(--danger)', borderColor: 'var(--danger)' }}
                onClick={() => onDelete(doc.id, doc.filename)}
              >Delete</button>
            </div>
          ))}
        </div>

        <button className="action-btn" style={{ marginTop: 16 }} onClick={onClose}>Close</button>
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
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [sidebarWidth, setSidebarWidth] = useState(280)
  const [selectedDocs, setSelectedDocs] = useState<string[]>([])
  const [sessions, setSessions] = useState<ChatSession[]>([])
  const [dailyTokenPct, setDailyTokenPct] = useState<number | null>(null)
  const [userId, setUserId] = useState<string>('')
  const [shareUrl, setShareUrl] = useState<string | null>(null)
  const [sharingId, setSharingId] = useState<string | null>(null)
  const bottomRef = useRef<HTMLDivElement>(null)
  const scrollRef = useRef<HTMLDivElement>(null)
  const stickToBottomRef = useRef(true)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const attachRef = useRef<HTMLInputElement>(null)
  const [attachedFile, setAttachedFile] = useState<File | null>(null)
  const [showOnboarding, setShowOnboarding] = useState(false)
  const [onboardingRequired, setOnboardingRequired] = useState(false)
  const [preferences, setPreferences] = useState<PreferencesState>(DEFAULT_PREFERENCES)
  const [showMemory, setShowMemory] = useState(false)
  const [showDocManager, setShowDocManager] = useState(false)
  const [memoryNotes, setMemoryNotes] = useState<MemoryNote[]>([])

  useEffect(() => {
    const w = localStorage.getItem('mv_sidebar_width')
    if (w) setSidebarWidth(Math.min(420, Math.max(220, Number(w))))
    const o = localStorage.getItem('mv_sidebar_open')
    if (o !== null) setSidebarOpen(o === '1')
  }, [])

  useEffect(() => { localStorage.setItem('mv_sidebar_width', String(sidebarWidth)) }, [sidebarWidth])
  useEffect(() => { localStorage.setItem('mv_sidebar_open', sidebarOpen ? '1' : '0') }, [sidebarOpen])

  useEffect(() => {
    getSupabase().auth.getSession().then(({ data: { session } }) => {
      if (!session) { window.location.href = '/login'; return }
      const uid = session.user.id
      setUserId(uid)
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
      getPreferences().then(r => {
        if (r.preferences) {
          setPreferences({ ...DEFAULT_PREFERENCES, ...r.preferences })
        } else {
          setOnboardingRequired(true); setShowOnboarding(true)
        }
      }).catch(() => {})
      listMemoryNotes().then(r => setMemoryNotes(r.notes || [])).catch(() => {})
    })
    loadDocs()
  }, [])

  useEffect(() => {
    const t = preferences.theme.toLowerCase()
    document.documentElement.setAttribute('data-theme', t)
    try { localStorage.setItem('mv_theme', t) } catch { /* ignore */ }
  }, [preferences.theme])

  const handleSavePreferences = async (prefs: PreferencesState) => {
    await savePreferences(prefs)
    setPreferences(prefs)
    setShowOnboarding(false)
  }

  const openMemory = () => {
    listMemoryNotes().then(r => setMemoryNotes(r.notes || [])).catch(() => {})
    setShowMemory(true)
  }

  const handleAddMemoryNote = async (content: string) => {
    const note = await addMemoryNote(content)
    setMemoryNotes(prev => [...prev, note])
  }

  const handleDeleteMemoryNote = async (id: string) => {
    setMemoryNotes(prev => prev.filter(n => n.id !== id))
    await deleteMemoryNote(id)
  }

  // Streaming appends a token to `messages` on every chunk, which used to
  // re-trigger a *smooth* scrollIntoView each time — dozens of overlapping
  // scroll animations per second is what made the chat feel laggy. Track
  // whether the user is already near the bottom and, if so, snap instantly
  // instead of restarting a smooth animation every token.
  useEffect(() => {
    if (stickToBottomRef.current) {
      bottomRef.current?.scrollIntoView({ behavior: 'auto', block: 'end' })
    }
  }, [messages, loading])

  const handleListScroll = () => {
    const el = scrollRef.current
    if (!el) return
    const distanceFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight
    stickToBottomRef.current = distanceFromBottom < 96
  }

  const showToast = (msg: string, type: Toast['type'] = 'info') => {
    setToast({ msg, type })
    setTimeout(() => setToast(null), 4500)
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

  const handleFolderChange = async (documentId: string, folder: string | null) => {
    setDocs(prev => prev.map(d => d.id === documentId ? { ...d, folder } : d))
    try {
      await setDocumentFolder(documentId, folder)
    } catch {
      showToast('Could not update folder', 'error')
      loadDocs(false)
    }
  }

  const handleDeleteDoc = async (documentId: string, filename: string) => {
    if (!window.confirm(`Delete "${filename}"? This removes it and its chunks permanently.`)) return
    const prevDocs = docs
    setDocs(prev => prev.filter(d => d.id !== documentId))
    setSelectedDocs(prev => prev.filter(id => id !== documentId))
    try {
      await deleteDocument(documentId)
      localStorage.setItem('mindvault_docs', JSON.stringify(prevDocs.filter(d => d.id !== documentId)))
    } catch {
      setDocs(prevDocs)
      showToast('Could not delete document', 'error')
    }
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

  // Ingestion (parse/chunk/embed) now runs in a background task on the
  // server -- /upload returns almost immediately with {status: "processing",
  // document_id}. This polls until each uploaded doc's chunk_count moves off
  // the -1 "processing" sentinel, or disappears entirely (background task
  // hit an error and cleaned the row up -- surfaced here as a failure toast
  // instead of a silent vanish).
  const pollUploads = (pending: { id: string; filename: string }[]) => {
    let remaining = pending
    let attempts = 0
    // Large/scanned documents can take several minutes to embed on a
    // constrained server CPU (double embedding pass: sentence-level for
    // semantic chunking, then chunk-level for storage) -- a short cap here
    // silently stops polling while the backend keeps working, leaving the
    // UI frozen on "Processing…" even after the document is actually done.
    // ~10 minutes at 4s covers that without polling too aggressively.
    const maxAttempts = 150
    const interval = setInterval(async () => {
      attempts++
      try {
        const data = await getDocuments()
        const freshDocs: Doc[] = data.documents || []
        setDocs(freshDocs)
        localStorage.setItem('mindvault_docs', JSON.stringify(freshDocs))
        remaining = remaining.filter(p => {
          const found = freshDocs.find(d => d.id === p.id)
          if (!found) {
            showToast(`${p.filename} failed to process`, 'error')
            return false
          }
          if (found.chunk_count !== -1) {
            showToast(`${p.filename} ready — ${found.chunk_count} chunks`, 'success')
            return false
          }
          return true
        })
      } catch { /* transient fetch failure -- try again next tick */ }

      if (remaining.length > 0 && attempts >= maxAttempts) {
        showToast(`Still processing ${remaining.map(r => r.filename).join(', ')} — reopen "Manage" to check later`, 'info')
      }
      if (remaining.length === 0 || attempts >= maxAttempts) clearInterval(interval)
    }, 4000)
  }

  const handleUpload = async (files: File[]) => {
    if (!files.length) return
    setUploading(true)
    const pending: { id: string; filename: string }[] = []
    for (let i = 0; i < files.length; i++) {
      const file = files[i]
      try {
        setUploadStatus(files.length > 1 ? `Uploading ${i + 1}/${files.length}: ${file.name.slice(0, 20)}...` : 'Uploading...')
        const res = await uploadDocument(file)
        pending.push({ id: res.document_id, filename: file.name })
      } catch {
        showToast(`Failed: ${file.name}`, 'error')
      }
    }
    localStorage.removeItem('mindvault_docs')
    await loadDocs(false)
    if (pending.length > 0) {
      setUploadStatus(`Processing ${pending.length} file${pending.length > 1 ? 's' : ''} in background…`)
      showToast(`${pending.length} file${pending.length > 1 ? 's' : ''} uploaded — processing`, 'info')
      pollUploads(pending)
    }
    setTimeout(() => { setUploading(false); setUploadStatus('') }, 2500)
  }

  const handleSend = useCallback(async (text?: string) => {
    const question = (text || input).trim()
    if (!question || loading || !sessionId) return

    stickToBottomRef.current = true
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

    const { getSupabase: _sb } = await import('../lib/supabase')
    const { data: { session: _session } } = await _sb().auth.getSession()
    const token = _session?.access_token || ''

    const onMeta = (meta: any) => {
      setMessages(prev => prev.map(m => m.id === assistantId ? {
        ...m,
        sources: meta.sources || [],
        chunks: meta.chunks || [],
        confidence: meta.confidence ?? undefined,
        intent: meta.intent,
        answer_type: meta.answer_type,
      } : m))
      if (!fileToSend) {
        const activeSession = sessions.find(s => s.id === sessionId)
        if (!activeSession || activeSession.name === 'New Chat') {
          renameSession(sessionId, question.slice(0, 45)).then(() => loadSessions()).catch(() => {})
        }
      }
    }
    const onToken = (tokenText: string) => {
      setMessages(prev => prev.map(m => m.id === assistantId
        ? { ...m, content: (m.content || '') + tokenText }
        : m))
    }
    const onWarning = (message: string) => {
      setMessages(prev => prev.map(m => m.id === assistantId ? { ...m, unverified: true } : m))
    }
    const onDone = (done: any) => {
      setMessages(prev => prev.map(m => m.id === assistantId ? {
        ...m,
        related_concepts: done.related_concepts || [],
        answer_type: done.answer_type ?? m.answer_type,
        tokens: done.tokens,
      } : m))
      if (done.tokens?.daily_used) setDailyTokenPct(done.tokens.daily_pct)
      setLoading(false)
    }
    const onError = (err: string) => {
      setMessages(prev => prev.map(m => m.id === assistantId ? { ...m, content: `Error: ${err}` } : m))
      showToast('Stream failed', 'error')
      setLoading(false)
    }

    const cancel = fileToSend
      ? streamQueryWithAttachment(question, sessionId, mode, selectedDocs, fileToSend, token, onMeta, onToken, onDone, onError, onWarning)
      : streamQuery(question, sessionId, mode, selectedDocs, token, onMeta, onToken, onDone, onError, onWarning)

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
    } catch { showToast('PDF export failed', 'error') }
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
    setSelectedDocs(prev => prev.includes(id) ? prev.filter(d => d !== id) : [...prev, id])
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
        onExport={handleExport} onExportPDF={handleExportPDF} onNewSession={handleNewSession} onClearSession={handleClearSession}
        open={sidebarOpen} onClose={() => setSidebarOpen(false)}
        width={sidebarWidth} onWidthChange={setSidebarWidth}
        selectedDocs={selectedDocs} onToggleDoc={handleToggleDoc}
        sessions={sessions} onSelectSession={handleSelectSession} onDeleteSession={handleDeleteSession}
        onShare={handleShare} sharingId={sharingId}
        onEditPreferences={() => { setOnboardingRequired(false); setShowOnboarding(true) }}
        onOpenMemory={openMemory}
        dailyTokenPct={dailyTokenPct}
        onOpenDocManager={() => { setShowDocManager(true); loadDocs(false) }}
      />

      {showOnboarding && (
        <PreferencesModal
          initial={preferences}
          memoryNotes={memoryNotes}
          dismissable={!onboardingRequired}
          onClose={() => setShowOnboarding(false)}
          onSave={handleSavePreferences}
          onAddMemoryNote={handleAddMemoryNote}
          onDeleteMemoryNote={handleDeleteMemoryNote}
        />
      )}

      {showMemory && (
        <MemoryModal
          notes={memoryNotes}
          onAdd={handleAddMemoryNote}
          onDelete={handleDeleteMemoryNote}
          onClose={() => setShowMemory(false)}
        />
      )}

      {showDocManager && (
        <DocumentManagerModal
          docs={docs}
          onSetFolder={handleFolderChange}
          onDelete={handleDeleteDoc}
          onClose={() => setShowDocManager(false)}
        />
      )}

      <main className="flex-1 flex flex-col overflow-hidden min-w-0" style={{ background: 'var(--bg)' }}>
        {/* Top bar */}
        <div className="flex items-center justify-between px-4 py-3 gap-3 flex-wrap"
          style={{ borderBottom: '1px solid var(--border)', background: 'var(--bg)' }}>
          <div className="flex items-center gap-3 flex-wrap">
            <button className="icon-btn" title="Vault (documents, chats, memory)" style={{ color: 'var(--text3)', fontSize: 18 }}
              onClick={() => setSidebarOpen(true)}>☰</button>
            <div className="segmented">
              {MODES.map(m => (
                <button key={m.value} className={`segmented-item ${mode === m.value ? 'active' : ''}`}
                  onClick={() => setMode(m.value)} title={m.desc}>{m.label}</button>
              ))}
            </div>
            <button className="graph-btn" onClick={handleViewFullGraph}>⬡ Graph</button>
          </div>
        </div>

        {/* Messages */}
        <div ref={scrollRef} onScroll={handleListScroll} className="flex-1 overflow-y-auto" style={{ padding: '24px 0' }}>
          {messages.length === 0 ? (
            <EmptyState onSuggest={(q) => handleSend(q)} docs={docs} />
          ) : (
            <div className="flex flex-col gap-7 max-w-3xl mx-auto px-4 md:px-6">
              {messages.map(msg => (
                <MessageBubble key={msg.id} msg={msg} onConceptClick={handleViewGraph} />
              ))}
              {loading && <TypingIndicator />}
              <div ref={bottomRef} />
            </div>
          )}
        </div>

        {/* Composer */}
        <div className="composer-wrap">
          {attachedFile && (
            <div style={{ maxWidth: 760, margin: '0 auto 8px' }}>
              <span className="chip" style={{
                color: 'var(--accent)', background: 'rgba(61,127,104,0.08)', border: '1px solid rgba(61,127,104,0.2)'
              }}>
                📎 {attachedFile.name}
                <button onClick={() => setAttachedFile(null)} className="tap-target"
                  style={{ background: 'none', border: 'none', color: 'var(--text3)', cursor: 'pointer', fontSize: 13, padding: 0 }}
                  title="Remove attachment">✕</button>
              </span>
            </div>
          )}
          <div className="composer-card flex gap-1 items-end">
            <input ref={attachRef} type="file" className="hidden"
              accept=".pdf,.txt,.md,.docx,.doc,.jpg,.jpeg,.png,.gif,.webp,.bmp,.tiff"
              onChange={(e) => {
                const f = e.target.files?.[0]
                if (f) setAttachedFile(f)
                if (attachRef.current) attachRef.current.value = ''
              }} />
            <button onClick={() => attachRef.current?.click()} disabled={loading || !sessionId}
              title="Attach a file or image for this question" className="icon-btn"
              style={{ color: attachedFile ? 'var(--accent)' : 'var(--text3)' }}>
              📎
            </button>
            <div className="flex-1">
              <textarea ref={textareaRef} value={input} onChange={e => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder={attachedFile ? "Ask something about this file..." : "Ask your vault anything..."}
                rows={1} className="vault-input" style={{ minHeight: 40, maxHeight: 140 }}
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
          <p className="eyebrow" style={{ textAlign: 'center', marginTop: 8, textTransform: 'none', letterSpacing: '0.02em' }}>
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
