import axios from 'axios'

if (!process.env.NEXT_PUBLIC_API_URL && process.env.NODE_ENV === 'production') {
  throw new Error('NEXT_PUBLIC_API_URL is not set')
}
const BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

const api = axios.create({
  baseURL: BASE,
  timeout: 180000,
  headers: { 'Content-Type': 'application/json' },
})

async function getToken(): Promise<string | null> {
  if (typeof window === 'undefined') return null
  const { getSupabase } = await import('./supabase')
  const { data: { session } } = await getSupabase().auth.getSession()
  return session?.access_token || null
}

api.interceptors.request.use(async (config) => {
  const token = await getToken()
  if (token) config.headers['Authorization'] = `Bearer ${token}`
  return config
})

export async function uploadDocument(file: File) {
  const form = new FormData()
  form.append('file', file)
  const token = await getToken()
  const res = await axios.post(`${BASE}/upload`, form, {
    headers: {
      'Content-Type': 'multipart/form-data',
      ...(token ? { Authorization: `Bearer ${token}` } : {})
    },
    timeout: 180000
  })
  return res.data
}

export async function queryKnowledge(question: string, sessionId: string, mode: string, documentIds: string[] = []) {
  const res = await api.post('/query', { question, session_id: sessionId, mode, document_ids: documentIds })
  return res.data
}

export async function queryWithAttachment(question: string, sessionId: string, mode: string, documentIds: string[], file: File) {
  const form = new FormData()
  form.append('file', file)
  form.append('question', question)
  form.append('session_id', sessionId)
  form.append('mode', mode)
  form.append('document_ids', JSON.stringify(documentIds))
  const token = await getToken()
  const res = await axios.post(`${BASE}/query-with-attachment`, form, {
    headers: {
      'Content-Type': 'multipart/form-data',
      ...(token ? { Authorization: `Bearer ${token}` } : {})
    },
    timeout: 180000
  })
  return res.data
}

export async function getDocuments() {
  const res = await api.get('/documents')
  return res.data
}

export async function setDocumentFolder(documentId: string, folder: string | null) {
  const res = await api.patch(`/documents/${documentId}/folder`, { folder })
  return res.data
}

export async function deleteDocument(documentId: string) {
  const res = await api.delete(`/documents/${documentId}`)
  return res.data
}

export async function searchMessages(query: string) {
  const res = await api.get('/sessions/search', { params: { q: query } })
  return res.data as { results: { session_id: string; role: string; content: string; timestamp: string }[] }
}

export async function exportSession(sessionId: string) {
  const res = await api.post('/export', { session_id: sessionId, format: 'markdown' })
  return res.data
}

export async function exportSessionPDF(sessionId: string): Promise<Blob> {
  const res = await api.post('/export', { session_id: sessionId, format: 'pdf' }, {
    responseType: 'blob'
  })
  // With responseType:'blob', axios delivers a 500's JSON error body as an
  // unreadable Blob instead of throwing -- silently masking the real error.
  // Detect that case and surface the actual backend message.
  if (res.data.type && res.data.type.includes('json')) {
    const text = await res.data.text()
    const parsed = JSON.parse(text)
    throw new Error(parsed.detail || 'PDF export failed')
  }
  return res.data
}

export async function getGraphTopic(topic: string) {
  const res = await api.get(`/graph/${encodeURIComponent(topic)}`)
  return res.data
}

export async function getFullGraph() {
  const res = await api.get('/graph')
  return res.data
}

export async function clearSession(sessionId: string) {
  const res = await api.delete(`/sessions/${sessionId}/messages`)
  return res.data
}
export async function listSessions() {
  const res = await api.get('/sessions')
  return res.data
}

export async function createSession() {
  const res = await api.post('/sessions')
  return res.data  // { session_id, name }
}

export async function renameSession(sessionId: string, name: string) {
  const res = await api.patch(`/sessions/${sessionId}/rename`, { name })
  return res.data
}

export async function deleteSession(sessionId: string) {
  const res = await api.delete(`/sessions/${sessionId}`)
  return res.data
}

export async function getSessionHistory(sessionId: string) {
  const res = await api.get(`/sessions/${sessionId}/history`)
  return res.data  // { history: [{role, content, timestamp}] }
}
export async function shareSession(sessionId: string) {
  const res = await api.post(`/sessions/${sessionId}/share`)
  return res.data  // { share_url, token }
}

export async function unshareSession(sessionId: string) {
  const res = await api.delete(`/sessions/${sessionId}/share`)
  return res.data
}

export async function getSharedSession(token: string) {
  const res = await axios.get(`${BASE}/share/${token}`)
  return res.data  // { name, created_at, messages }
}

export interface Preferences {
  name: string
  tone: string
  priorities: string[]
  system_prompt: string
  theme: string
}

export async function getPreferences() {
  const res = await api.get('/preferences')
  return res.data  // { preferences: Preferences | null }
}

export async function savePreferences(prefs: Preferences) {
  const res = await api.post('/preferences', prefs)
  return res.data
}

export async function listMemoryNotes() {
  const res = await api.get('/memory')
  return res.data  // { notes: { id, content, created_at }[] }
}

export async function addMemoryNote(content: string) {
  const res = await api.post('/memory', { content })
  return res.data
}

export async function deleteMemoryNote(noteId: string) {
  const res = await api.delete(`/memory/${noteId}`)
  return res.data
}

// Shared SSE line-reader -- both streamQuery (JSON body) and
// streamQueryWithAttachment (multipart body) parse the exact same
// meta/token/warning/done/error event shape, only the request differs.
function readSSE(
  res: Response,
  onMeta: (meta: any) => void,
  onToken: (text: string) => void,
  onDone: (data: any) => void,
  onError: (err: string) => void,
  onWarning?: (message: string) => void
) {
  return (async () => {
    if (!res.ok) { onError(`HTTP ${res.status}`); return }
    const reader = res.body!.getReader()
    const decoder = new TextDecoder()
    let buffer = ''

    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop() || ''
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue
        const json = line.slice(6).trim()
        if (!json) continue
        try {
          const evt = JSON.parse(json)
          if (evt.type === 'meta')  onMeta(evt)
          else if (evt.type === 'token') onToken(evt.text)
          else if (evt.type === 'warning') onWarning?.(evt.message)
          else if (evt.type === 'done')  onDone(evt)
          else if (evt.type === 'error') onError(evt.message)
        } catch {}
      }
    }
  })()
}

// Streaming query via SSE
export function streamQuery(
  question: string, sessionId: string, mode: string,
  documentIds: string[], token: string,
  onMeta: (meta: any) => void,
  onToken: (text: string) => void,
  onDone: (data: any) => void,
  onError: (err: string) => void,
  onWarning?: (message: string) => void
): () => void {
  const controller = new AbortController()

  fetch(`${BASE}/query/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
    body: JSON.stringify({ question, session_id: sessionId, mode, document_ids: documentIds }),
    signal: controller.signal,
  }).then((res) => readSSE(res, onMeta, onToken, onDone, onError, onWarning))
    .catch((err) => {
      if (err.name !== 'AbortError') onError(String(err))
    })

  return () => controller.abort()
}

// Streaming attachment query via SSE (multipart body, same event shape as streamQuery)
export function streamQueryWithAttachment(
  question: string, sessionId: string, mode: string,
  documentIds: string[], file: File, token: string,
  onMeta: (meta: any) => void,
  onToken: (text: string) => void,
  onDone: (data: any) => void,
  onError: (err: string) => void,
  onWarning?: (message: string) => void
): () => void {
  const controller = new AbortController()
  const form = new FormData()
  form.append('file', file)
  form.append('question', question)
  form.append('session_id', sessionId)
  form.append('mode', mode)
  form.append('document_ids', JSON.stringify(documentIds))

  fetch(`${BASE}/query-with-attachment/stream`, {
    method: 'POST',
    headers: { Authorization: `Bearer ${token}` },
    body: form,
    signal: controller.signal,
  }).then((res) => readSSE(res, onMeta, onToken, onDone, onError, onWarning))
    .catch((err) => {
      if (err.name !== 'AbortError') onError(String(err))
    })

  return () => controller.abort()
}