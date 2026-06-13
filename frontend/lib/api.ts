import axios from 'axios'

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

export async function getDocuments() {
  const res = await api.get('/documents')
  return res.data
}

export async function exportSession(sessionId: string) {
  const res = await api.post('/export', { session_id: sessionId, format: 'markdown' })
  return res.data
}

export async function getGraphTopic(topic: string) {
  const res = await api.get(`/graph/${encodeURIComponent(topic)}`)
  return res.data
}

export async function clearSession(sessionId: string) {
  const res = await api.delete(`/session/${sessionId}`)
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