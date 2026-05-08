import axios from 'axios'

const BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

const api = axios.create({
  baseURL: BASE,
  timeout: 180000,
  headers: { 'Content-Type': 'application/json' },
})

export async function uploadDocument(file: File) {
  const form = new FormData()
  form.append('file', file)
  const res = await api.post('/upload', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  return res.data
}

export async function queryKnowledge(question: string, sessionId: string, mode: string) {
  const res = await api.post('/query', {
    question,
    session_id: sessionId,
    mode,
  })
  return res.data
}

export async function getDocuments() {
  const res = await api.get('/documents')
  return res.data
}

export async function exportSession(sessionId: string) {
  const res = await api.post('/export', {
    session_id: sessionId,
    format: 'markdown',
  })
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
