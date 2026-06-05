import axios from 'axios'
import { getToken } from './supabase'

const BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

const api = axios.create({
  baseURL: BASE,
  timeout: 180000,
  headers: { 'Content-Type': 'application/json' },
})

// Attach token to every request
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

export async function queryKnowledge(question: string, sessionId: string, mode: string) {
  const res = await api.post('/query', { question, session_id: sessionId, mode })
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