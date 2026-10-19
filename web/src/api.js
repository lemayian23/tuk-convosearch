/**
 * API client for the student UI.
 *
 * Streaming uses fetch + ReadableStream rather than EventSource because the
 * question is sent as a POST body; EventSource can only GET.
 *
 * Location: web/src/api.js
 */

const BASE = (import.meta.env.VITE_API_BASE ?? '').replace(/\/+$/, '')

async function json(response) {
  if (!response.ok) {
    let detail = `Request failed (${response.status})`
    try {
      const body = await response.json()
      if (body?.detail) detail = typeof body.detail === 'string' ? body.detail : detail
    } catch {
      /* a non-JSON error body still means the request failed */
    }
    const error = new Error(detail)
    error.status = response.status
    throw error
  }
  return response.json()
}

export const getConfig = () => fetch(`${BASE}/api/config`).then(json)
export const getHealth = () => fetch(`${BASE}/api/health`).then(json)
export const getSessions = () => fetch(`${BASE}/api/sessions`).then(json)
export const getMessages = (sessionId) =>
  fetch(`${BASE}/api/sessions/${sessionId}/messages`).then(json)
export const removeSession = (sessionId) =>
  fetch(`${BASE}/api/sessions/${sessionId}`, { method: 'DELETE' }).then((r) => r.ok)

/**
 * Send a question and hand each server event to `onEvent` as it arrives.
 * Resolves when the stream ends; rejects on transport failure or a
 * non-2xx response (429 rate limit, 503 all model slots busy).
 */
export async function streamChat({ message, sessionId, pipeline = 'tuk', signal, onEvent }) {
  const response = await fetch(`${BASE}/api/chat/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message, session_id: sessionId, pipeline }),
    signal,
  })

  if (!response.ok || !response.body) {
    await json(response)
    return
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { value, done } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })

    let boundary
    while ((boundary = buffer.indexOf('\n\n')) !== -1) {
      const block = buffer.slice(0, boundary)
      buffer = buffer.slice(boundary + 2)
      const dataLine = block.split('\n').find((line) => line.startsWith('data:'))
      if (!dataLine) continue
      try {
        onEvent(JSON.parse(dataLine.slice(5).trim()))
      } catch {
        /* ignore keep-alive comments */
      }
    }
  }
}
