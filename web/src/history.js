/**
 * The list of conversations shown in the drawer, kept in this browser.
 *
 * Students have no login, so the only thing separating one student's history
 * from another is that a session id is an unguessable 128-bit hex string the
 * browser already holds. The server used to also expose every session on the
 * box at GET /api/sessions, which published those ids and, through them, every
 * transcript. The drawer now lists only the conversations started on this
 * device, so history does not follow a student to another phone either.
 *
 * Transcripts themselves stay server-side and are fetched by id.
 *
 * Location: web/src/history.js
 */

const HISTORY_KEY = 'tuk.history'
const MAX_ENTRIES = 30

function read() {
  try {
    const parsed = JSON.parse(window.localStorage.getItem(HISTORY_KEY) || '[]')
    return Array.isArray(parsed) ? parsed.filter((e) => e && e.session_id) : []
  } catch {
    // Private browsing, a full quota, or hand-edited storage.
    return []
  }
}

function write(entries) {
  try {
    window.localStorage.setItem(HISTORY_KEY, JSON.stringify(entries))
  } catch {
    /* the chat still works without a memory of past questions */
  }
}

export function readHistory() {
  return read()
}

/**
 * Record one finished exchange. The title stays the first question, because
 * that is what a student remembers ("exam dates"), matching the server.
 */
export function rememberSession({ sessionId, question, answered }) {
  if (!sessionId) return read()
  const existing = read().find((entry) => entry.session_id === sessionId)
  const entry = {
    session_id: sessionId,
    title: existing?.title || (question || '').slice(0, 120) || 'Untitled',
    message_count: (existing?.message_count || 0) + (answered ? 2 : 1),
    last_active_at: Date.now() / 1000,
  }
  const next = [entry, ...read().filter((e) => e.session_id !== sessionId)].slice(0, MAX_ENTRIES)
  write(next)
  return next
}
