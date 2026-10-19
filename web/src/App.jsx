/**
 * Student chat shell.
 *
 * Location: web/src/App.jsx
 */

import { useCallback, useEffect, useRef, useState } from 'react'
import Composer from './components/Composer'
import Drawer from './components/Drawer'
import Message from './components/Message'
import Sources from './components/Sources'
import { getConfig, getHealth, getMessages, streamChat } from './api'
import { readHistory, rememberSession } from './history'

const SESSION_KEY = 'tuk.session_id'
const PIPELINE_KEY = 'tuk.pipeline'

let sequence = 0
const nextKey = () => `turn-${++sequence}`

const PLAIN_MESSAGES = {
  429: 'You are sending questions faster than the server allows. Wait a moment.',
  503: 'The assistant is busy with other students. Please try again in a moment.',
  504: 'The model took too long to answer. Try a shorter question.',
  502: 'The assistant could not form an answer. Please try again.',
}

function readStored(key) {
  try {
    return window.localStorage.getItem(key)
  } catch {
    // Private browsing on a phone throws; the chat still works without memory.
    return null
  }
}

function writeStored(key, value) {
  try {
    if (value === null) window.localStorage.removeItem(key)
    else window.localStorage.setItem(key, value)
  } catch {
    /* non-fatal */
  }
}

export default function App() {
  const [config, setConfig] = useState(null)
  const [notice, setNotice] = useState('')
  const [messages, setMessages] = useState([])
  const [sessionId, setSessionId] = useState(() => readStored(SESSION_KEY))
  const [pipeline, setPipeline] = useState(() => readStored(PIPELINE_KEY) || 'tuk')
  const [sessions, setSessions] = useState(readHistory)
  const [streaming, setStreaming] = useState(false)
  const [drawerOpen, setDrawerOpen] = useState(false)

  const abort = useRef(null)
  const scroll = useRef(null)
  const pinned = useRef(true)
  const draftKey = useRef('')

  useEffect(() => {
    writeStored(SESSION_KEY, sessionId)
  }, [sessionId])

  useEffect(() => {
    writeStored(PIPELINE_KEY, pipeline)
  }, [pipeline])

  // One boot request for the runtime config; the UI renders without it.
  useEffect(() => {
    let alive = true
    getConfig()
      .then((value) => {
        if (!alive) return
        setConfig(value)
        setPipeline((current) =>
          !value.pipelines || value.pipelines[current]
            ? current
            : value.default_pipeline || 'tuk',
        )
      })
      .catch(() => alive && setNotice('The assistant is offline right now.'))
    return () => {
      alive = false
    }
  }, [])

  // Readiness is operator information, so it is fetched once in the
  // background and only surfaces when something is actually wrong.
  useEffect(() => {
    let alive = true
    getHealth()
      .then((health) => {
        if (!alive || health.status === 'ready') return
        if (!health.checks?.ollama_reachable || !health.checks?.model_present) {
          setNotice('The answer model is warming up. Answers may be slow.')
        } else if (!health.checks.corpus_indexed) {
          setNotice('No university documents are indexed yet.')
        } else if (!health.checks.index_matches_corpus) {
          setNotice('New documents are still being indexed.')
        }
      })
      .catch(() => {})
    return () => {
      alive = false
    }
  }, [])

  // A refresh must not blank a conversation the student was just reading.
  // The server keeps the history either way, so this is purely about what
  // the screen shows.
  useEffect(() => {
    if (!sessionId) return
    let alive = true
    getMessages(sessionId)
      .then((rows) => {
        if (!alive || !rows.length) return
        setMessages(
          rows.map((row) => ({
            id: nextKey(),
            role: row.role,
            content: row.content,
            streaming: false,
          })),
        )
      })
      .catch(() => {})
    return () => {
      alive = false
    }
    // Only on mount: later id changes come from openSession, which loads.
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  // Follow the answer as it types, but stop dragging the view back once the
  // student has scrolled up to re-read something.
  useEffect(() => {
    const node = scroll.current
    if (node && pinned.current) node.scrollTop = node.scrollHeight
  }, [messages])

  const patchDraft = useCallback((mutate) => {
    setMessages((current) =>
      current.map((turn) => (turn.id === draftKey.current ? mutate(turn) : turn)),
    )
  }, [])

  const stop = useCallback(() => {
    abort.current?.abort()
  }, [])

  const ask = useCallback(
    async (question) => {
      if (streaming) return
      const text = question.trim()
      if (!text) return

      pinned.current = true
      const draft = {
        id: nextKey(),
        role: 'assistant',
        content: '',
        streaming: true,
        sources: [],
        chunksFound: 0,
      }
      draftKey.current = draft.id
      setMessages((current) => [...current, { id: nextKey(), role: 'user', content: text }, draft])
      setStreaming(true)

      const controller = new AbortController()
      abort.current = controller

      // `sessionId` is still the previous conversation here; the id for this
      // one arrives on the `start` event, and the history entry needs it.
      let activeId = sessionId
      let answered = false

      const onEvent = (event) => {
        switch (event.type) {
          case 'start':
            activeId = event.session_id
            setSessionId(event.session_id)
            break
          case 'route':
            patchDraft((turn) => ({ ...turn, route: event.route }))
            break
          case 'sources':
            patchDraft((turn) => ({
              ...turn,
              sources: event.sources || [],
              chunksFound: event.chunks_found ?? 0,
            }))
            break
          case 'token':
            patchDraft((turn) => ({ ...turn, content: turn.content + (event.content || '') }))
            break
          case 'retry':
            patchDraft((turn) => ({
              ...turn,
              content: '',
              retryNote: 'That answer was not supported by the documents. Searching again…',
            }))
            break
          case 'error':
            patchDraft((turn) => ({ ...turn, error: event.message || 'Failed' }))
            break
          case 'done': {
            const context = event.context || {}
            answered = Boolean(context.answer)
            patchDraft((turn) => ({
              ...turn,
              // The streamed tokens can lag the finished context by a frame;
              // the server's own answer is authoritative.
              content: context.answer || turn.content,
              sources: context.sources || turn.sources,
              chunksFound: context.chunks_found ?? turn.chunksFound,
              route: context.route || turn.route,
              grounded: context.grounded,
              response_time: context.response_time,
              error: context.error ? 'No answer' : turn.error,
              streaming: false,
            }))
            break
          }
          case 'end':
            patchDraft((turn) => ({ ...turn, streaming: false }))
            break
          default:
            break
        }
      }

      try {
        await streamChat({
          message: text,
          sessionId,
          pipeline,
          signal: controller.signal,
          onEvent,
        })
      } catch (error) {
        const stopped = error.name === 'AbortError'
        patchDraft((turn) => ({
          ...turn,
          streaming: false,
          error: stopped
            ? 'Stopped'
            : PLAIN_MESSAGES[error.status] || 'The assistant could not answer.',
        }))
      } finally {
        patchDraft((turn) => ({ ...turn, streaming: false }))
        abort.current = null
        setStreaming(false)
        setSessions(rememberSession({ sessionId: activeId, question: text, answered }))
      }
    },
    [patchDraft, pipeline, sessionId, streaming],
  )

  const openSession = useCallback(async (id) => {
    abort.current?.abort()
    setSessionId(id)
    const rows = await getMessages(id).catch(() => [])
    setMessages(
      rows.map((row) => ({
        id: nextKey(),
        role: row.role,
        content: row.content,
        streaming: false,
      })),
    )
    pinned.current = true
  }, [])

  const startNew = useCallback(() => {
    abort.current?.abort()
    setMessages([])
    setSessionId(null)
    setDrawerOpen(false)
  }, [])

  const switchPipeline = useCallback(
    (name) => {
      if (name === pipeline) return
      setPipeline(name)
      // A session's history is fed to the model as context, so mixing a
      // fees conversation into a PubMed one would poison the next answer.
      startNew()
    },
    [pipeline, startNew],
  )

  const quickPrompts = config?.quick_prompts ?? []
  const maxChars = config?.max_question_chars ?? 4000
  const showIntro = messages.length === 0 && !streaming

  return (
    <div className="app">
      <header className="masthead">
        <button
          type="button"
          className="icon-button"
          onClick={() => setDrawerOpen((open) => !open)}
          aria-expanded={drawerOpen}
        >
          {drawerOpen ? 'Close' : 'History'}
        </button>
        <div className="mark">
          ConvoSearch
          <small>Technical University of Kenya</small>
        </div>
        <div className="spacer" />
        {config && Object.keys(config.pipelines).length > 1 && (
          <select
            className="icon-button"
            value={pipeline}
            aria-label="Choose assistant"
            onChange={(event) => switchPipeline(event.target.value)}
          >
            {Object.entries(config.pipelines).map(([name, description]) => (
              <option key={name} value={name} title={description}>
                {name === 'tuk' ? 'TU-K' : 'Research'}
              </option>
            ))}
          </select>
        )}
      </header>

      <div className="body">
        {drawerOpen && (
          <Drawer
            sessions={sessions}
            currentId={sessionId}
            onSelect={openSession}
            onNew={startNew}
            onClose={() => setDrawerOpen(false)}
          />
        )}

        <main className="thread">
          <div
            className="scroll"
            ref={scroll}
            onScroll={() => {
              const node = scroll.current
              if (node) pinned.current = node.scrollHeight - node.scrollTop - node.clientHeight < 120
            }}
          >
            <div className="stream">
              {showIntro && (
                <section className="intro">
                  <h1>Ask about TU-K</h1>
                  <p>
                    Fees, exams, registration, calendars and handbooks — answered from the
                    university documents the school has indexed, with the source named.
                  </p>
                  {notice && (
                    <p>
                      <span className="badge warn">{notice}</span>
                    </p>
                  )}
                  <div className="chips">
                    {quickPrompts.map((prompt) => (
                      <button type="button" className="chip" key={prompt} onClick={() => ask(prompt)}>
                        {prompt}
                      </button>
                    ))}
                  </div>
                </section>
              )}

              {messages.map((turn) => (
                <div className="turn-stack" key={turn.id}>
                  {turn.role === 'assistant' && turn.retryNote && !turn.content && (
                    <p className="meta">
                      <span className="badge">{turn.retryNote}</span>
                    </p>
                  )}
                  <Message turn={turn} />
                  {turn.role === 'assistant' && !turn.streaming && (
                    <Sources
                      sources={turn.sources}
                      chunksFound={turn.chunksFound}
                      open={false}
                    />
                  )}
                </div>
              ))}
            </div>
          </div>

          <Composer onSubmit={ask} onStop={stop} busy={streaming} maxLength={maxChars} />

          <p className="hint">
            {streaming ? 'Reading the university documents…' : 'Only what is in the indexed documents.'}
          </p>
        </main>
      </div>
    </div>
  )
}
