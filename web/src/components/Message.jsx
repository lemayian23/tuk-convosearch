/**
 * Renders one turn.
 *
 * The model's text is turned into React elements, never into HTML strings:
 * an answer can contain anything a document contained, so `innerHTML` would
 * be an injection point reachable by whoever uploaded that document.
 *
 * Location: web/src/components/Message.jsx
 */

const CITATION_SPLIT = /(\[[^\]\n]{1,80}\])/g
// Separate pattern for testing: a /g regex used with .test() carries
// lastIndex between calls and silently skips every other match.
const CITATION_WHOLE = /^\[[^\]\n]{1,80}\]$/

function inline(text, keyPrefix) {
  const parts = text.split(CITATION_SPLIT).filter(Boolean)
  return parts.map((part, index) =>
    CITATION_WHOLE.test(part) ? (
      <span className="cite" key={`${keyPrefix}-${index}`}>
        {part}
      </span>
    ) : (
      <span key={`${keyPrefix}-${index}`}>{part}</span>
    ),
  )
}

function blocks(text) {
  const lines = text.split('\n')
  const out = []
  let paragraph = []
  let bullets = []
  let numbered = []

  const flushParagraph = () => {
    if (paragraph.length) {
      out.push(<p key={`p-${out.length}`}>{inline(paragraph.join(' '), `p${out.length}`)}</p>)
      paragraph = []
    }
  }
  const flushList = (items, ordered) => {
    if (items.length) {
      const Tag = ordered ? 'ol' : 'ul'
      out.push(
        <Tag key={`l-${out.length}`}>
          {items.map((item, index) => (
            <li key={index}>{inline(item, `l${out.length}-${index}`)}</li>
          ))}
        </Tag>,
      )
      items.length = 0
    }
  }

  for (const raw of lines) {
    const line = raw.trimEnd()
    if (!line.trim()) {
      flushParagraph()
      flushList(bullets, false)
      flushList(numbered, true)
      continue
    }
    if (/^[-•*]\s+/.test(line)) {
      flushParagraph()
      flushList(numbered, true)
      bullets.push(line.replace(/^[-•*]\s+/, ''))
      continue
    }
    if (/^\d+[.)]\s+/.test(line)) {
      flushParagraph()
      flushList(bullets, false)
      numbered.push(line.replace(/^\d+[.)]\s+/, ''))
      continue
    }
    flushList(bullets, false)
    flushList(numbered, true)
    paragraph.push(line.trim())
  }

  flushParagraph()
  flushList(bullets, false)
  flushList(numbered, true)
  return out
}

export default function Message({ turn }) {
  const isUser = turn.role === 'user'

  return (
    <div className={`turn ${isUser ? 'user' : 'assistant'}`}>
      <div className={`bubble ${turn.streaming ? 'caret' : ''}`}>
        {blocks(turn.content || (turn.streaming ? '' : 'No answer was returned.'))}
      </div>

      {!isUser && !turn.streaming && (
        <div className="meta">
          {turn.route === 'off_topic' && <span className="badge warn">Off-topic</span>}
          {turn.route === 'small_talk' && <span className="badge">General</span>}
          {turn.grounded === false && turn.route !== 'off_topic' && (
            <span className="badge warn">Check this against the document</span>
          )}
          {turn.error && <span className="badge error">{turn.error}</span>}
          {typeof turn.response_time === 'number' && (
            <span>{turn.response_time.toFixed(1)}s</span>
          )}
        </div>
      )}
    </div>
  )
}
