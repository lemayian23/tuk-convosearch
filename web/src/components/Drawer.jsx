/**
 * Past conversations. Titles come from the first question, which is what a
 * student actually remembers ("exam dates"), not a timestamp.
 *
 * Location: web/src/components/Drawer.jsx
 */

function when(timestamp) {
  const days = (Date.now() / 1000 - timestamp) / 86400
  if (days < 1) return 'today'
  if (days < 2) return 'yesterday'
  if (days < 30) return `${Math.floor(days)} days ago`
  return new Date(timestamp * 1000).toLocaleDateString()
}

export default function Drawer({ sessions, currentId, onSelect, onNew, onClose }) {
  return (
    <aside className="drawer" aria-label="Past conversations">
      <button type="button" className="new-chat" onClick={onNew}>
        + New question
      </button>
      <h2>Recent</h2>
      {sessions.length === 0 ? (
        <p className="meta">Nothing yet. Your questions appear here.</p>
      ) : (
        <ul>
          {sessions.map((session) => (
            <li key={session.session_id}>
              <button
                type="button"
                className="entry"
                aria-current={session.session_id === currentId}
                onClick={() => {
                  onSelect(session.session_id)
                  onClose()
                }}
              >
                {session.title || 'Untitled'}
                <span>
                  {session.message_count} messages · {when(session.last_active_at)}
                </span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </aside>
  )
}
