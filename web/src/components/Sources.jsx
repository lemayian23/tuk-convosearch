/**
 * Source list for an answer. Collapsed by default on mobile so the answer
 * stays above the fold, but always present: an uncited answer from this
 * assistant is a bug, not a feature.
 *
 * Location: web/src/components/Sources.jsx
 */

export default function Sources({ sources, chunksFound, open }) {
  if (!sources || sources.length === 0) {
    return chunksFound === 0 ? (
      <p className="meta">
        <span className="badge warn">Nothing matched in the indexed documents</span>
      </p>
    ) : null
  }

  return (
    <details className="sources" open={open}>
      <summary>
        {sources.length === 1 ? '1 source document' : `${sources.length} source documents`}
      </summary>
      {sources.map((source, index) => (
        <div className="source-item" key={`${source.source}-${index}`}>
          <div className="name">{source.source}</div>
          {source.quote && <blockquote>{source.quote}</blockquote>}
        </div>
      ))}
    </details>
  )
}
