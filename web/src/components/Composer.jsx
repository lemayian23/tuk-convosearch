/**
 * Question box. Enter sends, Shift+Enter adds a line — and on small screens
 * the button reads "Ask" because a keyboard return is a long press.
 *
 * Location: web/src/components/Composer.jsx
 */

import { useEffect, useRef, useState } from 'react'

export default function Composer({ onSubmit, onStop, busy, maxLength, disabled }) {
  const [value, setValue] = useState('')
  const ref = useRef(null)

  useEffect(() => {
    if (!busy && ref.current) ref.current.focus()
  }, [busy])

  const submit = () => {
    const question = value.trim()
    if (!question || busy) return
    onSubmit(question)
    setValue('')
  }

  const onKeyDown = (event) => {
    if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault()
      submit()
    }
  }

  return (
    <form
      className="composer"
      onSubmit={(event) => {
        event.preventDefault()
        submit()
      }}
    >
      <label className="sr-only" htmlFor="question">
        Ask about TU-K
      </label>
      <textarea
        id="question"
        ref={ref}
        rows={1}
        value={value}
        placeholder="Ask about exams, fees, registration…"
        maxLength={maxLength}
        disabled={disabled}
        onChange={(event) => setValue(event.target.value)}
        onKeyDown={onKeyDown}
      />
      {busy ? (
        <button type="button" className="send stop" onClick={onStop}>
          Stop
        </button>
      ) : (
        <button type="submit" className="send" disabled={disabled || !value.trim()}>
          Ask
        </button>
      )}
    </form>
  )
}
