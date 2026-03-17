import { useEffect, useRef, useState } from 'react'
import type { AgentStep } from '../types'
import styles from './AgentLog.module.css'

interface AgentLogProps {
  steps: AgentStep[]
}

function stepIcon(type: AgentStep['step_type']): string {
  switch (type) {
    case 'thinking': return '\u2699'   // gear
    case 'tool_call': return '\u27a4'  // arrow
    case 'tool_result': return '\u2714' // check
    case 'done': return '\u2705'       // green check
    case 'error': return '\u274c'      // red x
  }
}

function stepClass(type: AgentStep['step_type']): string {
  switch (type) {
    case 'thinking': return styles.thinking
    case 'tool_call': return styles.toolCall
    case 'tool_result': return styles.toolResult
    case 'done': return styles.done
    case 'error': return styles.error
  }
}

function formatArgs(args: Record<string, unknown> | null): string | null {
  if (!args || Object.keys(args).length === 0) return null
  return JSON.stringify(args)
}

export default function AgentLog({ steps }: AgentLogProps) {
  const [collapsed, setCollapsed] = useState(false)
  const scrollRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [steps])

  // Auto-expand when new steps arrive
  useEffect(() => {
    if (steps.length > 0) setCollapsed(false)
  }, [steps.length])

  return (
    <section className={styles.panel} data-testid="agent-log">
      <div className={styles.header} onClick={() => setCollapsed(!collapsed)}>
        <h2>Agent Activity {steps.length > 0 && `(${steps.length})`}</h2>
        <span className={styles.toggle}>{collapsed ? '\u25b6 Show' : '\u25bc Hide'}</span>
      </div>

      {!collapsed && (
        <div className={styles.steps} ref={scrollRef}>
          {steps.length === 0 && (
            <p className={styles.empty}>Run a search to see agent activity here.</p>
          )}
          {steps.map((step, i) => (
            <div key={i} className={`${styles.step} ${stepClass(step.step_type)}`} data-testid="agent-step">
              <span className={`${styles.icon} ${step.step_type === 'thinking' ? styles.spinner : ''}`}>
                {stepIcon(step.step_type)}
              </span>
              <div className={styles.message}>
                <div>{step.message}</div>
                {step.step_type === 'tool_call' && step.tool_args && (
                  <div className={styles.args}>{formatArgs(step.tool_args)}</div>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </section>
  )
}

export { stepIcon, stepClass, formatArgs }
