import { describe, it, expect } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import AgentLog, { stepIcon, stepClass, formatArgs } from './AgentLog'
import type { AgentStep } from '../types'

const THINKING_STEP: AgentStep = {
  step_type: 'thinking',
  tool_name: null,
  tool_args: null,
  result_count: null,
  message: 'Analyzing query...',
}

const TOOL_CALL_STEP: AgentStep = {
  step_type: 'tool_call',
  tool_name: 'search_by_caption',
  tool_args: { query: 'beach sunset' },
  result_count: null,
  message: 'Calling search_by_caption',
}

const TOOL_RESULT_STEP: AgentStep = {
  step_type: 'tool_result',
  tool_name: 'search_by_caption',
  tool_args: null,
  result_count: 8,
  message: 'search_by_caption returned 8 results',
}

const DONE_STEP: AgentStep = {
  step_type: 'done',
  tool_name: null,
  tool_args: null,
  result_count: null,
  message: 'Search complete',
}

const ERROR_STEP: AgentStep = {
  step_type: 'error',
  tool_name: null,
  tool_args: null,
  result_count: null,
  message: 'LLM is not loaded',
}

describe('stepIcon', () => {
  it('returns gear for thinking', () => {
    expect(stepIcon('thinking')).toBe('\u2699')
  })

  it('returns arrow for tool_call', () => {
    expect(stepIcon('tool_call')).toBe('\u27a4')
  })

  it('returns check for tool_result', () => {
    expect(stepIcon('tool_result')).toBe('\u2714')
  })

  it('returns green check for done', () => {
    expect(stepIcon('done')).toBe('\u2705')
  })

  it('returns red x for error', () => {
    expect(stepIcon('error')).toBe('\u274c')
  })
})

describe('stepClass', () => {
  it('returns a class string for each step type', () => {
    expect(stepClass('thinking')).toBeTruthy()
    expect(stepClass('tool_call')).toBeTruthy()
    expect(stepClass('tool_result')).toBeTruthy()
    expect(stepClass('done')).toBeTruthy()
    expect(stepClass('error')).toBeTruthy()
  })

  it('returns different classes for different types', () => {
    const classes = new Set([
      stepClass('thinking'),
      stepClass('tool_call'),
      stepClass('tool_result'),
      stepClass('error'),
    ])
    expect(classes.size).toBe(4)
  })
})

describe('formatArgs', () => {
  it('returns null for null args', () => {
    expect(formatArgs(null)).toBeNull()
  })

  it('returns null for empty object', () => {
    expect(formatArgs({})).toBeNull()
  })

  it('returns JSON string for non-empty args', () => {
    expect(formatArgs({ query: 'beach' })).toBe('{"query":"beach"}')
  })
})

describe('AgentLog component', () => {
  it('shows empty message when no steps', () => {
    render(<AgentLog steps={[]} />)
    expect(screen.getByText('Run a search to see agent activity here.')).toBeInTheDocument()
  })

  it('renders thinking step', () => {
    render(<AgentLog steps={[THINKING_STEP]} />)
    expect(screen.getByText('Analyzing query...')).toBeInTheDocument()
  })

  it('renders tool_call step with args', () => {
    render(<AgentLog steps={[TOOL_CALL_STEP]} />)
    expect(screen.getByText('Calling search_by_caption')).toBeInTheDocument()
    expect(screen.getByText('{"query":"beach sunset"}')).toBeInTheDocument()
  })

  it('renders tool_result step', () => {
    render(<AgentLog steps={[TOOL_RESULT_STEP]} />)
    expect(screen.getByText('search_by_caption returned 8 results')).toBeInTheDocument()
  })

  it('renders done step', () => {
    render(<AgentLog steps={[DONE_STEP]} />)
    expect(screen.getByText('Search complete')).toBeInTheDocument()
  })

  it('renders error step', () => {
    render(<AgentLog steps={[ERROR_STEP]} />)
    expect(screen.getByText('LLM is not loaded')).toBeInTheDocument()
  })

  it('renders multiple steps in order', () => {
    const steps = [THINKING_STEP, TOOL_CALL_STEP, TOOL_RESULT_STEP, DONE_STEP]
    render(<AgentLog steps={steps} />)

    const stepElements = screen.getAllByTestId('agent-step')
    expect(stepElements).toHaveLength(4)
  })

  it('shows step count in header', () => {
    render(<AgentLog steps={[THINKING_STEP, DONE_STEP]} />)
    expect(screen.getByText(/Agent Activity.*\(2\)/)).toBeInTheDocument()
  })

  it('can be collapsed and expanded', () => {
    render(<AgentLog steps={[THINKING_STEP]} />)

    // Initially visible
    expect(screen.getByText('Analyzing query...')).toBeInTheDocument()

    // Click to collapse
    fireEvent.click(screen.getByText(/Agent Activity/))
    expect(screen.queryByText('Analyzing query...')).not.toBeInTheDocument()

    // Click to expand
    fireEvent.click(screen.getByText(/Agent Activity/))
    expect(screen.getByText('Analyzing query...')).toBeInTheDocument()
  })

  it('does not show args for non-tool_call steps', () => {
    render(<AgentLog steps={[TOOL_RESULT_STEP]} />)
    // tool_result should not render args even if they existed
    const argsElements = document.querySelectorAll('[class*="args"]')
    expect(argsElements).toHaveLength(0)
  })
})
