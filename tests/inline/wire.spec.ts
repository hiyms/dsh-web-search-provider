/**
 * Wire tests: the inline stream maps a fake gateway SSE stream onto the
 * harness StreamChunk contract — text, reasoning, tool calls, server-side
 * search skipping, terminal states, HTTP failures, transport aborts —
 * without ever throwing.
 */

import { afterEach, describe, expect, it, vi } from 'vitest'
import { inlineStream, inlineWireStream } from '../../src/wire.ts'
import type { InlineConfig } from '../../src/config.ts'
import type { SearchPlan, SearchPlanCandidate } from '../../src/plan.ts'
import { markAgentLoopRequest } from '@deepseek-ai/dsh-llm'
import type { GenerateOptions, Message, StreamChunk } from '@deepseek-ai/dsh-llm'

const encoder = new TextEncoder()

function sseBody(events: string[]): ReadableStream<Uint8Array> {
  return new ReadableStream({
    start(controller) {
      for (const event of events) controller.enqueue(encoder.encode(event))
      controller.close()
    },
  })
}

function event(type: string, data: unknown): string {
  return `event: ${type}\ndata: ${JSON.stringify(data)}\n\n`
}

const cfg: InlineConfig = {
  enabled: true,
  providers: [],
  includeSources: true,
  stripServerTools: true,
  idleTimeoutMs: 300_000,
  probe: false,
  probeTimeoutMs: 30_000,
}

const candidate: SearchPlanCandidate = {
  protocol: 'openai-responses',
  baseURL: 'https://gw.test/v1',
  model: 'm',
  apiKeyEnv: 'KEY',
  apiVersion: '2023-06-01',
}
const hooks = { resolveApiKey: async () => 'secret' }

function request(overrides: Partial<GenerateOptions> = {}): GenerateOptions {
  return markAgentLoopRequest({
    provider: 'p',
    model: 'deepseek-v4-flash',
    messages: [{ id: 'u1' as Message['id'], role: 'user', content: [{ type: 'text', text: 'hi' }], source: { kind: 'user' } }],
    ...overrides,
  })
}

async function collect(request: GenerateOptions): Promise<StreamChunk[]> {
  const chunks: StreamChunk[] = []
  for await (const chunk of inlineStream(request, candidate, hooks, cfg)) chunks.push(chunk)
  return chunks
}

function textOnlyStream(): string[] {
  return [
    event('response.output_item.added', { type: 'response.output_item.added', output_index: 0, item: { type: 'message', id: 'msg_1' } }),
    event('response.output_text.delta', { type: 'response.output_text.delta', output_index: 0, delta: 'Hello' }),
    event('response.output_text.delta', { type: 'response.output_text.delta', output_index: 0, delta: ' world' }),
    event('response.output_item.done', { type: 'response.output_item.done', output_index: 0, item: { type: 'message', id: 'msg_1', content: [{ type: 'output_text', text: 'Hello world' }] } }),
    event('response.completed', { type: 'response.completed', response: { status: 'completed', usage: { input_tokens: 10, output_tokens: 5, total_tokens: 15 } } }),
  ]
}

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('inlineStream', () => {
  it('maps a text-only stream to block/text-delta/usage/finish', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => new Response(sseBody(textOnlyStream()), { status: 200 })))
    const chunks = await collect(request())
    // Text slots open lazily on their first delta and never emit block-end
    // (the assembler assembles open text blocks from received deltas).
    expect(chunks.map(chunk => chunk.type)).toEqual(['block-start', 'text-delta', 'text-delta', 'usage', 'finish'])
    const [start, d1, d2, usage, finish] = chunks
    expect(start).toMatchObject({ type: 'block-start', index: 0, blockType: 'text' })
    expect(d1).toMatchObject({ type: 'text-delta', index: 0, text: 'Hello' })
    expect(d2).toMatchObject({ type: 'text-delta', index: 0, text: ' world' })
    expect(usage).toMatchObject({ type: 'usage', usage: { inputTokens: 10, outputTokens: 5 } })
    expect(finish).toMatchObject({ type: 'finish', reason: { kind: 'stop' } })
    expect('replayState' in (finish as { replayState?: unknown })).toBe(false)
  })

  it('skips web_search_call events and streams the answer normally', async () => {
    const stream = [
      event('response.output_item.added', { type: 'response.output_item.added', output_index: 0, item: { type: 'web_search_call', id: 'ws_1', action: { type: 'search', queries: ['node'] } } }),
      event('response.web_search_call.searching', { type: 'response.web_search_call.searching', item_id: 'ws_1' }),
      event('response.output_item.added', { type: 'response.output_item.added', output_index: 1, item: { type: 'message', id: 'msg_1' } }),
      event('response.output_text.delta', { type: 'response.output_text.delta', output_index: 1, delta: 'Node 22 is current' }),
      event('response.output_item.done', { type: 'response.output_item.done', output_index: 1, item: { type: 'message', id: 'msg_1', content: [{ type: 'output_text', text: 'Node 22 is current' }] } }),
      event('response.output_item.done', { type: 'response.output_item.done', output_index: 0, item: { type: 'web_search_call', id: 'ws_1', action: { type: 'search', queries: ['node'] } } }),
      event('response.completed', { type: 'response.completed', response: { status: 'completed', usage: { input_tokens: 3, output_tokens: 3, total_tokens: 6 } } }),
    ]
    vi.stubGlobal('fetch', vi.fn(async () => new Response(sseBody(stream), { status: 200 })))
    const chunks = await collect(request())
    const types = chunks.map(chunk => chunk.type)
    // No tool-call block: the server-side search stays invisible to the
    // harness, and the answer streams in the same turn.
    expect(types).not.toContain('tool-call-delta')
    expect(types.filter(t => t === 'block-start')).toHaveLength(1)
    expect(chunks.at(-1)).toMatchObject({ type: 'finish', reason: { kind: 'stop' } })
  })

  it('skips a pass-through function_call named web_search without slotting it', async () => {
    const stream = [
      event('response.output_item.added', { type: 'response.output_item.added', output_index: 0, item: { type: 'function_call', id: 'fc_1', call_id: 'call_00_1', name: 'web_search', status: 'in_progress' } }),
      event('response.function_call_arguments.delta', { type: 'response.function_call_arguments.delta', output_index: 0, delta: '{"queries":["news"]}' }),
      event('response.output_item.done', { type: 'response.output_item.done', output_index: 0, item: { type: 'function_call', id: 'fc_1', call_id: 'call_00_1', name: 'web_search', arguments: '{"queries":["news"]}' } }),
      event('response.output_item.added', { type: 'response.output_item.added', output_index: 1, item: { type: 'message', id: 'msg_1' } }),
      event('response.output_text.delta', { type: 'response.output_text.delta', output_index: 1, delta: 'Here is the news' }),
      event('response.output_item.done', { type: 'response.output_item.done', output_index: 1, item: { type: 'message', id: 'msg_1', content: [{ type: 'output_text', text: 'Here is the news' }] } }),
      event('response.completed', { type: 'response.completed', response: { status: 'completed', usage: { input_tokens: 3, output_tokens: 3, total_tokens: 6 } } }),
    ]
    vi.stubGlobal('fetch', vi.fn(async () => new Response(sseBody(stream), { status: 200 })))
    const chunks = await collect(request())
    const types = chunks.map(chunk => chunk.type)
    expect(types).not.toContain('tool-call-delta')
    expect(chunks.at(-1)).toMatchObject({ type: 'finish', reason: { kind: 'stop' } })
  })

  it('produces a tool-call block and finish tool-calls for local function calls', async () => {
    const stream = [
      event('response.output_item.added', { type: 'response.output_item.added', output_index: 0, item: { type: 'function_call', id: 'fc_1', call_id: 'call_00_1', name: 'bash' } }),
      event('response.function_call_arguments.delta', { type: 'response.function_call_arguments.delta', output_index: 0, delta: '{"cmd":' }),
      event('response.function_call_arguments.delta', { type: 'response.function_call_arguments.delta', output_index: 0, delta: '"ls"}' }),
      event('response.output_item.done', { type: 'response.output_item.done', output_index: 0, item: { type: 'function_call', id: 'fc_1', call_id: 'call_00_1', name: 'bash', arguments: '{"cmd":"ls"}' } }),
      event('response.completed', { type: 'response.completed', response: { status: 'completed', usage: { input_tokens: 1, output_tokens: 1, total_tokens: 2 } } }),
    ]
    vi.stubGlobal('fetch', vi.fn(async () => new Response(sseBody(stream), { status: 200 })))
    const chunks = await collect(request())
    expect(chunks.map(chunk => chunk.type)).toEqual(['block-start', 'tool-call-delta', 'tool-call-delta', 'block-end', 'usage', 'finish'])
    const toolCall = chunks[3] as Extract<StreamChunk, { type: 'block-end' }>
    expect(toolCall.block).toMatchObject({ type: 'tool-call', id: 'call_00_1|fc_1', name: 'bash', arguments: '{"cmd":"ls"}' })
    expect(chunks.at(-1)).toMatchObject({ type: 'finish', reason: { kind: 'tool-calls' } })
  })

  it('removes speculative sandbox escalation arguments from full-access tool calls', async () => {
    const argumentsText = JSON.stringify({
      command: 'Get-ChildItem C:\\',
      sandbox_permissions: 'require_escalated',
      justification: 'Need full filesystem access',
    })
    const stream = [
      event('response.output_item.added', { type: 'response.output_item.added', output_index: 0, item: { type: 'function_call', id: 'fc_1', call_id: 'call_00_1', name: 'powershell' } }),
      event('response.function_call_arguments.delta', { type: 'response.function_call_arguments.delta', output_index: 0, delta: argumentsText }),
      event('response.output_item.done', { type: 'response.output_item.done', output_index: 0, item: { type: 'function_call', id: 'fc_1', call_id: 'call_00_1', name: 'powershell', arguments: argumentsText } }),
      event('response.completed', { type: 'response.completed', response: { status: 'completed', usage: { input_tokens: 1, output_tokens: 1, total_tokens: 2 } } }),
    ]
    vi.stubGlobal('fetch', vi.fn(async () => new Response(sseBody(stream), { status: 200 })))
    const chunks = await collect(request())
    const toolCall = chunks.find(chunk => chunk.type === 'block-end') as Extract<StreamChunk, { type: 'block-end' }>
    expect(toolCall.block).toMatchObject({
      type: 'tool-call',
      name: 'powershell',
      arguments: JSON.stringify({ command: 'Get-ChildItem C:\\' }),
    })
  })

  it('preserves sandbox escalation arguments when retrying a real DSH denial', async () => {
    const argumentsText = JSON.stringify({
      command: 'Get-Content C:\\protected\\file.txt',
      sandbox_permissions: 'require_escalated',
      justification: 'Retry after the sandbox denial',
    })
    const stream = [
      event('response.output_item.added', { type: 'response.output_item.added', output_index: 0, item: { type: 'function_call', id: 'fc_2', call_id: 'call_00_2', name: 'powershell' } }),
      event('response.function_call_arguments.delta', { type: 'response.function_call_arguments.delta', output_index: 0, delta: argumentsText }),
      event('response.output_item.done', { type: 'response.output_item.done', output_index: 0, item: { type: 'function_call', id: 'fc_2', call_id: 'call_00_2', name: 'powershell', arguments: argumentsText } }),
      event('response.completed', { type: 'response.completed', response: { status: 'completed', usage: { input_tokens: 1, output_tokens: 1, total_tokens: 2 } } }),
    ]
    const denial = {
      id: 'u-denial' as Message['id'],
      role: 'user' as const,
      content: [{
        type: 'tool-result' as const,
        toolCallId: 'call_previous|fc_previous',
        content: [{ type: 'text' as const, text: '[sandbox: file access denied under full-access] C:\\protected\\file.txt' }],
        isError: true,
      }],
      source: { kind: 'user' as const },
    } as Message
    vi.stubGlobal('fetch', vi.fn(async () => new Response(sseBody(stream), { status: 200 })))
    const chunks = await collect(request({ messages: [denial] }))
    const toolCall = chunks.find(chunk => chunk.type === 'block-end') as Extract<StreamChunk, { type: 'block-end' }>
    expect(toolCall.block).toMatchObject({
      type: 'tool-call',
      name: 'powershell',
      arguments: argumentsText,
    })
  })

  it('streams reasoning deltas', async () => {
    const stream = [
      event('response.output_item.added', { type: 'response.output_item.added', output_index: 0, item: { type: 'reasoning', id: 'rs_1' } }),
      event('response.reasoning_text.delta', { type: 'response.reasoning_text.delta', output_index: 0, delta: 'thinking' }),
      event('response.output_item.done', { type: 'response.output_item.done', output_index: 0, item: { type: 'reasoning', id: 'rs_1', content: [{ type: 'reasoning_text', text: 'thinking' }] } }),
      event('response.completed', { type: 'response.completed', response: { status: 'completed', usage: { input_tokens: 1, output_tokens: 1, total_tokens: 2 } } }),
    ]
    vi.stubGlobal('fetch', vi.fn(async () => new Response(sseBody(stream), { status: 200 })))
    const chunks = await collect(request())
    expect(chunks).toContainEqual(expect.objectContaining({ type: 'reasoning-delta', text: 'thinking' }))
  })

  it('maps HTTP 401 to an error finish with AUTH', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => new Response('denied', { status: 401 })))
    const chunks = await collect(request())
    const finish = chunks.at(-1) as Extract<StreamChunk, { type: 'finish' }>
    expect(finish.type).toBe('finish')
    expect(finish.reason.kind).toBe('error')
    if (finish.reason.kind === 'error') expect(finish.reason.failure.code).toBe('AUTH')
  })

  it('cancels the response body on a non-2xx reply', async () => {
    const body = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(encoder.encode('denied'))
        controller.close()
      },
    })
    const cancelSpy = vi.spyOn(body, 'cancel')
    vi.stubGlobal('fetch', vi.fn(async () => new Response(body, { status: 429, headers: { 'retry-after': '2' } })))
    const chunks = await collect(request())
    const finish = chunks.at(-1) as Extract<StreamChunk, { type: 'finish' }>
    expect(finish.reason.kind).toBe('error')
    if (finish.reason.kind === 'error') {
      expect(finish.reason.failure.code).toBe('RATE_LIMIT')
      expect(finish.reason.failure.providerRetryAfterMs).toBe(2000)
    }
    expect(cancelSpy).toHaveBeenCalled()
  })

  it('maps response.failed to an error finish', async () => {
    const stream = [
      event('response.failed', { type: 'response.failed', response: { status: 'failed', error: { message: 'upstream exploded' } } }),
    ]
    vi.stubGlobal('fetch', vi.fn(async () => new Response(sseBody(stream), { status: 200 })))
    const chunks = await collect(request())
    const finish = chunks.at(-1) as Extract<StreamChunk, { type: 'finish' }>
    expect(finish.reason.kind).toBe('error')
  })

  it('maps incomplete to max-tokens finish', async () => {
    const stream = [
      event('response.incomplete', { type: 'response.incomplete', response: { status: 'incomplete', usage: { input_tokens: 1, output_tokens: 1, total_tokens: 2 } } }),
    ]
    vi.stubGlobal('fetch', vi.fn(async () => new Response(sseBody(stream), { status: 200 })))
    const chunks = await collect(request())
    expect(chunks.at(-1)).toMatchObject({ type: 'finish', reason: { kind: 'max-tokens' } })
  })

  it('yields an error finish on transport failure instead of throwing', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => { throw new Error('fetch failed: ECONNREFUSED') }))
    const chunks = await collect(request())
    const finish = chunks.at(-1) as Extract<StreamChunk, { type: 'finish' }>
    expect(finish.reason.kind).toBe('error')
    if (finish.reason.kind === 'error') expect(finish.reason.failure.code).toBe('TRANSPORT')
  })

  it('closes open slots and yields a transport error when the stream ends without a terminal event', async () => {
    const stream = [
      event('response.output_item.added', { type: 'response.output_item.added', output_index: 0, item: { type: 'function_call', id: 'fc_1', call_id: 'call_00_1', name: 'bash' } }),
      event('response.function_call_arguments.delta', { type: 'response.function_call_arguments.delta', output_index: 0, delta: '{"cmd":' }),
      // no terminal event: the tool slot must be closed by the transport error
    ]
    vi.stubGlobal('fetch', vi.fn(async () => new Response(sseBody(stream), { status: 200 })))
    const chunks = await collect(request())
    const types = chunks.map(chunk => chunk.type)
    expect(types).toContain('block-end')
    const finish = chunks.at(-1) as Extract<StreamChunk, { type: 'finish' }>
    expect(finish.reason.kind).toBe('error')
    if (finish.reason.kind === 'error') expect(finish.reason.failure.code).toBe('TRANSPORT')
  })

  it('yields an aborted finish when the request signal aborts', async () => {
    const controller = new AbortController()
    vi.stubGlobal('fetch', vi.fn(async (_url, init) => {
      const signal = (init as RequestInit).signal
      return new Response(new ReadableStream({
        start(c) {
          c.enqueue(encoder.encode(event('response.output_item.added', { type: 'response.output_item.added', output_index: 0, item: { type: 'message', id: 'msg_1' } })))
          c.enqueue(encoder.encode(event('response.output_text.delta', { type: 'response.output_text.delta', output_index: 0, delta: 'partial' })))
          signal?.addEventListener('abort', () => c.error(new Error('aborted')))
        },
      }), { status: 200 })
    }))
    const req = request({ signal: controller.signal })
    const chunks: StreamChunk[] = []
    for await (const chunk of inlineStream(req, candidate, hooks, cfg)) {
      chunks.push(chunk)
      if (chunk.type === 'block-start') controller.abort()
    }
    const finish = chunks.at(-1) as Extract<StreamChunk, { type: 'finish' }>
    expect(finish.reason.kind).toBe('aborted')
  })

  it('yields a TIMEOUT finish when the idle watchdog trips', async () => {
    vi.stubGlobal('fetch', vi.fn(async (_url, init) => {
      const signal = (init as RequestInit).signal
      return new Response(new ReadableStream({
        start(c) {
          c.enqueue(encoder.encode(event('response.output_item.added', { type: 'response.output_item.added', output_index: 0, item: { type: 'message', id: 'msg_1' } })))
          signal?.addEventListener('abort', () => c.error(new Error('aborted')))
        },
      }), { status: 200 })
    }))
    const localCfg: InlineConfig = { ...cfg, idleTimeoutMs: 50 }
    const chunks: StreamChunk[] = []
    for await (const chunk of inlineStream(request(), candidate, hooks, localCfg)) chunks.push(chunk)
    const finish = chunks.at(-1) as Extract<StreamChunk, { type: 'finish' }>
    expect(finish.type).toBe('finish')
    expect(finish.reason.kind).toBe('error')
    if (finish.reason.kind === 'error') expect(finish.reason.failure.code).toBe('TIMEOUT')
  })

  it('yields an error finish when resolveApiKey throws instead of throwing', async () => {
    const failingHooks = { resolveApiKey: async () => { throw new Error('credential service down') } }
    vi.stubGlobal('fetch', vi.fn(async () => { throw new Error('should never be called') }))
    const chunks: StreamChunk[] = []
    for await (const chunk of inlineStream(request(), candidate, failingHooks, cfg)) chunks.push(chunk)
    const finish = chunks.at(-1) as Extract<StreamChunk, { type: 'finish' }>
    expect(finish.type).toBe('finish')
    expect(finish.reason.kind).toBe('error')
  })

  it('yields an aborted finish when the request aborts during preflight', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => { throw new Error('should never be called') }))
    const controller = new AbortController()
    const slowHooks = {
      resolveApiKey: async () => {
        await new Promise((resolve) => setTimeout(resolve, 50))
        return 'secret'
      },
    }
    const req = request({ signal: controller.signal })
    const chunks: StreamChunk[] = []
    const draining = (async () => {
      for await (const chunk of inlineStream(req, candidate, slowHooks, cfg)) chunks.push(chunk)
    })()
    setTimeout(() => controller.abort(), 10)
    await draining
    const finish = chunks.at(-1) as Extract<StreamChunk, { type: 'finish' }>
    expect(finish.reason.kind).toBe('aborted')
  })

  it('yields TIMEOUT when the watchdog trips while fetch is still waiting for headers', async () => {
    vi.stubGlobal('fetch', vi.fn(async (_url, init) => new Promise((_resolve, reject) => {
      const signal = (init as RequestInit).signal
      signal?.addEventListener('abort', () => reject(new DOMException('The operation was aborted', 'AbortError')))
    })))
    const localCfg: InlineConfig = { ...cfg, idleTimeoutMs: 50 }
    const chunks: StreamChunk[] = []
    for await (const chunk of inlineStream(request(), candidate, hooks, localCfg)) chunks.push(chunk)
    const finish = chunks.at(-1) as Extract<StreamChunk, { type: 'finish' }>
    expect(finish.reason.kind).toBe('error')
    if (finish.reason.kind === 'error') expect(finish.reason.failure.code).toBe('TIMEOUT')
  })

  it('yields TIMEOUT when the watchdog trips during credential resolution', async () => {
    const fetchMock = vi.fn(async () => { throw new Error('should never be called') })
    vi.stubGlobal('fetch', fetchMock)
    const hangingHooks = { resolveApiKey: async () => new Promise<string | undefined>(() => undefined) }
    const localCfg: InlineConfig = { ...cfg, idleTimeoutMs: 50 }
    const chunks: StreamChunk[] = []
    for await (const chunk of inlineStream(request(), candidate, hangingHooks, localCfg)) chunks.push(chunk)
    const finish = chunks.at(-1) as Extract<StreamChunk, { type: 'finish' }>
    expect(finish.type).toBe('finish')
    expect(finish.reason.kind).toBe('error')
    if (finish.reason.kind === 'error') expect(finish.reason.failure.code).toBe('TIMEOUT')
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it('maps a content-filter incomplete reply to an error finish instead of max-tokens', async () => {
    const stream = [
      event('response.output_item.added', { type: 'response.output_item.added', output_index: 0, item: { type: 'message', id: 'msg_1' } }),
      event('response.output_text.delta', { type: 'response.output_text.delta', output_index: 0, delta: 'partial' }),
      event('response.incomplete', { type: 'response.incomplete', response: { status: 'incomplete', incomplete_details: { reason: 'content_filter' }, usage: { input_tokens: 1, output_tokens: 1, total_tokens: 2 } } }),
    ]
    vi.stubGlobal('fetch', vi.fn(async () => new Response(sseBody(stream), { status: 200 })))
    const chunks = await collect(request())
    const finish = chunks.at(-1) as Extract<StreamChunk, { type: 'finish' }>
    expect(finish.reason.kind).toBe('error')
    if (finish.reason.kind === 'error') expect(finish.reason.failure.message).toContain('content_filter')
  })

  it('skips a non-object SSE payload instead of crashing the stream', async () => {
    const stream = [
      'event: message\ndata: null\n\n',
      event('response.output_item.added', { type: 'response.output_item.added', output_index: 0, item: { type: 'message', id: 'msg_1' } }),
      event('response.output_text.delta', { type: 'response.output_text.delta', output_index: 0, delta: 'hi' }),
      event('response.output_item.done', { type: 'response.output_item.done', output_index: 0, item: { type: 'message', id: 'msg_1', content: [{ type: 'output_text', text: 'hi' }] } }),
      event('response.completed', { type: 'response.completed', response: { status: 'completed', usage: { input_tokens: 1, output_tokens: 1, total_tokens: 2 } } }),
    ]
    vi.stubGlobal('fetch', vi.fn(async () => new Response(sseBody(stream), { status: 200 })))
    const chunks = await collect(request())
    expect(chunks.at(-1)).toMatchObject({ type: 'finish', reason: { kind: 'stop' } })
  })
})

describe('inlineWireStream', () => {
  it('dispatches to the Responses wire for a responses candidate', async () => {
    const plan = { settle: async () => candidate } as unknown as SearchPlan
    const stream = textOnlyStream()
    const fetchMock = vi.fn(async () => new Response(sseBody(stream), { status: 200 }))
    vi.stubGlobal('fetch', fetchMock)
    const chunks: StreamChunk[] = []
    for await (const chunk of inlineWireStream(request(), plan, hooks, cfg)) chunks.push(chunk)
    expect(chunks.at(-1)).toMatchObject({ type: 'finish', reason: { kind: 'stop' } })
    const [url] = fetchMock.mock.calls[0] as [string]
    expect(url).toContain('/responses')
  })

  it('serves the wire body with the candidate spelling the probe verified', async () => {
    const versionedCandidate: SearchPlanCandidate = { ...candidate, webSearchToolType: 'web_search_2025_08_26' }
    const plan = { settle: async () => versionedCandidate } as unknown as SearchPlan
    const fetchMock = vi.fn(async () => new Response(sseBody(textOnlyStream()), { status: 200 }))
    vi.stubGlobal('fetch', fetchMock)
    const chunks: StreamChunk[] = []
    for await (const chunk of inlineWireStream(request(), plan, hooks, cfg)) chunks.push(chunk)
    expect(chunks.at(-1)).toMatchObject({ type: 'finish', reason: { kind: 'stop' } })
    const [, init] = fetchMock.mock.calls[0] as [string, RequestInit]
    expect((JSON.parse(String(init.body)) as { tools: unknown[] }).tools).toContainEqual({ type: 'web_search_2025_08_26' })
  })

  it('dispatches to the Anthropic wire when the settled candidate is anthropic', async () => {
    const anthropicCandidate: SearchPlanCandidate = { ...candidate, protocol: 'anthropic-messages' }
    const plan = { settle: async () => anthropicCandidate } as unknown as SearchPlan
    const stream = [
      event('message_start', { type: 'message_start', message: { id: 'msg_1', usage: { input_tokens: 1, output_tokens: 0 } } }),
      event('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text' } }),
      event('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: 'hi' } }),
      event('content_block_stop', { type: 'content_block_stop', index: 0 }),
      event('message_delta', { type: 'message_delta', delta: { stop_reason: 'end_turn' }, usage: { output_tokens: 1 } }),
      event('message_stop', { type: 'message_stop' }),
    ]
    const fetchMock = vi.fn(async () => new Response(sseBody(stream), { status: 200 }))
    vi.stubGlobal('fetch', fetchMock)
    const chunks: StreamChunk[] = []
    for await (const chunk of inlineWireStream(request(), plan, hooks, cfg)) chunks.push(chunk)
    expect(chunks.at(-1)).toMatchObject({ type: 'finish', reason: { kind: 'stop' } })
    const [url] = fetchMock.mock.calls[0] as [string]
    expect(url).toContain('/messages')
  })

  it('yields an error finish when the plan fails to settle', async () => {
    const plan = { settle: async () => { throw new Error('native web search is disabled: nope') } } as unknown as SearchPlan
    vi.stubGlobal('fetch', vi.fn(async () => { throw new Error('should never be called') }))
    const chunks: StreamChunk[] = []
    for await (const chunk of inlineWireStream(request(), plan, hooks, cfg)) chunks.push(chunk)
    const finish = chunks.at(-1) as Extract<StreamChunk, { type: 'finish' }>
    expect(finish.reason.kind).toBe('error')
  })
})
