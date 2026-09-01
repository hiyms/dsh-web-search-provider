/**
 * Serialization tests: dsh Message blocks and tool schemas project onto the
 * OpenAI Responses input vocabulary with the exact shapes the gateway
 * requires (reasoning pass-back, call_id pairing, versioned web tool).
 */

import { describe, expect, it } from 'vitest'
import { buildAnthropicWireBody, buildWireBody, flattenText, pairedToolCallIds, serializeMessage, shortHash, splitCallId, wireTools } from '../../src/serialize.ts'
import { RESPONSES_WEB_SEARCH_TOOL_TYPE } from '../../src/plan.ts'
import type { Message } from '@deepseek-ai/dsh-llm'
import { markAgentLoopRequest } from '@deepseek-ai/dsh-llm'

function message(role: Message['role'], blocks: unknown[]): Message {
  return { id: `m-${role}-1`, role, content: blocks as Message['content'], source: { kind: 'user' } as Message['source'] }
}

describe('shortHash', () => {
  it('is stable and 8 chars', () => {
    const a = shortHash('hello world')
    expect(a).toBe(shortHash('hello world'))
    expect(a).toHaveLength(8)
  })

  it('differs for different inputs', () => {
    expect(shortHash('hello')).not.toBe(shortHash('world'))
  })
})

describe('splitCallId', () => {
  it('splits pi-ai style call ids', () => {
    expect(splitCallId('call_00_abc|fc_1234')).toEqual({ callId: 'call_00_abc', itemId: 'fc_1234' })
  })

  it('synthesizes an item id for foreign call ids', () => {
    const { callId, itemId } = splitCallId('call_99')
    expect(callId).toBe('call_99')
    expect(itemId).toMatch(/^fc_/)
  })
})

describe('flattenText', () => {
  it('joins text blocks and skips others', () => {
    expect(flattenText([{ type: 'text', text: 'a' }, { type: 'reasoning', text: 'r' }, { type: 'text', text: 'b' }])).toBe('ab')
  })
})

describe('serializeMessage', () => {
  it('serializes a plain user text message', () => {
    const result = serializeMessage(message('user', [{ type: 'text', text: 'hello' }]))
    expect(result).toEqual([{ role: 'user', content: [{ type: 'input_text', text: 'hello' }] }])
  })

  describe('pairedToolCallIds', () => {
    it('keeps only call ids with both replay halves', () => {
      const messages = [
        message('assistant', [
          { type: 'tool-call', id: 'call_paired|fc_1', name: 'bash', arguments: '{}' },
          { type: 'tool-call', id: 'call_orphan|fc_2', name: 'web_search', arguments: '{}' },
        ]),
        message('user', [
          { type: 'tool-result', toolCallId: 'call_paired|fc_1', content: [{ type: 'text', text: 'ok' }] },
          { type: 'tool-result', toolCallId: 'call_result_only|fc_3', content: [{ type: 'text', text: 'stale' }] },
        ]),
      ]
      expect([...pairedToolCallIds(messages)]).toEqual(['call_paired'])
    })
  })

  it('serializes tool results as function_call_output', () => {
    const result = serializeMessage(message('user', [{ type: 'tool-result', toolCallId: 'call_00_abc|fc_1', content: [{ type: 'text', text: '42' }] }]))
    expect(result).toEqual([{ type: 'function_call_output', call_id: 'call_00_abc', output: '42' }])
  })

  it('prefixes error tool results', () => {
    const result = serializeMessage(message('user', [{ type: 'tool-result', toolCallId: 'c|fc_1', content: [{ type: 'text', text: 'boom' }], isError: true }]))
    expect((result[0] as { output: string }).output).toBe('[Error] boom')
  })

  it('serializes assistant text with a stable message id', () => {
    const result = serializeMessage(message('assistant', [{ type: 'text', text: 'hi' }]))
    const item = result[0] as { type: string; id: string; role: string; content: Array<{ type: string; text: string }> }
    expect(item.type).toBe('message')
    expect(item.id).toMatch(/^msg_/)
    expect(item.role).toBe('assistant')
    expect(item.content).toEqual([{ type: 'output_text', text: 'hi', annotations: [] }])
  })

  it('serializes reasoning blocks with a synthesized id', () => {
    const result = serializeMessage(message('assistant', [{ type: 'reasoning', text: 'think' }]))
    const item = result[0] as { type: string; id: string; content: Array<{ type: string; text: string }> }
    expect(item.type).toBe('reasoning')
    expect(item.id).toMatch(/^rs_/)
    expect(item.content).toEqual([{ type: 'reasoning_text', text: 'think' }])
  })

  it('serializes tool calls as function_call with split ids', () => {
    const result = serializeMessage(message('assistant', [{ type: 'tool-call', id: 'call_00_abc|fc_1234', name: 'bash', arguments: '{"cmd":"ls"}' }]))
    expect(result[0]).toEqual({
      type: 'function_call',
      call_id: 'call_00_abc',
      id: 'fc_1234',
      name: 'bash',
      arguments: '{"cmd":"ls"}',
      status: 'completed',
    })
  })

  it('maps invalid replay item ids to stable short ids', () => {
    const itemId = 'tool call/with invalid characters'
    const result = serializeMessage(message('assistant', [{ type: 'tool-call', id: `call_00_abc|${itemId}`, name: 'bash', arguments: '{}' }]))

    expect(result[0]).toMatchObject({
      call_id: 'call_00_abc',
      id: `fc_${shortHash(itemId)}`,
    })
  })

  it('maps oversized replay item ids to stable short ids', () => {
    const itemId = `fc_${'a'.repeat(62)}`
    const result = serializeMessage(message('assistant', [{ type: 'tool-call', id: `call_00_abc|${itemId}`, name: 'bash', arguments: '{}' }]))
    const replayItemId = (result[0] as { id: string }).id

    expect(replayItemId).toBe(`fc_${shortHash(itemId)}`)
    expect(replayItemId.length).toBeLessThanOrEqual(64)
  })

  it('throws UnsupportedContentError on image content', () => {
    expect(() => serializeMessage(message('user', [{ type: 'image', attachment: {} as never }]))).toThrow(/image/)
  })
})

describe('wireTools', () => {
  it('maps function tools and appends the server-side web tool', () => {
    const result = wireTools([{ name: 'bash', description: 'run', parameters: {} }], true, 'web_search')
    expect(result).toEqual([
      { type: 'function', name: 'bash', description: 'run', parameters: {} },
      { type: 'web_search' },
    ])
  })

  it('appends the versioned spelling when the candidate carries it', () => {
    const result = wireTools([{ name: 'bash', description: 'run', parameters: {} }], true, 'web_search_2025_08_26')
    expect(result).toEqual([
      { type: 'function', name: 'bash', description: 'run', parameters: {} },
      { type: 'web_search_2025_08_26' },
    ])
  })

  it('strips all shadowed function variants when configured', () => {
    const result = wireTools([
      { name: 'web_search', description: 'search', parameters: {} },
      { name: 'open_page', description: 'open', parameters: {} },
      { name: 'find_in_page', description: 'find', parameters: {} },
    ], true, 'web_search')
    expect(result).toEqual([{ type: 'web_search' }])
  })

  it('keeps the shadowed function variants when stripping is off', () => {
    const result = wireTools([
      { name: 'web_search', description: 'search', parameters: {} },
      { name: 'open_page', description: 'open', parameters: {} },
      { name: 'find_in_page', description: 'find', parameters: {} },
    ], false, 'web_search')
    expect(result).toHaveLength(4)
  })
})

describe('buildWireBody', () => {
  it('builds the complete request body', () => {
    const request = markAgentLoopRequest({
      provider: 'opencode-go-response',
      model: 'deepseek-v4-flash',
      messages: [message('user', [{ type: 'text', text: 'hi' }])],
      system: 'sys',
      tools: [{ name: 'bash', description: 'run', parameters: {} }],
      maxTokens: 100,
      temperature: 0.5,
      sessionId: 'sess-1234567890abcdef',
    })
    const body = buildWireBody(request, { includeSources: true, stripServerTools: true }, 'configured-model', 'web_search') as Record<string, unknown>
    expect(body.model).toBe('configured-model')
    expect(body.stream).toBe(true)
    expect(body.include).toEqual(['web_search_call.action.sources'])
    expect(body.max_output_tokens).toBe(100)
    expect(body.temperature).toBe(0.5)
    expect(body.prompt_cache_key).toBe('sess-1234567890abcdef')
    expect((body.input as unknown[])[0]).toEqual({ role: 'user', content: [{ type: 'input_text', text: 'sys' }] })
    expect((body.tools as unknown[]).at(-1)).toEqual({ type: 'web_search' })
  })

  it('serves the resolved plan model, not the loop request model', () => {
    const request = markAgentLoopRequest({ provider: 'p', model: 'loop-model', messages: [] })
    const body = buildWireBody(request, { includeSources: false, stripServerTools: true }, 'configured-model', 'web_search') as Record<string, unknown>
    expect(body.model).toBe('configured-model')
  })

  it('honors a maxTokens below 16 verbatim instead of flooring it upward', () => {
    const request = markAgentLoopRequest({ provider: 'p', model: 'm', messages: [], maxTokens: 4 })
    const body = buildWireBody(request, { includeSources: false, stripServerTools: true }, 'm', 'web_search') as Record<string, unknown>
    expect(body.max_output_tokens).toBe(4)
  })

  it('floors a zero maxTokens at 1 instead of sending an invalid cap', () => {
    const request = markAgentLoopRequest({ provider: 'p', model: 'm', messages: [], maxTokens: 0 })
    const body = buildWireBody(request, { includeSources: false, stripServerTools: true }, 'm', 'web_search') as Record<string, unknown>
    expect(body.max_output_tokens).toBe(1)
  })

  it('omits optional fields when absent', () => {
    const request = markAgentLoopRequest({ provider: 'p', model: 'm', messages: [] })
    const body = buildWireBody(request, { includeSources: false, stripServerTools: true }, 'm', 'web_search') as Record<string, unknown>
    expect(body.include).toBeUndefined()
    expect(body.max_output_tokens).toBeUndefined()
    expect(body.prompt_cache_key).toBeUndefined()
  })

  it('drops orphaned replay calls and results while preserving paired calls', () => {
    const request = markAgentLoopRequest({
      provider: 'p',
      model: 'm',
      messages: [
        message('assistant', [
          { type: 'tool-call', id: 'call_paired|fc_1', name: 'bash', arguments: '{}' },
          { type: 'tool-call', id: 'call_orphan|fc_2', name: 'web_search', arguments: '{}' },
        ]),
        message('user', [
          { type: 'tool-result', toolCallId: 'call_paired|fc_1', content: [{ type: 'text', text: 'ok' }] },
          { type: 'tool-result', toolCallId: 'call_result_only|fc_3', content: [{ type: 'text', text: 'stale' }] },
        ]),
      ],
    })
    const body = buildWireBody(request, { includeSources: false, stripServerTools: true }, 'm', 'web_search') as {
      input: Array<{ type?: string; call_id?: string }>
    }
    expect(body.input.filter(item => item.type === 'function_call').map(item => item.call_id)).toEqual(['call_paired'])
    expect(body.input.filter(item => item.type === 'function_call_output').map(item => item.call_id)).toEqual(['call_paired'])
  })
})

describe('buildAnthropicWireBody', () => {
  it('puts the system prompt top-level and strips the web_search function variant', () => {
    const request = markAgentLoopRequest({
      provider: 'p',
      model: 'deepseek-v4-flash',
      messages: [message('user', [{ type: 'text', text: 'hi' }])],
      system: 'sys',
      tools: [{ name: 'web_search', description: 'search', parameters: {} }, { name: 'bash', description: 'run', parameters: {} }],
      maxTokens: 100,
    })
    const body = buildAnthropicWireBody(request, { stripServerTools: true }, 'configured-model') as Record<string, unknown>
    expect(body.model).toBe('configured-model')
    expect(body.system).toEqual([{ type: 'text', text: 'sys' }])
    expect(body.stream).toBe(true)
    expect(body.max_tokens).toBe(100)
    const tools = body.tools as Array<Record<string, unknown>>
    // function variant stripped; server-side tool appended
    expect(tools.filter(tool => tool.name === 'web_search')).toHaveLength(1)
    expect(tools.find(tool => tool.name === 'web_search')).toMatchObject({ type: 'web_search_20250305', max_uses: 5 })
    expect(tools.find(tool => tool.name === 'bash')).toEqual({ name: 'bash', description: 'run', input_schema: {} })
  })

  it('serializes tool results as tool_result blocks and reasoning as plain text', () => {
    const request = markAgentLoopRequest({
      provider: 'p',
      model: 'm',
      messages: [
        message('assistant', [
          { type: 'reasoning', text: 'think' },
          { type: 'text', text: 'answer' },
          { type: 'tool-call', id: 'call_00_x|fc_1', name: 'bash', arguments: '{"cmd":"ls"}' },
        ]),
        message('user', [{ type: 'tool-result', toolCallId: 'call_00_x|fc_1', content: [{ type: 'text', text: 'ok' }] }]),
      ],
    })
    const body = buildAnthropicWireBody(request, { stripServerTools: true }, 'm') as Record<string, unknown>
    const messages = body.messages as Array<Record<string, unknown>>
    const assistant = messages.find(m => m.role === 'assistant') as { content: Array<Record<string, unknown>> }
    // reasoning became plain text; tool call became tool_use
    expect(assistant.content.map(block => block.type)).toEqual(['text', 'text', 'tool_use'])
    expect(assistant.content[2]).toMatchObject({ type: 'tool_use', id: 'call_00_x', name: 'bash', input: { cmd: 'ls' } })
    const toolResult = messages.find(m => m.role === 'user') as { content: Array<Record<string, unknown>> }
    expect(toolResult.content[0]).toMatchObject({ type: 'tool_result', tool_use_id: 'call_00_x', content: 'ok' })
  })

  it('honors a maxTokens below 16 verbatim instead of flooring it upward', () => {
    const request = markAgentLoopRequest({
      provider: 'p',
      model: 'm',
      messages: [message('user', [{ type: 'text', text: 'hi' }])],
      maxTokens: 4,
    })
    const body = buildAnthropicWireBody(request, { stripServerTools: true }, 'm') as Record<string, unknown>
    expect(body.max_tokens).toBe(4)
  })

  it('floors a zero maxTokens at 1 instead of sending an invalid cap', () => {
    const request = markAgentLoopRequest({
      provider: 'p',
      model: 'm',
      messages: [message('user', [{ type: 'text', text: 'hi' }])],
      maxTokens: 0,
    })
    const body = buildAnthropicWireBody(request, { stripServerTools: true }, 'm') as Record<string, unknown>
    expect(body.max_tokens).toBe(1)
  })

  it('keeps tool_use ids paired across rounds for native Anthropic ids', () => {
    // A native Anthropic tool_use id (no `|` separator) must serialize back
    // to the same id so the follow-up tool_result pairs with it.
    const assistantReq = markAgentLoopRequest({
      provider: 'p',
      model: 'm',
      messages: [
        message('assistant', [{ type: 'tool-call', id: 'toolu_01ABC', name: 'bash', arguments: '{"cmd":"ls"}' }]),
      ],
    })
    const assistantBody = buildAnthropicWireBody(assistantReq, { stripServerTools: true }, 'm') as Record<string, unknown>
    const assistantMsg = (assistantBody.messages as Array<Record<string, unknown>>).find(m => m.role === 'assistant') as { content: Array<Record<string, unknown>> }
    expect(assistantMsg.content[0]).toMatchObject({ type: 'tool_use', id: 'toolu_01ABC' })

    const resultReq = markAgentLoopRequest({
      provider: 'p',
      model: 'm',
      messages: [
        message('user', [{ type: 'tool-result', toolCallId: 'toolu_01ABC', content: [{ type: 'text', text: 'ok' }] }]),
      ],
    })
    const resultBody = buildAnthropicWireBody(resultReq, { stripServerTools: true }, 'm') as Record<string, unknown>
    const resultMsg = (resultBody.messages as Array<Record<string, unknown>>).find(m => m.role === 'user') as { content: Array<Record<string, unknown>> }
    expect(resultMsg.content[0]).toMatchObject({ type: 'tool_result', tool_use_id: 'toolu_01ABC' })
  })
})
