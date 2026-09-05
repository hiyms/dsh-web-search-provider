/**
 * Anthropic Messages wire for the inline web search plugin: intercepts an
 * agent-loop model call, builds a Messages streaming request carrying the
 * server-side `web_search_20250305` tool, and maps the Anthropic event
 * stream onto the harness StreamChunk contract. Server-side search blocks
 * (`server_tool_use` / `web_search_tool_result`) are skipped — the gateway
 * executes the search inside the same request and the model answers from
 * its results in the same turn. The generator NEVER throws; every failure
 * becomes a finish chunk.
 * @module dsh-web-search-provider/wire-anthropic
 */

import type { GenerateOptions, StreamChunk, TokenUsage } from '@deepseek-ai/dsh-llm'
import { attributionHeaders, ToolCallId } from '@deepseek-ai/dsh-llm'
import type { SearchPlanCandidate } from './plan.ts'
import { abortedFinish, classifyHttpStatus, classifyWireError, errorFinish, parseRetryAfterMs } from './failure.ts'
import { abortable } from './http.ts'
import { buildAnthropicWireBody } from './serialize.ts'
import { parseSse } from './sse.ts'
import { IdleWatchdog } from './watchdog.ts'
import type { InlineConfig } from './config.ts'
import type { InlineHooks } from './wire.ts'

/** Path appended to the endpoint base. */
const MESSAGES_ENDPOINT = '/messages'

/** One in-flight Anthropic content block, keyed by its stream index. */
interface AnthropicSlot {
  /** Local, monotonically increasing block index for the harness. */
  index: number
  blockType: 'text' | 'tool-call' | 'reasoning'
  /** Harness tool-call id for tool slots (the Anthropic tool_use id). */
  id?: string
  name?: string
  text: string
  arguments: string
}

/** Map an Anthropic usage object onto the harness TokenUsage shape. */
export function mapAnthropicUsage(usage: {
  readonly input_tokens?: number
  readonly output_tokens?: number
  readonly cache_read_input_tokens?: number
  readonly cache_creation_input_tokens?: number
  readonly output_tokens_details?: { readonly thinking_tokens?: number }
}): TokenUsage {
  return {
    inputTokens: usage.input_tokens ?? 0,
    outputTokens: usage.output_tokens ?? 0,
    ...usage.cache_read_input_tokens !== undefined && usage.cache_read_input_tokens > 0
      ? { cacheReadTokens: usage.cache_read_input_tokens }
      : {},
    ...usage.cache_creation_input_tokens !== undefined && usage.cache_creation_input_tokens > 0
      ? { cacheWriteTokens: usage.cache_creation_input_tokens }
      : {},
    ...usage.output_tokens_details?.thinking_tokens !== undefined
      ? { reasoningTokens: usage.output_tokens_details.thinking_tokens }
      : {},
  }
}

/**
 * Run one inline model call through the Anthropic Messages API with the
 * server-side web search tool injected. Never throws; always ends with a
 * finish chunk.
 * @param request - the frozen agent-loop request.
 * @param candidate - the settled endpoint facts (probe-verified).
 * @param hooks - credential hooks.
 * @param cfg - the current plugin configuration.
 * @returns the StreamChunk stream for the loop's assembler.
 */
export async function* inlineAnthropicStream(
  request: GenerateOptions,
  candidate: SearchPlanCandidate,
  hooks: InlineHooks,
  cfg: InlineConfig,
): AsyncGenerator<StreamChunk> {
  // The controller exists for the whole call so an abort during preflight is
  // not missed; the watchdog runs from preflight on, so a credential
  // resolution that hangs forever trips it instead of wedging the request.
  const controller = new AbortController()
  const onAbort = (): void => controller.abort()
  request.signal?.addEventListener('abort', onAbort, { once: true })
  let watchdog: IdleWatchdog | undefined
  try {
    if (request.signal?.aborted ?? false) {
      yield abortedFinish()
      return
    }
    watchdog = new IdleWatchdog(cfg.idleTimeoutMs)
    watchdog.signal.addEventListener('abort', onAbort, { once: true })
    let apiKey: string | undefined
    try {
      apiKey = await abortable(hooks.resolveApiKey(candidate.apiKeyEnv), controller.signal)
    } catch (error) {
      if (request.signal?.aborted ?? false) {
        yield abortedFinish()
        return
      }
      if (watchdog.signal.aborted === true) {
        yield errorFinish({ message: 'inline stream idle timeout', code: 'TIMEOUT' })
        return
      }
      yield errorFinish(classifyWireError(error))
      return
    }
    if (apiKey === undefined || apiKey.length === 0) {
      yield errorFinish({ message: `no API key for "${candidate.apiKeyEnv}"`, code: 'AUTH' })
      return
    }
    // The idle window covers the wire phase; reset the bound consumed by the
    // key resolution.
    watchdog.reset()

    let response: Response
    try {
      response = await fetch(`${candidate.baseURL.replace(/\/+$/, '')}${MESSAGES_ENDPOINT}`, {
        method: 'POST',
        redirect: 'error',
        signal: controller.signal,
        headers: {
          'x-api-key': apiKey,
          authorization: `Bearer ${apiKey}`,
          'anthropic-version': candidate.apiVersion,
          'content-type': 'application/json',
          accept: 'text/event-stream',
          ...attributionHeaders(),
        },
        body: JSON.stringify(buildAnthropicWireBody(request, cfg, candidate.model)),
      })
    } catch (error) {
      if (request.signal?.aborted === true) {
        yield abortedFinish()
        return
      }
      if (watchdog.signal.aborted === true) {
        // Idle watchdog tripped while fetch was still waiting (headers or
        // body): map to a retryable TIMEOUT, not UNKNOWN.
        yield errorFinish({ message: 'inline stream idle timeout', code: 'TIMEOUT' })
        return
      }
      yield errorFinish(classifyWireError(error))
      return
    }
    if (!response.ok) {
      // Refuse the error body outright: nothing in it can serve the request,
      // and the connection should not stay open.
      void response.body?.cancel().catch(() => undefined)
      yield errorFinish(classifyHttpStatus(
        response.status,
        parseRetryAfterMs(response.headers.get('retry-after')),
        response.headers.get('x-request-id') ?? undefined,
        'Messages API',
      ))
      return
    }
    if (response.body === null) {
      yield errorFinish(classifyWireError(new Error('empty response body')))
      return
    }

    const slots = new Map<number, AnthropicSlot>()
    let localIndex = 0
    let stopReason: string | undefined
    let sawText = false
    let sawToolCall = false
    // Real Anthropic streams split usage: input-side fields arrive on
    // message_start.message.usage, output-side on message_delta.usage.
    let inputUsage: {
      input_tokens?: number
      cache_read_input_tokens?: number
      cache_creation_input_tokens?: number
    } | undefined

    const closeOpenSlots = function* (): Generator<StreamChunk> {
      for (const [streamIndex, slot] of slots) {
        if (slot.blockType === 'tool-call') {
          yield { type: 'block-end', index: slot.index, block: { type: 'tool-call', id: ToolCallId(slot.id ?? 'tool'), name: slot.name ?? '', arguments: slot.arguments } }
        } else if (slot.blockType === 'reasoning') {
          yield { type: 'block-end', index: slot.index, block: { type: 'reasoning', text: slot.text } }
        }
        slots.delete(streamIndex)
      }
    }

    try {
      let sawMessageStop = false
      for await (const parsed of parseSse(response.body)) {
        watchdog.reset()
        // A non-object payload (e.g. `data: null`) carries no event fields;
        // skipping it keeps a malformed event from crashing the stream.
        if (typeof parsed.data !== 'object' || parsed.data === null) continue
        const data = parsed.data as {
          type?: string
          index?: number
          content_block?: { type?: string; id?: string; name?: string; input?: unknown }
          delta?: { type?: string; text?: string; thinking?: string; partial_json?: string; stop_reason?: string }
          message?: { id?: string; usage?: unknown }
          usage?: {
            input_tokens?: number
            output_tokens?: number
            cache_read_input_tokens?: number
            cache_creation_input_tokens?: number
            output_tokens_details?: { thinking_tokens?: number }
          }
          error?: { type?: string; message?: string }
        }
        switch (data.type) {
          case 'message_start': {
            // Input-side usage (input_tokens and cache counts) lives here on
            // real Anthropic endpoints; merged with message_delta below.
            const usage = data.message?.usage as {
              input_tokens?: number
              cache_read_input_tokens?: number
              cache_creation_input_tokens?: number
            } | undefined
            if (usage !== undefined) inputUsage = usage
            break
          }
          case 'content_block_start': {
            const blockType = data.content_block?.type
            if (blockType === 'text') {
              slots.set(data.index ?? -1, { index: localIndex++, blockType: 'text', text: '', arguments: '' })
              yield { type: 'block-start', index: slots.get(data.index ?? -1)?.index ?? 0, blockType: 'text' }
            } else if (blockType === 'thinking' || blockType === 'redacted_thinking') {
              slots.set(data.index ?? -1, { index: localIndex++, blockType: 'reasoning', text: '', arguments: '' })
              yield { type: 'block-start', index: slots.get(data.index ?? -1)?.index ?? 0, blockType: 'reasoning' }
            } else if (blockType === 'tool_use') {
              sawToolCall = true
              slots.set(data.index ?? -1, {
                index: localIndex++,
                blockType: 'tool-call',
                id: data.content_block?.id ?? 'tool',
                name: data.content_block?.name,
                text: '',
                arguments: '',
              })
              yield { type: 'block-start', index: slots.get(data.index ?? -1)?.index ?? 0, blockType: 'tool-call' }
            }
            // server_tool_use / web_search_tool_result: the server-side search
            // ran inside this request — never slot them.
            break
          }
          case 'content_block_delta': {
            const slot = slots.get(data.index ?? -1)
            if (slot === undefined) break
            const deltaType = data.delta?.type
            if (deltaType === 'text_delta') {
              const text = data.delta?.text ?? ''
              if (text.length === 0) break
              sawText = true
              slot.text += text
              yield { type: 'text-delta', index: slot.index, text }
            } else if (deltaType === 'thinking_delta') {
              const text = data.delta?.thinking ?? ''
              if (text.length === 0) break
              slot.text += text
              yield { type: 'reasoning-delta', index: slot.index, text }
            } else if (deltaType === 'input_json_delta') {
              const json = data.delta?.partial_json ?? ''
              if (json.length === 0) break
              slot.arguments += json
              yield { type: 'tool-call-delta', index: slot.index, id: ToolCallId(slot.id ?? 'tool'), name: slot.name, argumentsDelta: json }
            }
            // signature_delta: ignored (no harness reasoning signature).
            break
          }
          case 'content_block_stop': {
            const slot = slots.get(data.index ?? -1)
            if (slot === undefined) break
            if (slot.blockType === 'tool-call') {
              yield { type: 'block-end', index: slot.index, block: { type: 'tool-call', id: ToolCallId(slot.id ?? 'tool'), name: slot.name ?? '', arguments: slot.arguments } }
            } else if (slot.blockType === 'reasoning') {
              yield { type: 'block-end', index: slot.index, block: { type: 'reasoning', text: slot.text } }
            }
            // Text slots never emit block-end (assembler uses deltas).
            slots.delete(data.index ?? -1)
            break
          }
          case 'message_delta': {
            const usage = data.usage
            if (usage !== undefined) {
              yield { type: 'usage', usage: mapAnthropicUsage({
                ...inputUsage,
                ...usage,
              }) }
            }
            if (data.delta?.stop_reason !== undefined) stopReason = data.delta.stop_reason
            break
          }
          case 'message_stop': {
            sawMessageStop = true
            // Close any slot the stream left open (e.g. after an aborted
            // turn), mirroring the Responses wire's closeOpenSlots paths.
            yield* closeOpenSlots()
            if (!sawToolCall && !sawText) {
              yield errorFinish({ message: 'the provider returned an empty response', code: 'EMPTY_RESPONSE' })
              return
            }
            if (stopReason === 'max_tokens') {
              yield { type: 'finish', reason: { kind: 'max-tokens' } }
              return
            }
            if (stopReason === 'tool_use' || sawToolCall) {
              yield { type: 'finish', reason: { kind: 'tool-calls' } }
              return
            }
            if (stopReason === 'refusal' || stopReason === 'sensitive') {
              yield errorFinish({ message: `the provider refused: ${stopReason}`, code: 'UNKNOWN' })
              return
            }
            yield { type: 'finish', reason: { kind: 'stop' } }
            return
          }
          case 'error': {
            sawMessageStop = true
            yield* closeOpenSlots()
            const message = data.error?.message ?? 'anthropic stream error'
            yield errorFinish(classifyWireError(new Error(message)))
            return
          }
          default:
            // ping, and unknown events: ignored.
            break
        }
      }
      if (!sawMessageStop) {
        yield* closeOpenSlots()
        yield errorFinish(classifyWireError(new Error('stream ended without a terminal event')))
      }
    } catch (error) {
      if (request.signal?.aborted ?? false) {
        yield abortedFinish()
        return
      }
      if (watchdog.signal.aborted === true) {
        yield* closeOpenSlots()
        yield errorFinish({ message: 'inline stream idle timeout', code: 'TIMEOUT' })
        return
      }
      yield* closeOpenSlots()
      yield errorFinish(classifyWireError(error))
    }
  } finally {
    request.signal?.removeEventListener('abort', onAbort)
    watchdog?.dispose()
  }
}
