/**
 * The short-circuit wire: intercepts an agent-loop model call, builds a
 * streaming request carrying the server-side `web_search` tool, and maps the
 * SSE events onto the harness StreamChunk contract. The generator NEVER
 * throws — every failure becomes a finish chunk, exactly what the loop's
 * catch-less `for await` requires. {@link inlineWireStream} settles the plan
 * first and then dispatches to the protocol the VERIFIED candidate speaks, so
 * the wire always matches the probe verdict (never the chat route snapshot).
 *
 * Server-side `web_search_call` items are skipped: the gateway executes the
 * search inside the same request and the model answers from its results in
 * the same turn, so the harness never sees a tool call. (UI search cards
 * were removed — the added latency outweighed the visual value.)
 * @module dsh-web-search-provider/wire
 */

import type { CallId, GenerateOptions, StreamChunk } from '@deepseek-ai/dsh-llm'
import { attributionHeaders } from '@deepseek-ai/dsh-llm'
import type { SearchPlan, SearchPlanCandidate } from './plan.ts'
import { WEB_SEARCH_TOOL_TYPE } from './plan.ts'
import { abortedFinish, classifyHttpStatus, classifyWireError, errorFinish, parseRetryAfterMs } from './failure.ts'
import { abortable } from './http.ts'
import { mapUsage } from './usage.ts'
import type { WireUsage } from './usage.ts'
import { buildWireBody, flattenText } from './serialize.ts'
import { parseSse } from './sse.ts'
import { IdleWatchdog } from './watchdog.ts'
import type { InlineConfig } from './config.ts'
import { inlineAnthropicStream } from './wire-anthropic.ts'

/** Path appended to the endpoint base. */
const RESPONSES_ENDPOINT = '/responses'

/** Marker emitted by DSH when a sandboxed tool call is denied file access. */
const SANDBOX_DENIAL_MARKER = '[sandbox: file access denied under'

/** Credential and cancellation hooks the stream resolves per operation. */
export interface InlineHooks {
  /** Resolve one credential reference through the credentials seam. */
  resolveApiKey: (apiKeyEnv: string) => Promise<string | undefined>
}

/** One in-flight server output item, keyed by the server output_index. */
interface Slot {
  /** Local, monotonically increasing block index for the harness. */
  index: number
  blockType: 'text' | 'tool-call' | 'reasoning'
  /** Harness tool-call id (`call_id|item_id`) for tool slots. */
  id?: string
  name?: string
  text: string
  arguments: string
  /** Whether a block-start was emitted (text slots open lazily). */
  opened?: boolean
}

/** Whether any message carries image content (unsupported on this wire). */
export function contentHasImages(request: GenerateOptions): boolean {
  return request.messages.some(message => message.content.some(block => block.type === 'image'))
}

/**
 * Whether the current turn is a retry immediately following a real DSH
 * sandbox denial.
 */
function hasGroundedSandboxEscalation(request: GenerateOptions): boolean {
  const last = request.messages.at(-1)
  if (last?.role !== 'user') return false
  return last.content.some(block =>
    block.type === 'tool-result'
    && flattenText(block.content).includes(SANDBOX_DENIAL_MARKER),
  )
}

/**
 * Strip speculative sandbox escalation fields while preserving sanctioned
 * retries grounded in the immediately preceding tool result.
 */
function sanitizeSandboxEscalationArguments(argumentsText: string, grounded: boolean): string {
  if (grounded) return argumentsText
  let parsed: unknown
  try {
    parsed = JSON.parse(argumentsText)
  } catch {
    return argumentsText
  }
  if (typeof parsed !== 'object' || parsed === null) return argumentsText
  const args = parsed as Record<string, unknown>
  if (!('sandbox_permissions' in args) && !('justification' in args)) return argumentsText
  delete args.sandbox_permissions
  delete args.justification
  return JSON.stringify(args)
}

/**
 * Run one inline model call through whichever wire the settled candidate
 * speaks. The plan is settled HERE — once — so the protocol choice follows
 * the probe verdict instead of the chat route snapshot taken before the
 * probe ran.
 * @param request - the frozen agent-loop request.
 * @param plan - the search plan owning endpoint facts.
 * @param hooks - credential hooks.
 * @param cfg - the current plugin configuration.
 * @returns the StreamChunk stream for the loop's assembler.
 */
export async function* inlineWireStream(
  request: GenerateOptions,
  plan: SearchPlan,
  hooks: InlineHooks,
  cfg: InlineConfig,
): AsyncGenerator<StreamChunk> {
  if (request.signal?.aborted ?? false) {
    yield abortedFinish()
    return
  }
  let candidate: SearchPlanCandidate
  try {
    // The settle is bounded by the probe timeouts; racing it against the
    // request signal makes a user abort land promptly instead of waiting out
    // the probe verdict.
    candidate = request.signal === undefined
      ? await plan.settle()
      : await abortable(plan.settle(), request.signal)
  } catch (error) {
    if (request.signal?.aborted ?? false) {
      yield abortedFinish()
      return
    }
    yield errorFinish(classifyWireError(error))
    return
  }
  if (candidate.protocol === 'anthropic-messages') {
    yield* inlineAnthropicStream(request, candidate, hooks, cfg)
  } else {
    yield* inlineStream(request, candidate, hooks, cfg)
  }
}

/**
 * Run one inline model call: fetch the Responses endpoint with the
 * server-side web tool injected and translate the SSE stream into harness
 * chunks. Never throws; always ends with a finish chunk.
 * @param request - the frozen agent-loop request.
 * @param candidate - the settled endpoint facts (probe-verified).
 * @param hooks - credential hooks.
 * @param cfg - the current plugin configuration.
 * @returns the StreamChunk stream for the loop's assembler.
 */
export async function* inlineStream(
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
  const groundedSandboxEscalation = hasGroundedSandboxEscalation(request)
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
      // Never throw out of the generator: a failing credential hook becomes
      // an error finish exactly like every other failure path.
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
      response = await fetch(`${candidate.baseURL.replace(/\/+$/, '')}${RESPONSES_ENDPOINT}`, {
        method: 'POST',
        redirect: 'error',
        signal: controller.signal,
        headers: {
          authorization: `Bearer ${apiKey}`,
          'content-type': 'application/json',
          accept: 'text/event-stream',
          ...attributionHeaders(),
          ...request.sessionId !== undefined
            ? { 'x-client-request-id': String(request.sessionId), session_id: String(request.sessionId) }
            : {},
        },
        body: JSON.stringify(buildWireBody(request, cfg, candidate.model, candidate.webSearchToolType ?? WEB_SEARCH_TOOL_TYPE)),
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
      ))
      return
    }
    if (response.body === null) {
      yield errorFinish(classifyWireError(new Error('empty response body')))
      return
    }

    const slots = new Map<number, Slot>()
    const open = new Set<number>()
    let localIndex = 0

    const closeOpenSlots = function* (): Generator<StreamChunk> {
      for (const outputIndex of open) {
        const slot = slots.get(outputIndex)
        if (slot === undefined) continue
        open.delete(outputIndex)
        if (slot.blockType === 'tool-call') {
          yield {
            type: 'block-end',
            index: slot.index,
            block: {
              type: 'tool-call',
              id: slot.id as CallId,
              name: slot.name ?? '',
              arguments: sanitizeSandboxEscalationArguments(slot.arguments, groundedSandboxEscalation),
            },
          }
        } else if (slot.blockType === 'reasoning') {
          yield { type: 'block-end', index: slot.index, block: { type: 'reasoning', text: slot.text } }
        } else {
          yield { type: 'block-end', index: slot.index, block: { type: 'text', text: slot.text } }
        }
      }
    }

    try {
      let sawTerminal = false
      for await (const parsed of parseSse(response.body)) {
        watchdog.reset()
        // A non-object payload (e.g. `data: null`) carries no event fields;
        // skipping it keeps a malformed event from crashing the stream.
        if (typeof parsed.data !== 'object' || parsed.data === null) continue
        const data = parsed.data as {
          type?: string
          output_index?: number
          item?: { type?: string; id?: string; call_id?: string; name?: string; arguments?: string; content?: Array<{ type?: string; text?: string }> }
          response?: { status?: string; usage?: WireUsage; incomplete_details?: { reason?: string }; error?: { message?: string } }
          delta?: string
          text?: string
          arguments?: string
        }
        switch (data.type) {
          case 'response.output_item.added': {
            // Server-side web_search_call items are skipped entirely: the
            // search already ran on the provider inside this request.
            if (data.item?.type === 'web_search_call') break
            const itemType = data.item?.type
            if (itemType !== 'message' && itemType !== 'function_call' && itemType !== 'reasoning') break
            // Defensive: the gateway may pass server-side web_search tool
            // calls through as function_call items (e.g. when a system role
            // is present). Treat them like web_search_call — never slot them —
            // so the harness never dispatches a server-side tool locally.
            if (itemType === 'function_call'
              && (data.item?.name === 'web_search' || data.item?.name === 'open_page' || data.item?.name === 'find_in_page')) break
            const outputIndex = data.output_index ?? -1
            const slot: Slot = {
              index: localIndex++,
              blockType: itemType === 'function_call' ? 'tool-call' : itemType === 'message' ? 'text' : 'reasoning',
              ...itemType === 'function_call' ? { id: `${data.item?.call_id ?? 'call'}|${data.item?.id ?? 'item'}` } : {},
              ...itemType === 'function_call' ? { name: data.item?.name } : {},
              text: '',
              arguments: '',
            }
            slots.set(outputIndex, slot)
            if (slot.blockType === 'tool-call' || slot.blockType === 'reasoning') {
              open.add(outputIndex)
              yield { type: 'block-start', index: slot.index, blockType: slot.blockType }
            }
            // Text slots open lazily on their first delta (see below).
            break
          }
          case 'response.output_text.delta': {
            const slot = slots.get(data.output_index ?? -1)
            if (slot === undefined || slot.blockType !== 'text') break
            const delta = data.delta ?? ''
            if (delta.length === 0) break
            slot.text += delta
            if (slot.opened !== true) {
              slot.opened = true
              // Text slots are never closed by closeOpenSlots: the assembler
              // assembles open text blocks from received deltas.
              yield { type: 'block-start', index: slot.index, blockType: 'text' }
            }
            yield { type: 'text-delta', index: slot.index, text: delta }
            break
          }
          case 'response.reasoning_text.delta': {
            const slot = slots.get(data.output_index ?? -1)
            if (slot === undefined || slot.blockType !== 'reasoning') break
            const delta = data.delta ?? ''
            if (delta.length === 0) break
            slot.text += delta
            yield { type: 'reasoning-delta', index: slot.index, text: delta }
            break
          }
          case 'response.function_call_arguments.delta': {
            const slot = slots.get(data.output_index ?? -1)
            if (slot === undefined || slot.blockType !== 'tool-call') break
            const delta = data.delta ?? ''
            if (delta.length === 0) break
            slot.arguments += delta
            yield { type: 'tool-call-delta', index: slot.index, id: slot.id as CallId, name: slot.name, argumentsDelta: delta }
            break
          }
          case 'response.output_item.done': {
            const outputIndex = data.output_index ?? -1
            if (data.item?.type === 'web_search_call') break
            const slot = slots.get(outputIndex)
            if (slot === undefined) break
            if (slot.blockType === 'tool-call') {
              if (!open.has(outputIndex)) break
              open.delete(outputIndex)
              const item = data.item
              yield {
                type: 'block-end',
                index: slot.index,
                block: {
                  type: 'tool-call',
                  id: slot.id as CallId,
                  name: item?.name ?? slot.name ?? '',
                  arguments: sanitizeSandboxEscalationArguments(
                    item?.arguments ?? slot.arguments,
                    groundedSandboxEscalation,
                  ),
                },
              }
            } else if (slot.blockType === 'reasoning') {
              if (!open.has(outputIndex)) break
              open.delete(outputIndex)
              yield { type: 'block-end', index: slot.index, block: { type: 'reasoning', text: slot.text } }
            }
            // Text slots never emit block-end: the assembler assembles them
            // from the deltas it received.
            break
          }
          case 'response.completed':
          case 'response.incomplete': {
            sawTerminal = true
            yield* closeOpenSlots()
            const usage = data.response?.usage
            if (usage !== undefined) yield { type: 'usage', usage: mapUsage(usage) }
            const hasToolCall = [...slots.values()].some(slot => slot.blockType === 'tool-call')
            if (data.type === 'response.incomplete') {
              const reason = data.response?.incomplete_details?.reason
              // `max_output_tokens` is the truncation the harness asks for;
              // any other reason (e.g. content_filter) is a provider refusal.
              if (reason !== undefined && reason !== 'max_output_tokens') {
                yield errorFinish({ message: `the provider stopped the response early: ${reason}`, code: 'UNKNOWN' })
                return
              }
              yield { type: 'finish', reason: { kind: 'max-tokens' } }
              return
            }
            if (hasToolCall) {
              yield { type: 'finish', reason: { kind: 'tool-calls' } }
              return
            }
            const anyText = [...slots.values()].some(slot => slot.text.length > 0)
            if (!anyText) {
              yield errorFinish({ message: 'the provider returned an empty response', code: 'EMPTY_RESPONSE' })
              return
            }
            yield { type: 'finish', reason: { kind: 'stop' } }
            return
          }
          case 'response.failed': {
            sawTerminal = true
            yield* closeOpenSlots()
            yield errorFinish(classifyWireError(new Error(data.response?.error?.message ?? 'provider request failed')))
            return
          }
          case 'error': {
            sawTerminal = true
            yield* closeOpenSlots()
            const message = typeof parsed.data === 'object' && parsed.data !== null
              && 'message' in parsed.data && typeof parsed.data.message === 'string'
              ? parsed.data.message
              : 'provider stream error'
            yield errorFinish(classifyWireError(new Error(message)))
            return
          }
          default:
            // ping, web_search_call.*, content_part.*, unknown events: ignored.
            break
        }
      }
      if (!sawTerminal) {
        yield* closeOpenSlots()
        yield errorFinish(classifyWireError(new Error('stream ended without a terminal event')))
      }
    } catch (error) {
      if (request.signal?.aborted ?? false) {
        yield abortedFinish()
        return
      }
      if (watchdog.signal.aborted === true) {
        // Idle watchdog tripped: map to a retryable TIMEOUT, not UNKNOWN.
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
