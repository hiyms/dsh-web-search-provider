/**
 * Composition tests of the inline plugin: the narrow gate, the llm/stream
 * short-circuit registration, the prompt section, and plan rebuild
 * semantics. The runtime is faked exactly like tests/apply.spec.ts; probe is
 * disabled so no network runs.
 */

import { afterEach, describe, expect, it, vi } from 'vitest'
import { apply, WEB_SEARCH_SETTINGS_NAMESPACE } from '../../src/index.ts'
import type { InlineConfig } from '../../src/config.ts'
import type { GenerateOptions, Message, StreamChunk } from '@deepseek-ai/dsh-llm'
import { markAgentLoopRequest } from '@deepseek-ai/dsh-llm'
import type { Context } from '@deepseek-ai/cordis'

vi.mock('@deepseek-ai/dsh-settings', () => ({ installSettingsSection: undefined }))

interface FakeRuntime {
  ctx: Context
  listener: ((request: GenerateOptions, next: () => unknown) => unknown) | undefined
  sectionNames: string[]
  settingsDocument: Record<string, unknown>
  /** The registered prompt section object (name + dynamic text provider). */
  promptSection: { name: string; text: () => string } | undefined
  installedSettingsSections: string[]
  /** Commit a change to one settings namespace, as the settings service would. */
  triggerSettingsChange(ns: string): void
}

const config: InlineConfig = {
  enabled: true,
  providers: [],
  includeSources: true,
  stripServerTools: true,
  idleTimeoutMs: 300_000,
  probe: false,
  probeTimeoutMs: 30_000,
}

interface FakeSettings {
  settings: unknown
  installedSections: string[]
  triggerChange(ns: string): void
}

function fakeSettings(document: Record<string, unknown>): FakeSettings {
  const watchers = new Map<string, () => void>()
  const installedSections: string[] = []
  const settings = {
    get: (ns: unknown) => document[String(ns)],
    register: (ns: unknown, schema: (value: unknown) => unknown, options?: { base?: unknown }) => {
      const namespace = String(ns)
      return {
        // Re-resolve on every read so a mutated document is visible to the
        // source thunk after a committed change.
        get: () => schema({ ...options?.base, ...document[namespace] }),
        watch: (callback: () => void) => {
          watchers.set(namespace, callback)
          return () => { watchers.delete(namespace) }
        },
        update: async () => undefined,
        replace: async () => undefined,
      }
    },
    installSection: (
      _owner: Context,
      ns: unknown,
      schema: (value: unknown) => unknown,
      entry: unknown,
      hooks: { setSource(source: () => unknown): void; onChange(): void },
    ) => {
      installedSections.push(String(ns))
      const scope = settings.register(ns, schema, { base: entry })
      hooks.setSource(scope.get)
      hooks.onChange()
      scope.watch(hooks.onChange)
    },
  }
  return { settings, installedSections, triggerChange: (ns) => watchers.get(ns)?.() }
}

/** Mutable default-model selection; `current: null` simulates an unsettled
 * boot or a transient route gap. */
type SelectionRef = { current: { provider: string; model: string } | null }

function buildRuntime(
  overrides: Partial<FakeRuntime> = {},
  selectionRef: SelectionRef = { current: { provider: 'opencode-go-response', model: 'deepseek-v4-flash' } },
  extraSettings: Record<string, unknown> = {},
): FakeRuntime {
  let listener: FakeRuntime['listener']
  const listeners = new Map<string, (request: GenerateOptions, next: () => unknown) => unknown>()
  const sectionNames: string[] = []
  let promptSection: FakeRuntime['promptSection']
  const settingsDocument: Record<string, unknown> = {
    [WEB_SEARCH_SETTINGS_NAMESPACE]: {},
    // The chat route profile: currentChatRoute() reads `llm-pi-ai` providers.
    'llm-pi-ai': { providers: { 'opencode-go-response': { api: 'openai-responses' } } },
    ...extraSettings,
  }
  const fake = fakeSettings(settingsDocument)
  const store = new Map<string, unknown>([
    ['settings', fake.settings],
    // Mutable selection so route-switch tests can move between providers; a
    // null ref simulates a boot where the default-model service has not
    // settled a selection yet (route facts undetectable), read live so a
    // test can create and close a route gap mid-run.
    ['agentDefaultModel', {
      currentSelection: () => selectionRef.current === null
        ? undefined
        : { provider: selectionRef.current.provider, model: selectionRef.current.model },
    }],
    // Credentials seam so resolveApiKey never falls back to the process env.
    ['credentials', { resolve: async () => ({ value: 'secret' }) }],
  ])
  const ctx = {
    get: (name: string) => store.get(name),
    // The settings seam reads `ctx.fiber.state` to skip change callbacks
    // while a fiber is unloading; a live fiber is what this fake is.
    fiber: { state: 1 },
    logger: { info: () => undefined, warn: () => undefined },
    on: (event: string, handler: (request: GenerateOptions, next: () => unknown) => unknown) => {
      if (event === 'llm/stream') listener = handler
      listeners.set(event, handler)
      return () => { listeners.delete(event) }
    },
    waterfall: (_self: unknown, event: string, payload: GenerateOptions, next: () => unknown) => {
      const handler = listeners.get(event)
      return handler === undefined ? next() : handler(payload, next)
    },
    systemPrompt: {
      section: (section: { name: string; text?: () => string }) => {
        sectionNames.push(section.name)
        promptSection = section as { name: string; text: () => string }
        return () => undefined
      },
    },
    inject: (servicesToInject: string[], callback: (sctx: Context) => void) => {
      if (servicesToInject.every(name => store.has(name))) callback(ctx as unknown as Context)
    },
    effect: (callback: () => unknown) => {
      const disposer = callback()
      return () => { if (typeof disposer === 'function') disposer() }
    },
  }
  // Attach store entries as context properties so injected service contexts
  // expose the same property API as the Harness runtime.
  for (const [name, service] of store) (ctx as Record<string, unknown>)[name] = service
  // `listener` must be a live binding: it is assigned by ctx.on when apply
  // runs, after this object is constructed, so a snapshot would stay undefined.
  return {
    ctx: ctx as unknown as Context,
    get listener() { return listener },
    sectionNames,
    installedSettingsSections: fake.installedSections,
    settingsDocument,
    get promptSection() { return promptSection },
    triggerSettingsChange: (ns) => fake.triggerChange(ns),
    ...overrides,
  }
}

function request(overrides: Partial<GenerateOptions> = {}): GenerateOptions {
  return markAgentLoopRequest({
    provider: 'opencode-go-response',
    model: 'deepseek-v4-flash',
    messages: [{ id: 'u1' as Message['id'], role: 'user', content: [{ type: 'text', text: 'hi' }], source: { kind: 'user' } }],
    ...overrides,
  })
}

async function drain(stream: AsyncIterable<StreamChunk> | undefined): Promise<StreamChunk[]> {
  if (stream === undefined) return []
  const chunks: StreamChunk[] = []
  for await (const chunk of stream) chunks.push(chunk)
  return chunks
}

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('web-search-provider apply', () => {
  it('uses the settings provider instance API when legacy helpers are absent', () => {
    const runtime = buildRuntime()
    apply(runtime.ctx, config)
    expect(runtime.installedSettingsSections).toEqual([WEB_SEARCH_SETTINGS_NAMESPACE])
  })

  it('registers an llm/stream listener and the prompt section', () => {
    const runtime = buildRuntime()
    apply(runtime.ctx, config)
    expect(runtime.listener).toBeTypeOf('function')
    expect(runtime.sectionNames).toContain('tool:web-search-provider')
  })

  it('short-circuits agent-loop requests on the whitelisted route', async () => {
    const runtime = buildRuntime()
    apply(runtime.ctx, config)
    const next = vi.fn(() => 'next-value')
    const stream = [
      'event: response.output_item.added\ndata: {"type":"response.output_item.added","output_index":0,"item":{"type":"message","id":"msg_1"}}\n\n',
      'event: response.output_text.delta\ndata: {"type":"response.output_text.delta","output_index":0,"delta":"Node 22 is current"}\n\n',
      'event: response.output_item.done\ndata: {"type":"response.output_item.done","output_index":0,"item":{"type":"message","id":"msg_1","content":[{"type":"output_text","text":"Node 22 is current"}]}}\n\n',
      'event: response.completed\ndata: {"type":"response.completed","response":{"status":"completed","usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}}\n\n',
    ].join('')
    vi.stubGlobal('fetch', vi.fn(async () => new Response(stream, { status: 200 })))
    const result = runtime.listener?.(request(), next)
    expect(result).not.toBe('next-value')
    const chunks = await drain(result as AsyncIterable<StreamChunk>)
    expect(chunks.at(-1)).toMatchObject({ type: 'finish', reason: { kind: 'stop' } })
    expect(next).not.toHaveBeenCalled()
  })

  it('passes non-loop requests through to next()', () => {
    const runtime = buildRuntime()
    apply(runtime.ctx, config)
    const plain = { provider: 'p', model: 'm', messages: [] }
    const next = vi.fn(() => 'next-value')
    expect(runtime.listener?.(plain as GenerateOptions, next)).toBe('next-value')
    expect(next).toHaveBeenCalledTimes(1)
  })

  it('passes compaction and title requests through to next()', () => {
    const runtime = buildRuntime()
    apply(runtime.ctx, config)
    const next = vi.fn(() => 'next-value')
    expect(runtime.listener?.(request({ purpose: 'compaction' }), next)).toBe('next-value')
    expect(runtime.listener?.(request({ purpose: 'session-title' }), next)).toBe('next-value')
    expect(next).toHaveBeenCalledTimes(2)
  })

  it('passes image-bearing requests through to next()', () => {
    const runtime = buildRuntime()
    apply(runtime.ctx, config)
    const next = vi.fn(() => 'next-value')
    const withImage = request({ messages: [{ id: 'u1' as Message['id'], role: 'user', content: [{ type: 'image', attachment: {} as never }], source: { kind: 'user' } }] })
    expect(runtime.listener?.(withImage, next)).toBe('next-value')
    expect(next).toHaveBeenCalledTimes(1)
  })

  it('honors the enabled switch', () => {
    const runtime = buildRuntime()
    apply(runtime.ctx, { ...config, enabled: false })
    const next = vi.fn(() => 'next-value')
    expect(runtime.listener?.(request(), next)).toBe('next-value')
  })

  it('honors the provider whitelist', () => {
    const runtime = buildRuntime()
    apply(runtime.ctx, { ...config, providers: ['other-route'] })
    const next = vi.fn(() => 'next-value')
    expect(runtime.listener?.(request(), next)).toBe('next-value')
  })

  it('never serves a whitelisted provider that is not the current chat route', async () => {
    const fetchMock = vi.fn(async () => { throw new Error('should never be called') })
    vi.stubGlobal('fetch', fetchMock)
    const runtime = buildRuntime()
    // The whitelist names both routes, but the plan can only ever derive from
    // the CURRENT chat route: a request from a non-current provider must not
    // be served with the current route's endpoint facts.
    apply(runtime.ctx, { ...config, providers: ['opencode-go-response', 'other-route'] })
    const next = vi.fn(() => 'next-value')
    expect(runtime.listener?.(request({ provider: 'other-route' }), next)).toBe('next-value')
    expect(next).toHaveBeenCalledTimes(1)
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it('serves a whitelisted provider when it is the current chat route', async () => {
    const runtime = buildRuntime()
    apply(runtime.ctx, { ...config, providers: ['opencode-go-response'] })
    const next = vi.fn(() => 'next-value')
    const stream = [
      'event: response.output_item.added\ndata: {"type":"response.output_item.added","output_index":0,"item":{"type":"message","id":"msg_1"}}\n\n',
      'event: response.output_text.delta\ndata: {"type":"response.output_text.delta","output_index":0,"delta":"Node 22 is current"}\n\n',
      'event: response.output_item.done\ndata: {"type":"response.output_item.done","output_index":0,"item":{"type":"message","id":"msg_1","content":[{"type":"output_text","text":"Node 22 is current"}]}}\n\n',
      'event: response.completed\ndata: {"type":"response.completed","response":{"status":"completed","usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}}\n\n',
    ].join('')
    vi.stubGlobal('fetch', vi.fn(async () => new Response(stream, { status: 200 })))
    const result = runtime.listener?.(request(), next)
    expect(result).not.toBe('next-value')
    const chunks = await drain(result as AsyncIterable<StreamChunk>)
    expect(chunks.at(-1)).toMatchObject({ type: 'finish', reason: { kind: 'stop' } })
    expect(next).not.toHaveBeenCalled()
  })

  it('never probes while the plugin is disabled', async () => {
    const fetchMock = vi.fn(async () => { throw new Error('should never be called') })
    vi.stubGlobal('fetch', fetchMock)
    const runtime = buildRuntime()
    apply(runtime.ctx, { ...config, enabled: false })
    const next = vi.fn(() => 'next-value')
    expect(runtime.listener?.(request(), next)).toBe('next-value')
    expect(next).toHaveBeenCalledTimes(1)
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it('applies a settings change committed while the route was briefly unavailable', async () => {
    const selection: SelectionRef = { current: { provider: 'deepseek', model: 'deepseek-v4-flash' } }
    const runtime = buildRuntime({}, selection, {
      'llm-pi-ai': { providers: { deepseek: { api: 'openai-completions', baseURL: 'https://old.example' } } },
    })
    apply(runtime.ctx, config)
    const next = vi.fn(() => 'next-value')
    const stream = [
      'event: response.output_item.added\ndata: {"type":"response.output_item.added","output_index":0,"item":{"type":"message","id":"msg_1"}}\n\n',
      'event: response.output_text.delta\ndata: {"type":"response.output_text.delta","output_index":0,"delta":"Node 22 is current"}\n\n',
      'event: response.output_item.done\ndata: {"type":"response.output_item.done","output_index":0,"item":{"type":"message","id":"msg_1","content":[{"type":"output_text","text":"Node 22 is current"}]}}\n\n',
      'event: response.completed\ndata: {"type":"response.completed","response":{"status":"completed","usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}}\n\n',
    ].join('')
    const fetchMock = vi.fn(async () => new Response(stream, { status: 200 }))
    vi.stubGlobal('fetch', fetchMock)
    // First request serves through the old endpoint.
    const first = runtime.listener?.(request({ provider: 'deepseek' }), next)
    await drain(first as AsyncIterable<StreamChunk>)
    expect((fetchMock.mock.calls[0] as [string])[0]).toBe('https://old.example/responses')
    // The settings commit lands while the selection is temporarily null.
    selection.current = null
    runtime.settingsDocument[WEB_SEARCH_SETTINGS_NAMESPACE] = { baseURL: 'https://new.example' }
    runtime.triggerSettingsChange(WEB_SEARCH_SETTINGS_NAMESPACE)
    selection.current = { provider: 'deepseek', model: 'deepseek-v4-flash' }
    // The next request must rebuild with the NEW config, not reuse the stale
    // plan snapshot (same route, old baseURL).
    const second = runtime.listener?.(request({ provider: 'deepseek' }), next)
    await drain(second as AsyncIterable<StreamChunk>)
    expect((fetchMock.mock.calls[1] as [string])[0]).toBe('https://new.example/responses')
  })

  it('defers the plan to the first request when the route is unsettled at attach', async () => {
    const fetchMock = vi.fn(async () => { throw new Error('should never be called') })
    vi.stubGlobal('fetch', fetchMock)
    // No default-model selection yet: the attach-time settings change cannot
    // resolve a route, so nothing is probed and the plan waits for a request.
    const runtime = buildRuntime({}, { current: null })
    apply(runtime.ctx, { ...config, probe: true })
    const next = vi.fn(() => 'next-value')
    expect(runtime.listener?.(request(), next)).toBe('next-value')
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it('never probes for requests outside the gate', async () => {
    const fetchMock = vi.fn(async () => { throw new Error('should never be called') })
    vi.stubGlobal('fetch', fetchMock)
    const runtime = buildRuntime()
    apply(runtime.ctx, config)
    const next = vi.fn(() => 'next-value')
    runtime.listener?.(request({ purpose: 'compaction' }), next)
    runtime.listener?.(request({ provider: 'unrelated-route' }), next)
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it('serves a chat-completions route through its sibling candidates', async () => {
    const selection: SelectionRef = { current: { provider: 'deepseek', model: 'deepseek-v4-flash' } }
    const runtime = buildRuntime({}, selection, {
      'llm-pi-ai': { providers: { deepseek: { api: 'openai-completions', baseURL: 'https://api.deepseek.com' } } },
    })
    apply(runtime.ctx, config)
    const next = vi.fn(() => 'next-value')
    const stream = [
      'event: response.output_item.added\ndata: {"type":"response.output_item.added","output_index":0,"item":{"type":"message","id":"msg_1"}}\n\n',
      'event: response.output_text.delta\ndata: {"type":"response.output_text.delta","output_index":0,"delta":"Node 22 is current"}\n\n',
      'event: response.output_item.done\ndata: {"type":"response.output_item.done","output_index":0,"item":{"type":"message","id":"msg_1","content":[{"type":"output_text","text":"Node 22 is current"}]}}\n\n',
      'event: response.completed\ndata: {"type":"response.completed","response":{"status":"completed","usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}}\n\n',
    ].join('')
    vi.stubGlobal('fetch', vi.fn(async () => new Response(stream, { status: 200 })))
    // The route speaks chat-completions (not search-capable on its own wire),
    // but its DeepSeek-host siblings give the plan candidates: the request is
    // served rather than passed through.
    const result = runtime.listener?.(request({ provider: 'deepseek' }), next)
    expect(result).not.toBe('next-value')
    const chunks = await drain(result as AsyncIterable<StreamChunk>)
    expect(chunks.at(-1)).toMatchObject({ type: 'finish', reason: { kind: 'stop' } })
    expect(next).not.toHaveBeenCalled()
  })

  it('keeps the prompt section empty while the plugin cannot serve', async () => {
    const runtime = buildRuntime()
    apply(runtime.ctx, { ...config, enabled: false })
    expect(runtime.promptSection?.text()).toBe('')
  })

  it('fills the prompt section once a plan can serve', async () => {
    const runtime = buildRuntime()
    apply(runtime.ctx, config)
    // The route is settled in this fake, so the plan is ready right after
    // attach: the guidance appears.
    expect(runtime.promptSection?.text()).toContain('web_search')
  })

  it('keeps the prompt section empty while the probe has not settled', async () => {
    const probingConfig: InlineConfig = { ...config, probe: true, probeTimeoutMs: 1000 }
    const runtime = buildRuntime()
    // The probe never settles: the plan stays in `probing`, and the model
    // must not be told web_search is available before the verdict is in.
    vi.stubGlobal('fetch', vi.fn(async () => new Promise(() => undefined)))
    apply(runtime.ctx, probingConfig)
    expect(runtime.promptSection?.text()).toBe('')
  })

  it('rebuilds the plan when the chat route changes and never serves an unknown gateway', async () => {
    const selection: SelectionRef = { current: { provider: 'opencode-go-response', model: 'deepseek-v4-flash' } }
    const runtime = buildRuntime({}, selection)
    apply(runtime.ctx, config)
    const next = vi.fn(() => 'next-value')
    const stream = [
      'event: response.output_item.added\ndata: {"type":"response.output_item.added","output_index":0,"item":{"type":"message","id":"msg_1"}}\n\n',
      'event: response.output_text.delta\ndata: {"type":"response.output_text.delta","output_index":0,"delta":"Node 22 is current"}\n\n',
      'event: response.output_item.done\ndata: {"type":"response.output_item.done","output_index":0,"item":{"type":"message","id":"msg_1","content":[{"type":"output_text","text":"Node 22 is current"}]}}\n\n',
      'event: response.completed\ndata: {"type":"response.completed","response":{"status":"completed","usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}}\n\n',
    ].join('')
    vi.stubGlobal('fetch', vi.fn(async () => new Response(stream, { status: 200 })))
    // Route A speaks openai-responses: short-circuit.
    const first = runtime.listener?.(request(), next)
    expect(first).not.toBe('next-value')
    // Switch to an unknown-gateway route: the plan is rebuilt, no sibling
    // candidates exist, so the request passes through and no endpoint is
    // probed with the wrong protocol.
    selection.current = { provider: 'opencode-go', model: 'deepseek-v4-flash' }
    const second = runtime.listener?.(request({ provider: 'opencode-go' }), next)
    expect(second).toBe('next-value')
    expect(next).toHaveBeenCalled()
  })

  it('does not re-probe when a settings change leaves candidates identical', async () => {
    const probingConfig: InlineConfig = { ...config, probe: true, probeTimeoutMs: 30_000 }
    // A spelling-independent 401 fails the probe with exactly one request per
    // plan build, keeping the call-count assertions meaningful under the
    // two-spelling probe fallback.
    const fetchMock = vi.fn(async () => new Response(JSON.stringify({ error: { message: 'invalid api key' } }), { status: 401 }))
    vi.stubGlobal('fetch', fetchMock)
    const runtime = buildRuntime()
    apply(runtime.ctx, probingConfig)
    // The plan builds (and probes once) at settings attach.
    await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1))
    // A no-op edit for the plan (idle bound, sources, tool stripping) must
    // not start a second probe.
    runtime.settingsDocument[WEB_SEARCH_SETTINGS_NAMESPACE] = { idleTimeoutMs: 600_000 }
    runtime.triggerSettingsChange(WEB_SEARCH_SETTINGS_NAMESPACE)
    await new Promise((resolve) => setTimeout(resolve, 20))
    expect(fetchMock).toHaveBeenCalledTimes(1)
  })

  it('rebuilds the plan when the probe knobs change, recovering a failed plan', async () => {
    const probingConfig: InlineConfig = { ...config, probe: true, probeTimeoutMs: 30_000 }
    // A spelling-independent 401 fails the probe with exactly one request per
    // plan build, keeping the call-count assertions meaningful under the
    // two-spelling probe fallback.
    const fetchMock = vi.fn(async () => new Response(JSON.stringify({ error: { message: 'invalid api key' } }), { status: 401 }))
    vi.stubGlobal('fetch', fetchMock)
    const runtime = buildRuntime()
    apply(runtime.ctx, probingConfig)
    await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1))
    // The first plan failed (the endpoint never confirms search); raising the
    // probe bound must rebuild and re-probe even though the candidates are
    // the same.
    runtime.settingsDocument[WEB_SEARCH_SETTINGS_NAMESPACE] = { probeTimeoutMs: 60_000 }
    runtime.triggerSettingsChange(WEB_SEARCH_SETTINGS_NAMESPACE)
    await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2))
  })
})
