/**
 * Detection of the provider route the harness currently chats with: the
 * default-model selection names a route and model, the `llm-pi-ai` settings
 * section carries the route's profile, and the pi-ai catalog supplies the
 * defaults (wire protocol, base URL) a profile that overrides nothing inherits.
 * The probe and plan modules read this to decide which search-capable
 * protocol to use without any configuration.
 * @module dsh-web-search-provider/current-provider
 */

import type { Context } from '@deepseek-ai/cordis'
import type { SettingsNamespace } from '@deepseek-ai/dsh-settings'
import { builtinProviders, getBuiltinModels } from '@earendil-works/pi-ai/providers/all'
import type { BuiltinProvider } from '@earendil-works/pi-ai/providers/all'

/** Settings namespace of the harness's pi-ai LLM adapter (its `providers` dict). */
const LLM_PI_AI_NAMESPACE = 'llm-pi-ai' as SettingsNamespace

/** The resolved chat-route facts this package consumes. */
export interface CurrentChatRoute {
  /** Registered provider route key (the llm-pi-ai `providers` dict key). */
  readonly provider: string
  /** Provider-owned model id. */
  readonly model: string
  /** Wire protocol the route speaks, when determinable. */
  readonly api?: string
  /** Endpoint base the route's requests go to. */
  readonly baseURL?: string
  /** Credential reference the route resolves its key through. */
  readonly apiKeyEnv?: string
}

/** Catalog providers indexed by route id. The catalog is static for a given
 * pi-ai version; constructing it per call would rebuild ~30 provider objects
 * on every route read (several per request). */
let catalogCache: ReadonlyMap<string, { readonly baseUrl?: string }> | undefined

function catalogById(): ReadonlyMap<string, { readonly baseUrl?: string }> {
  if (catalogCache === undefined) {
    const map = new Map<string, { readonly baseUrl?: string }>()
    for (const provider of builtinProviders()) map.set(provider.id, provider)
    catalogCache = map
  }
  return catalogCache
}

/** Per-provider model tables, cached the same way. */
const modelTableCache = new Map<string, readonly { readonly id: string; readonly api?: string; readonly baseUrl?: string }[]>()

function modelTableOf(provider: string): readonly { readonly id: string; readonly api?: string; readonly baseUrl?: string }[] {
  const cached = modelTableCache.get(provider)
  if (cached !== undefined) return cached
  try {
    const models = getBuiltinModels(provider as BuiltinProvider)
    modelTableCache.set(provider, models)
    return models
  } catch {
    // A provider entry whose model table fails to materialize is treated as
    // unknown rather than failing route detection for every other route.
    return []
  }
}

/**
 * The catalog facts for one route+model: the model's own wire protocol and
 * base URL. A route pi-ai does not ship, or a model it does not list, yields
 * no facts — the profile is then the whole answer.
 * @param provider - provider route key.
 * @param model - model id.
 * @returns the catalog protocol and base URL, when known.
 */
function catalogModelFacts(provider: string, model: string): { api?: string; baseUrl?: string } {
  const catalog = catalogById().get(provider)
  if (catalog === undefined) return {}
  const found = modelTableOf(provider).find(candidate => candidate.id === model)
  if (found === undefined) return {}
  return { api: found.api, baseUrl: found.baseUrl }
}

/** Read one route profile from the llm-pi-ai settings section, defensively narrowed. */
function profileFacts(section: unknown, provider: string): { api?: string; baseURL?: string; apiKeyEnv?: string } | undefined {
  if (typeof section !== 'object' || section === null) return undefined
  const providers = (section as Record<string, unknown>)['providers']
  if (typeof providers !== 'object' || providers === null) return undefined
  const profile = (providers as Record<string, unknown>)[provider]
  if (typeof profile !== 'object' || profile === null) return undefined
  const record = profile as Record<string, unknown>
  const stringField = (key: string): string | undefined =>
    typeof record[key] === 'string' && (record[key] as string).length > 0 ? record[key] as string : undefined
  return {
    ...stringField('api') === undefined ? {} : { api: stringField('api') },
    ...stringField('baseURL') === undefined ? {} : { baseURL: stringField('baseURL') },
    ...stringField('apiKeyEnv') === undefined ? {} : { apiKeyEnv: stringField('apiKeyEnv') },
  }
}

/**
 * Read the route the harness currently chats with. Absent any of the inputs
 * (no default-model service, no settings service, no selection) the answer is
 * `undefined` and the caller must fall back to explicit configuration.
 * @param ctx - plugin context; the `agentDefaultModel` and `settings`
 *   services are read optionally through `ctx.get`.
 * @returns the current chat route, or `undefined` when undetectable.
 */
export function currentChatRoute(ctx: Context): CurrentChatRoute | undefined {
  const selection = ctx.get('agentDefaultModel')?.currentSelection()
  if (selection === undefined) return undefined
  const profile = profileFacts(ctx.get('settings')?.get(LLM_PI_AI_NAMESPACE), selection.provider)
  const catalog = catalogModelFacts(selection.provider, selection.model)
  return {
    provider: selection.provider,
    model: selection.model,
    ...profile?.api === undefined ? {} : { api: profile.api },
    ...profile?.api === undefined && catalog.api !== undefined ? { api: catalog.api } : {},
    ...profile?.baseURL === undefined ? {} : { baseURL: profile.baseURL },
    ...profile?.baseURL === undefined && catalog.baseUrl !== undefined ? { baseURL: catalog.baseUrl } : {},
    ...profile?.apiKeyEnv === undefined ? {} : { apiKeyEnv: profile.apiKeyEnv },
  }
}
