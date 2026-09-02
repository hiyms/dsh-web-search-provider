/**
 * The inline web-search plugin: short-circuits agent-loop model calls on
 * the `llm/stream` waterfall, injecting the server-side `web_search` tool
 * into the wire request so search executes inside the model's own turn.
 * The narrow gate keeps every other request on the normal adapter path.
 * @module dsh-web-search-provider
 */

import type { Context } from '@deepseek-ai/cordis'
// Bring the `systemPrompt` service declaration (dsh-agent augmentation) into
// the type graph: module augmentations only apply when their module is part
// of the program.
import type {} from '@deepseek-ai/dsh-agent'
import type { GenerateOptions, StreamChunk } from '@deepseek-ai/dsh-llm'
import { isAgentLoopRequest } from '@deepseek-ai/dsh-llm'
import { credentialRef } from '@deepseek-ai/dsh-credentials'
import * as dshSettings from '@deepseek-ai/dsh-settings'
import type { SettingsNamespace } from '@deepseek-ai/dsh-settings'
import { launchEnvironmentOf } from '@deepseek-ai/dsh-launch-environment'
import { resolveCandidates, sameCandidates, SearchPlan } from './plan.ts'
import type { PlanConfig, SearchPlanCandidate } from './plan.ts'
import { probeCandidate } from './probe.ts'
import { currentChatRoute } from './current-provider.ts'
import type { CurrentChatRoute } from './current-provider.ts'
import { Config } from './config.ts'
import type { InlineConfig } from './config.ts'
import { contentHasImages, inlineWireStream } from './wire.ts'
import type { InlineHooks } from './wire.ts'

/** Cordis plugin name used by loader diagnostics. */
export const name = 'web-search-provider'

/** Services the plugin hooks into; `credentials` is declared so the
 * credential seam is settled before apply resolves keys. */
export const inject = ['llm', 'systemPrompt', 'settings', 'credentials']

/** Settings namespace carrying this plugin's section. */
export const WEB_SEARCH_SETTINGS_NAMESPACE = 'web-search-provider' as SettingsNamespace

/** Schema of the plugin's settings section, exported for composition consumers. */
export { Config } from './config.ts'
export type { InlineConfig } from './config.ts'

interface SettingsSectionHooks {
  setSource(source: () => InlineConfig): void
  onChange(): void
}

interface InstanceSettingsInstaller {
  installSection(
    owner: Context,
    namespace: SettingsNamespace,
    schema: typeof Config,
    entry: InlineConfig,
    hooks: SettingsSectionHooks,
  ): void
}

interface LegacySettingsModule {
  installSettingsSection?(
    owner: Context,
    namespace: SettingsNamespace,
    schema: typeof Config,
    entry: InlineConfig,
    hooks: SettingsSectionHooks,
  ): void
}

function isInstanceSettingsInstaller(value: unknown): value is InstanceSettingsInstaller {
  return typeof value === 'object'
    && value !== null
    && 'installSection' in value
    && typeof value.installSection === 'function'
}

function installWebSearchSettings(
  ctx: Context,
  config: InlineConfig,
  hooks: SettingsSectionHooks,
): void {
  const legacyInstaller = (dshSettings as LegacySettingsModule).installSettingsSection
  if (legacyInstaller !== undefined) {
    legacyInstaller(ctx, WEB_SEARCH_SETTINGS_NAMESPACE, Config, config, hooks)
    return
  }
  ctx.inject(['settings'], (settingsCtx) => {
    if (!isInstanceSettingsInstaller(settingsCtx.settings)) {
      throw new Error('web-search-provider: settings service does not support section installation')
    }
    settingsCtx.settings.installSection(ctx, WEB_SEARCH_SETTINGS_NAMESPACE, Config, config, hooks)
  })
}

/**
 * Register the inline short-circuit. The plan (candidates plus probe
 * verdict) follows the settings section; the listener reads the CURRENT
 * plan and config per request.
 * @param ctx - context whose `llm` events and `systemPrompt` receive the
 *   registrations; both are effect-scoped and unregister on dispose.
 * @param config - the composition entry config, used as the settings base layer.
 */
export function apply(ctx: Context, config: InlineConfig): void {
  let current: () => InlineConfig = () => config
  // The plan is built when the settings section attaches IF the chat route
  // is already detectable (so the probe verdict and the prompt guidance are
  // ready before the first conversation turn); otherwise it is deferred to
  // the FIRST model request that passes the gate. The deferral matters: at
  // apply time the settings document (llm-pi-ai section) and the credentials
  // seam may not be settled yet, so route detection returns unknowns and the
  // probe would target the wrong endpoint with the wrong key (observed at
  // boot: api/baseURL/key all unknown). By first request the harness has
  // already driven the LLM, so the route and credential facts are guaranteed
  // settled.
  let currentPlan: SearchPlan | undefined
  // The route snapshot the current plan was built for: model/provider
  // switches in the web UI must rebuild the plan (and re-probe) instead of
  // reusing a stale candidate set.
  let planRoute: CurrentChatRoute | undefined
  // The probe knobs the current plan was built with: a probe/probeTimeoutMs
  // change must rebuild even when the candidate set is unchanged (e.g. to
  // recover a failed plan with a longer timeout or probing off).
  let planProbe: { enabled: boolean; timeoutMs: number } | undefined

  const hooks: InlineHooks = {
    resolveApiKey: async (apiKeyEnv) => {
      const credentials = ctx.get('credentials')
      if (credentials !== undefined) return (await credentials.resolve(credentialRef(apiKeyEnv)))?.value
      const ambient = launchEnvironmentOf(ctx).get(apiKeyEnv)
      return ambient !== undefined && ambient.value.length > 0 ? ambient.value : undefined
    },
  }

  /** The live plan, built on first use and rebuilt when the chat route moves. */
  function plan(): SearchPlan {
    const route = currentChatRoute(ctx)
    if (currentPlan !== undefined && sameRoute(route, planRoute)) return currentPlan
    currentPlan = buildPlan(ctx, current(), hooks)
    planRoute = route
    planProbe = probeSettings(current())
    reportRoute(ctx)
    reportPlan(ctx, currentPlan)
    return currentPlan
  }

  installWebSearchSettings(ctx, config, {
    setSource: (source) => {
      current = source
    },
    onChange: () => {
      const cfg = current()
      if (!cfg.enabled) {
        // Disabled: drop the plan and its probe so nothing runs and the
        // prompt section (which reads the plan) goes dark.
        currentPlan = undefined
        planRoute = undefined
        planProbe = undefined
        return
      }
      // Route facts may not be settled at attach time (the llm-pi-ai
      // section or the credentials seam may still be initializing), or the
      // selection may be briefly unavailable mid-session: without a route
      // the plan cannot resolve candidates, and a STALE plan must not
      // survive the gap — drop it so the next request rebuilds with the
      // current config (a later restore of the same route snapshot would
      // otherwise reuse the old baseURL/model).
      const route = currentChatRoute(ctx)
      if (route === undefined) {
        currentPlan = undefined
        planRoute = undefined
        planProbe = undefined
        return
      }
      // Resolve the candidate set WITHOUT starting probes; only a real
      // difference (candidates, or the probe knobs that decide their
      // verdict) rebuilds the plan. This keeps no-op edits (idleTimeoutMs,
      // includeSources, …) from wasting a probe round-trip.
      const candidates = resolveCandidates(ctx, planConfigOf(cfg))
      const probe = probeSettings(cfg)
      if (currentPlan === undefined
        || planProbe === undefined
        || !sameCandidates(candidates, currentPlan.candidates)
        || probe.enabled !== planProbe.enabled
        || probe.timeoutMs !== planProbe.timeoutMs) {
        currentPlan = buildPlan(ctx, cfg, hooks)
        planRoute = route
        planProbe = probe
        reportPlan(ctx, currentPlan)
      }
    },
  })

  ctx.on('llm/stream', (request: GenerateOptions, next: () => AsyncIterable<StreamChunk>) => {
    const cfg = current()
    // Zero-cost gate first: disabled plugins, non-loop requests, purposed
    // calls, provider mismatches, and image-bearing requests never build a
    // plan and never start a probe.
    if (!preflight(request, cfg, ctx)) return next()
    const p = plan()
    if (!p.available()) return next()
    return inlineWireStream(request, p, hooks, cfg)
  })

  ctx.systemPrompt.section({
    name: 'tool:web-search-provider',
    order: 115,
    // The guidance is only true while the plugin actually serves: with the
    // plugin disabled or the plan failed, an empty section keeps the model
    // from being steered toward a web_search tool that does not exist.
    text: () => servingPrompt(current, currentPlan),
  })
}

/**
 * The web-search guidance text, or an empty string while the plugin cannot
 * serve: disabled, no plan yet, or a plan that has not settled on a
 * VERIFIED candidate (probing or failed). Only a ready plan — the state
 * `available()` admits for serving — may tell the model the tool exists.
 * @param current - the authoritative config source.
 * @param plan - the live plan, when one exists.
 * @returns the section text to contribute to this assembly.
 */
function servingPrompt(current: () => InlineConfig, plan: SearchPlan | undefined): string {
  if (!current().enabled || plan?.chosenCandidate() === undefined) return ''
  return '## Web Search\n\n'
    + 'The web_search tool runs natively on the model provider inside the same request: '
    + 'when you call it, the search executes server-side and its results are immediately '
    + 'available for you to answer from. Use web_search when the user asks for current or '
    + 'online information. Prefer web_search over guessing when freshness matters. '
    + 'Cite the relevant URLs as markdown links in your answer.'
}

/**
 * Announce the plan's verdict through the harness logger.
 * @param ctx - the plugin context whose logger receives the line.
 * @param plan - the plan whose settled verdict to announce.
 */
function reportPlan(ctx: Context, plan: SearchPlan): void {
  void plan.settled.then(() => {
    const chosen = plan.chosenCandidate()
    if (chosen === undefined) {
      ctx.logger.warn('[web-search-provider] %s', plan.failureReason() ?? 'web search is disabled')
      return
    }
    ctx.logger.info(
      '[web-search-provider] serving inline web search through %s at %s (model %s, key %s)',
      chosen.protocol,
      chosen.baseURL,
      chosen.model,
      chosen.apiKeyEnv,
    )
  })
}

/** Report the detected chat route through the harness logger. */
function reportRoute(ctx: Context): void {
  const route = currentChatRoute(ctx)
  if (route === undefined) {
    ctx.logger.warn('[web-search-provider] chat route undetectable; inline web search stays on the adapter path')
    return
  }
  ctx.logger.info(
    '[web-search-provider] chat route %s (api=%s, baseURL=%s, key=%s)',
    route.provider,
    route.api ?? 'unknown',
    route.baseURL ?? 'unknown',
    route.apiKeyEnv ?? 'unknown',
  )
}

/** Build one plan from the section; `hooks` is referenced lazily via closure. */
function buildPlan(ctx: Context, cfg: InlineConfig, hooks: InlineHooks): SearchPlan {
  return new SearchPlan(
    resolveCandidates(ctx, planConfigOf(cfg)),
    (candidate: SearchPlanCandidate) => probeCandidate(candidate, hooks.resolveApiKey, cfg.probeTimeoutMs),
    cfg.probe,
  )
}

/** Project the settings section onto the plan's config surface. */
function planConfigOf(cfg: InlineConfig): PlanConfig {
  return {
    ...cfg.baseURL !== undefined && cfg.baseURL.length > 0 ? { baseURL: cfg.baseURL } : {},
    ...cfg.model !== undefined && cfg.model.length > 0 ? { model: cfg.model } : {},
    ...cfg.apiKeyEnv !== undefined && cfg.apiKeyEnv.length > 0 ? { apiKeyEnv: cfg.apiKeyEnv } : {},
    probe: cfg.probe,
    probeTimeoutMs: cfg.probeTimeoutMs,
  }
}

/** The probe knobs deciding whether a candidate set needs a new verdict. */
function probeSettings(cfg: InlineConfig): { enabled: boolean; timeoutMs: number } {
  return { enabled: cfg.probe, timeoutMs: cfg.probeTimeoutMs }
}

/** Whether two route snapshots are identical for plan rebuilding purposes. */
function sameRoute(left: CurrentChatRoute | undefined, right: CurrentChatRoute | undefined): boolean {
  if (left === undefined || right === undefined) return left === right
  return left.provider === right.provider
    && left.model === right.model
    && left.api === right.api
    && left.baseURL === right.baseURL
    && left.apiKeyEnv === right.apiKeyEnv
}

/**
 * The narrow gate: short-circuit only agent-loop conversation requests on a
 * whitelisted route; everything else keeps the normal adapter path. Runs
 * before the plan is built so disabled plugins and unrelated requests never
 * trigger a probe.
 */
function preflight(request: GenerateOptions, cfg: InlineConfig, ctx: Context): boolean {
  if (!isAgentLoopRequest(request)) return false
  if (request.purpose !== undefined) return false
  if (!cfg.enabled) return false
  if (!providerAllowed(request, cfg, ctx)) return false
  if (contentHasImages(request)) return false
  return true
}

/**
 * Provider whitelist; empty config follows the current chat route. The
 * request provider must ALWAYS be the current chat route's provider: the
 * plan's candidates (baseURL/model/key) are derived from that route, so
 * serving any other provider would send its request to the wrong endpoint.
 * The whitelist only restricts which of the route's providers may be served;
 * it can never extend serving to a provider the plan was not built for.
 */
function providerAllowed(request: GenerateOptions, cfg: InlineConfig, ctx: Context): boolean {
  const route = currentChatRoute(ctx)
  if (route === undefined || route.provider !== request.provider) return false
  return cfg.providers.length === 0 || cfg.providers.includes(request.provider)
}
