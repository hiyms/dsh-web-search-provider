# dsh-web-search-provider

[![npm version](https://img.shields.io/npm/v/dsh-web-search-provider)](https://www.npmjs.com/package/dsh-web-search-provider)

[简体中文](./README.zh.md)

Network search support for the Deepseek Harness, powered by the model provider's server-side capability.

Using this plugin requires the model provider to use the **OpenAI Responses API** or an **Anthropic-compatible Messages API**, and to **provide a web search capability**.

> [!TIP]
> The Anthropic-compatible Messages API only provides a search capability, while the OpenAI Responses API additionally supports viewing a URL's page content / finding specific content within a URL's page. **Providers on the OpenAI Responses API are recommended.**

> [!TIP]
> Verified to work with: the Deepseek official Messages API (the DSH built-in Deepseek provider), the OpenCode Go Messages API, the OpenCode Go Response API, and an OpenAI relay station's Response API.

> [!WARNING]
> The DSH built-in OpenCode Go provider uses different API types for different models. For example, for its Deepseek V4 Flash/Pro models, to use this plugin's capability you need to manually add a custom provider of the Response / Message API type.

> [!WARNING]
> Some model providers may charge extra for the web search capability this plugin uses.

This plugin does not provide its support through the `web_search` tool inside the Deepseek Harness, so you can use it alongside other web search provider plugins (such as the built-in `web-search-deepseek`).

## Comparing with web-search-deepseek

> TL;DR Compared with `web-search-deepseek`, this plugin has a clear advantage in both token consumption and time.

`web-search-deepseek` is the Deepseek Harness's built-in search plugin: when the model in a session calls the `web_search` tool, it creates a new internal session to run the search and returns several URLs with their content summaries.

Compared with `web-search-deepseek`, the AI in this plugin's sessions sends the network search request directly to the server, making **each individual search faster**. Also, the model can directly view / search page content (only with the **OpenAI Responses API**), reducing the tokens the AI spends on bash/curl to read a page's full content.


Using the Deepseek official Messages API's `deepseek-v4-flash model with high think`, each plugin performed 5 specified searches per run, tested 10 times:

| Plugin | Average tokens | Average time |
|:---:|:---:|:---:|
| web-search-deepseek | 4,446 | 47.5s |
| **web-search-provider** | **822** | **14.5s** |

Time is measured from the answer duration shown in the Deepseek Harness. Tokens are the average of the "output" tokens reported on the Deepseek Open Platform, using two different API tokens (because the Deepseek Harness does not count tokens consumed inside tool calls).

The results are for reference only. On tasks that need more web search content, this plugin is expected to show a more pronounced advantage.

## Install
```sh
dsh plugin add dsh-web-search-provider
```

------

Or install from source:
```sh
pnpm install
```

Reference the built plugin from a `cordis.yml` overlay:

```yaml
- insert:
    - id: web-search-provider
      name: 'dsh-web-search-provider'
      config:
        enabled: true
        probe: true
```

## Config

Settings section `web-search-provider` (a live namespace: edits reach the next request). All fields are optional unless noted.

| Key | Default | Meaning |
|---|---|---|
| `enabled` | `true` | Master switch; `false` sends every request down the normal adapter path. |
| `providers` | `[]` | Provider whitelist (llm-pi-ai route keys). A request is served only when its provider is also the current chat route (the plan derives its endpoint facts from that route); the whitelist restricts which of its providers may be served. Empty = serve the current chat route. |
| `baseURL` | route | Endpoint base override for the route's candidates; `/responses` or `/messages` is appended. |
| `model` | loop model | Model override; used by the capability probe and by the served wire request. |
| `apiKeyEnv` | route ref | Credential reference resolved per operation. |
| `includeSources` | `true` | Append `include: ['web_search_call.action.sources']` to Responses wire requests. |
| `stripServerTools` | `true` | Strip the function-tool variants (`web_search`/`open_page`/`find_in_page`) from the wire tools. |
| `idleTimeoutMs` | `300000` | Idle bound for one inline request, in milliseconds. |
| `probe` | `true` | Verify the endpoint executes native search before serving. |
| `probeTimeoutMs` | `30000` | Bound on one probe request, in milliseconds. |

```yaml
- id: web-search-provider
  name: 'dsh-web-search-provider'
  config:
    enabled: true
    probe: true
```


## Known limitations

- DSH `0.1.1-rc.2` and the `0.1.2-alpha.2` development line are supported; the plugin adapts to either settings registration API.
- For providers that use the Response API / Message API **but do not provide a search capability**, the session **errors on first use**. Keep sending messages / open a new session and DSH keeps working without this plugin's capability.
- When the AI uses this plugin's web search capability, there is **no visible search-call UI**. The usage may appear as multiple consecutive thinking segments, with the tail of a thinking segment containing something like "*Let me make search queries*".

## Contributing

### Building
```sh
pnpm install
pnpm build        # tsc emits lib/types, tsdown bundles lib/index.js
pnpm test         # vitest unit suite
```
