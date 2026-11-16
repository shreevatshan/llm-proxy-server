# LLM Proxy Server

📦 **GitHub:** [shreevatshan/llm-proxy-server](https://github.com/shreevatshan/llm-proxy-server)

A unified proxy server that exposes **OpenAI-compatible**, **Anthropic-compatible**, **Azure OpenAI-compatible** and **TypeSafe-compatible** API endpoints for multiple LLM providers including Azure OpenAI, AWS Bedrock, Google Gemini, OpenAI, Ollama, LlamaCpp, and any custom server. Supports Chat Completions, Embeddings, Images, Audio, the **OpenAI Responses API**, the **OpenAI Decisions API**, the **TypeSafe System One API**, the **Anthropic Messages API**, and **Azure OpenAI deployment-based URLs**.

## Architecture

The server runs **five services** in a single container:

| Service | Default Port | Purpose |
|---------|-------------|--------|
| **OpenAI API** | `11440` | OpenAI-compatible API endpoints (`/v1/chat/completions`, `/v1/models`, etc.) |
| **Anthropic API** | `2027` | Anthropic Messages API endpoints (`/v1/messages`, `/v1/models`) |
| **Azure OpenAI API** | `11439` | Azure OpenAI-compatible deployment-based endpoints (`/openai/deployments/{provider}/{deployment}/chat/completions`, etc.) |
| **TypeSafe API** | `11366` | TypeSafe System One API (`/v1/systemone`, `/v1/models`), TypeSafe-native for the TypeSafe SDKs |
| **Management** | `8765` | Admin dashboard, user login, provider configuration |

## Quick Start

```bash
docker run -d \
  -p 11440:11440 \
  -p 2027:2027 \
  -p 11439:11439 \
  -p 11366:11366 \
  -p 8765:8765 \
  -v ./data:/llm-proxy-server/data \
  -e JWT_SECRET_KEY=your-secret-key-change-this \
  -e LLMPROXY_ADMIN_USERNAME=admin \
  -e LLMPROXY_ADMIN_PASSWORD=admin123 \
  shreevatshan/llm-proxy-server:latest
```

- Admin dashboard: `http://localhost:8765/admin`
- OpenAI API: `http://localhost:11440/v1`
- Anthropic API: `http://localhost:2027/v1`
- Azure OpenAI API: `http://localhost:11439/openai`
- TypeSafe API: `http://localhost:11366` (the TypeSafe SDK's `base_url`)

## Configuration

### Using Docker Compose

Create a `docker-compose.yaml`:

```yaml
services:
  llm-proxy-server:
    image: shreevatshan/llm-proxy-server:latest
    container_name: llm-proxy-server
    ports:
      - "11440:11440"  # OpenAI API
      - "2027:2027"    # Anthropic API
      - "11439:11439"  # Azure OpenAI API
      - "11366:11366"  # TypeSafe System One API
      - "8765:8765"    # Management panel
    volumes:
      - ./data:/llm-proxy-server/data
    environment:
      # Database Configuration
      - DATABASE_URL=sqlite+aiosqlite:///./data/llm_proxy.db

      # JWT Authentication
      - JWT_SECRET_KEY=your-secret-key-change-this-in-production

      # Server Configuration
      - LLMPROXY_HOST=0.0.0.0
      - OPENAI_SERVER_PORT=11440
      - ANTHROPIC_SERVER_PORT=2027
      - AZURE_OPENAI_SERVER_PORT=11439
      - TYPESAFE_SERVER_PORT=11366
      - MANAGEMENT_SERVER_PORT=8765

      # Admin Configuration
      - LLMPROXY_ADMIN_ENABLED=true
      - LLMPROXY_ADMIN_USERNAME=admin
      - LLMPROXY_ADMIN_EMAIL=admin@localhost
      - LLMPROXY_ADMIN_PASSWORD=admin123

      # Zoho OAuth Configuration (optional - for SSO login)
      #- ZOHO_CLIENT_ID=your-zoho-client-id
      #- ZOHO_CLIENT_SECRET=your-zoho-client-secret
      #- ZOHO_REDIRECT_URI=http://localhost:8765/auth/zoho/callback

      # Notification Webhook (optional - for system event notifications)
      #- NOTIFICATION_WEBHOOK_URL=https://your-webhook-endpoint.com/notifications

      # OpenTelemetry Tracing (optional)
      #- OTEL_EXPORTER_OTLP_ENDPOINT=https://your-otlp-endpoint
      #- OTEL_SERVICE_NAME=llm-proxy-server
      #- OTEL_EXPORTER_OTLP_HEADERS=api-key=your-api-key
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "python", "-c", "import requests; requests.get('http://localhost:8765/health')"]
      interval: 30s
      timeout: 10s
      retries: 3
```

Run with:
```bash
docker-compose up -d
```

This is the minimal single-service setup. `docker-compose.yaml.example` in the
repository has the full stack, including the `searxng` and `fourget` search
backends described under [Web Search Interception](#web-search-interception).

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `JWT_SECRET_KEY` | **Yes** | - | Secret key for JWT token signing. Required — the app refuses to start if unset. Generate a random value (e.g. `python -c "import secrets; print(secrets.token_urlsafe(48))"`). |
| `COOKIE_SECURE` | No | `true` | Set the `Secure` flag on auth/CSRF cookies so browsers only send them over HTTPS. **If you serve the management panel over plain HTTP on a non-localhost host, set this to `false`** — otherwise browsers drop the cookies and login silently loops. |
| `DATABASE_URL` | No | `sqlite+aiosqlite:///./data/llm_proxy.db` | Database connection string |
| `LLMPROXY_HOST` | No | `0.0.0.0` | Host to bind all servers to |
| `LLMPROXY_DOMAIN` | No | `localhost` | Public domain/host used for building callback and external URLs |
| `OPENAI_SERVER_PORT` | No | `11440` | Port for OpenAI API server |
| `ANTHROPIC_SERVER_PORT` | No | `2027` | Port for Anthropic API server |
| `AZURE_OPENAI_SERVER_PORT` | No | `11439` | Port for Azure OpenAI API server |
| `TYPESAFE_SERVER_PORT` | No | `11366` | Port for the TypeSafe (System One) API server |
| `MANAGEMENT_SERVER_PORT` | No | `8765` | Port for management panel |
| `LLMPROXY_ADMIN_ENABLED` | No | `false` | Enable admin dashboard (disabled by default) |
| `LLMPROXY_ADMIN_USERNAME` | No | `admin` | Default admin username (only when admin enabled) |
| `LLMPROXY_ADMIN_EMAIL` | No | `admin@localhost` | Default admin email (only when admin enabled) |
| `LLMPROXY_ADMIN_PASSWORD` | Required if admin enabled | - | Admin password. Set a strong value; do not use `admin123`. |
| `ZOHO_CLIENT_ID` | No | - | Zoho OAuth client ID for SSO login |
| `ZOHO_CLIENT_SECRET` | No | - | Zoho OAuth client secret for SSO login |
| `ZOHO_REDIRECT_URI` | No | `http://localhost:8765/auth/zoho/callback` | Zoho OAuth redirect URI |
| `NOTIFICATION_WEBHOOK_URL` | No | - | Webhook endpoint for system event notifications (user signups, etc.) |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | No | - | OpenTelemetry endpoint URL |
| `OTEL_SERVICE_NAME` | No | `llm-proxy-server` | Service name for tracing |
| `OTEL_EXPORTER_OTLP_HEADERS` | No | - | Additional OTLP headers |

## Usage

### 1. Configure Providers

Access the admin dashboard at `http://localhost:8765/admin` and add your providers.

Supported provider types:
- **Azure OpenAI** — Azure-hosted OpenAI models
- **AWS Bedrock** — Bedrock models (supports both OpenAI and Anthropic API formats)
- **Google AI** — Gemini models
- **Custom** — Any OpenAI-compatible, Anthropic-compatible and/or TypeSafe System One-compatible server (OpenAI, Ollama, LlamaCpp, vLLM, TypeSafe, OpenRouter, etc.)

Custom providers can be configured to support the **OpenAI API**, the **Anthropic API** and/or the **TypeSafe System One API**: each has its own base URL field (**OpenAI base URL**, **Anthropic base URL**, **System One base URL**), and a filled field means the server serves that API natively (see [7. TypeSafe System One API](#7-typesafe-system-one-api)).

### 2. OpenAI API

Use the OpenAI-compatible API on port **11440**:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11440/v1",
    api_key="your-user-token"  # Get from dashboard
)

response = client.chat.completions.create(
    model="azure:primary/gpt-4",  # or bedrock:production/claude-3, etc.
    messages=[
        {"role": "user", "content": "Hello!"}
    ]
)

print(response.choices[0].message.content)
```

### 3. Anthropic API

Use the Anthropic Messages API on port **2027** for providers that support it (AWS Bedrock, custom Anthropic-compatible servers):

```python
from anthropic import Anthropic

client = Anthropic(
    base_url="http://localhost:2027",
    api_key="your-user-token"
)

message = client.messages.create(
    model="bedrock:default/us.anthropic.claude-sonnet-4-5",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Hello!"}
    ]
)

print(message.content[0].text)
```

Streaming is also supported:

```python
with client.messages.stream(
    model="bedrock:default/us.anthropic.claude-sonnet-4-5",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Write a haiku about programming."}]
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
```

### 4. Azure OpenAI API

Use the Azure OpenAI-compatible deployment-based API on port **11439**. This provides Azure OpenAI-style URLs with `/openai/deployments/{provider}/{deployment}/...` paths, and preserves the upstream model name in responses (e.g. `gpt-4.1-2025-04-14` instead of the proxy model alias).

```python
from openai import AzureOpenAI

client = AzureOpenAI(
    azure_endpoint="http://localhost:11439",
    api_key="your-user-token",
    api_version="2024-06-01"
)

response = client.chat.completions.create(
    model="gpt-4",  # deployment name
    messages=[
        {"role": "user", "content": "Hello!"}
    ]
)

print(response.choices[0].message.content)
```

#### Azure OpenAI API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/openai/deployments/{provider}/{deployment}/chat/completions` | POST | Chat completions |
| `/openai/deployments/{provider}/{deployment}/completions` | POST | Text completions |
| `/openai/deployments/{provider}/{deployment}/embeddings` | POST | Create embeddings |
| `/openai/deployments/{provider}/{deployment}/images/generations` | POST | Generate images |
| `/openai/deployments/{provider}/{deployment}/audio/speech` | POST | Text to speech |
| `/openai/deployments/{provider}/{deployment}/audio/transcriptions` | POST | Transcribe audio |
| `/openai/deployments/{provider}/{deployment}/audio/translations` | POST | Translate audio |
| `/openai/deployments/{provider}/responses` | POST | Create response |
| `/openai/deployments/{provider}/responses/{response_id}` | GET | Get response |
| `/openai/deployments/{provider}/responses/{response_id}` | DELETE | Delete response |
| `/openai/deployments/{provider}` | GET | List deployments |
| `/openai/models` | GET | List all Azure models |

### 5. Responses API

The proxy also supports the [OpenAI Responses API](https://platform.openai.com/docs/api-reference/responses) for custom providers (OpenAI, Ollama, LlamaCpp, and custom OpenAI-compatible servers). Azure, Google, and Bedrock providers are not supported for Responses API.

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11440/v1",
    api_key="your-user-token"
)

response = client.responses.create(
    model="openai:primary/gpt-4o",
    input="Explain quantum computing in simple terms."
)

print(response.output_text)
```

#### Responses API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/v1/responses` | POST | Create a response (supports streaming) |
| `/v1/responses/{response_id}` | GET | Retrieve a response by ID |
| `/v1/responses/{response_id}` | DELETE | Delete a response |
| `/v1/responses/{response_id}/cancel` | POST | Cancel an in-progress response |
| `/v1/responses/{response_id}/input_items` | GET | List input items for a response |
| `/v1/responses/input_tokens` | POST | Count input tokens |
| `/v1/responses/compact` | POST | Compact a response |

### 6. Decisions API

The proxy relays the [OpenAI Decisions API](https://developers.openai.com/api/docs/guides/decisions): typed answers (`predicate` probabilities, `choice` picks and `score` ratings) about shared input from a decision model such as `gpt-6-luna`. The request body is forwarded as sent, with `model` mapped to the upstream id, and the upstream `{model, answers, usage}` object is relayed unchanged except that `model` echoes the name you sent (on the Azure OpenAI port the upstream `model` string is kept). The call is non-streaming and counts as one request toward rate limits.

```python
from openai import OpenAI  # openai-python >= 3.26.0 (>= 3.37.0 for developer/system/tool input items, input_json and image instructions)

client = OpenAI(
    base_url="http://localhost:11440/v1",
    api_key="your-user-token"
)

decision = client.decisions.create(
    model="openai:primary/gpt-6-luna",
    input="The package arrived with a broken screen. I want my money back.",
    questions=[
        {"type": "predicate", "name": "damaged", "instructions": "Does the customer report a damaged item?"},
        {
            "type": "choice",
            "name": "queue",
            "instructions": "Which team should handle this ticket?",
            "choices": [
                {"value": "billing", "description": "Refunds and payments"},
                {"value": "shipping", "description": "Delivery and damaged items"},
            ],
        },
        {
            "type": "score",
            "name": "urgency",
            "instructions": "How urgent is this ticket?",
            "levels": [{"label": "low"}, {"label": "medium"}, {"label": "high"}],
        },
    ],
)

print(decision.answers)
```

```bash
curl http://localhost:11440/v1/decisions \
  -H "Authorization: Bearer your-user-token" \
  -H "Content-Type: application/json" \
  -d '{"model": "openai:primary/gpt-6-luna",
       "input": "The package arrived with a broken screen.",
       "questions": [{"type": "predicate", "name": "damaged", "instructions": "Is the item damaged?"}]}'
```

On the Azure OpenAI port, use the plain `OpenAI` client with the v1 base URL and a Bearer token: `OpenAI(base_url="http://localhost:11439/openai/v1/", api_key="your-user-token")`. Dated `AzureOpenAI(api_version=...)` clients and the `api-key` header are not supported for this endpoint.

#### Decisions API Endpoints

| Endpoint | Port | Description |
|---|---|---|
| `/v1/decisions` | 11440 (OpenAI) | Create a decision |
| `/openai/v1/decisions` | 11439 (Azure OpenAI) | Create a decision (Azure v1 path) |
| `/openai/v1/decisions`, `/azure-openai/openai/v1/decisions` | 8765 (Management) | The same two endpoints through the management port |

**Supported backends:**
- Custom providers with the OpenAI API whose upstream serves the OpenAI-shaped `POST {base_url}/decisions` (OpenAI and compatible gateways).
- Azure providers, for deployments (e.g. `gpt-6-luna`) whose deployment row selects `/decisions` in the provider form. It is never selected by default, and a deployment whose row does not select it gets `400 unsupported_api`. Deployments not listed in any row (dynamic discovery, legacy rows) are not gated. Requests go to `{endpoint}/openai/v1/decisions`.
- Google, Bedrock, PlatformAI, Anthropic-only and System One-only custom providers and Claude deployments on Foundry answer `400 unsupported_api`. A custom OpenAI provider whose server has no `/decisions` returns that server's own error (usually 404). Upstreams that use a different decisions request shape (e.g. OpenRouter's `/api/alpha/decisions`, Perplexity) are not supported; TypeSafe-shaped **System One** upstreams, including Azure Foundry's Microsoft-Decision-1, are served on their own port instead (see [7. TypeSafe System One API](#7-typesafe-system-one-api)).

When a prefix-less model name such as `gpt-6-luna` is served by several providers, requests rotate across all of them, including ones that cannot serve decisions. Use the full model id (e.g. `openai:primary/gpt-6-luna`), or add a model mapping that pins the name to a decisions-capable provider.

### 7. TypeSafe System One API

The proxy serves the [TypeSafe System One API](https://docs.typesafe.ai/api/system-one.md) — typed answers (`noul`, `choice`, `score`, `box`) to named questions about a `state` — on its own **TypeSafe port (11366)**, which behaves like `api.typesafe.ai` so the official TypeSafe SDKs work unchanged. It is a native passthrough: the request body is forwarded as sent, with `model` mapped to the upstream id, and the upstream `{model, answers, usage}` object is relayed unchanged except that `model` echoes the name you sent. The call is non-streaming (`stream` gets a 400), counts as one request toward rate limits, and is not retried by the proxy (the SDK retries 408/429/5xx itself).

**Set up an upstream.** Add a **Custom** provider and fill in its **System One base URL**, which **includes `/v1`** — the proxy posts to `{System One base URL}/systemone` and lists models from `{System One base URL}/models`. Leave the **OpenAI base URL** blank for System One-only hosts, so their models are offered only on the TypeSafe port. For gateways that serve both APIs, one entry can carry both URLs: each discovered model is offered on the API whose listing named it (models neither listing names, e.g. ones you call by full id, are offered on both).

| Upstream | System One base URL |
|---|---|
| TypeSafe | `https://api.typesafe.ai/v1` |
| OpenRouter | `https://openrouter.ai/api/v1` |
| Fireworks | `https://api.fireworks.ai/inference/v1` |
| LLM Gateway, DigitalOcean, Requesty | their `/v1` base URL |
| Self-hosted vLLM, SGLang, Ollama, Strands | `http://host:port/v1` |

**Microsoft-Decision-1 on Azure Foundry.** Add the deployment to an **Azure** provider with the **Foundry** backend and select **/systemone (Microsoft-Decision-1)** in its deployment row. Ticking it on a new row unticks the default `/chat/completions` and `/responses`; keep the row to `/systemone` alone so the deployment is offered only on the TypeSafe port. Deployments that are discovered or in legacy rows are never offered there. The proxy posts to `{resource endpoint}/providers/microsoft/v1/systemone` (no `api-version`; a project-scoped `/api/projects/<p>` suffix is trimmed, other path prefixes are kept) with the provider's key in the `api-key` header. Call it with your **deployment name** as `model` (e.g. `Decision-1`), not the model name `microsoft-decision-1`; Foundry reports the underlying model in responses, and the proxy replaces it with the name you sent. Decision-1 answers `noul`, `choice` and `score` questions on text or JSON `state`.

Through the API, set `systemone_base_url` and include `"systemone"` in `supported_apis` (rows that list `"systemone"` without a `systemone_base_url` use `base_url`). Models are discovered from `{System One base URL}/models` in either the TypeSafe (`{"models": [...]}`) or the OpenAI (`{"data": [...]}`) shape; a provider with an OpenAI or Anthropic URL as well lists those models too. A provider created through the API without `supported_apis` gets them from its filled URLs. Hosts that list no System One models can still be called by full model id, e.g. `custom:openrouter/typesafe/jev-1.14`.

**Use the TypeSafe SDK** with the proxy as its `base_url` (**without** `/v1` — the SDK appends `/v1/systemone` and `/v1/models`) and your proxy API key:

```python
from typesafe_sdk import Choice, Noul, TypeSafeClient  # pip install "typesafe-sdk[zstd]"

with TypeSafeClient(
    base_url="http://localhost:11366",   # or http://localhost:8765/typesafe
    api_key="your-user-token",
    timeout=60,                          # the SDK default is 10 s
) as client:
    print([m.name for m in client.models.list().models])
    result = client.system_one(
        model="custom:typesafe/jev-latest",   # or a bare name such as "jev-latest"
        state="I was charged twice. Please help.",
        questions={
            "billing": Noul(instructions="Is this about billing?"),
            "tone": Choice(instructions="What is the tone?", criteria={"calm": None, "angry": None}),
        },
    )
    print(result.nouls["billing"].noul, result.choices["tone"].choice)
```

The environment variables work too: `TYPESAFE_BASE_URL=http://localhost:11366` and `TYPESAFE_API_KEY=your-user-token`. Raise the SDK's 10 s default timeout: each retry of a timed-out call is billed again upstream.

```bash
curl http://localhost:11366/v1/systemone \
  -H "Authorization: Bearer your-user-token" \
  -H "Content-Type: application/json" \
  -d '{"model": "jev-latest",
       "state": "My screen arrived cracked.",
       "questions": {"damaged": {"type": "noul", "instructions": "Is the item damaged?"}}}'
```

#### TypeSafe API Endpoints

| Endpoint | Port | Description |
|---|---|---|
| `POST /v1/systemone` | 11366 (TypeSafe) | System One request |
| `GET /v1/models` | 11366 (TypeSafe) | System One models you may use, as `{"models": [{"name", "description", "release_date"}]}` |
| `/typesafe/v1/systemone`, `/typesafe/v1/models` | 8765 (Management) | The same endpoints through the management port |

System One is **not** served on the OpenAI (11440) or Azure OpenAI (11439) ports.

**Behaviour:**
- **Supported backends:** custom providers with a System One base URL, and Azure deployments whose row selects `/systemone` (Microsoft-Decision-1). Other Azure deployments, Google, Bedrock, PlatformAI and custom providers without a System One URL answer `400` when named by full id; a bare name that no System One provider serves is a `404`. Bare names rotate only over System One providers. On the other ports, models of a System One-only provider are not listed and a bare name only such a provider serves is a `404`.
- **Headers:** only `TypeSafe-Danger-Zone-Images` and `TypeSafe-Danger-Zone-Long-Context` are forwarded upstream. `x-typesafe-request-id`, `retry-after`, `retry-after-ms`, `x-ratelimit-limit`, `x-ratelimit-remaining`, `x-ratelimit-reset` and `x-ratelimit-{limit,remaining,reset}-{requests,tokens}` are relayed back. Every `/v1/systemone` and `/v1/models` response, and every error, carries an `x-typesafe-request-id` (the upstream's when it sent one).
- **Errors** use TypeSafe's shape, `{"detail": {"error_type": "...", "message": "..."}}` (`detail[]` for 422 validation errors): `authentication_error` (403 without a key, 401 for an invalid one), `permission_error` (403), `not_found_error` (404), `invalid_request_error` (400), `rate_limit_error` (429), `request_too_large` (413), `api_usage_error` (415, 405, and 400 for an undecodable compressed body), `overloaded_error` (503, 529), `timeout_error` (504), `api_error` (other 5xx). Only `authentication_error` and `api_usage_error` are confirmed against api.typesafe.ai. Upstream TypeSafe errors are relayed verbatim; other upstream error shapes are converted.
- **Compressed bodies:** `Content-Encoding` `gzip`, `x-gzip` or `zstd` (one layer, as the SDKs send) is accepted on this port only, up to 32 MiB compressed and 32 MiB decoded, with a zstd window of at most 8 MiB; the key is checked before a compressed body is read. `br` and stacked codings get `415`, corrupt data `400`, oversized compressed bodies `413`. Uncompressed bodies are handled as on the other ports.
- **Quota:** a request is charged once it passes validation; `stream`, a model not served over System One, and NaN/Infinity or unpaired surrogates in the body get a `400` without being charged.
- **Not supported yet:** image uploads through the SDK, which always sends them as `multipart/form-data` (`415`); raw-HTTP callers can send `assets` inline as JSON with `data_base64`.
- **Model mappings** apply on the TypeSafe port only when the mapping has **TypeSafe** ticked in its APIs (off by default).

### Model Naming Format

All models follow the pattern: `{provider_key}/{model_name}`

Examples:
- `azure:primary/gpt-4`
- `bedrock:production/anthropic.claude-3-5-sonnet-20240620-v1:0`
- `google:primary/gemini-2.0-flash-exp`
- `openai:primary/gpt-4o`
- `ollama:local/llama3.2`

## Web Search Interception

When a client sends a web search tool, the proxy can strip it, run the searches
itself against a self-hosted metasearch engine, and feed the results back to the
model. Two backend types are supported, and the admin can register any number of
instances of each:

| Backend type | Compose service | Base URL to configure |
|---------|-----------------|-----------------------|
| SearXNG | `searxng` | `http://searxng:8080` |
| 4get | `fourget` | `http://fourget` |

Both services are defined in `docker-compose.yaml.example` on
`llm-proxy-network` with no host port published, so they are reachable only
from the proxy container. SearXNG refuses to start without a real
`SEARXNG_SECRET`; generate one with
`python -c "import secrets; print(secrets.token_hex(32))"`. Its settings
(JSON output enabled, engine timeouts, unreliable engines removed) are inlined
as a compose `config`, which needs Docker Compose 2.23 or newer.

### How it works

1. A request arrives with a web search tool. Detected forms:
   - **Anthropic Messages:** server tools such as `web_search_20250305`, or a
     schema-less tool named `web_search` (as sent by Claude Code).
   - **Chat Completions:** the `web_search_options` field.
   - **Responses API:** any tool whose type starts with `web_search` (e.g. `web_search`, `web_search_preview`).
   - On any surface, a function tool named `web_search_proxy` (or
     `litellm_web_search`, for LiteLLM compatibility).
2. The proxy replaces it with an internal `web_search_proxy` function tool and
   calls the upstream model.
3. When the model calls that tool, the proxy runs the queries concurrently on
   the caller's backend and feeds the results back, repeating until the model
   answers or the round limit is reached. Duplicate queries are skipped.
4. The client sees one request and one response. On the Anthropic and
   Responses surfaces, searches come back as native `server_tool_use` /
   `web_search_tool_result` blocks or `web_search_call` items, so clients
   render them as they would a provider-hosted search. Streaming is supported,
   and token usage is summed across all rounds.

Intercepted responses carry an `x-llmproxy-websearch: rounds=<n>;queries=<n>`
header.

### Configuration

**These settings are not environment variables.** Everything is configured in
the admin dashboard under **Web Search**, and changes take effect immediately,
without a restart:

1. Under **Backends**, press *Add backend*: give it a name, pick its type, enter
   its base URL and press *Run test*. Each backend keeps its own search
   parameters. The first backend becomes the default.
2. Under **Defaults**, choose the default backend and the result limits.
3. Under **Interception**, turn it on and choose which API surfaces and
   providers it applies to.

| Setting | Default | Who sets it | Description |
|---------|---------|-------------|-------------|
| Enabled | off | admin | Master switch |
| Apply to | all surfaces | admin | Anthropic Messages, Chat Completions, Responses |
| Providers | none | admin | Provider instances interception applies to (`*` for all) |
| Timeout | 10s | admin | Per-search timeout |
| Max agentic loops | 3 | admin | Search rounds before the model must answer |
| Max queries per turn | 5 | admin | Queries accepted from a single model turn |
| Default backend | first added | admin | Where searches go for users who did not pick one |
| Max results | 5 | admin default, user | Results returned per query (1–20) |
| Max snippet chars | 500 | admin default, user | Snippet length per result (50–5000) |
| SearXNG: categories / language / safesearch / time range / engines | `general` / `auto` / moderate / any / default | per backend, user | Passed through to SearXNG |
| 4get: scraper / language / country / safesearch / time range | `ddg` / — / — / moderate / any | per backend, user | Upstream engine 4get scrapes, its locale and filters |

Interception only runs when it is enabled, the default backend is enabled, and
both the request's API surface and its provider are selected. Otherwise the web
search tool is passed to the upstream provider unchanged.

#### Backends

- **Enabled:** a disabled backend is never used. The default backend cannot be
  disabled or deleted; choose another default first.
- **Users can pick this backend:** when off, the backend disappears from the
  users' list. It can still be the default.
- A backend's type cannot change after it is created; add a new backend instead.
- Deleting a backend returns the users who picked it to the default and removes
  their parameters for it. Turning it off or hiding it only moves them to the
  default while it stays that way; their choice comes back with the backend.

#### User preferences

Each user has a **Web Search** tab in their dashboard where they can:

- pick another backend among the enabled, user-selectable ones (the URL is
  never shown to them),
- override the backend's search parameters, and the result limits.

Anything a user leaves empty follows the admin default: the default backend,
that backend's parameters, and the default limits. Parameters are saved per
backend, so switching backends and back keeps them. Engines and categories are
one choice, as are 4get's scraper with its language and country: a user who sets
either takes the whole group, so it never mixes with the backend's own. Users
cannot turn interception off or change the admin-only settings above.

Endpoints, all under the management server:

| Endpoint | Who | Purpose |
|----------|-----|---------|
| `GET` / `PUT /admin/websearch/settings` | admin | Interception settings, default backend and limits |
| `GET` / `POST` / `PUT` / `DELETE /admin/websearch/backends[?backend_id=]` | admin | List, add, replace, delete backends |
| `PUT /admin/websearch/backends/toggle?backend_id=` | admin | Turn a backend on or off |
| `POST /admin/websearch/test` | admin | Run one query on an unsaved backend, a stored one, or the default |
| `POST /admin/websearch/engines` | admin | A SearXNG instance's engines and categories |
| `GET` / `PUT` / `DELETE /auth/websearch` | user | The caller's choices, available backends and effective settings; reset |
| `GET /auth/websearch/engines?backend_id=` | user | Engines and categories of a pickable SearXNG backend (cached 10 minutes) |

#### Upgrading from a single backend

Before multiple backends, `PUT /admin/websearch/settings` took the backend's URL
and parameters directly. On the first start after upgrading, the stored SearXNG
and 4get configurations each become a backend (named `SearXNG` and `4get`) with
their parameters, and the one that was selected becomes the default. The
database is backed up first to `llm_proxy.db.pre-websearch-backends-<time>.bak`.
The old columns are left in place, so the previous release still finds its
settings if you roll back. The settings endpoint now takes `default_backend_id`
instead; backends are edited through `/admin/websearch/backends`.

## API Documentation

Each server exposes interactive Swagger UI documentation:

| Server | Swagger UI | OpenAPI JSON |
|--------|-----------|-------------|
| OpenAI API | `http://localhost:11440/docs` | `http://localhost:11440/openapi.json` |
| Anthropic API | `http://localhost:2027/docs` | `http://localhost:2027/openapi.json` |
| Azure OpenAI API | `http://localhost:11439/docs` | `http://localhost:11439/openapi.json` |
| TypeSafe API | `http://localhost:11366/docs` | `http://localhost:11366/openapi.json` |
| Management | `http://localhost:8765/docs` | `http://localhost:8765/openapi.json` |

## Authentication

The LLM Proxy Server supports multiple authentication methods:

### Standard User Authentication

Users can sign up and log in with email/password.

### Zoho OAuth SSO (Optional)

Enable Single Sign-On with Zoho:

```yaml
environment:
  - ZOHO_CLIENT_ID=1000.XXXXXXXXXXXXXXXXXXXXXX
  - ZOHO_CLIENT_SECRET=your-zoho-client-secret
  - ZOHO_REDIRECT_URI=http://localhost:8765/auth/zoho/callback
```

Once configured, users can click **"Login with Zoho"** to authenticate with their Zoho account. The integration automatically creates user accounts on first login and works with all Zoho data centers.

### Admin Authentication

Admin users have full access to provider configuration and user management:

- Default admin credentials are set via environment variables
- Access admin dashboard at `http://localhost:8765/admin`
- Login with `LLMPROXY_ADMIN_USERNAME` and `LLMPROXY_ADMIN_PASSWORD`

**Security Note**: Change default admin credentials in production!

## Notification Webhooks

The LLM Proxy Server can send webhook notifications for various system events, making it easy to integrate with messaging platforms like Slack, Discord, Microsoft Teams, or custom notification systems.

### Configuration

Enable webhooks by setting the webhook URL:

```yaml
environment:
  - NOTIFICATION_WEBHOOK_URL=https://your-webhook-endpoint.com/notifications
```

If not configured, webhooks are silently skipped with no errors.

### Supported Events

#### User Signup Notifications

Receive notifications when new users sign up through any method:
- **Manual signup** — User registers via signup page (requires admin approval)
- **OAuth signup** — User logs in with Zoho OAuth for the first time (auto-approved)
- **Admin-created** — Admin creates user via dashboard (auto-approved)

**Payload Format** (optimized for messaging platforms):

```json
{
  "text": "🔔 New User Signup\n\n👤 **Username:** john_doe\n📧 **Email:** john@example.com\n🔐 **Signup Method:** Manual Signup\n📊 **Status:** ⏳ Pending Approval\n🆔 **User ID:** 123"
}
```

The text format is compatible with Slack, Discord, Microsoft Teams, and other messaging webhooks.

#### Daily Usage Report

When `NOTIFICATION_WEBHOOK_URL` is set, a usage summary is posted every day at **08:00 in the configured `TIMEZONE`**. It contains:
- Total request count for yesterday and for the last 30 days
- Top 5 users by request count, for yesterday and for the last 30 days
- Top 5 models by request count, for yesterday and for the last 30 days

"Last 30 days" means the 30 complete days ending yesterday, so every figure is final. It does not include today's partial counts, unlike the dashboard's 30-day view. If the server is down at 08:00, that day's report is skipped and not sent later.

**Payload Format** (same text format as the signup notification):

```json
{
  "text": "📊 Daily Usage Report — 2026-10-15\n🌐 Domain: llm.example.com\n\n📈 Total Requests\n• Yesterday (2026-10-15): 1,234\n• Last 30 days (2026-09-16 → 2026-10-15): 45,678\n\n👤 Top 5 Users — Yesterday\n1. alice — 400\n...\n\n🤖 Top 5 Models — Last 30 Days\n1. openai/gpt-4o — 12,345\n..."
}
```

### Integration Examples

**Slack Incoming Webhook:**
```yaml
environment:
  - NOTIFICATION_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL
```

**Discord Webhook:**
```yaml
environment:
  - NOTIFICATION_WEBHOOK_URL=https://discord.com/api/webhooks/YOUR_WEBHOOK_ID/YOUR_WEBHOOK_TOKEN
```

**Microsoft Teams Webhook:**
```yaml
environment:
  - NOTIFICATION_WEBHOOK_URL=https://outlook.office.com/webhook/YOUR_WEBHOOK_URL
```

### Testing

Use a webhook testing service to verify notifications:
1. Get a test URL from [webhook.site](https://webhook.site/) or [requestbin.com](https://requestbin.com/)
2. Set `NOTIFICATION_WEBHOOK_URL` to the test URL
3. Create a test user or perform a signup
4. Check the test site for the webhook payload

### Error Handling

- **No URL configured**: Webhooks skipped silently (no error)
- **Request failures**: Errors logged but don't block the triggering operation
- **Timeout**: 10 seconds per webhook request
- All webhook activity is logged for monitoring and troubleshooting

Webhook behavior is summarized in the section above. (A dedicated webhook setup
guide is not yet part of the repository.)

## Observability

### OpenTelemetry Integration

Enable distributed tracing by setting environment variables:

```yaml
environment:
  - OTEL_EXPORTER_OTLP_ENDPOINT=https://your-otlp-collector:4318
  - OTEL_SERVICE_NAME=llm-proxy-server
  - OTEL_EXPORTER_OTLP_HEADERS=api-key=your-api-key
```

### Health Check

Each server exposes a `/health` endpoint:

```bash
curl http://localhost:11440/health  # OpenAI API
curl http://localhost:2027/health   # Anthropic API
curl http://localhost:11439/health  # Azure OpenAI API
curl http://localhost:11366/health  # TypeSafe API
curl http://localhost:8765/health   # Management
```

## Data Persistence

The container uses a volume mount for data persistence:

```bash
-v ./data:/llm-proxy-server/data
```

This directory contains:
- `llm_proxy.db` - SQLite database with user accounts, provider configurations, and model cache

## Logs

View container logs:

```bash
docker logs -f llm-proxy-server
```
