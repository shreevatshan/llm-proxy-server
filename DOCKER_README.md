# LLM Proxy Server

📦 **GitHub:** [shreevatshan/llm-proxy-server](https://github.com/shreevatshan/llm-proxy-server)

A unified proxy server that exposes **OpenAI-compatible**, **Anthropic-compatible**, and **Azure OpenAI-compatible** API endpoints for multiple LLM providers including Azure OpenAI, AWS Bedrock, Google Gemini, OpenAI, Ollama, LlamaCpp, and any custom server. Supports Chat Completions, Embeddings, Images, Audio, the **OpenAI Responses API**, the **Anthropic Messages API**, and **Azure OpenAI deployment-based URLs**.

## Architecture

The server runs **four services** in a single container:

| Service | Default Port | Purpose |
|---------|-------------|--------|
| **OpenAI API** | `11440` | OpenAI-compatible API endpoints (`/v1/chat/completions`, `/v1/models`, etc.) |
| **Anthropic API** | `2027` | Anthropic Messages API endpoints (`/v1/messages`, `/v1/models`) |
| **Azure OpenAI API** | `11439` | Azure OpenAI-compatible deployment-based endpoints (`/openai/deployments/{provider}/{deployment}/chat/completions`, etc.) |
| **Management** | `8765` | Admin dashboard, user login, provider configuration |

## Quick Start

```bash
docker run -d \
  -p 11440:11440 \
  -p 2027:2027 \
  -p 11439:11439 \
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
- **Custom** — Any OpenAI-compatible and/or Anthropic-compatible server (OpenAI, Ollama, LlamaCpp, vLLM, etc.)

Custom providers can be configured to support **OpenAI API**, **Anthropic API**, or **both**.

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
model. Two backends are supported and either can be selected at runtime:

| Backend | Compose service | Base URL to configure |
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
   the configured backend and feeds the results back, repeating until the model
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
the admin dashboard under **Web Search**: pick the backend, enter its base URL,
press *Run test* to confirm connectivity, then enable interception and choose
which API surfaces and providers it applies to. Each backend stores its own URL
and parameters, so switching between them keeps both configurations. Changes
take effect immediately, without a restart.

| Setting | Default | Description |
|---------|---------|-------------|
| Apply to | all surfaces | Anthropic Messages, Chat Completions, Responses |
| Providers | none | Provider instances interception applies to (`*` for all) |
| Max results | 5 | Results returned per query |
| Max snippet chars | 500 | Snippet length per result |
| Timeout | 10s | Per-search timeout |
| Max agentic loops | 3 | Search rounds before the model must answer |
| Max queries per turn | 5 | Queries accepted from a single model turn |
| SearXNG: categories / language / safesearch / time range / engines | `general` / `auto` / moderate / any / default | Passed through to SearXNG |
| 4get: scraper / language / country | `ddg` / — / — | Upstream engine 4get scrapes, plus locale |

Interception only runs when it is enabled, a base URL is set for the selected
backend, and both the request's API surface and its provider are selected.
Otherwise the web search tool is passed to the upstream provider unchanged.

## API Documentation

Each server exposes interactive Swagger UI documentation:

| Server | Swagger UI | OpenAPI JSON |
|--------|-----------|-------------|
| OpenAI API | `http://localhost:11440/docs` | `http://localhost:11440/openapi.json` |
| Anthropic API | `http://localhost:2027/docs` | `http://localhost:2027/openapi.json` |
| Azure OpenAI API | `http://localhost:11439/docs` | `http://localhost:11439/openapi.json` |
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
