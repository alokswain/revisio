# Revisio

A local Python chat assistant with a responsive browser UI, backed by OpenAI's
Chat Completions API. Ask questions and continue the conversation with follow-ups.

## Run locally

Requires Python 3.10+ and an OpenAI API key with API billing enabled.
A ChatGPT subscription does not include API usage.

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY="your-api-key"
python app.py
```

Open http://127.0.0.1:8000. Without a key, the UI opens but cannot send questions.
Enter your real key directly in your terminal, never in the browser or source code.
The app reads environment variables; it does not automatically load `.env` files.

Optional environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `OPENAI_MODEL` | `gpt-4.1-mini` | An OpenAI model supporting Chat Completions |
| `PORT` | `8000` | Local server port |

## Behavior

- Keys remain on the Python server; questions are sent to OpenAI.
- The most recent 39 messages provide context for follow-up questions.
- Conversations live in browser memory and disappear on refresh or New chat.
- Failed questions can be edited or retried without duplicating chat history.
- Replies are displayed as plain text, never executed as HTML.
- Fonts and icons are loaded from Google Fonts and unpkg; system fallbacks work offline.

## Test

```sh
source .venv/bin/activate
python -m unittest -v
```

Tests mock OpenAI requests and require neither a real key nor network access.

## Deployment

This is a local, single-user app. The Flask development server binds only to
`127.0.0.1`. Before exposing it publicly, add authentication, request rate limits,
HTTPS, and a production WSGI server; otherwise visitors could spend your API quota.


