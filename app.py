import os
from pathlib import Path

import requests
from flask import Flask, jsonify, request, send_from_directory
from openai import APIError, APITimeoutError, AuthenticationError, PermissionDeniedError, RateLimitError

from langchain_agent import SYSTEM_PROMPT, ask as ask_langchain


ROOT = Path(__file__).parent
app = Flask(__name__, static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 128 * 1024


@app.get("/")
def index():
    return send_from_directory(ROOT / "static", "index.html")


@app.get("/api/config")
def config():
    return jsonify(
        configured=bool(os.environ.get("OPENAI_API_KEY")),
        model=os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"),
    )


@app.post("/api/chat")
def chat():
    payload = request.get_json(silent=True)
    agent = payload.get("agent", "openai") if isinstance(payload, dict) else "openai"
    if agent not in ("openai", "langchain"):
        return jsonify(error="Choose the openai or langchain agent."), 400
    messages = payload.get("messages") if isinstance(payload, dict) else None
    if not isinstance(messages, list) or not 1 <= len(messages) <= 40:
        return jsonify(error="Send between 1 and 40 messages."), 400
    for position, message in enumerate(messages):
        expected_role = "user" if position % 2 == 0 else "assistant"
        if (
            not isinstance(message, dict)
            or message.get("role") != expected_role
            or not isinstance(message.get("content"), str)
            or not message["content"].strip()
            or len(message["content"]) > 12000
        ):
            return jsonify(error="Messages must alternate user and assistant, with 1-12000 characters each."), 400
    if messages[-1]["role"] != "user":
        return jsonify(error="The last message must be a question from the user."), 400

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return jsonify(error="Set OPENAI_API_KEY on the server, then restart the app."), 503

    try:
        if agent == "langchain":
            answer = ask_langchain(
                messages,
                model=os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"),
                api_key=api_key,
            )
            return jsonify(answer=answer)
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"),
                "messages": [{"role": "system", "content": SYSTEM_PROMPT}]
                + [{"role": message["role"], "content": message["content"]} for message in messages],
                "max_completion_tokens": 2048,
            },
            timeout=(10, 90),
        )
        response.raise_for_status()
        answer = response.json()["choices"][0]["message"]["content"]
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("Empty model response")
        return jsonify(answer=answer)
    except (requests.Timeout, APITimeoutError):
        return jsonify(error="The response timed out. Please try again."), 504
    except RateLimitError:
        return jsonify(error="OpenAI rate limit or billing quota reached. Check your API account and try again."), 429
    except (AuthenticationError, PermissionDeniedError):
        return jsonify(error="OpenAI rejected the server credentials. Check OPENAI_API_KEY and model access."), 502
    except APIError:
        return jsonify(error="OpenAI could not complete the request. Please try again."), 502
    except requests.HTTPError as error:
        upstream_status = error.response.status_code if error.response is not None else None
        if upstream_status == 429:
            return jsonify(error="OpenAI rate limit or billing quota reached. Check your API account and try again."), 429
        if upstream_status in (401, 403):
            return jsonify(error="OpenAI rejected the server credentials. Check OPENAI_API_KEY and model access."), 502
        return jsonify(error="OpenAI could not complete the request. Please try again."), 502
    except (requests.RequestException, ValueError, KeyError, IndexError, TypeError):
        return jsonify(error="Could not get a response from OpenAI. Please try again."), 502


@app.errorhandler(413)
def too_large(error):
    return jsonify(error="This conversation is too large. Start a new chat."), 413


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "8000")))