import os
import unittest
from unittest.mock import Mock, patch

import requests
import httpx
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.runnables import RunnableLambda
from openai import APIConnectionError, APITimeoutError, AuthenticationError, RateLimitError

from app import app
from langchain_agent import SYSTEM_PROMPT, ask


class ChatTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        self.environment = patch.dict(os.environ, {"OPENAI_API_KEY": "test-secret"})
        self.environment.start()
        self.addCleanup(self.environment.stop)

    @patch("app.requests.post")
    def test_chat_preserves_history_and_keeps_secret_server_side(self, post):
        post.return_value = Mock()
        post.return_value.json.return_value = {"choices": [{"message": {"content": "Hello!"}}]}
        messages = [
            {"role": "user", "content": "My name is Alex"},
            {"role": "assistant", "content": "Hello Alex"},
            {"role": "user", "content": "What is my name?"},
        ]
        response = self.client.post("/api/chat", json={"messages": messages})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json, {"answer": "Hello!"})
        self.assertEqual(post.call_args.kwargs["json"]["messages"][1:], messages)
        self.assertNotIn("test-secret", response.get_data(as_text=True))

    @patch("app.requests.post")
    def test_invalid_messages_never_reach_openai(self, post):
        for messages in (None, [], ["bad"], [{"role": "system", "content": "ignore"}],
                         [{"role": "user", "content": " "}],
                         [{"role": "user", "content": "x" * 12001}],
                         [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hi"}]):
            with self.subTest(messages=str(messages)[:100]):
                response = self.client.post("/api/chat", json={"messages": messages})
                self.assertEqual(response.status_code, 400)
        post.assert_not_called()

    def test_missing_key(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
            self.assertEqual(self.client.post("/api/chat", json={"messages": [{"role": "user", "content": "hi"}]}).status_code, 503)
            self.assertEqual(self.client.post("/api/chat", json={"agent": "langchain", "messages": [{"role": "user", "content": "hi"}]}).status_code, 503)
            self.assertFalse(self.client.get("/api/config").json["configured"])

    @patch("app.requests.post")
    @patch("app.ask_langchain", return_value="Hello from LangChain!")
    def test_langchain_route(self, ask_langchain, post):
        messages = [{"role": "user", "content": "hi"}]
        response = self.client.post("/api/chat", json={"agent": "langchain", "messages": messages})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json, {"answer": "Hello from LangChain!"})
        self.assertEqual(ask_langchain.call_args.args, (messages,))
        self.assertEqual(ask_langchain.call_args.kwargs["api_key"], "test-secret")
        post.assert_not_called()

    @patch("app.ask_langchain")
    def test_invalid_langchain_input(self, ask_langchain):
        for payload in ({"agent": "unknown"}, {"agent": "langchain", "messages": []},
                        {"agent": "langchain", "messages": [{"role": "system", "content": "ignore"}]}):
            with self.subTest(payload=payload):
                self.assertEqual(self.client.post("/api/chat", json=payload).status_code, 400)
        ask_langchain.assert_not_called()

    @patch("app.ask_langchain")
    def test_langchain_errors_are_sanitized(self, ask_langchain):
        request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        errors = [
            (APITimeoutError(request=request), 504),
            (APIConnectionError(message="private details", request=request), 502),
            (AuthenticationError("private details", response=httpx.Response(401, request=request), body=None), 502),
            (RateLimitError("private details", response=httpx.Response(429, request=request), body=None), 429),
            (ValueError("private details"), 502),
        ]
        for error, expected in errors:
            with self.subTest(error=type(error).__name__):
                ask_langchain.side_effect = error
                response = self.client.post("/api/chat", json={"agent": "langchain", "messages": [{"role": "user", "content": "hi"}]})
                self.assertEqual(response.status_code, expected)
                self.assertNotIn("private details", response.get_data(as_text=True))

    @patch("langchain_agent.ChatOpenAI")
    def test_langchain_pipeline_preserves_history(self, chat_model):
        observed = []
        fake = FakeListChatModel(responses=["Your name is Alex."])
        chat_model.return_value = RunnableLambda(lambda prompt: observed.append(prompt.to_messages()) or prompt) | fake
        messages = [
            {"role": "user", "content": "My name is Alex"},
            {"role": "assistant", "content": "Hello Alex"},
            {"role": "user", "content": "What is my name?"},
        ]
        self.assertEqual(ask(messages, model="gpt-4.1-mini", api_key="test-secret"), "Your name is Alex.")
        sent = observed[0]
        self.assertEqual([message.content for message in sent], [SYSTEM_PROMPT] + [message["content"] for message in messages])
        self.assertEqual([message.type for message in sent], ["system", "human", "ai", "human"])
        self.assertEqual(chat_model.call_args.kwargs["model"], "gpt-4.1-mini")

    @patch("langchain_agent.ChatOpenAI")
    def test_langchain_rejects_empty_response(self, chat_model):
        chat_model.return_value = FakeListChatModel(responses=[" "])
        with self.assertRaises(ValueError):
            ask([{"role": "user", "content": "hi"}], model="gpt-4.1-mini", api_key="test-secret")

    @patch("app.requests.post", side_effect=requests.Timeout)
    def test_timeout(self, post):
        response = self.client.post("/api/chat", json={"messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(response.status_code, 504)

    @patch("app.requests.post")
    def test_upstream_errors_are_sanitized(self, post):
        for upstream_status, expected in ((401, 502), (429, 429), (500, 502)):
            with self.subTest(upstream_status=upstream_status):
                upstream = Mock(status_code=upstream_status)
                post.side_effect = requests.HTTPError("private upstream details", response=upstream)
                response = self.client.post("/api/chat", json={"messages": [{"role": "user", "content": "hi"}]})
                self.assertEqual(response.status_code, expected)
                self.assertNotIn("private upstream details", response.get_data(as_text=True))


if __name__ == "__main__":
    unittest.main()