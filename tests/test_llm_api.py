import json
import unittest
from unittest.mock import patch, MagicMock

import requests
from book_generator.llm_api import (
    call_llm_api,
    _call_llama_cpp_api_internal,
)


class TestLlmApi(unittest.TestCase):
    @patch("book_generator.llm_api.save_to_cache")
    @patch("book_generator.llm_api.load_from_cache")
    @patch("book_generator.llm_api._call_llama_cpp_api_internal")
    def test_call_llm_api(
        self,
        mock_call_llama_cpp,
        mock_load_from_cache,
        mock_save_to_cache,
    ):
        mock_load_from_cache.return_value = None
        mock_call_llama_cpp.return_value = "successful response"

        config = {
            "api_settings": {"provider": "llama_cpp"},
        }
        prompt = "test prompt"

        result = call_llm_api(prompt, config)

        self.assertEqual(result, "successful response")
        self.assertEqual(mock_call_llama_cpp.call_count, 1)
        mock_save_to_cache.assert_called_once_with(prompt, "successful response", "api_cache", None)

    @patch("requests.post")
    def test_llama_cpp_non_streaming_native_completion_endpoint(self, mock_post):
        mock_tokenize_resp = MagicMock()
        mock_tokenize_resp.status_code = 200
        mock_tokenize_resp.json.return_value = {"tokens": [1, 2, 3, 4]}

        mock_completion_resp = MagicMock()
        mock_completion_resp.status_code = 200
        mock_completion_resp.json.return_value = {
            "content": "Generated story text",
            "stop": True,
        }
        mock_post.side_effect = [mock_tokenize_resp, mock_completion_resp]

        config = {
            "api_settings": {
                "provider": "llama_cpp",
                "llama_cpp": {
                    "base_url": "http://localhost:8080",
                    "endpoint": "/completion",
                    "model": "llama-3",
                    "temperature": 0.7,
                },
            },
            "debug_options": {"verbose_debug": False},
        }

        result = _call_llama_cpp_api_internal("Write a chapter.", config)
        self.assertEqual(result, "Generated story text")
        self.assertEqual(mock_post.call_count, 2)
        # Check completion call args
        completion_call = mock_post.call_args_list[1]
        args, kwargs = completion_call
        self.assertEqual(args[0], "http://localhost:8080/completion")
        self.assertEqual(kwargs["json"]["prompt"], "Write a chapter.")
        self.assertEqual(kwargs["json"]["temperature"], 0.7)
        self.assertFalse(kwargs["stream"])

    @patch("requests.post")
    def test_llama_cpp_chat_completions_default(self, mock_post):
        mock_tokenize_resp = MagicMock()
        mock_tokenize_resp.status_code = 200
        mock_tokenize_resp.json.return_value = {"tokens": [1, 2]}

        mock_completion_resp = MagicMock()
        mock_completion_resp.status_code = 200
        mock_completion_resp.json.return_value = {
            "choices": [{"message": {"content": "Chat completion story text"}}],
        }
        mock_post.side_effect = [mock_tokenize_resp, mock_completion_resp]

        config = {
            "api_settings": {
                "provider": "llama_cpp",
                "llama_cpp": {
                    "base_url": "http://localhost:8080",
                    "model": "llama-3",
                    "temperature": 0.7,
                },
            },
            "debug_options": {"verbose_debug": False},
        }

        result = _call_llama_cpp_api_internal("Write a chapter.", config)
        self.assertEqual(result, "Chat completion story text")
        self.assertEqual(mock_post.call_count, 2)
        completion_call = mock_post.call_args_list[1]
        args, kwargs = completion_call
        self.assertEqual(args[0], "http://localhost:8080/v1/chat/completions")
        self.assertEqual(kwargs["json"]["messages"], [{"role": "user", "content": "Write a chapter."}])
        self.assertEqual(kwargs["json"]["model"], "llama-3")
        self.assertEqual(kwargs["json"]["temperature"], 0.7)
        self.assertFalse(kwargs["stream"])

    @patch("requests.post")
    def test_llama_cpp_non_streaming_openai_format(self, mock_post):
        mock_tokenize_resp = MagicMock()
        mock_tokenize_resp.status_code = 404

        mock_completion_resp = MagicMock()
        mock_completion_resp.status_code = 200
        mock_completion_resp.json.return_value = {
            "choices": [{"text": "OpenAI format text"}]
        }
        mock_post.side_effect = [mock_tokenize_resp, mock_completion_resp]

        config = {
            "api_settings": {
                "llama_cpp": {
                    "base_url": "http://localhost:8080",
                },
            },
            "debug_options": {"verbose_debug": False},
        }

        result = _call_llama_cpp_api_internal("Test prompt", config)
        self.assertEqual(result, "OpenAI format text")

    @patch("requests.post")
    def test_llama_cpp_streaming_success(self, mock_post):
        mock_tokenize_resp = MagicMock()
        mock_tokenize_resp.status_code = 200
        mock_tokenize_resp.json.return_value = {"tokens": [1, 2]}

        mock_stream_resp = MagicMock()
        mock_stream_resp.status_code = 200
        mock_stream_resp.iter_lines.return_value = [
            b'data: {"content": "Once "}',
            b'data: {"content": "upon "}',
            b'data: {"content": "a time.", "stop": true}',
        ]
        mock_post.side_effect = [mock_tokenize_resp, mock_stream_resp]

        config = {
            "api_settings": {
                "llama_cpp": {
                    "base_url": "http://localhost:8080",
                },
            },
            "debug_options": {"verbose_debug": True},
        }

        result = _call_llama_cpp_api_internal("Tell a tale.", config)
        self.assertEqual(result, "Once upon a time.")

    @patch("time.sleep")
    @patch("requests.post")
    def test_llama_cpp_retry_on_failure(self, mock_post, mock_sleep):
        mock_tokenize_resp = MagicMock()
        mock_tokenize_resp.status_code = 200
        mock_tokenize_resp.json.return_value = {"tokens": [1]}

        mock_err_resp = MagicMock()
        mock_err_resp.status_code = 500
        http_err = requests.exceptions.HTTPError(response=mock_err_resp)

        mock_ok_resp = MagicMock()
        mock_ok_resp.status_code = 200
        mock_ok_resp.json.return_value = {"content": "Recovered response"}

        mock_post.side_effect = [mock_tokenize_resp, http_err, mock_ok_resp]

        config = {
            "api_settings": {
                "llama_cpp": {
                    "base_url": "http://localhost:8080",
                    "max_retries": 2,
                    "retry_delay_seconds": 1,
                },
            },
            "debug_options": {"verbose_debug": False},
        }

        result = _call_llama_cpp_api_internal("Retry prompt", config)
        self.assertEqual(result, "Recovered response")
        mock_sleep.assert_called_once_with(1)

    @patch("requests.post")
    def test_llama_cpp_disable_reasoning_payload(self, mock_post):
        mock_tokenize_resp = MagicMock()
        mock_tokenize_resp.status_code = 200
        mock_tokenize_resp.json.return_value = {"tokens": [1]}

        mock_completion_resp = MagicMock()
        mock_completion_resp.status_code = 200
        mock_completion_resp.json.return_value = {
            "content": "<think>Thinking about chapter</think>Final chapter content",
            "stop": True,
        }
        mock_post.side_effect = [mock_tokenize_resp, mock_completion_resp]

        config = {
            "api_settings": {
                "llama_cpp": {
                    "base_url": "http://localhost:8080",
                    "model": "deepseek-r1",
                    "disable_reasoning": True,
                },
            },
            "debug_options": {"verbose_debug": False},
        }

        result = _call_llama_cpp_api_internal("Write chapter", config)
        self.assertEqual(result, "Final chapter content")

        # Verify payload options
        completion_call = mock_post.call_args_list[1]
        _, kwargs = completion_call
        self.assertEqual(kwargs["json"]["reasoning_budget"], 0)
        self.assertEqual(kwargs["json"]["chat_template_kwargs"], {"enable_thinking": False})

    @patch("requests.post")
    def test_llama_cpp_streaming_disable_reasoning_strips_thinking_tags(self, mock_post):
        mock_tokenize_resp = MagicMock()
        mock_tokenize_resp.status_code = 200
        mock_tokenize_resp.json.return_value = {"tokens": [1]}

        mock_stream_resp = MagicMock()
        mock_stream_resp.status_code = 200
        mock_stream_resp.iter_lines.return_value = [
            b'data: {"content": "<think>Thinking"}',
            b'data: {"content": " process...</think>Actual output", "stop": true}',
        ]
        mock_post.side_effect = [mock_tokenize_resp, mock_stream_resp]

        config = {
            "api_settings": {
                "llama_cpp": {
                    "base_url": "http://localhost:8080",
                    "disable_reasoning": True,
                },
            },
            "debug_options": {"verbose_debug": True},
        }

        result = _call_llama_cpp_api_internal("Generate", config)
        self.assertEqual(result, "Actual output")


    @patch("requests.post")
    def test_llama_cpp_enabled_reasoning_strips_thinking_tags(self, mock_post):
        mock_tokenize_resp = MagicMock()
        mock_tokenize_resp.status_code = 200
        mock_tokenize_resp.json.return_value = {"tokens": [1]}

        mock_completion_resp = MagicMock()
        mock_completion_resp.status_code = 200
        mock_completion_resp.json.return_value = {
            "content": "<think>Complex reasoning about plot</think>Clean story output",
            "stop": True,
        }
        mock_post.side_effect = [mock_tokenize_resp, mock_completion_resp]

        config = {
            "api_settings": {
                "llama_cpp": {
                    "base_url": "http://localhost:8080",
                    "model": "gemma4:26B",
                    "disable_reasoning": False,
                },
            },
            "debug_options": {"verbose_debug": False},
        }

        result = _call_llama_cpp_api_internal("Write chapter", config)
        self.assertEqual(result, "Clean story output")

        # Verify payload does NOT inject reasoning_budget: 0 or enable_thinking: False
        completion_call = mock_post.call_args_list[1]
        _, kwargs = completion_call
        self.assertNotIn("reasoning_budget", kwargs["json"])
        self.assertNotIn("chat_template_kwargs", kwargs["json"])

    @patch("requests.post")
    def test_llama_cpp_streaming_reasoning_content_delta_handled(self, mock_post):
        mock_tokenize_resp = MagicMock()
        mock_tokenize_resp.status_code = 200
        mock_tokenize_resp.json.return_value = {"tokens": [1]}

        mock_stream_resp = MagicMock()
        mock_stream_resp.status_code = 200
        mock_stream_resp.iter_lines.return_value = [
            b'data: {"choices": [{"delta": {"reasoning_content": "Thinking step 1... "}}]}',
            b'data: {"choices": [{"delta": {"reasoning_content": "Thinking step 2."}}]}',
            b'data: {"choices": [{"delta": {"content": "Story chapter text."}}], "stop": true}',
        ]
        mock_post.side_effect = [mock_tokenize_resp, mock_stream_resp]

        config = {
            "api_settings": {
                "llama_cpp": {
                    "base_url": "http://localhost:8080",
                    "disable_reasoning": False,
                },
            },
            "debug_options": {"verbose_debug": True},
        }

        result = _call_llama_cpp_api_internal("Generate", config)
        self.assertEqual(result, "Story chapter text.")

    @patch("requests.post")
    def test_llama_cpp_non_streaming_reasoning_content_handled(self, mock_post):
        mock_tokenize_resp = MagicMock()
        mock_tokenize_resp.status_code = 200
        mock_tokenize_resp.json.return_value = {"tokens": [1]}

        mock_completion_resp = MagicMock()
        mock_completion_resp.status_code = 200
        mock_completion_resp.json.return_value = {
            "choices": [{
                "message": {
                    "content": "<think>Internal thought</think>Chapter text.",
                    "reasoning_content": "Model thought process here",
                }
            }]
        }
        mock_post.side_effect = [mock_tokenize_resp, mock_completion_resp]

        config = {
            "api_settings": {
                "llama_cpp": {
                    "base_url": "http://localhost:8080",
                    "disable_reasoning": False,
                },
            },
            "debug_options": {"verbose_debug": False},
        }

        result = _call_llama_cpp_api_internal("Generate", config)
        self.assertEqual(result, "Chapter text.")

    @patch("requests.post")
    def test_llama_cpp_no_model_custom_config_omits_overrides(self, mock_post):
        mock_tokenize_resp = MagicMock()
        mock_tokenize_resp.status_code = 200
        mock_tokenize_resp.json.return_value = {"tokens": [1]}

        mock_completion_resp = MagicMock()
        mock_completion_resp.status_code = 200
        mock_completion_resp.json.return_value = {
            "content": "Server default model response",
            "stop": True,
        }
        mock_post.side_effect = [mock_tokenize_resp, mock_completion_resp]

        # Only base_url and connection params provided, all model configs omitted
        config = {
            "api_settings": {
                "llama_cpp": {
                    "base_url": "http://localhost:8080",
                    "max_retries": 1,
                    "retry_delay_seconds": 1,
                    "request_timeout_seconds": 30,
                },
            },
            "debug_options": {"verbose_debug": False},
        }

        result = _call_llama_cpp_api_internal("Generate text", config)
        self.assertEqual(result, "Server default model response")

        completion_call = mock_post.call_args_list[1]
        _, kwargs = completion_call
        payload = kwargs["json"]

        self.assertNotIn("model", payload)
        self.assertNotIn("temperature", payload)
        self.assertNotIn("n_ctx", payload)
        self.assertNotIn("reasoning_budget", payload)
        self.assertNotIn("chat_template_kwargs", payload)

    @patch("requests.post")
    def test_llama_cpp_model_custom_config_included_when_set(self, mock_post):
        mock_tokenize_resp = MagicMock()
        mock_tokenize_resp.status_code = 200
        mock_tokenize_resp.json.return_value = {"tokens": [1]}

        mock_completion_resp = MagicMock()
        mock_completion_resp.status_code = 200
        mock_completion_resp.json.return_value = {
            "content": "Custom model response",
            "stop": True,
        }
        mock_post.side_effect = [mock_tokenize_resp, mock_completion_resp]

        config = {
            "api_settings": {
                "llama_cpp": {
                    "base_url": "http://localhost:8080",
                    "model": "my-custom-model",
                    "temperature": 0.5,
                    "context_window_size": 8192,
                    "llm_options": {
                        "n_predict": 512,
                    },
                },
            },
            "debug_options": {"verbose_debug": False},
        }

        result = _call_llama_cpp_api_internal("Generate text", config)
        self.assertEqual(result, "Custom model response")

        completion_call = mock_post.call_args_list[1]
        _, kwargs = completion_call
        payload = kwargs["json"]

        self.assertEqual(payload.get("model"), "my-custom-model")
        self.assertEqual(payload.get("temperature"), 0.5)
        self.assertEqual(payload.get("n_ctx"), 8192)
        self.assertEqual(payload.get("n_predict"), 512)


if __name__ == "__main__":
    unittest.main()

