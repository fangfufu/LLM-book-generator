import hashlib
import json
import logging
import os
import pathlib
import re


def _check_for_repetition(
    current_text,
    repetition_check_config,
    full_response_text_parts_for_check=None,
):
    """
    Checks for repeated sentences or paragraphs in the generated text.

    Args:
        current_text (str): The current chunk of text from the LLM stream.
        repetition_check_config (dict): Configuration for repetition checks.
        full_response_text_parts_for_check (list, optional): A list containing
            the history of text parts for more robust checking. Defaults to None.

    Returns:
        bool: True if a repetition is detected, False otherwise.
    """
    min_sentence_length = repetition_check_config.get(
        "min_sentence_length_for_repetition_check", 10
    )

    # Check 0: Hacky way to remove leaked tags
    if "start_of_turn" in current_text or "end_of_turn" in current_text:
        return True

    # Check 1: Exact repetition of the current chunk in the history
    # max_history_size = repetition_check_config.get("max_history_size", 5)
    # if (
    #     full_response_text_parts_for_check is not None
    #     and len(current_text) > min_sentence_length
    # ):
    #     # Count occurrences of the current chunk in the recent history
    #     recent_history = full_response_text_parts_for_check[-max_history_size:]
    #     if recent_history.count(current_text) > 1:
    #         logging.warning(
    #             f"Repetition detected (exact chunk): Found chunk '{current_text}' multiple times in recent history."
    #         )
    #         return True
            
    # Check 2: Repetition of sentences using the full text
    # This check is more expensive, so it's done on the accumulated text
    full_text = "".join(full_response_text_parts_for_check or [])
    if len(full_text) < min_sentence_length * 2:  # Not enough text to check
        return False

    # Simple sentence tokenization
    sentences = [
        s.strip() for s in full_text.replace("!", ".").replace("?", ".").split(".") if s
    ]
    long_sentences = [s for s in sentences if len(s) > min_sentence_length]

    if len(long_sentences) > len(set(long_sentences)):
        # Find the repeated sentence for logging purposes
        seen = set()
        for sentence in long_sentences:
            if sentence in seen:
                logging.warning(
                    f"Repetition detected (sentence): Sentence '{sentence}' is repeated."
                )
                return True
            seen.add(sentence)

    return False
import sys
import time

import requests
from dotenv import load_dotenv
from transformers import AutoTokenizer

from book_generator.utils import sanitize_filename
from book_generator.constants import REPETITION_DETECTED


def setup_environment():
    """Loads environment variables from .env file."""
    load_dotenv()
    logging.info("Environment variables loaded.")


# --- Caching Mechanism ---
def get_cache_path(prompt_text, cache_dir, cache_prefix=None):
    """Generates the cache file path, optionally prepending a prefix."""
    prompt_hash = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
    pathlib.Path(cache_dir).mkdir(parents=True, exist_ok=True)

    filename_base = prompt_hash
    if cache_prefix:
        # Sanitize the prefix to make it filename-safe and limit length
        # Use a shorter length limit for prefixes to avoid overly long filenames
        sanitized_prefix = sanitize_filename(cache_prefix, max_length=50)
        if sanitized_prefix:  # Ensure sanitization didn't result in an empty string
            filename_base = f"{sanitized_prefix}_{prompt_hash}"
            logging.debug(f"Using cache prefix: '{sanitized_prefix}'")
        else:
            logging.error(
                f"Cache prefix '{cache_prefix}' sanitized to empty string. Using hash only."
            )

    return pathlib.Path(cache_dir) / f"{filename_base}.json"


def load_from_cache(prompt_text, cache_dir, cache_prefix=None):
    """Loads response from cache if available."""
    cache_file = get_cache_path(prompt_text, cache_dir, cache_prefix)
    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cached_data = json.load(f)
            if "prompt" in cached_data and "response" in cached_data:
                # Log the actual filename for clarity
                logging.info(f"Cache hit for file: {cache_file.name}")
                return cached_data["response"]
            else:
                logging.error(f"Invalid cache file format: {cache_file}. Ignoring.")
                return None
        except Exception as e:
            logging.error(f"Error reading cache file {cache_file}: {e}")
            return None
    logging.debug(f"Cache miss for file: {cache_file.name}")
    return None


def save_to_cache(prompt_text, response_text, cache_dir, cache_prefix=None):
    """Saves the API response to the cache."""
    cache_file = get_cache_path(prompt_text, cache_dir, cache_prefix)
    try:
        cache_data = {"prompt": prompt_text, "response": response_text}
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=4)
        logging.info(f"Response saved to cache: {cache_file}")
    except Exception as e:
        logging.error(f"Error saving response to cache file {cache_file}: {e}")


# --- LLM API Interaction ---

def _clean_response_text(text, strip_thinking=True):
    """
    Cleans generated text by stripping leaked special tokens and optionally reasoning/thinking blocks.

    Args:
        text (str): The raw text from the LLM.
        strip_thinking (bool): Whether to strip <think>...</think> blocks.

    Returns:
        str: Cleaned text.
    """
    if not text:
        return text
    cleaned = text
    if strip_thinking:
        cleaned = re.sub(r"(?is)<think>.*?</think>", "", cleaned)
        cleaned = re.sub(r"(?is)<\|channel>thought.*?<channel\|>", "", cleaned)
        cleaned = re.sub(r"(?is)<\|channel\|>thought.*?<\|channel\|>", "", cleaned)
        cleaned = re.sub(r"(?i)</?think>", "", cleaned)
    # Strip any leaked channel tokens or chat template delimiters
    cleaned = re.sub(r"<\|?channel\|?>[a-zA-Z0-9_-]*", "", cleaned)
    cleaned = re.sub(r"<channel\|?>", "", cleaned)
    cleaned = (
        cleaned.replace("</end_of_turn>", "")
        .replace("<end_of_turn>", "")
        .replace("</start_of_turn>", "")
        .replace("<start_of_turn>", "")
        .replace("<bos>", "")
        .replace("<eos>", "")
        .strip()
    )
    return cleaned


class StreamThoughtParser:
    """
    Parses a stream of text chunks to separate inline <think>...</think>
    or channel-based reasoning blocks from response text in real-time.
    """

    def __init__(self):
        self.in_think_block = False
        self.pending_tag = ""
        self._current_end_tag = None

    def process_chunk(self, chunk_text):
        """
        Takes raw text and yields tuples of (is_thinking: bool, text: str).
        """
        if not chunk_text:
            return []

        text = self.pending_tag + chunk_text
        self.pending_tag = ""
        results = []

        while text:
            if not self.in_think_block:
                m = re.search(r"(<think>|<\|channel\|?>thought)", text, re.IGNORECASE)
                if m:
                    idx = m.start()
                    before = text[:idx]
                    if before:
                        results.append((False, before))
                    self.in_think_block = True
                    matched_tag = m.group(1).lower()
                    if "<think>" in matched_tag:
                        self._current_end_tag = "</think>"
                    else:
                        self._current_end_tag = "<channel|>"
                    text = text[m.end():]
                else:
                    match = re.search(r"(<[a-zA-Z0-9_|/-]*)$", text)
                    if match and len(match.group(1)) <= 25:
                        before = text[:match.start()]
                        self.pending_tag = text[match.start():]
                        if before:
                            results.append((False, before))
                        break
                    else:
                        results.append((False, text))
                        break
            else:
                end_pattern = (
                    r"(</think>|<\|?channel\|?>)"
                    if not self._current_end_tag
                    else (re.escape(self._current_end_tag) + r"|</think>|<\|?channel\|?>")
                )
                m = re.search(end_pattern, text, re.IGNORECASE)
                if m:
                    idx = m.start()
                    before = text[:idx]
                    if before:
                        results.append((True, before))
                    self.in_think_block = False
                    self._current_end_tag = None
                    text = text[m.end():]
                else:
                    match = re.search(r"(<[a-zA-Z0-9_|/-]*)$", text)
                    if match and len(match.group(1)) <= 25:
                        before = text[:match.start()]
                        self.pending_tag = text[match.start():]
                        if before:
                            results.append((True, before))
                        break
                    else:
                        results.append((True, text))
                        break

        return results

    def finalize(self):
        """Flushes any pending partial tag text."""
        if self.pending_tag:
            res = [(self.in_think_block, self.pending_tag)]
            self.pending_tag = ""
            return res
        return []


def _call_llama_cpp_api_internal(prompt, config, cache_prefix=None):
    """
    Internal function to call the llama.cpp API.
    Assumes caching is handled by the caller.

    Args:
        prompt (str): The prompt to send to the API.
        config (dict): The application configuration.
        cache_prefix (str, optional): Contextual prefix, caching is external.

    Returns:
        str or None: The API response text, or None if an error occurred.
    """
    api_settings_conf = config.get("api_settings", {})
    llama_cpp_config = api_settings_conf.get(
        "llama_cpp", api_settings_conf.get("llama.cpp", {})
    )

    default_max_retries = api_settings_conf.get("default_max_retries", 3)
    default_retry_delay = api_settings_conf.get("default_retry_delay_seconds", 5)

    base_url = llama_cpp_config.get("base_url", "http://localhost:8080")
    model_name = llama_cpp_config.get("model", None)
    model_display = model_name or "default"
    tokenizer_model_name = llama_cpp_config.get("tokenizer_model", None)

    max_retries = int(llama_cpp_config.get("max_retries", default_max_retries))
    retry_delay = int(llama_cpp_config.get("retry_delay_seconds", default_retry_delay))
    request_timeout = int(
        llama_cpp_config.get("request_timeout_seconds", 120)
    )

    endpoint = llama_cpp_config.get("endpoint")
    if endpoint:
        api_url = f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"
    elif base_url.rstrip("/").endswith("/chat/completions") or base_url.rstrip("/").endswith("/completion"):
        api_url = base_url
    elif base_url.rstrip("/").endswith("/v1"):
        api_url = f"{base_url.rstrip('/')}/chat/completions"
    else:
        api_url = f"{base_url.rstrip('/')}/v1/chat/completions"

    is_chat_endpoint = "chat/completions" in api_url

    verbose_debug = config.get("debug_options", {}).get("verbose_debug", False)
    stream_llama_cpp = verbose_debug

    if is_chat_endpoint:
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "stream": stream_llama_cpp,
        }
    else:
        payload = {
            "prompt": prompt,
            "stream": stream_llama_cpp,
        }
    if model_name:
        payload["model"] = model_name

    # Get all llm_options from the config
    llm_options = llama_cpp_config.get("llm_options", {})
    if llm_options:
        logging.info(f"Applying llama.cpp llm_options: {llm_options}")
        for key, value in llm_options.items():
            if value is not None:
                payload[key] = value
                logging.debug(f"Set llama.cpp option '{key}': {value}")

    # Check for standalone temperature if not in llm_options
    if "temperature" not in payload and "temperature" in llama_cpp_config:
        payload["temperature"] = float(llama_cpp_config["temperature"])

    # Handle standalone context_window_size / n_ctx
    if "n_ctx" not in payload and "context_window_size" in llama_cpp_config:
        context_window_size = llama_cpp_config.get("context_window_size")
        if context_window_size:
            try:
                payload["n_ctx"] = int(context_window_size)
            except (ValueError, TypeError):
                logging.error(
                    f"Invalid 'context_window_size' value: {context_window_size}. It must be an integer. Ignoring."
                )

    disable_reasoning = bool(llama_cpp_config.get("disable_reasoning", False))
    if disable_reasoning:
        if "reasoning_budget" not in payload:
            payload["reasoning_budget"] = 0
        if "chat_template_kwargs" not in payload:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        elif (
            isinstance(payload["chat_template_kwargs"], dict)
            and "enable_thinking" not in payload["chat_template_kwargs"]
        ):
            payload["chat_template_kwargs"]["enable_thinking"] = False

    if verbose_debug:
        logging.info(f"llama.cpp API Prompt for model '{model_display}':\n{prompt}")

    # Attempt token counting: first via llama.cpp /tokenize endpoint, then fall back to AutoTokenizer
    token_counted = False
    try:
        tokenize_url = f"{base_url.rstrip('/')}/tokenize"
        tok_payload = {"content": prompt}
        if model_name:
            tok_payload["model"] = model_name
        tok_res = requests.post(
            tokenize_url,
            headers={"Content-Type": "application/json"},
            json=tok_payload,
            timeout=5,
        )
        if tok_res.status_code == 200:
            tok_data = tok_res.json()
            tokens = tok_data.get("tokens", [])
            logging.info(
                f"llama.cpp server-side token count for prompt (model: '{model_display}'): {len(tokens)} tokens."
            )
            token_counted = True
    except Exception as e_tok:
        logging.debug(f"llama.cpp server-side tokenize request failed: {e_tok}")

    if not token_counted and tokenizer_model_name:
        try:
            logging.debug(
                f"Loading tokenizer: {tokenizer_model_name} for llama.cpp prompt token count."
            )
            hugging_face_token = os.getenv("HF_TOKEN")
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_model_name, token=hugging_face_token
            )
            token_ids = tokenizer.encode(prompt)
            num_tokens = len(token_ids)
            logging.info(
                f"llama.cpp client-side token count for prompt (tokenizer: '{tokenizer_model_name}', model: '{model_display}'): {num_tokens} tokens."
            )
        except Exception as e_token_llamacpp:
            logging.error(
                f"Could not count tokens for llama.cpp prompt using tokenizer '{tokenizer_model_name}' (model: '{model_display}'): {e_token_llamacpp}",
                exc_info=True,
            )

    for attempt in range(max_retries):
        try:
            response = requests.post(
                api_url,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=request_timeout,
                stream=stream_llama_cpp,
            )
            response.raise_for_status()

            if stream_llama_cpp:
                logging.info(f"Streaming llama.cpp response for model '{model_display}':")
                full_response_text_parts = []
                full_reasoning_parts = []
                thought_parser = StreamThoughtParser()
                stream_section = None
                repetition_check_config = config.get("repetition_check", {})
                print(f"\n--- llama.cpp Stream ({model_display}) ---")
                for line in response.iter_lines():
                    if line:
                        decoded_line = (
                            line.decode("utf-8") if isinstance(line, bytes) else line
                        )
                        if decoded_line.startswith("data: "):
                            json_str = decoded_line[6:].strip()
                        elif decoded_line.startswith("data:"):
                            json_str = decoded_line[5:].strip()
                        else:
                            json_str = decoded_line.strip()

                        if json_str == "[DONE]":
                            break

                        try:
                            chunk = json.loads(json_str)
                            if "error" in chunk:
                                logging.error(
                                    f"llama.cpp API error during stream for model '{model_display}': {chunk['error']}"
                                )
                                print(f"\n--- End llama.cpp Stream (Error) ---")
                                return None

                            # 1. Process separate reasoning fields (OpenAI reasoning_content or llama.cpp thinking)
                            reasoning_part = ""
                            if "thinking" in chunk:
                                reasoning_part = chunk.get("thinking", "")
                            elif "choices" in chunk and chunk["choices"]:
                                choice = chunk["choices"][0]
                                if "delta" in choice:
                                    reasoning_part = (
                                        choice["delta"].get("reasoning_content")
                                        or choice["delta"].get("reasoning")
                                        or ""
                                    )

                            if reasoning_part:
                                full_reasoning_parts.append(reasoning_part)
                                if stream_section != "thinking":
                                    print("\n[Thinking] ", end="", flush=True)
                                    stream_section = "thinking"
                                print(reasoning_part, end="", flush=True)

                            # 2. Process content chunks (which may contain inline <think> tags)
                            raw_content_part = ""
                            if "content" in chunk:
                                raw_content_part = chunk.get("content", "")
                            elif "choices" in chunk and chunk["choices"]:
                                choice = chunk["choices"][0]
                                if "text" in choice:
                                    raw_content_part = choice.get("text", "")
                                elif "delta" in choice and "content" in choice["delta"]:
                                    raw_content_part = choice["delta"].get("content") or ""
                            elif "response" in chunk:
                                raw_content_part = chunk.get("response", "")

                            if raw_content_part:
                                segments = thought_parser.process_chunk(raw_content_part)
                                for is_thinking, segment in segments:
                                    if is_thinking:
                                        full_reasoning_parts.append(segment)
                                        if stream_section != "thinking":
                                            print("\n[Thinking] ", end="", flush=True)
                                            stream_section = "thinking"
                                        print(segment, end="", flush=True)
                                    else:
                                        full_response_text_parts.append(segment)
                                        if stream_section != "response":
                                            print("\n[Response] ", end="", flush=True)
                                            stream_section = "response"
                                        print(segment, end="", flush=True)

                                        # Only check for repetition on response text, not internal thoughts
                                        if repetition_check_config.get("enabled", False):
                                            if _check_for_repetition(
                                                segment,
                                                repetition_check_config,
                                                full_response_text_parts,
                                            ):
                                                logging.warning(
                                                    "Repetition detected. Terminating and retrying."
                                                )
                                                print(
                                                    "\n--- End llama.cpp Stream (Repetition Detected) ---"
                                                )
                                                return REPETITION_DETECTED

                            is_done = chunk.get("stop", False) or (
                                "choices" in chunk
                                and chunk["choices"]
                                and chunk["choices"][0].get("finish_reason") is not None
                            )
                            if is_done:
                                for is_thinking, segment in thought_parser.finalize():
                                    if is_thinking:
                                        full_reasoning_parts.append(segment)
                                    else:
                                        full_response_text_parts.append(segment)

                                print(f"\n--- End llama.cpp Stream (Done) ---")
                                logging.info(
                                    f"llama.cpp API stream completed for model '{model_display}'."
                                )
                                full_reasoning_text = "".join(full_reasoning_parts).strip()
                                if full_reasoning_text:
                                    logging.info(
                                        f"llama.cpp API reasoning process captured ({len(full_reasoning_text)} chars)."
                                    )
                                    logging.debug(f"Reasoning text:\n{full_reasoning_text}")

                                final_text = "".join(full_response_text_parts)
                                return _clean_response_text(final_text, strip_thinking=True)
                        except json.JSONDecodeError:
                            logging.error(
                                f"Error decoding JSON chunk from llama.cpp stream: {decoded_line}"
                            )
                            print(f"\n--- End llama.cpp Stream (JSON Error) ---")
                            return None

                for is_thinking, segment in thought_parser.finalize():
                    if is_thinking:
                        full_reasoning_parts.append(segment)
                    else:
                        full_response_text_parts.append(segment)

                print(f"\n--- End llama.cpp Stream (Done) ---")
                full_reasoning_text = "".join(full_reasoning_parts).strip()
                if full_reasoning_text:
                    logging.info(
                        f"llama.cpp API reasoning process captured ({len(full_reasoning_text)} chars)."
                    )
                    logging.debug(f"Reasoning text:\n{full_reasoning_text}")

                final_text = "".join(full_response_text_parts)
                return (
                    _clean_response_text(final_text, strip_thinking=True)
                    if full_response_text_parts
                    else None
                )
            else:  # Not streaming
                response_data = response.json()
                if "error" in response_data:
                    logging.error(
                        f"llama.cpp API error for model '{model_display}': {response_data['error']}"
                    )
                    return None

                response_text = None
                if "content" in response_data:
                    response_text = response_data["content"]
                elif "choices" in response_data and response_data["choices"]:
                    choice = response_data["choices"][0]
                    if "text" in choice:
                        response_text = choice.get("text")
                    elif "message" in choice and "content" in choice["message"]:
                        response_text = choice["message"].get("content")
                elif "response" in response_data:
                    response_text = response_data["response"]

                reasoning_text = None
                if "choices" in response_data and response_data["choices"]:
                    choice = response_data["choices"][0]
                    if "message" in choice:
                        reasoning_text = (
                            choice["message"].get("reasoning_content")
                            or choice["message"].get("reasoning")
                        )
                elif "thinking" in response_data:
                    reasoning_text = response_data.get("thinking")

                if response_text:
                    inline_thoughts = re.findall(
                        r"(?is)<think>(.*?)</think>", response_text
                    )
                    if inline_thoughts:
                        combined_inline = "\n".join(
                            t.strip() for t in inline_thoughts if t.strip()
                        )
                        if reasoning_text:
                            reasoning_text = f"{reasoning_text}\n{combined_inline}"
                        else:
                            reasoning_text = combined_inline

                if reasoning_text:
                    logging.info(
                        f"llama.cpp API reasoning process captured for model '{model_display}' ({len(reasoning_text)} chars)."
                    )
                    logging.debug(f"Reasoning text:\n{reasoning_text}")

                if response_text is not None:
                    logging.info(
                        f"llama.cpp API call successful for model '{model_display}'."
                    )
                    return _clean_response_text(response_text, strip_thinking=True)
                else:
                    logging.error(
                        f"llama.cpp API response for model '{model_display}' did not contain expected content key. Attempt {attempt + 1}/{max_retries}. Data: {response_data}"
                    )

        except requests.exceptions.HTTPError as e:
            logging.error(
                f"llama.cpp API call (model '{model_display}') attempt {attempt + 1} failed with HTTPError: {e}. Status: {e.response.status_code if e.response is not None else 'unknown'}",
                exc_info=True,
            )
        except requests.exceptions.RequestException as e:
            logging.error(
                f"llama.cpp API call (model '{model_display}') attempt {attempt + 1} failed: {e}",
                exc_info=True,
            )

        if attempt < max_retries - 1:
            logging.info(f"Retrying llama.cpp API call in {retry_delay} seconds...")
            time.sleep(retry_delay)
        else:
            logging.error(
                f"llama.cpp API call for model '{model_display}' failed after {max_retries} attempts."
            )
            if api_settings_conf.get("terminate_on_api_failure", False):
                logging.critical(
                    "Terminating program due to repeated API failures."
                )
                sys.exit(1)
            return None
    return None


def call_llm_api(prompt, config, cache_prefix=None):
    """
    Calls the configured LLM API (llama.cpp), using caching and handling retries.
    Includes special handling for repetition detection.
    """
    cache_dir = config.get("cache_dir", "api_cache")
    cached_response = load_from_cache(prompt, cache_dir, cache_prefix)
    if cached_response is not None:
        return cached_response

    api_settings = config.get("api_settings", {})
    api_provider = api_settings.get("provider", "llama_cpp")
    logging.info(
        f"Calling {api_provider.upper()} API... (Cache Prefix: {cache_prefix or 'None'})"
    )

    repetition_check_config = config.get("repetition_check", {})
    max_retries_repetition = repetition_check_config.get(
        "max_retries_on_repetition", 2
    )
    repetition_retry_delay = repetition_check_config.get(
        "repetition_retry_delay_seconds", 5
    )
    response_text = None

    for attempt in range(max_retries_repetition + 1):
        if api_provider in ("llama_cpp", "llama.cpp"):
            response_text = _call_llama_cpp_api_internal(prompt, config, cache_prefix)
        else:
            logging.error(f"Unsupported API provider: {api_provider}")
            return None

        if response_text == REPETITION_DETECTED:
            logging.warning(
                f"Repetition detected in response. Attempt {attempt + 1}/{max_retries_repetition + 1}."
            )
            if attempt < max_retries_repetition:
                logging.info(
                    f"Retrying after {repetition_retry_delay} seconds due to repetition..."
                )
                time.sleep(repetition_retry_delay)
                # Invalidate cache for this specific prompt before retrying
                cache_file = get_cache_path(prompt, cache_dir, cache_prefix)
                if cache_file.exists():
                    os.remove(cache_file)
            else:
                logging.error(
                    "Max retries for repetition reached. Returning None."
                )
                return None
        else:
            # If not a repetition or another error, break the loop
            break

    if response_text is not None and response_text != REPETITION_DETECTED:
        save_to_cache(prompt, response_text, cache_dir, cache_prefix)

    return response_text
