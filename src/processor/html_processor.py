"""
Markdown to HTML processor module.
Receives raw LLM markdown output and processes it into clean HTML.
"""

import hashlib
import logging
import pathlib
import re
import time
from typing import Any

import markdown
from lxml import html

from src.utils import sanitize_filename


def clean_markdown_text(markdown_text: str) -> str:
    """Pre-processes markdown by stripping and cleaning up excessive newlines."""
    if not markdown_text:
        return ""
    cleaned = re.sub(r"\n{3,}", "\n\n", markdown_text.strip())
    if cleaned != markdown_text.strip():
        logging.debug("Cleaned excessive newlines from Markdown content.")
    return cleaned


def save_intermediate_html(
    html_content: str,
    cleaned_markdown: str,
    config: dict[str, Any] | None,
    context_label: str | None = None,
) -> pathlib.Path | None:
    """Saves intermediate HTML for debugging if enabled in configuration."""
    if not config or not config.get("debug_options", {}).get(
        "save_intermediate_html", False
    ):
        return None

    debug_html_dir = (
        pathlib.Path(config.get("cache_dir", "output/api_cache")) / "debug_html"
    )
    debug_html_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    content_hash = hashlib.sha1(cleaned_markdown.encode("utf-8")).hexdigest()[:8]
    if context_label:
        context_hint = sanitize_filename(context_label, max_length=80)
    else:
        context_hint = "unknown_context"

    html_filename = (
        debug_html_dir / f"md_to_html_{context_hint}_{timestamp}_{content_hash}.html"
    )
    try:
        with open(html_filename, "w", encoding="utf-8") as f_html:
            f_html.write("<!DOCTYPE html>\n<html>\n<head><meta charset='UTF-8'>")
            f_html.write(
                "<style>.arithmatex { border: 1px dotted blue; padding: 2px; }</style>"
            )
            f_html.write("</head>\n<body>\n")
            f_html.write(html_content)
            f_html.write("\n</body>\n</html>")
        logging.info(f"Saved intermediate HTML to: {html_filename}")
        return html_filename
    except Exception as e_save:
        logging.error(f"Could not save intermediate HTML to {html_filename}: {e_save}")
        return None


def convert_markdown_to_raw_html(
    markdown_text: str,
    config: dict[str, Any] | None = None,
    context_label: str | None = None,
) -> tuple[str, str]:
    """
    Converts markdown to raw HTML string with configured extensions.
    Returns a tuple of (cleaned_markdown, raw_html_content).
    """
    cleaned_markdown = clean_markdown_text(markdown_text)
    if not cleaned_markdown:
        return "", ""

    extensions = [
        "extra",  # Includes tables, footnotes, abbreviations, etc.
        "sane_lists",
        "fenced_code",
        "pymdownx.arithmatex",  # For LaTeX math $$...$$ and $...$
        "pymdownx.superfences",  # Improved fenced code blocks
        "pymdownx.details",
        "pymdownx.mark",
    ]
    extension_configs = {"pymdownx.arithmatex": {"generic": True}}
    md_converter = markdown.Markdown(
        extensions=extensions, extension_configs=extension_configs
    )
    html_content = md_converter.convert(cleaned_markdown)
    logging.debug(f"Generated HTML (first 500 chars): {html_content[:500]}...")

    save_intermediate_html(html_content, cleaned_markdown, config, context_label)
    return cleaned_markdown, html_content


def clean_html_tree(html_content: str) -> html.HtmlElement:
    """
    Parses HTML string into an lxml HtmlElement tree rooted at <div>,
    and removes empty or <br>-only <p> tags while preserving tails.
    """
    parser = html.HTMLParser(encoding="utf-8")
    html_wrapper_str = f"<div>{html_content}</div>"
    try:
        html_tree_root = html.fromstring(
            html_wrapper_str.encode("utf-8"), parser=parser
        )
    except UnicodeDecodeError:
        logging.error(
            "UTF-8 decoding failed for HTML string, trying 'latin-1' fallback."
        )
        html_tree_root = html.fromstring(
            html_wrapper_str.encode("latin-1"), parser=parser
        )

    paragraphs_to_remove = []
    for p_tag in html_tree_root.xpath(".//p"):
        text_content = p_tag.text_content().strip()
        children = p_tag.getchildren()

        is_completely_empty = not text_content and not children

        only_br_children = False
        if not text_content and children:
            all_children_are_br = True
            for child in children:
                if child.tag != "br":
                    all_children_are_br = False
                    break
            only_br_children = all_children_are_br

        if is_completely_empty or only_br_children:
            logging.debug(
                f"Found <p> tag {'empty' if is_completely_empty else 'containing only <br> tags'}. "
                f"Scheduling for removal. HTML snippet: "
                f"{html.tostring(p_tag, encoding='unicode', pretty_print=False)[:100]}"
            )
            paragraphs_to_remove.append(p_tag)

    if paragraphs_to_remove:
        logging.info(
            f"Removing {len(paragraphs_to_remove)} empty or <br>-only <p> tags."
        )
        for p_tag in paragraphs_to_remove:
            parent = p_tag.getparent()
            if parent is not None:
                if p_tag.tail and p_tag.tail.strip():
                    previous_sibling = p_tag.getprevious()
                    if previous_sibling is not None:
                        previous_sibling.tail = (
                            previous_sibling.tail or ""
                        ) + p_tag.tail
                    else:
                        parent.text = (parent.text or "") + p_tag.tail
                parent.remove(p_tag)

    return html_tree_root


def process_markdown_to_html(
    markdown_text: str,
    config: dict[str, Any] | None = None,
    context_label: str | None = None,
) -> tuple[str, str, html.HtmlElement]:
    """
    Complete processing pipeline:
    1. Pre-process and clean markdown text.
    2. Convert markdown to HTML with pymdown extensions.
    3. Save intermediate HTML if configured.
    4. Parse and clean HTML tree.

    Returns:
        Tuple of (cleaned_markdown, html_content, html_tree_root).
    """
    cleaned_markdown, html_content = convert_markdown_to_raw_html(
        markdown_text, config=config, context_label=context_label
    )
    html_tree_root = clean_html_tree(html_content)
    return cleaned_markdown, html_content, html_tree_root


def markdown_to_html(
    markdown_text: str,
    config: dict[str, Any] | None = None,
    context_label: str | None = None,
) -> str:
    """Convenience helper returning serialized processed HTML."""
    _, _, html_tree_root = process_markdown_to_html(
        markdown_text, config=config, context_label=context_label
    )
    return str(html.tostring(html_tree_root, encoding="unicode"))
