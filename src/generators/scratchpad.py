"""
This module manages the scratchpad system for book generation.
It maintains character and location notes in dedicated markdown files,
short summaries, and chapter drafts, enabling state tracking and resumption
upon interruption.
"""

import hashlib
import logging
import pathlib
import re

from src.llm import call_llm_api
from src.utils import sanitize_filename


def get_scratchpad_dir(config: dict) -> pathlib.Path:
    """
    Computes and creates the topic- and model-specific scratchpad directory.
    Uses the same subfolder naming scheme as api_cache (<sanitized_model_name>/<topic_dir_name>).
    """
    if "scratchpad_dir_resolved" in config and isinstance(
        config["scratchpad_dir_resolved"], pathlib.Path
    ):
        return config["scratchpad_dir_resolved"]

    api_settings = config.get("api_settings", {})
    api_provider = api_settings.get("provider", "llama_cpp")
    generation_params = config.get("generation_params", {})
    main_topic = generation_params.get("main_topic", "untitled_topic")

    sanitized_topic = sanitize_filename(main_topic, 64)
    topic_hash = hashlib.sha1(main_topic.encode("utf-8")).hexdigest()[:8]
    topic_dir_name = f"{sanitized_topic}_{topic_hash}"

    if api_provider in ("llama_cpp", "llama.cpp"):
        llama_cpp_conf = api_settings.get(
            "llama_cpp", api_settings.get("llama.cpp", {})
        )
        model_name = llama_cpp_conf.get("model") or "llama_cpp_default_model"
    else:
        model_name = "unknown_api_provider_model"
    sanitized_model_name = sanitize_filename(model_name, 30)

    base_scratchpad_dir = (
        api_settings.get("base_scratchpad_dir")
        or config.get("base_scratchpad_dir")
        or config.get("scratchpad_dir")
        or "scratchpad"
    )

    scratchpad_dir = (
        pathlib.Path(base_scratchpad_dir) / sanitized_model_name / topic_dir_name
    )
    scratchpad_dir.mkdir(parents=True, exist_ok=True)
    config["scratchpad_dir"] = str(scratchpad_dir)
    config["scratchpad_dir_resolved"] = scratchpad_dir
    return scratchpad_dir


def get_character_dir(scratchpad_dir: pathlib.Path) -> pathlib.Path:
    """Returns the character subfolder inside scratchpad."""
    char_dir = scratchpad_dir / "character"
    char_dir.mkdir(parents=True, exist_ok=True)
    return char_dir


def get_location_dir(scratchpad_dir: pathlib.Path) -> pathlib.Path:
    """Returns the location subfolder inside scratchpad."""
    loc_dir = scratchpad_dir / "location"
    loc_dir.mkdir(parents=True, exist_ok=True)
    return loc_dir


def get_chapters_dir(scratchpad_dir: pathlib.Path) -> pathlib.Path:
    """Returns the chapters subfolder inside scratchpad."""
    chap_dir = scratchpad_dir / "chapters"
    chap_dir.mkdir(parents=True, exist_ok=True)
    return chap_dir


def has_initialized_scratchpad(scratchpad_dir: pathlib.Path) -> bool:
    """Checks if character and location summaries exist and are populated."""
    char_summary_file = get_character_dir(scratchpad_dir) / "summary.md"
    loc_summary_file = get_location_dir(scratchpad_dir) / "summary.md"

    if char_summary_file.exists() and loc_summary_file.exists():
        char_text = char_summary_file.read_text(encoding="utf-8").strip()
        loc_text = loc_summary_file.read_text(encoding="utf-8").strip()
        return bool(char_text and loc_text)
    return False


def init_scratchpad(
    scratchpad_dir: pathlib.Path,
    character_list: list[dict[str, str]] | None,
    location_list: list[dict[str, str]] | None,
):
    """
    Initializes the scratchpad folder structure with separate markdown files
    for each character and location, plus short summary.md files.
    """
    char_dir = get_character_dir(scratchpad_dir)
    loc_dir = get_location_dir(scratchpad_dir)
    get_chapters_dir(scratchpad_dir)

    # Initialize characters
    char_summary_lines = ["# Characters Summary\n"]
    if character_list:
        for char in character_list:
            name = char.get("name", "").strip()
            desc = char.get("description", "").strip()
            if not name:
                continue

            char_summary_lines.append(f"* **{name}**: {desc}")
            char_file = char_dir / f"{sanitize_filename(name, max_length=50)}.md"
            if not char_file.exists():
                char_content = f"""# {name}

## Short Summary
{desc}

## Character Notes & Development
- **Role & Significance:** {desc}
- **Status:** Active
- **Key Relationships & Traits:** Consistent with story background.
- **Story Developments:**
  - Initial introduction.
"""
                char_file.write_text(char_content, encoding="utf-8")
                logging.info(f"Created scratchpad character note: {char_file.name}")

    char_summary_path = char_dir / "summary.md"
    if (
        not char_summary_path.exists()
        or not char_summary_path.read_text(encoding="utf-8").strip()
    ):
        char_summary_path.write_text(
            "\n".join(char_summary_lines) + "\n", encoding="utf-8"
        )
        logging.info(
            f"Initialized character summary in scratchpad: {char_summary_path}"
        )

    # Initialize locations
    loc_summary_lines = ["# Locations Summary\n"]
    if location_list:
        for loc in location_list:
            name = loc.get("name", "").strip()
            desc = loc.get("description", "").strip()
            if not name:
                continue

            loc_summary_lines.append(f"* **{name}**: {desc}")
            loc_file = loc_dir / f"{sanitize_filename(name, max_length=50)}.md"
            if not loc_file.exists():
                loc_content = f"""# {name}

## Short Summary
{desc}

## Location Details & Atmosphere
- **Significance:** {desc}
- **Atmosphere & Features:** Key physical setting in the narrative.
- **Story Developments:**
  - Initial introduction.
"""
                loc_file.write_text(loc_content, encoding="utf-8")
                logging.info(f"Created scratchpad location note: {loc_file.name}")

    loc_summary_path = loc_dir / "summary.md"
    if (
        not loc_summary_path.exists()
        or not loc_summary_path.read_text(encoding="utf-8").strip()
    ):
        loc_summary_path.write_text(
            "\n".join(loc_summary_lines) + "\n", encoding="utf-8"
        )
        logging.info(f"Initialized location summary in scratchpad: {loc_summary_path}")


def load_character_summary_text(scratchpad_dir: pathlib.Path) -> str:
    """Loads the character summary markdown text."""
    summary_file = get_character_dir(scratchpad_dir) / "summary.md"
    if summary_file.exists():
        return summary_file.read_text(encoding="utf-8").strip()
    return ""


def load_location_summary_text(scratchpad_dir: pathlib.Path) -> str:
    """Loads the location summary markdown text."""
    summary_file = get_location_dir(scratchpad_dir) / "summary.md"
    if summary_file.exists():
        return summary_file.read_text(encoding="utf-8").strip()
    return ""


def load_characters_from_scratchpad(
    scratchpad_dir: pathlib.Path,
) -> list[dict[str, str]]:
    """Loads character list from character/summary.md or character markdown files."""
    summary_text = load_character_summary_text(scratchpad_dir)
    characters = []
    if summary_text:
        for line in summary_text.splitlines():
            line = line.strip()
            match = re.match(r"^[\*\-]\s*\*\*(.*?)\*\*:\s*(.*)", line)
            if match:
                characters.append(
                    {
                        "name": match.group(1).strip(),
                        "description": match.group(2).strip(),
                    }
                )

    if not characters:
        char_dir = get_character_dir(scratchpad_dir)
        for f in sorted(char_dir.glob("*.md")):
            if f.name == "summary.md":
                continue
            content = f.read_text(encoding="utf-8")
            name_match = re.search(r"^#\s*(.+)$", content, re.MULTILINE)
            summary_match = re.search(r"##\s*Short Summary\s*\n+([^\n#]+)", content)
            name = name_match.group(1).strip() if name_match else f.stem
            desc = summary_match.group(1).strip() if summary_match else ""
            characters.append({"name": name, "description": desc})

    return characters


def load_locations_from_scratchpad(
    scratchpad_dir: pathlib.Path,
) -> list[dict[str, str]]:
    """Loads location list from location/summary.md or location markdown files."""
    summary_text = load_location_summary_text(scratchpad_dir)
    locations = []
    if summary_text:
        for line in summary_text.splitlines():
            line = line.strip()
            match = re.match(r"^[\*\-]\s*\*\*(.*?)\*\*:\s*(.*)", line)
            if match:
                locations.append(
                    {
                        "name": match.group(1).strip(),
                        "description": match.group(2).strip(),
                    }
                )

    if not locations:
        loc_dir = get_location_dir(scratchpad_dir)
        for f in sorted(loc_dir.glob("*.md")):
            if f.name == "summary.md":
                continue
            content = f.read_text(encoding="utf-8")
            name_match = re.search(r"^#\s*(.+)$", content, re.MULTILINE)
            summary_match = re.search(r"##\s*Short Summary\s*\n+([^\n#]+)", content)
            name = name_match.group(1).strip() if name_match else f.stem
            desc = summary_match.group(1).strip() if summary_match else ""
            locations.append({"name": name, "description": desc})

    return locations


def detect_relevant_entities(
    scratchpad_dir: pathlib.Path,
    chapter_title: str,
    chapter_summary: str,
    chapter_text: str,
) -> tuple[list[str], list[str]]:
    """
    Detects which characters and locations from the scratchpad are featured or referenced
    in the chapter summary or draft.
    """
    all_characters = load_characters_from_scratchpad(scratchpad_dir)
    all_locations = load_locations_from_scratchpad(scratchpad_dir)

    search_corpus = f"{chapter_title}\n{chapter_summary}\n{chapter_text}".lower()

    detected_chars = []
    for char in all_characters:
        name = char.get("name", "").strip()
        if not name:
            continue
        # Exact match of full name or name parts
        if re.search(r"\b" + re.escape(name.lower()) + r"\b", search_corpus):
            detected_chars.append(name)
        else:
            # Check individual name tokens if length > 3 (e.g., first or last name)
            tokens = [t for t in re.split(r"\s+", name) if len(t) > 3]
            if any(
                re.search(r"\b" + re.escape(t.lower()) + r"\b", search_corpus)
                for t in tokens
            ):
                detected_chars.append(name)

    detected_locs = []
    for loc in all_locations:
        name = loc.get("name", "").strip()
        if not name:
            continue
        if re.search(r"\b" + re.escape(name.lower()) + r"\b", search_corpus):
            detected_locs.append(name)
        else:
            tokens = [t for t in re.split(r"\s+", name) if len(t) > 3]
            if any(
                re.search(r"\b" + re.escape(t.lower()) + r"\b", search_corpus)
                for t in tokens
            ):
                detected_locs.append(name)

    # Fallback if detection found nothing
    if not detected_chars and all_characters:
        detected_chars = [c["name"] for c in all_characters[:4] if c.get("name")]
    if not detected_locs and all_locations:
        detected_locs = [loc["name"] for loc in all_locations[:3] if loc.get("name")]

    return detected_chars, detected_locs


def load_specific_character_notes(
    scratchpad_dir: pathlib.Path, character_names: list[str]
) -> str:
    """Loads and formats specific character markdown files for prompt injection."""
    char_dir = get_character_dir(scratchpad_dir)
    notes_blocks = []

    for name in character_names:
        safe_name = sanitize_filename(name, max_length=50)
        char_file = char_dir / f"{safe_name}.md"
        if char_file.exists():
            content = char_file.read_text(encoding="utf-8").strip()
            notes_blocks.append(content)
        else:
            # Fallback search across directory
            found = False
            for f in char_dir.glob("*.md"):
                if f.name == "summary.md":
                    continue
                if safe_name.lower() in f.stem.lower():
                    notes_blocks.append(f.read_text(encoding="utf-8").strip())
                    found = True
                    break
            if not found:
                notes_blocks.append(f"# {name}\nNo detailed notes recorded yet.")

    if not notes_blocks:
        return ""

    joined_notes = "\n\n---\n\n".join(notes_blocks)
    return f"""--- Specific Character Notes (from Scratchpad) ---
{joined_notes}
--- End Specific Character Notes ---"""


def load_specific_location_notes(
    scratchpad_dir: pathlib.Path, location_names: list[str]
) -> str:
    """Loads and formats specific location markdown files for prompt injection."""
    loc_dir = get_location_dir(scratchpad_dir)
    notes_blocks = []

    for name in location_names:
        safe_name = sanitize_filename(name, max_length=50)
        loc_file = loc_dir / f"{safe_name}.md"
        if loc_file.exists():
            content = loc_file.read_text(encoding="utf-8").strip()
            notes_blocks.append(content)
        else:
            found = False
            for f in loc_dir.glob("*.md"):
                if f.name == "summary.md":
                    continue
                if safe_name.lower() in f.stem.lower():
                    notes_blocks.append(f.read_text(encoding="utf-8").strip())
                    found = True
                    break
            if not found:
                notes_blocks.append(f"# {name}\nNo detailed notes recorded yet.")

    if not notes_blocks:
        return ""

    joined_notes = "\n\n---\n\n".join(notes_blocks)
    return f"""--- Specific Location Notes (from Scratchpad) ---
{joined_notes}
--- End Specific Location Notes ---"""


def save_chapter_pass(
    scratchpad_dir: pathlib.Path,
    chapter_num: int,
    chapter_title: str,
    pass_number: int,
    content: str,
):
    """Saves a chapter pass output to scratchpad/chapters."""
    chap_dir = get_chapters_dir(scratchpad_dir)
    safe_title = sanitize_filename(chapter_title, max_length=40)
    file_path = (
        chap_dir / f"chapter_{chapter_num:02d}_{safe_title}_pass{pass_number}.md"
    )
    file_path.write_text(content, encoding="utf-8")
    logging.info(
        f"Saved Chapter {chapter_num} Pass {pass_number} to scratchpad: {file_path.name}"
    )


def save_chapter_summary(
    scratchpad_dir: pathlib.Path,
    chapter_num: int,
    chapter_title: str,
    summary: str,
):
    """Saves a chapter's final summary to scratchpad/chapters."""
    chap_dir = get_chapters_dir(scratchpad_dir)
    safe_title = sanitize_filename(chapter_title, max_length=40)
    file_path = chap_dir / f"chapter_{chapter_num:02d}_{safe_title}_summary.md"
    file_path.write_text(summary, encoding="utf-8")
    logging.info(f"Saved Chapter {chapter_num} summary to scratchpad: {file_path.name}")


def load_chapter_from_scratchpad(
    scratchpad_dir: pathlib.Path, chapter_num: int
) -> dict[str, str] | None:
    """
    Checks if a completed chapter (Pass 2) already exists in scratchpad.
    Returns a dict with 'content' and 'summary' if found, enabling resumption upon interruption.
    """
    chap_dir = get_chapters_dir(scratchpad_dir)
    pass2_files = list(chap_dir.glob(f"chapter_{chapter_num:02d}_*_pass2.md"))
    if not pass2_files:
        return None

    pass2_file = pass2_files[0]
    content = pass2_file.read_text(encoding="utf-8").strip()
    if not content or "Content generation failed" in content:
        return None

    # Look for matching summary
    summary = ""
    summary_files = list(chap_dir.glob(f"chapter_{chapter_num:02d}_*_summary.md"))
    if summary_files:
        summary = summary_files[0].read_text(encoding="utf-8").strip()

    return {"content": content, "summary": summary}


def update_scratchpad_entities(
    config: dict,
    scratchpad_dir: pathlib.Path,
    chapter_title: str,
    chapter_content: str,
    chapter_num: int,
):
    """
    Prompts the LLM to update character and location markdown files and summaries
    based on the events of the latest chapter.
    """
    char_summary_text = load_character_summary_text(scratchpad_dir)
    loc_summary_text = load_location_summary_text(scratchpad_dir)

    prompt = f"""
You are an assistant tracking characters, locations, and continuity for a novel.
Review the following chapter text for Chapter {chapter_num} titled '{chapter_title}'.

--- Current Characters Overview ---
{char_summary_text}

--- Current Locations Overview ---
{loc_summary_text}

--- Chapter Content ---
{chapter_content}
--- End Chapter Content ---

Your task is to update our scratchpad notes based on what happened in this chapter.
1. Identify characters who appeared or had developments in this chapter. For each, give an updated 1-2 sentence summary and concise bullet point notes on what happened to them / their current status in Chapter {chapter_num}.
2. If any new significant character was introduced, provide their full name, short 1-2 sentence summary, and initial notes.
3. Identify locations that appeared in this chapter. For each, give an updated 1-2 sentence summary and concise bullet point notes on events, atmosphere, or changes that took place there in Chapter {chapter_num}.
4. If any new significant physical location was introduced, provide its name, short 1-2 sentence summary, and initial notes.

Format your response strictly using this Markdown structure:

--- CHARACTERS UPDATE ---
* **Character Name**: Updated 1-2 sentence summary.
  - Notes: What happened to them, their actions, discoveries, or status in Chapter {chapter_num}.

--- LOCATIONS UPDATE ---
* **Location Name**: Updated 1-2 sentence summary.
  - Notes: Key events, description, or changes at this location in Chapter {chapter_num}.
--- END UPDATE ---

Provide only the formatted updates. Do not add introductory text.
Output in British English.
"""

    safe_title = sanitize_filename(chapter_title, max_length=30)
    cache_prefix = f"scratchpad_update_ch{chapter_num:02d}_{safe_title}"
    response = call_llm_api(prompt, config, cache_prefix=cache_prefix)

    if not response or not response.strip():
        logging.warning(f"No scratchpad updates generated for Chapter {chapter_num}.")
        return

    _apply_scratchpad_updates(scratchpad_dir, response, chapter_num)


def _apply_scratchpad_updates(
    scratchpad_dir: pathlib.Path, update_text: str, chapter_num: int
):
    """Parses LLM update text and applies updates to character and location markdown files."""
    char_dir = get_character_dir(scratchpad_dir)
    loc_dir = get_location_dir(scratchpad_dir)

    # Split into characters section and locations section
    char_part = ""
    loc_part = ""

    if "--- CHARACTERS UPDATE ---" in update_text:
        remainder = update_text.split("--- CHARACTERS UPDATE ---", 1)[1]
        if "--- LOCATIONS UPDATE ---" in remainder:
            char_part, loc_part = remainder.split("--- LOCATIONS UPDATE ---", 1)
        else:
            char_part = remainder
    elif "--- LOCATIONS UPDATE ---" in update_text:
        loc_part = update_text.split("--- LOCATIONS UPDATE ---", 1)[1]

    if "--- END UPDATE ---" in loc_part:
        loc_part = loc_part.split("--- END UPDATE ---", 1)[0]
    elif "--- END UPDATE ---" in char_part and not loc_part:
        char_part = char_part.split("--- END UPDATE ---", 1)[0]

    def _parse_entity_items(text_block: str) -> list[dict[str, str]]:
        items = []
        current_item = None
        for line in text_block.splitlines():
            line_str = line.strip()
            # Match bullet item: * **Name**: Description
            m = re.match(r"^[\*\-]\s*\*\*(.*?)\*\*:\s*(.*)", line_str)
            if m:
                if current_item:
                    items.append(current_item)
                current_item = {
                    "name": m.group(1).strip(),
                    "summary": m.group(2).strip(),
                    "notes": [],
                }
            elif current_item and line_str.startswith(("-", "*", "•", "Notes:")):
                cleaned_note = re.sub(r"^[-*•\s]*(?:Notes:\s*)?", "", line_str).strip()
                if cleaned_note:
                    current_item["notes"].append(cleaned_note)
            elif current_item and line_str:
                current_item["notes"].append(line_str)

        if current_item:
            items.append(current_item)
        return items

    char_items = _parse_entity_items(char_part)
    loc_items = _parse_entity_items(loc_part)

    # Update characters
    for item in char_items:
        name = item["name"]
        summary = item["summary"]
        notes_str = "; ".join(item["notes"]) if item["notes"] else summary

        safe_name = sanitize_filename(name, max_length=50)
        char_file = char_dir / f"{safe_name}.md"

        if char_file.exists():
            existing = char_file.read_text(encoding="utf-8")
            # Update summary if present
            if "## Short Summary" in existing:
                existing = re.sub(
                    r"## Short Summary\n.*?(?=\n##|\Z)",
                    f"## Short Summary\n{summary}\n",
                    existing,
                    flags=re.DOTALL,
                )
            # Append chapter note to developments
            new_note = f"\n  - **Chapter {chapter_num}:** {notes_str}"
            if "## Story Developments" in existing:
                existing = existing.replace(
                    "## Story Developments", f"## Story Developments{new_note}"
                )
            else:
                existing += f"\n\n## Story Developments\n{new_note}\n"
            char_file.write_text(existing, encoding="utf-8")
            logging.info(f"Updated scratchpad character note: {char_file.name}")
        else:
            new_content = f"""# {name}

## Short Summary
{summary}

## Character Notes & Development
- **Role & Significance:** {summary}
- **Status:** Active
- **Story Developments:**
  - **Chapter {chapter_num}:** {notes_str}
"""
            char_file.write_text(new_content, encoding="utf-8")
            logging.info(f"Added new character to scratchpad: {char_file.name}")

    # Rebuild character summary.md
    if char_items:
        _sync_summary_file(char_dir, "Characters Summary", char_items)

    # Update locations
    for item in loc_items:
        name = item["name"]
        summary = item["summary"]
        notes_str = "; ".join(item["notes"]) if item["notes"] else summary

        safe_name = sanitize_filename(name, max_length=50)
        loc_file = loc_dir / f"{safe_name}.md"

        if loc_file.exists():
            existing = loc_file.read_text(encoding="utf-8")
            if "## Short Summary" in existing:
                existing = re.sub(
                    r"## Short Summary\n.*?(?=\n##|\Z)",
                    f"## Short Summary\n{summary}\n",
                    existing,
                    flags=re.DOTALL,
                )
            new_note = f"\n  - **Chapter {chapter_num}:** {notes_str}"
            if "## Story Developments" in existing:
                existing = existing.replace(
                    "## Story Developments", f"## Story Developments{new_note}"
                )
            else:
                existing += f"\n\n## Story Developments\n{new_note}\n"
            loc_file.write_text(existing, encoding="utf-8")
            logging.info(f"Updated scratchpad location note: {loc_file.name}")
        else:
            new_content = f"""# {name}

## Short Summary
{summary}

## Location Details & Atmosphere
- **Significance:** {summary}
- **Story Developments:**
  - **Chapter {chapter_num}:** {notes_str}
"""
            loc_file.write_text(new_content, encoding="utf-8")
            logging.info(f"Added new location to scratchpad: {loc_file.name}")

    # Rebuild location summary.md
    if loc_items:
        _sync_summary_file(loc_dir, "Locations Summary", loc_items)


def _sync_summary_file(
    folder: pathlib.Path, heading: str, updated_items: list[dict[str, str]]
):
    """Merges updated items into the folder's summary.md."""
    summary_file = folder / "summary.md"
    existing_map = {}
    if summary_file.exists():
        text = summary_file.read_text(encoding="utf-8")
        for line in text.splitlines():
            m = re.match(r"^[\*\-]\s*\*\*(.*?)\*\*:\s*(.*)", line.strip())
            if m:
                existing_map[m.group(1).strip()] = m.group(2).strip()

    # Apply updates
    for item in updated_items:
        existing_map[item["name"]] = item["summary"]

    lines = [f"# {heading}\n"]
    for name, desc in existing_map.items():
        lines.append(f"* **{name}**: {desc}")

    summary_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logging.info(
        f"Synchronized scratchpad {summary_file.name} ({len(existing_map)} entries)"
    )
