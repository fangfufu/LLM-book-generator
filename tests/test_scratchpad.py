import pathlib
import tempfile
import unittest
from unittest.mock import patch

from src.generators.fiction import (
    generate_fiction_chapter_pass1,
    generate_fiction_chapter_pass2,
)
from src.generators.scratchpad import (
    detect_relevant_entities,
    get_character_dir,
    get_location_dir,
    get_scratchpad_dir,
    has_initialized_scratchpad,
    init_scratchpad,
    load_chapter_from_scratchpad,
    load_character_summary_text,
    load_characters_from_scratchpad,
    load_location_summary_text,
    load_locations_from_scratchpad,
    load_specific_character_notes,
    load_specific_location_notes,
    save_chapter_pass,
    save_chapter_summary,
    update_scratchpad_entities,
)


class TestScratchpad(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_scratchpad = pathlib.Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_get_scratchpad_dir_naming_scheme(self):
        config = {
            "api_settings": {
                "provider": "llama_cpp",
                "base_scratchpad_dir": str(self.base_scratchpad),
                "llama_cpp": {"model": "test_model:latest"},
            },
            "generation_params": {"main_topic": "Operation Reptilian Infiltration"},
        }

        scratchpad_dir = get_scratchpad_dir(config)
        self.assertTrue(scratchpad_dir.exists())
        # Should be inside base_scratchpad / sanitized_model / topic_dir
        self.assertIn("test_model_latest", str(scratchpad_dir))
        self.assertIn("Operation_Reptilian_Infiltration", str(scratchpad_dir))

    def test_init_scratchpad_and_loading(self):
        scratchpad_dir = self.base_scratchpad / "test_scratchpad"
        scratchpad_dir.mkdir(parents=True, exist_ok=True)

        self.assertFalse(has_initialized_scratchpad(scratchpad_dir))

        characters = [
            {"name": "James Smith", "description": "Veteran MI6 handler."},
            {"name": "Elena Vance", "description": "Undercover field agent."},
        ]
        locations = [
            {"name": "Whitehall Bunker", "description": "Secret command center."},
            {
                "name": "Vauxhall Cross",
                "description": "MI6 Headquarters on the Thames.",
            },
        ]

        init_scratchpad(scratchpad_dir, characters, locations)
        self.assertTrue(has_initialized_scratchpad(scratchpad_dir))

        # Check character files
        char_dir = get_character_dir(scratchpad_dir)
        self.assertTrue((char_dir / "summary.md").exists())
        self.assertTrue((char_dir / "James_Smith.md").exists())
        self.assertTrue((char_dir / "Elena_Vance.md").exists())

        # Check location files
        loc_dir = get_location_dir(scratchpad_dir)
        self.assertTrue((loc_dir / "summary.md").exists())
        self.assertTrue((loc_dir / "Whitehall_Bunker.md").exists())
        self.assertTrue((loc_dir / "Vauxhall_Cross.md").exists())

        # Test loading
        loaded_chars = load_characters_from_scratchpad(scratchpad_dir)
        self.assertEqual(len(loaded_chars), 2)
        self.assertEqual(loaded_chars[0]["name"], "James Smith")
        self.assertEqual(loaded_chars[1]["name"], "Elena Vance")

        loaded_locs = load_locations_from_scratchpad(scratchpad_dir)
        self.assertEqual(len(loaded_locs), 2)
        self.assertEqual(loaded_locs[0]["name"], "Whitehall Bunker")
        self.assertEqual(loaded_locs[1]["name"], "Vauxhall Cross")

        char_summary_text = load_character_summary_text(scratchpad_dir)
        self.assertIn("James Smith", char_summary_text)
        self.assertIn("Elena Vance", char_summary_text)

        loc_summary_text = load_location_summary_text(scratchpad_dir)
        self.assertIn("Whitehall Bunker", loc_summary_text)
        self.assertIn("Vauxhall Cross", loc_summary_text)

    def test_detect_relevant_entities(self):
        scratchpad_dir = self.base_scratchpad / "test_detect"
        scratchpad_dir.mkdir(parents=True, exist_ok=True)

        characters = [
            {"name": "James Smith", "description": "Veteran MI6 handler."},
            {"name": "Elena Vance", "description": "Undercover field agent."},
            {"name": "Marcus Kane", "description": "Rogue operative."},
        ]
        locations = [
            {"name": "Whitehall Bunker", "description": "Secret command center."},
            {
                "name": "Vauxhall Cross",
                "description": "MI6 Headquarters on the Thames.",
            },
            {
                "name": "Berlin Safehouse",
                "description": "Covert apartment in Kreuzberg.",
            },
        ]
        init_scratchpad(scratchpad_dir, characters, locations)

        chapter_title = "The Breach at Whitehall"
        chapter_summary = (
            "Elena enters the Whitehall Bunker to retrieve classified cipher files."
        )
        chapter_draft = (
            "Elena crept quietly down the reinforced corridor of the Whitehall Bunker."
        )

        detected_chars, detected_locs = detect_relevant_entities(
            scratchpad_dir, chapter_title, chapter_summary, chapter_draft
        )

        self.assertIn("Elena Vance", detected_chars)
        self.assertNotIn("Marcus Kane", detected_chars)
        self.assertIn("Whitehall Bunker", detected_locs)
        self.assertNotIn("Berlin Safehouse", detected_locs)

    def test_load_specific_notes(self):
        scratchpad_dir = self.base_scratchpad / "test_notes"
        scratchpad_dir.mkdir(parents=True, exist_ok=True)

        characters = [{"name": "Elena Vance", "description": "Undercover field agent."}]
        locations = [
            {"name": "Whitehall Bunker", "description": "Secret command center."}
        ]
        init_scratchpad(scratchpad_dir, characters, locations)

        char_notes = load_specific_character_notes(scratchpad_dir, ["Elena Vance"])
        self.assertIn("Specific Character Notes", char_notes)
        self.assertIn("Elena Vance", char_notes)

        loc_notes = load_specific_location_notes(scratchpad_dir, ["Whitehall Bunker"])
        self.assertIn("Specific Location Notes", loc_notes)
        self.assertIn("Whitehall Bunker", loc_notes)

    def test_chapter_pass_saving_and_resumption(self):
        scratchpad_dir = self.base_scratchpad / "test_resume"
        scratchpad_dir.mkdir(parents=True, exist_ok=True)

        # Before saving, no chapter should be found
        self.assertIsNone(load_chapter_from_scratchpad(scratchpad_dir, 1))

        # Save Pass 1
        save_chapter_pass(
            scratchpad_dir,
            1,
            "The Beginning",
            pass_number=1,
            content="Draft pass 1 text.",
        )
        self.assertIsNone(load_chapter_from_scratchpad(scratchpad_dir, 1))

        # Save Pass 2 & Summary
        save_chapter_pass(
            scratchpad_dir,
            1,
            "The Beginning",
            pass_number=2,
            content="Enriched pass 2 text.",
        )
        save_chapter_summary(
            scratchpad_dir, 1, "The Beginning", summary="Summary of the beginning."
        )

        # Now load_chapter_from_scratchpad should find completed chapter
        loaded = load_chapter_from_scratchpad(scratchpad_dir, 1)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["content"], "Enriched pass 2 text.")
        self.assertEqual(loaded["summary"], "Summary of the beginning.")

    @patch("src.generators.scratchpad.call_llm_api")
    def test_update_scratchpad_entities(self, mock_llm):
        scratchpad_dir = self.base_scratchpad / "test_updates"
        scratchpad_dir.mkdir(parents=True, exist_ok=True)

        characters = [{"name": "Elena Vance", "description": "Undercover field agent."}]
        locations = [
            {"name": "Whitehall Bunker", "description": "Secret command center."}
        ]
        init_scratchpad(scratchpad_dir, characters, locations)

        mock_llm.return_value = """--- CHARACTERS UPDATE ---
* **Elena Vance**: Wounded field agent who uncovered deep conspiracy.
  - Notes: Sustained arm injury in shoot-out and escaped with microfiche.
* **Arthur Pendelton**: Corrupt senior cabinet minister.
  - Notes: Revealed as the mole passing intelligence.

--- LOCATIONS UPDATE ---
* **Whitehall Bunker**: Heavily compromised command facility now under lockdown.
  - Notes: Fire broke out in server room after Elena's escape.
--- END UPDATE ---
"""
        config = {"cache_dir": str(self.base_scratchpad)}
        update_scratchpad_entities(
            config,
            scratchpad_dir,
            chapter_title="Fire in the Deep",
            chapter_content="Gunfire echoed through the halls...",
            chapter_num=1,
        )

        char_dir = get_character_dir(scratchpad_dir)
        loc_dir = get_location_dir(scratchpad_dir)

        # Check Elena Vance updated
        elena_text = (char_dir / "Elena_Vance.md").read_text(encoding="utf-8")
        self.assertIn("Wounded field agent", elena_text)
        self.assertIn("Sustained arm injury", elena_text)

        # Check new character Arthur Pendelton created
        self.assertTrue((char_dir / "Arthur_Pendelton.md").exists())
        arthur_text = (char_dir / "Arthur_Pendelton.md").read_text(encoding="utf-8")
        self.assertIn("Corrupt senior cabinet minister", arthur_text)

        # Check location updated
        loc_text = (loc_dir / "Whitehall_Bunker.md").read_text(encoding="utf-8")
        self.assertIn("Heavily compromised", loc_text)
        self.assertIn("Fire broke out", loc_text)

        # Check summary.md synchronized
        char_summary = (char_dir / "summary.md").read_text(encoding="utf-8")
        self.assertIn("Arthur Pendelton", char_summary)

    @patch("src.generators.fiction.call_llm_api")
    def test_two_pass_fiction_generation(self, mock_llm):
        mock_llm.side_effect = [
            "Pass 1 Chapter Draft",
            "Pass 2 Enriched Chapter Content",
        ]

        config = {"cache_dir": str(self.base_scratchpad)}

        pass1_out = generate_fiction_chapter_pass1(
            config=config,
            previous_chapters_summary="Previous events...",
            chapter_title="The Cold Arrival",
            chapter_summary="Elena arrives in London in the rain.",
            character_summary="- Elena: Agent",
            location_summary="- London: City",
            writing_tone="Suspenseful",
            chapter_num=1,
        )
        self.assertEqual(pass1_out, "Pass 1 Chapter Draft")

        pass2_out = generate_fiction_chapter_pass2(
            config=config,
            previous_chapters_summary="Previous events...",
            chapter_title="The Cold Arrival",
            chapter_summary="Elena arrives in London in the rain.",
            pass1_draft=pass1_out,
            character_notes="Elena has a titanium watch and trench coat.",
            location_notes="London is enveloped in heavy smog and rain.",
            writing_tone="Suspenseful",
            chapter_num=1,
        )
        self.assertEqual(pass2_out, "Pass 2 Enriched Chapter Content")
        self.assertEqual(mock_llm.call_count, 2)

        # Verify Pass 2 prompt doesn't contain the broad summaries, but contains specific notes
        pass2_call = mock_llm.call_args_list[1]
        prompt_arg = pass2_call[0][0]
        self.assertIn("Pass 1 Chapter Draft", prompt_arg)
        self.assertIn("titanium watch", prompt_arg)
        self.assertIn("heavy smog", prompt_arg)

    @patch("src.orchestrator.assemble_marketing_docx")
    @patch("src.orchestrator.assemble_docx")
    @patch("src.orchestrator.generate_book_blurb")
    @patch("src.orchestrator.generate_back_matter")
    @patch("src.orchestrator.generate_front_matter")
    @patch("src.orchestrator.save_summary_to_markdown")
    @patch("src.orchestrator.generate_overall_summary")
    @patch("src.orchestrator.update_scratchpad_entities")
    @patch("src.orchestrator.summarize_fiction_chapter")
    @patch("src.orchestrator.generate_fiction_chapter_pass2")
    @patch("src.orchestrator.generate_fiction_chapter_pass1")
    @patch("src.orchestrator.generate_fiction_chapter_outline")
    @patch("src.orchestrator.generate_location_list")
    @patch("src.orchestrator.generate_character_list")
    @patch("src.orchestrator.generate_overall_story")
    def test_orchestrator_fiction_resumption(
        self,
        mock_story,
        mock_char_list,
        mock_loc_list,
        mock_outline,
        mock_pass1,
        mock_pass2,
        mock_summary,
        mock_update_ent,
        mock_overall_sum,
        mock_save_sum_md,
        mock_front_matter,
        mock_back_matter,
        mock_marketing,
        mock_assemble_docx,
        mock_assemble_marketing,
    ):
        from src.orchestrator import run_generation_process_fiction

        scratchpad_dir = self.base_scratchpad / "test_orch_resume"
        scratchpad_dir.mkdir(parents=True, exist_ok=True)

        config = {
            "scratchpad_dir_resolved": scratchpad_dir,
            "generation_params": {
                "book_title": "Test Book",
                "writing_tone": "Neutral",
                "author_name": "Author",
                "author_gender": "male",
                "fiction_chapter_count": 2,
            },
        }

        mock_story.return_value = "Overall master plot."
        mock_char_list.return_value = [{"name": "Elena Vance", "description": "Agent"}]
        mock_loc_list.return_value = [{"name": "Whitehall", "description": "Bunker"}]
        mock_outline.return_value = [
            {"title": "Chapter One", "summary": "First chapter summary"},
            {"title": "Chapter Two", "summary": "Second chapter summary"},
        ]
        mock_pass1.return_value = "Pass 1 Draft Text"
        mock_pass2.return_value = "Pass 2 Final Text"
        mock_summary.return_value = "Generated new summary"

        # First run: should generate chapter 1 and chapter 2
        run_generation_process_fiction(
            config, self.base_scratchpad, self.base_scratchpad
        )

        self.assertEqual(mock_pass1.call_count, 2)
        self.assertEqual(mock_pass2.call_count, 2)
        self.assertTrue(has_initialized_scratchpad(scratchpad_dir))
        self.assertIsNotNone(load_chapter_from_scratchpad(scratchpad_dir, 1))
        self.assertIsNotNone(load_chapter_from_scratchpad(scratchpad_dir, 2))

        # Reset call counts
        mock_pass1.reset_mock()
        mock_pass2.reset_mock()
        mock_char_list.reset_mock()
        mock_loc_list.reset_mock()

        # Second run (simulating restarting after interruption / completion):
        # Chapters are already completed in scratchpad -> pass1 and pass2 should NOT be called!
        # Character and location lists should be loaded from scratchpad without calling generation APIs!
        run_generation_process_fiction(
            config, self.base_scratchpad, self.base_scratchpad
        )

        self.assertEqual(mock_pass1.call_count, 0)
        self.assertEqual(mock_pass2.call_count, 0)
        self.assertEqual(mock_char_list.call_count, 0)
        self.assertEqual(mock_loc_list.call_count, 0)


if __name__ == "__main__":
    unittest.main()
