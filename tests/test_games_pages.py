"""Source and initial-render regressions for the NeqSim Games pages."""

import ast
from pathlib import Path
import unittest

from streamlit.testing.v1 import AppTest

from studio.navigation import destination_by_key


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GAMES_PAGE = PROJECT_ROOT / "pages" / "34_NeqSim_Games.py"
PHASE_PAGE = PROJECT_ROOT / "pages" / "37_NeqSim_Phase_Equilibrium_Lab.py"


class GamesPagesTest(unittest.TestCase):
    """Keep the shared menu, flashcards, and native phase game discoverable."""

    def test_games_hub_is_the_single_studio_destination(self):
        destination = destination_by_key("games")
        source = GAMES_PAGE.read_text(encoding="utf-8")

        self.assertTrue(destination.available)
        self.assertEqual(destination.page, "pages/34_NeqSim_Games.py")
        self.assertIn("pages/36_NeqSim_Plant_Operator.py", source)
        self.assertIn("pages/37_NeqSim_Phase_Equilibrium_Lab.py", source)
        self.assertIn("Create a flashcard deck", source)
        self.assertIn("Download custom deck JSON", source)
        self.assertIn(
            "candidate_deck = custom_deck((*existing, new_card))",
            source,
        )

    def test_both_pages_are_valid_python(self):
        ast.parse(GAMES_PAGE.read_text(encoding="utf-8"))
        phase_source = PHASE_PAGE.read_text(encoding="utf-8")
        ast.parse(phase_source)
        self.assertIn("run_phase_challenge(", phase_source)
        self.assertIn("Phase compositions & K-values", phase_source)
        self.assertIn("last_run.controls == current_controls", phase_source)

    def test_games_hub_initial_render_does_not_start_native_calculation(self):
        app = AppTest.from_file(str(GAMES_PAGE)).run(timeout=30)

        if app.exception:
            details = "\n".join(str(item.value) for item in app.exception)
            self.fail(f"Games hub raised exceptions:\n{details}")
        button_labels = [button.label for button in app.button]
        self.assertIn("Open Plant Operator", button_labels)
        self.assertIn("Open Phase Equilibrium Lab", button_labels)
        self.assertIn("Reveal answer", button_labels)

    def test_flashcard_reveal_and_self_check_update_mastery(self):
        app = AppTest.from_file(str(GAMES_PAGE)).run(timeout=30)
        reveal = next(
            button for button in app.button if button.label == "Reveal answer"
        )

        app = reveal.click().run(timeout=30)
        self.assertIn(
            "The equilibrium phases, their amounts, compositions, and "
            "properties at specified temperature and pressure.",
            [item.value for item in app.success],
        )
        knew = next(
            button for button in app.button if button.label == "I knew this"
        )
        app = knew.click().run(timeout=30)

        metrics = {item.label: item.value for item in app.metric}
        self.assertEqual(metrics["Reviewed"], "1")
        self.assertEqual(metrics["Mastery"], "100%")

    def test_phase_lab_initial_render_waits_for_player(self):
        app = AppTest.from_file(str(PHASE_PAGE)).run(timeout=30)

        if app.exception:
            details = "\n".join(str(item.value) for item in app.exception)
            self.fail(f"Phase lab raised exceptions:\n{details}")
        self.assertIn(
            "▶ Run native TP flash",
            [button.label for button in app.button],
        )
        self.assertEqual(
            [slider.label for slider in app.slider],
            ["Temperature [°C]", "Pressure [bara]"],
        )

    def test_phase_lab_winning_controls_publish_current_evidence(self):
        app = AppTest.from_file(
            str(PHASE_PAGE),
            default_timeout=90,
        ).run()
        app.slider[0].set_value(50.0)
        app.slider[1].set_value(80.0)
        run_button = next(
            button
            for button in app.button
            if button.label == "▶ Run native TP flash"
        )

        app = run_button.click().run(timeout=90)

        if app.exception:
            details = "\n".join(str(item.value) for item in app.exception)
            self.fail(f"Phase lab winning attempt raised exceptions:\n{details}")
        self.assertIn("Phase window captured", [item.value for item in app.success])
        metrics = {item.label: item.value for item in app.metric}
        self.assertEqual(metrics["Score"], "1000/1000")
        self.assertEqual(metrics["Attempts"], "1")
        self.assertEqual(metrics["Best score"], "1000/1000")


if __name__ == "__main__":
    unittest.main()
