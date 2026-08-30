"""Tests for portable NeqSim flashcard game decks."""

import json
import unittest

from process_chat.flashcard_game import (
    MAX_DECK_CARDS,
    PHASE_EQUILIBRIUM_DECK,
    Flashcard,
    FlashcardDeck,
    custom_deck,
    deck_from_json,
    deck_to_json,
    score_flashcards,
    validate_deck,
)


class FlashcardGameTest(unittest.TestCase):
    """Keep decks portable, bounded, unique, and honestly scored."""

    def test_builtin_phase_deck_covers_equilibrium_and_properties(self):
        deck = validate_deck(PHASE_EQUILIBRIUM_DECK)

        self.assertGreaterEqual(len(deck.cards), 10)
        self.assertIn("Phase equilibrium", {card.topic for card in deck.cards})
        self.assertIn("Fluid properties", {card.topic for card in deck.cards})
        self.assertIn("advanced", {card.difficulty for card in deck.cards})

    def test_versioned_json_round_trip_preserves_cards(self):
        encoded = deck_to_json(PHASE_EQUILIBRIUM_DECK)
        decoded = deck_from_json(encoded)

        self.assertEqual(decoded, PHASE_EQUILIBRIUM_DECK)
        self.assertEqual(json.loads(encoded)["schema_version"], 1)

    def test_invalid_and_duplicate_cards_fail_loudly(self):
        duplicate = Flashcard("same", "Topic", "Question", "Answer")
        with self.assertRaisesRegex(ValueError, "ids must be unique"):
            validate_deck(
                FlashcardDeck("Duplicates", "", (duplicate, duplicate))
            )
        with self.assertRaisesRegex(ValueError, "invalid JSON"):
            deck_from_json("not-json")
        with self.assertRaisesRegex(ValueError, "Difficulty"):
            validate_deck(
                FlashcardDeck(
                    "Bad difficulty",
                    "",
                    (
                        Flashcard(
                            "one",
                            "Topic",
                            "Question",
                            "Answer",
                            difficulty="impossible",
                        ),
                    ),
                )
            )
        with self.assertRaisesRegex(ValueError, "schema version"):
            validate_deck(
                FlashcardDeck(
                    "Bad schema",
                    "",
                    (Flashcard("one", "Topic", "Question", "Answer"),),
                    schema_version=True,
                )
            )

    def test_custom_builder_rejects_card_and_json_size_overflow(self):
        cards = tuple(
            Flashcard(
                f"custom-{index}",
                "Topic",
                "Question",
                "Answer",
            )
            for index in range(MAX_DECK_CARDS + 1)
        )
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            custom_deck(cards)

        oversized_but_field_valid = FlashcardDeck(
            "Oversized JSON",
            "",
            tuple(
                Flashcard(
                    f"large-{index}",
                    "Topic",
                    "Q" * 1_000,
                    "A" * 2_000,
                    "E" * 2_000,
                )
                for index in range(30)
            ),
        )
        with self.assertRaisesRegex(ValueError, "JSON is too large"):
            deck_to_json(oversized_but_field_valid)

    def test_score_counts_only_reviewed_known_cards(self):
        deck = PHASE_EQUILIBRIUM_DECK
        score = score_flashcards(
            deck,
            {
                deck.cards[0].card_id: True,
                deck.cards[1].card_id: False,
            },
        )

        self.assertEqual(score.reviewed, 2)
        self.assertEqual(score.mastered, 1)
        self.assertEqual(score.review_again, 1)
        self.assertEqual(score.mastery_pct, 50.0)
        with self.assertRaisesRegex(ValueError, "unknown flashcard ids"):
            score_flashcards(deck, {"not-in-deck": True})


if __name__ == "__main__":
    unittest.main()
