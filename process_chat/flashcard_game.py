"""Reusable, validated flashcard decks for NeqSim learning games."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Iterable, Mapping


FLASHCARD_SCHEMA_VERSION = 1
MAX_DECK_CARDS = 100
MAX_DECK_JSON_BYTES = 100_000
VALID_DIFFICULTIES = frozenset({"foundation", "applied", "advanced"})


@dataclass(frozen=True)
class Flashcard:
    """One two-sided learning card."""

    card_id: str
    topic: str
    prompt: str
    answer: str
    explanation: str = ""
    difficulty: str = "foundation"


@dataclass(frozen=True)
class FlashcardDeck:
    """A portable flashcard game deck."""

    name: str
    description: str
    cards: tuple[Flashcard, ...]
    schema_version: int = FLASHCARD_SCHEMA_VERSION


@dataclass(frozen=True)
class FlashcardScore:
    """Current self-assessment score for a deck session."""

    reviewed: int
    mastered: int
    review_again: int
    mastery_pct: float
    label: str


def _clean_text(value, field_name: str, *, maximum: int) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be text.")
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} cannot be empty.")
    if len(cleaned) > maximum:
        raise ValueError(f"{field_name} cannot exceed {maximum} characters.")
    return cleaned


def validate_card(card: Flashcard) -> Flashcard:
    """Return a normalized card or reject unsafe/ambiguous content."""
    if not isinstance(card, Flashcard):
        raise ValueError("Cards must use the Flashcard schema.")
    difficulty = str(card.difficulty).strip().lower()
    if difficulty not in VALID_DIFFICULTIES:
        raise ValueError(
            "Difficulty must be foundation, applied, or advanced."
        )
    if card.explanation is not None and not isinstance(card.explanation, str):
        raise ValueError("Explanation must be text.")
    explanation = str(card.explanation or "").strip()
    if len(explanation) > 2_000:
        raise ValueError("Explanation cannot exceed 2000 characters.")
    return Flashcard(
        card_id=_clean_text(card.card_id, "Card id", maximum=80),
        topic=_clean_text(card.topic, "Topic", maximum=80),
        prompt=_clean_text(card.prompt, "Prompt", maximum=1_000),
        answer=_clean_text(card.answer, "Answer", maximum=2_000),
        explanation=explanation,
        difficulty=difficulty,
    )


def validate_deck(deck: FlashcardDeck) -> FlashcardDeck:
    """Validate deck size, identifiers, and all card fields."""
    if not isinstance(deck, FlashcardDeck):
        raise ValueError("Deck must use the FlashcardDeck schema.")
    if (
        isinstance(deck.schema_version, bool)
        or not isinstance(deck.schema_version, int)
        or deck.schema_version != FLASHCARD_SCHEMA_VERSION
    ):
        raise ValueError("Unsupported flashcard schema version.")
    name = _clean_text(deck.name, "Deck name", maximum=120)
    description = str(deck.description or "").strip()[:1_000]
    cards = tuple(validate_card(card) for card in deck.cards)
    if not cards:
        raise ValueError("A flashcard deck must contain at least one card.")
    if len(cards) > MAX_DECK_CARDS:
        raise ValueError(
            f"A flashcard deck cannot exceed {MAX_DECK_CARDS} cards."
        )
    card_ids = [card.card_id for card in cards]
    if len(card_ids) != len(set(card_ids)):
        raise ValueError("Flashcard ids must be unique within a deck.")
    return FlashcardDeck(
        name=name,
        description=description,
        cards=cards,
        schema_version=FLASHCARD_SCHEMA_VERSION,
    )


def deck_to_json(deck: FlashcardDeck) -> str:
    """Serialize a validated deck to portable, versioned JSON."""
    normalized = validate_deck(deck)
    payload = {
        "schema_version": normalized.schema_version,
        "name": normalized.name,
        "description": normalized.description,
        "cards": [asdict(card) for card in normalized.cards],
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def deck_from_json(payload: str | bytes) -> FlashcardDeck:
    """Load a bounded, versioned deck from JSON."""
    if isinstance(payload, bytes):
        raw_bytes = payload
        try:
            payload = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("Flashcard deck must be UTF-8 JSON.") from exc
    elif isinstance(payload, str):
        raw_bytes = payload.encode("utf-8")
    else:
        raise ValueError("Flashcard deck must be JSON text.")
    if len(raw_bytes) > MAX_DECK_JSON_BYTES:
        raise ValueError("Flashcard deck JSON is too large.")
    try:
        raw = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError("Flashcard deck contains invalid JSON.") from exc
    if not isinstance(raw, dict):
        raise ValueError("Flashcard deck JSON must contain one object.")
    raw_cards = raw.get("cards")
    if not isinstance(raw_cards, list):
        raise ValueError("Flashcard deck cards must be a list.")
    if len(raw_cards) > MAX_DECK_CARDS:
        raise ValueError(
            f"A flashcard deck cannot exceed {MAX_DECK_CARDS} cards."
        )
    cards: list[Flashcard] = []
    for raw_card in raw_cards:
        if not isinstance(raw_card, dict):
            raise ValueError("Every flashcard must be an object.")
        try:
            cards.append(
                Flashcard(
                    card_id=raw_card["card_id"],
                    topic=raw_card["topic"],
                    prompt=raw_card["prompt"],
                    answer=raw_card["answer"],
                    explanation=raw_card.get("explanation", ""),
                    difficulty=raw_card.get("difficulty", "foundation"),
                )
            )
        except KeyError as exc:
            raise ValueError(
                f"Flashcard is missing required field: {exc.args[0]}."
            ) from exc
    return validate_deck(
        FlashcardDeck(
            name=raw.get("name", ""),
            description=raw.get("description", ""),
            cards=tuple(cards),
            schema_version=raw.get("schema_version"),
        )
    )


def score_flashcards(
    deck: FlashcardDeck,
    marks: Mapping[str, bool],
) -> FlashcardScore:
    """Score only reviewed cards; unknown identifiers are rejected."""
    normalized = validate_deck(deck)
    known_ids = {card.card_id for card in normalized.cards}
    unknown = set(marks) - known_ids
    if unknown:
        raise ValueError(
            "Marks contain unknown flashcard ids: " + ", ".join(sorted(unknown))
        )
    if any(not isinstance(value, bool) for value in marks.values()):
        raise ValueError("Flashcard marks must be boolean self-assessments.")
    reviewed = len(marks)
    mastered = sum(1 for value in marks.values() if value)
    review_again = reviewed - mastered
    mastery_pct = 100.0 * mastered / reviewed if reviewed else 0.0
    label = (
        "Mastered"
        if reviewed == len(normalized.cards) and mastery_pct >= 90.0
        else "Building confidence"
        if mastery_pct >= 60.0
        else "Keep practicing"
    )
    return FlashcardScore(
        reviewed=reviewed,
        mastered=mastered,
        review_again=review_again,
        mastery_pct=mastery_pct,
        label=label,
    )


def custom_deck(cards: Iterable[Flashcard]) -> FlashcardDeck:
    """Build the standard session-local custom deck."""
    return validate_deck(
        FlashcardDeck(
            name="My custom NeqSim deck",
            description="A user-authored flashcard game exported from NeqSim Games.",
            cards=tuple(cards),
        )
    )


PHASE_EQUILIBRIUM_DECK = validate_deck(
    FlashcardDeck(
        name="Phase equilibrium & fluid properties",
        description=(
            "Foundation and applied cards covering TP flashes, phase splits, "
            "K-values, equations of state, and key fluid properties."
        ),
        cards=(
            Flashcard(
                "phase-tp-flash",
                "Phase equilibrium",
                "What does a TP flash calculate?",
                "The equilibrium phases, their amounts, compositions, and "
                "properties at specified temperature and pressure.",
                "The total feed composition is fixed while NeqSim minimizes "
                "the thermodynamic equilibrium condition.",
            ),
            Flashcard(
                "phase-dew-point",
                "Phase equilibrium",
                "What is the hydrocarbon dew point?",
                "The condition where the first infinitesimal liquid hydrocarbon "
                "phase forms from a gas.",
                difficulty="foundation",
            ),
            Flashcard(
                "phase-bubble-point",
                "Phase equilibrium",
                "What is the bubble point?",
                "The condition where the first infinitesimal gas bubble forms from a liquid.",
                difficulty="foundation",
            ),
            Flashcard(
                "phase-k-value",
                "Phase equilibrium",
                "For Kᵢ = yᵢ/xᵢ, what does Kᵢ > 1 mean?",
                "Component i preferentially partitions to the gas phase.",
                "yᵢ is gas-phase mole fraction and xᵢ is liquid-phase mole fraction.",
                difficulty="applied",
            ),
            Flashcard(
                "phase-retrograde",
                "Phase equilibrium",
                "What is retrograde condensation?",
                "Liquid dropout caused by pressure reduction or compression-path "
                "changes in a gas-condensate system, even without ordinary "
                "cooling below a normal boiling point.",
                difficulty="advanced",
            ),
            Flashcard(
                "property-z-factor",
                "Fluid properties",
                "What does gas compressibility factor Z correct in PV = ZnRT?",
                "It corrects ideal-gas behavior for real-fluid intermolecular effects.",
                difficulty="foundation",
            ),
            Flashcard(
                "property-density",
                "Fluid properties",
                "At fixed temperature, what usually happens to gas density as pressure rises?",
                "Gas density generally increases, although the real-fluid "
                "relationship depends on Z and phase behavior.",
                difficulty="applied",
            ),
            Flashcard(
                "property-viscosity",
                "Fluid properties",
                "Which property measures resistance to shear flow?",
                "Dynamic viscosity, commonly reported in cP or Pa·s.",
                difficulty="foundation",
            ),
            Flashcard(
                "property-cp",
                "Fluid properties",
                "What does isobaric heat capacity Cp describe?",
                "The enthalpy change per unit temperature change at constant pressure.",
                difficulty="applied",
            ),
            Flashcard(
                "model-eos",
                "Thermodynamic models",
                "Why must a phase-equilibrium result identify its equation of state?",
                "Different models and mixing rules predict fugacity, phase "
                "boundaries, and properties differently.",
                "The game uses SRK with mixing rule 2 for a reproducible training case.",
                difficulty="applied",
            ),
            Flashcard(
                "balance-beta",
                "Validation",
                "What should all equilibrium phase mole fractions sum to?",
                "One, within numerical tolerance.",
                difficulty="foundation",
            ),
            Flashcard(
                "validation-missing",
                "Validation",
                "Should a missing native property be treated as a passing game check?",
                "No. Missing or non-finite native evidence must block the affected check.",
                difficulty="applied",
            ),
        ),
    )
)
