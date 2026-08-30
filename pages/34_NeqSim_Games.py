"""NeqSim Games hub with simulation challenges and reusable flashcards."""

from __future__ import annotations

import html
import os
import sys

import pandas as pd
import streamlit as st


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from process_chat.flashcard_game import (  # noqa: E402
    PHASE_EQUILIBRIUM_DECK,
    Flashcard,
    FlashcardDeck,
    custom_deck,
    deck_from_json,
    deck_to_json,
    score_flashcards,
    validate_card,
)
from theme import apply_theme, theme_toggle  # noqa: E402


CUSTOM_CARDS_KEY = "neqsim_games_custom_cards"
FLASHCARD_DECK_KEY = "neqsim_games_flashcard_deck"
FLASHCARD_INDEX_KEY = "neqsim_games_flashcard_index"
FLASHCARD_REVEALED_KEY = "neqsim_games_flashcard_revealed"
FLASHCARD_MARKS_KEY = "neqsim_games_flashcard_marks"


def _initialize_state() -> None:
    st.session_state.setdefault(CUSTOM_CARDS_KEY, [])
    st.session_state.setdefault(FLASHCARD_DECK_KEY, "Built-in phase deck")
    st.session_state.setdefault(FLASHCARD_INDEX_KEY, 0)
    st.session_state.setdefault(FLASHCARD_REVEALED_KEY, False)
    st.session_state.setdefault(FLASHCARD_MARKS_KEY, {})


def _reset_flashcard_session() -> None:
    st.session_state[FLASHCARD_INDEX_KEY] = 0
    st.session_state[FLASHCARD_REVEALED_KEY] = False
    st.session_state[FLASHCARD_MARKS_KEY] = {}


def _custom_cards() -> tuple[Flashcard, ...]:
    cards = st.session_state.get(CUSTOM_CARDS_KEY, [])
    return tuple(card for card in cards if isinstance(card, Flashcard))


def _active_deck() -> FlashcardDeck | None:
    if st.session_state[FLASHCARD_DECK_KEY] == "Built-in phase deck":
        return PHASE_EQUILIBRIUM_DECK
    cards = _custom_cards()
    return custom_deck(cards) if cards else None


def _mark_card(deck: FlashcardDeck, mastered: bool) -> None:
    index = int(st.session_state[FLASHCARD_INDEX_KEY]) % len(deck.cards)
    marks = dict(st.session_state[FLASHCARD_MARKS_KEY])
    marks[deck.cards[index].card_id] = bool(mastered)
    st.session_state[FLASHCARD_MARKS_KEY] = marks
    st.session_state[FLASHCARD_INDEX_KEY] = (index + 1) % len(deck.cards)
    st.session_state[FLASHCARD_REVEALED_KEY] = False


st.set_page_config(
    page_title="NeqSim Games",
    page_icon="images/neqsimlogocircleflat.png",
    layout="wide",
)
apply_theme()
theme_toggle()
_initialize_state()

st.markdown(
    """
<style>
    .block-container { max-width: 1450px; padding-top: 1.2rem; padding-bottom: 3rem; }
    .games-hero {
        padding: 1.6rem 1.8rem;
        border-radius: 18px;
        color: #f7fcff;
        background: linear-gradient(125deg, #0b2f4a, #176f78);
        box-shadow: 0 16px 38px rgba(8, 43, 65, 0.17);
        margin-bottom: 1rem;
    }
    .games-hero h1 { color: #ffffff !important; margin: 0; letter-spacing: -0.035em; }
    .games-hero p { color: #d9f2f3 !important; max-width: 850px; margin: 0.65rem 0 0; }
    .game-card {
        min-height: 190px;
        padding: 1.05rem 1.15rem;
        border: 1px solid rgba(24, 83, 110, 0.23);
        border-radius: 14px;
        background: rgba(244, 250, 252, 0.94);
        margin-bottom: 0.6rem;
    }
    .game-card h3 { color: #143d56 !important; margin-top: 0.15rem; }
    .game-card p { color: #4b6c7d !important; line-height: 1.5; }
    .flashcard-shell {
        min-height: 260px;
        padding: 1.4rem;
        border: 1px solid rgba(23, 71, 116, 0.27);
        border-radius: 16px;
        background: linear-gradient(145deg, rgba(247, 252, 254, 0.98), rgba(233, 246, 246, 0.94));
    }
    .flash-topic { color: #16747a !important; font-weight: 750; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.08em; }
    .flash-prompt { color: #153b53 !important; font-size: 1.35rem; font-weight: 700; line-height: 1.4; }
    @media (max-width: 600px) {
        .block-container { padding-left: 0.85rem; padding-right: 0.85rem; }
        .games-hero { padding: 1.2rem; }
    }
</style>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### NeqSim Games")
    st.caption("Simulation challenges and learning decks")
    if st.button("← NeqSim Studio", use_container_width=True):
        st.switch_page("pages/00_NeqSim_Studio.py")
    st.divider()
    st.caption(
        "All game scores are educational. Native calculations remain visible "
        "and are not design certification."
    )

st.markdown(
    """
<section class="games-hero" aria-labelledby="games-title">
  <h1 id="games-title">NeqSim Games</h1>
  <p>
    Learn thermodynamics and process engineering by changing real model inputs,
    reading native evidence, and practicing concepts with reusable flashcard decks.
  </p>
</section>
""",
    unsafe_allow_html=True,
)

simulation_tab, flashcard_tab, builder_tab = st.tabs(
    ["Simulation games", "Flashcard game", "Create a flashcard deck"]
)

with simulation_tab:
    st.subheader("Choose a native NeqSim challenge")
    plant_column, phase_column = st.columns(2, gap="large")
    with plant_column:
        st.markdown(
            """
<article class="game-card">
  <div aria-hidden="true">🎛️</div>
  <h3>Plant Operator</h3>
  <p>Increase gas throughput while balancing compression, cooling, export conditions, and conservation.</p>
  <small>Process simulation · operating decisions</small>
</article>
""",
            unsafe_allow_html=True,
        )
        if st.button("Open Plant Operator", use_container_width=True):
            st.switch_page("pages/36_NeqSim_Plant_Operator.py")
    with phase_column:
        st.markdown(
            """
<article class="game-card">
  <div aria-hidden="true">🧪</div>
  <h3>Phase Equilibrium Lab</h3>
  <p>Find a narrow retrograde-condensate window using temperature, pressure, phase split, density, viscosity, and Z.</p>
  <small>TP flash · equilibrium and properties</small>
</article>
""",
            unsafe_allow_html=True,
        )
        if st.button("Open Phase Equilibrium Lab", use_container_width=True):
            st.switch_page("pages/37_NeqSim_Phase_Equilibrium_Lab.py")

with flashcard_tab:
    header_column, reset_column = st.columns([4, 1])
    with header_column:
        st.subheader("Practice a deck")
    with reset_column:
        if st.button("Reset progress", use_container_width=True):
            _reset_flashcard_session()
            st.rerun()

    st.radio(
        "Deck",
        ("Built-in phase deck", "My custom deck"),
        horizontal=True,
        key=FLASHCARD_DECK_KEY,
        on_change=_reset_flashcard_session,
    )
    deck = _active_deck()
    if deck is None:
        st.info(
            "Your custom deck is empty. Add or import cards in "
            "**Create a flashcard deck**."
        )
    else:
        index = int(st.session_state[FLASHCARD_INDEX_KEY]) % len(deck.cards)
        card = deck.cards[index]
        marks = {
            card_id: value
            for card_id, value in dict(
                st.session_state[FLASHCARD_MARKS_KEY]
            ).items()
            if card_id in {item.card_id for item in deck.cards}
        }
        score = score_flashcards(deck, marks)
        metric_1, metric_2, metric_3 = st.columns(3)
        metric_1.metric("Card", f"{index + 1}/{len(deck.cards)}")
        metric_2.metric("Reviewed", score.reviewed)
        metric_3.metric("Mastery", f"{score.mastery_pct:.0f}%")
        st.progress(score.reviewed / len(deck.cards))
        st.markdown(
            f"""
<section class="flashcard-shell" aria-label="Current flashcard">
  <p class="flash-topic">{html.escape(card.topic)} · {html.escape(card.difficulty)}</p>
  <p class="flash-prompt">{html.escape(card.prompt)}</p>
</section>
""",
            unsafe_allow_html=True,
        )
        if not st.session_state[FLASHCARD_REVEALED_KEY]:
            if st.button("Reveal answer", type="primary", use_container_width=True):
                st.session_state[FLASHCARD_REVEALED_KEY] = True
                st.rerun()
        else:
            st.success(card.answer)
            if card.explanation:
                st.caption(card.explanation)
            knew_column, review_column = st.columns(2)
            with knew_column:
                if st.button("I knew this", use_container_width=True):
                    _mark_card(deck, True)
                    st.rerun()
            with review_column:
                if st.button("Review again", use_container_width=True):
                    _mark_card(deck, False)
                    st.rerun()
        st.caption(f"Session status: {score.label}")

with builder_tab:
    st.subheader("Build a reusable flashcard game")
    st.write(
        "Create a session deck, practice it immediately, and download the "
        "versioned JSON to share or import later."
    )
    with st.form("neqsim_flashcard_builder", clear_on_submit=True):
        topic = st.text_input("Topic", placeholder="e.g. Gas compressibility")
        prompt = st.text_area("Question / front")
        answer = st.text_area("Answer / back")
        explanation = st.text_area("Optional explanation")
        difficulty = st.selectbox(
            "Difficulty",
            ("foundation", "applied", "advanced"),
        )
        add_card = st.form_submit_button("Add card to custom deck")
    if add_card:
        try:
            existing = _custom_cards()
            used_ids = {card.card_id for card in existing}
            next_number = len(existing) + 1
            card_id = f"custom-{next_number}"
            while card_id in used_ids:
                next_number += 1
                card_id = f"custom-{next_number}"
            new_card = validate_card(
                Flashcard(
                    card_id=card_id,
                    topic=topic,
                    prompt=prompt,
                    answer=answer,
                    explanation=explanation,
                    difficulty=difficulty,
                )
            )
            candidate_deck = custom_deck((*existing, new_card))
            # Guarantee that anything stored by the builder can immediately
            # round-trip through the same bounded JSON import contract.
            deck_to_json(candidate_deck)
        except ValueError as error:
            st.error(str(error))
        else:
            st.session_state[CUSTOM_CARDS_KEY] = list(candidate_deck.cards)
            _reset_flashcard_session()
            st.success("Card added. Select **My custom deck** to play it.")

    uploaded_deck = st.file_uploader(
        "Import a flashcard deck",
        type=("json",),
        help="Imports the validated NeqSim flashcard schema v1.",
    )
    import_column, clear_column = st.columns(2)
    with import_column:
        if st.button(
            "Import uploaded deck",
            disabled=uploaded_deck is None,
            use_container_width=True,
        ):
            try:
                imported = deck_from_json(uploaded_deck.getvalue())
            except ValueError as error:
                st.error(str(error))
            else:
                st.session_state[CUSTOM_CARDS_KEY] = list(imported.cards)
                _reset_flashcard_session()
                st.success(f"Imported {len(imported.cards)} cards.")
                st.rerun()
    with clear_column:
        if st.button("Clear custom deck", use_container_width=True):
            st.session_state[CUSTOM_CARDS_KEY] = []
            _reset_flashcard_session()
            st.rerun()

    custom_cards = _custom_cards()
    st.metric("Custom cards", len(custom_cards))
    if custom_cards:
        session_deck = custom_deck(custom_cards)
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Topic": card.topic,
                        "Question": card.prompt,
                        "Difficulty": card.difficulty,
                    }
                    for card in custom_cards
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.download_button(
            "Download custom deck JSON",
            data=deck_to_json(session_deck),
            file_name="neqsim_flashcard_deck.json",
            mime="application/json",
            use_container_width=True,
        )

st.divider()
st.caption(
    "NeqSim Games · native simulation evidence for calculated challenges · "
    "portable schema-v1 flashcard decks for knowledge practice."
)
