import json
import textwrap
from datetime import datetime, timedelta
from pathlib import Path
from typing import List
from uuid import uuid4

import streamlit as st
from PIL import Image, ImageColor, ImageDraw, ImageFont


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
GALLERY_FILE = DATA_DIR / "idea_gallery.json"
IMAGES_DIR = DATA_DIR / "idea_images"


@st.cache_resource(show_spinner=False)
def _load_font():
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", 24)
    except Exception:
        return ImageFont.load_default()


def ensure_storage():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)


def load_gallery() -> List[dict]:
    ensure_storage()
    if not GALLERY_FILE.exists():
        return []

    try:
        with GALLERY_FILE.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError:
        st.warning("Could not read gallery history. Starting fresh for this session.")
        return []


def save_gallery(entries: List[dict]):
    ensure_storage()
    with GALLERY_FILE.open("w", encoding="utf-8") as handle:
        json.dump(entries, handle, indent=2)


def week_label_for(date_value: datetime) -> str:
    year, week_num, _ = date_value.isocalendar()
    return f"{year}-W{week_num:02d}"


def generate_image(idea: str, contributor: str, week_label: str) -> Path:
    ensure_storage()
    width, height = 900, 560
    background = Image.new("RGB", (width, height), color="#0b1b38")
    draw = ImageDraw.Draw(background, "RGBA")

    idea_seed = abs(hash((idea, week_label))) % (2 ** 32)
    palette = [
        "#00a2ff",
        "#ff7d00",
        "#ffd166",
        "#06d6a0",
        "#ef476f",
    ]

    for i in range(6):
        color = palette[(idea_seed + i) % len(palette)]
        opacity = 80 + (idea_seed + i * 13) % 60
        x0 = (idea_seed // (i + 3)) % width
        y0 = (idea_seed // (i + 5)) % height
        x1 = min(width, x0 + 180 + (idea_seed % 120))
        y1 = min(height, y0 + 120 + (idea_seed % 90))
        draw.ellipse((x0 - 90, y0 - 40, x1, y1), fill=(*ImageColor.getrgb(color), opacity))

    font = _load_font()
    try:
        header_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 30)
    except Exception:
        header_font = font

    title = f"Week {week_label}: Laugh & Launch"
    draw.text((40, 40), title, fill="white", font=header_font)
    draw.text((40, 90), f"Contributor: {contributor}", fill="#b6c8ff", font=font)

    wrapped = textwrap.fill(idea, width=28)
    text_y = 150
    for line in wrapped.split("\n"):
        draw.text((60, text_y), line, fill="white", font=font)
        text_y += font.getbbox(line)[3] + 10

    footer = "Professional polish • Funny twist • Future feature"
    draw.text((40, height - 60), footer, fill="#ffe6a7", font=font)

    filename = f"idea_{uuid4().hex}.png"
    image_path = IMAGES_DIR / filename
    background.save(image_path, format="PNG")
    return image_path


def add_entry(entries: List[dict], idea: str, contributor: str) -> dict:
    now = datetime.utcnow()
    week_label = week_label_for(now)
    contributor = (contributor or "").strip() or "Anonymous"
    image_path = generate_image(idea, contributor, week_label)

    entry = {
        "id": uuid4().hex,
        "idea": idea.strip(),
        "contributor": contributor,
        "created_at": now.isoformat(),
        "week": week_label,
        "image_path": str(image_path.relative_to(DATA_DIR)),
        "votes": 0,
    }
    entries.append(entry)
    save_gallery(entries)
    return entry


def parse_date(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value)
    except Exception:
        return datetime.utcnow()


def vote_for_entry(entries: List[dict], entry_id: str):
    for entry in entries:
        if entry["id"] == entry_id:
            entry["votes"] = entry.get("votes", 0) + 1
            save_gallery(entries)
            st.toast("Vote recorded!", icon="🎉")
            return
    st.warning("Could not find that image to vote on.")


def render_entry(entry: dict, data_root: Path):
    image_file = data_root / entry["image_path"]
    caption = f"{entry['idea']}\n— {entry['contributor']} (Votes: {entry.get('votes', 0)})"
    if image_file.exists():
        st.image(str(image_file), caption=caption)
    else:
        st.info("Image file missing. Generate again to refresh the gallery entry.")
        return
    if st.button("👍 Vote", key=f"vote_{entry['id']}"):
        gallery = load_gallery()
        vote_for_entry(gallery, entry["id"])


st.title("Weekly Idea Generation Gallery")
st.write(
    "Submit fresh ideas for new app functionality and watch the page conjure a professional "
    "yet funny visual for everyone to enjoy. Each week's ideas are saved so you can revisit "
    "the creativity and vote on your favourites."
)

st.sidebar.success("Share your idea and add a vote to the visuals you love!")

gallery_entries = load_gallery()

st.subheader("Share an idea for this week")
with st.form("idea_submission", clear_on_submit=True):
    idea_text = st.text_area("What should we build next?", placeholder="A pipeline that explains results with memes ...")
    contributor = st.text_input("Your name (optional)")
    submitted = st.form_submit_button("Generate & save image")

if submitted:
    if idea_text.strip():
        new_entry = add_entry(gallery_entries, idea_text, contributor)
        st.success("Your idea is saved and the image has been added to the gallery!")
        st.image(str(DATA_DIR / new_entry["image_path"]), caption="Freshly generated")
    else:
        st.warning("Please add an idea before submitting.")

now = datetime.utcnow()
last_week_cutoff = now - timedelta(days=7)
recent_entries = [
    entry for entry in gallery_entries if parse_date(entry.get("created_at", "")).replace(tzinfo=None) >= last_week_cutoff
]
recent_entries.sort(key=lambda item: item.get("votes", 0), reverse=True)

st.subheader("This week's spotlight (last 7 days)")
if recent_entries:
    for entry in recent_entries:
        render_entry(entry, DATA_DIR)
else:
    st.info("No ideas from the last week yet. Be the first to add one!")

week_options = sorted({entry["week"] for entry in gallery_entries}, reverse=True)
st.subheader("Browse past weeks")
if week_options:
    selected_week = st.selectbox("Pick a week to revisit", week_options, index=0)
    week_entries = [entry for entry in gallery_entries if entry["week"] == selected_week]
    week_entries.sort(key=lambda item: parse_date(item.get("created_at", "")), reverse=True)

    for entry in week_entries:
        render_entry(entry, DATA_DIR)
else:
    st.info("No archived weeks yet. Submit an idea to start building the gallery.")
