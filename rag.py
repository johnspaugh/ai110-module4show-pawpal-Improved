"""
Retrieval for PawPal+ (the "R" in RAG).

Loads the pet-care guides in knowledge/, splits them into one chunk per "##"
section, and finds the chunks that best match a pet. Scoring is plain keyword
overlap weighted by how rare each word is (TF-IDF style), so it needs no
extra libraries and gives the same answer every run.
"""

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("pawpal.rag")

KNOWLEDGE_DIR = Path(__file__).resolve().parent / "knowledge"

STAGE_WEIGHT = 3.0  # life-stage words (senior, adult, kitten) matter most
NOTE_MIN_IDF = 1.8  # only distinctive words from the owner's notes steer the search
MIN_SCORE = 1.5  # below this a chunk is not relevant enough to return

# Words that carry no meaning for matching.
STOPWORDS = {
    "the", "and", "for", "are", "with", "that", "this", "from", "have", "has",
    "their", "they", "them", "can", "not", "but", "all", "any", "your", "you",
    "about", "into", "than", "then", "also", "very", "more", "most", "such",
    "his", "her", "its", "was", "were", "who", "get", "got", "does", "need",
    "needs", "usually", "often", "should", "keep", "per", "each", "one",
}

# Different words people use for the same thing.
SYNONYMS = {
    "medicine": "medication",
    "med": "medication",
    "meds": "medication",
    "pill": "medication",
    "tablet": "medication",
    "older": "senior",
    "elderly": "senior",
    "pup": "puppy",
}

# One task per line: Title | minutes | priority | frequency | essential | HH:MM | category
TASK_LINE = re.compile(
    r"^-\s*(?P<title>[^|]+?)\s*\|\s*(?P<minutes>\d+)\s*\|\s*"
    r"(?P<priority>low|medium|high)\s*\|\s*"
    r"(?P<frequency>daily|weekly|monthly|once)\s*\|\s*"
    r"(?P<essential>yes|no)\s*\|\s*(?P<start>\d{2}:\d{2})"
    r"(?:\s*\|\s*(?P<category>\w+))?\s*$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Chunk:
    """One section of one guide, with the file it came from so it can be cited."""

    source: str  # file name, e.g. "dog_care.md"
    heading: str  # section title, e.g. "Senior dogs"
    text: str  # the section body

    @property
    def label(self) -> str:
        """Short citation string, e.g. "dog_care.md: Senior dogs"."""
        return f"{self.source}: {self.heading}"

    @property
    def species(self) -> str:
        """Species this chunk is for: "dog", "cat", "other", or "any"."""
        first = Path(self.source).stem.split("_")[0]
        return first if first in {"dog", "cat", "other"} else "any"

    @property
    def prose(self) -> str:
        """The section text without the structured task lines (those are data, not search text)."""
        lines = [ln for ln in self.text.splitlines() if not TASK_LINE.match(ln.strip())]
        return "\n".join(lines)

    def task_suggestions(self) -> List[Dict]:
        """Parse the structured task lines in this chunk into dicts."""
        found = []
        for line in self.text.splitlines():
            m = TASK_LINE.match(line.strip())
            if m:
                found.append({
                    "title": m["title"].strip(),
                    "duration_minutes": int(m["minutes"]),
                    "priority": m["priority"].lower(),
                    "frequency": m["frequency"].lower(),
                    "essential": m["essential"].lower() == "yes",
                    "start_time": m["start"],
                    "category": (m["category"] or "other").lower(),
                })
        return found


def tokenize(text: str) -> List[str]:
    """Lowercase words with stopwords removed, plurals trimmed and synonyms merged."""
    words = []
    for word in re.findall(r"[a-z]+", text.lower()):
        if len(word) < 3 or word in STOPWORDS:
            continue
        if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]
        words.append(SYNONYMS.get(word, word))
    return words


def load_knowledge(directory: Path = KNOWLEDGE_DIR) -> List[Chunk]:
    """Read every .md guide in the directory and split it into "##" sections."""
    chunks: List[Chunk] = []
    for path in sorted(Path(directory).glob("*.md")):
        text = path.read_text(encoding="utf-8")
        # Anything before the first "##" is the file's title and format note.
        for section in re.split(r"^## ", text, flags=re.MULTILINE)[1:]:
            heading, _, body = section.partition("\n")
            chunks.append(Chunk(path.name, heading.strip(), body.strip()))
    logger.info("loaded %d knowledge chunks from %s", len(chunks), directory)
    return chunks


def notes_for_pet(notes: str, pet_name: str, all_pet_names: List[str]) -> str:
    """Keep the sentences of the owner's notes that are about this pet.

    A sentence is kept if it names this pet, or names no pet at all (a general
    note). Sentences that name only other pets are dropped, so a note about
    one pet's medicine does not change the search for another pet.
    """
    kept = []
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", notes):
        named = [
            n for n in all_pet_names
            if re.search(r"\b" + re.escape(n) + r"\b", sentence, re.IGNORECASE)
        ]
        if sentence.strip() and (not named or pet_name in named):
            kept.append(sentence.strip())
    return " ".join(kept)


def build_query(species: str, age: int) -> str:
    """Turn a pet's species and age into a search query naming its life stage."""
    species = species.strip().lower()
    if species in {"dog", "cat"}:
        if age >= (8 if species == "dog" else 10):
            stage = "senior older"
        elif age < 1:
            stage = "puppy kitten young" if species == "dog" else "kitten young"
        else:
            stage = "adult"
    else:
        stage = "small"
    return f"{species} {stage}"


class Retriever:
    """Keyword search over the knowledge chunks."""

    def __init__(self, chunks: Optional[List[Chunk]] = None) -> None:
        self.chunks = chunks if chunks is not None else load_knowledge()
        self._counts = [
            Counter(tokenize(c.heading + " " + c.heading + " " + c.prose)) for c in self.chunks
        ]
        doc_freq: Counter = Counter()
        for counts in self._counts:
            doc_freq.update(counts.keys())
        n = max(len(self.chunks), 1)
        self._idf = {t: math.log(1 + n / df) for t, df in doc_freq.items()}

    def search(
        self, query: str, species: str = "any", notes: str = "",
        k: int = 3, min_score: float = MIN_SCORE,
    ) -> List[Tuple[Chunk, float]]:
        """Return up to k (chunk, score) pairs, best first, for chunks that fit the species.

        query names the pet's species and life stage. Its words count
        STAGE_WEIGHT times; words from the owner's notes count once, and only
        distinctive ones (rare across the guides) are used, so common words
        like "morning" or "walk" do not pull in unrelated guidance. Chunks
        written for another species are skipped, and chunks scoring below
        min_score are dropped, so a query with no real match returns [].
        """
        # The species word is handled by the species filter below, so it must not
        # add score (otherwise every dog chunk matches every dog).
        species = species.strip().lower()
        species_terms = set(tokenize(species))
        stage_terms = set(tokenize(query)) - species_terms
        note_terms = {
            t for t in tokenize(notes) if self._idf.get(t, 0.0) >= NOTE_MIN_IDF
        } - stage_terms - species_terms
        weights = {t: 1.0 for t in note_terms}
        weights.update({t: STAGE_WEIGHT for t in stage_terms})

        scored = []
        for chunk, counts in zip(self.chunks, self._counts):
            if chunk.species not in ("any", species):
                continue
            score = sum(
                w * self._idf.get(t, 0.0) * (1 + math.log(counts[t]))
                for t, w in weights.items() if counts.get(t)
            )
            if score >= min_score:
                scored.append((chunk, round(score, 3)))
        scored.sort(key=lambda pair: -pair[1])
        results = scored[:k]
        logger.info(
            "retrieve query=%r notes=%r species=%s -> %s",
            query, notes, species, [(c.label, s) for c, s in results] or "no match",
        )
        return results
