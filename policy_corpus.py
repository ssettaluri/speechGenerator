"""
Step 4 — Policy corpus tool.
Stores historical speeches with alignment, topic, and sentiment.
Provides retrieval by topic and/or alignment to feed context into the agent loop.
Database: SQLite at corpus.db (local, no external deps beyond stdlib).
"""

import os
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Respect DB_PATH env var so Docker can mount a persistent volume
DB_PATH = Path(os.environ.get("DB_PATH", Path(__file__).parent / "corpus.db"))

VALID_SENTIMENTS = {"positive", "neutral", "negative", "inspirational", "cautionary"}
VALID_ALIGNMENTS = {"left", "center-left", "center", "center-right", "right"}


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

DDL = """
CREATE TABLE IF NOT EXISTS speeches (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    alignment   TEXT    NOT NULL CHECK(alignment IN ('left','center-left','center','center-right','right')),
    topic       TEXT    NOT NULL,
    sentiment   TEXT    NOT NULL CHECK(sentiment IN ('positive','neutral','negative','inspirational','cautionary')),
    sample_text TEXT    NOT NULL,
    source      TEXT,
    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_alignment ON speeches(alignment);
CREATE INDEX IF NOT EXISTS idx_topic     ON speeches(topic);
CREATE INDEX IF NOT EXISTS idx_sentiment ON speeches(sentiment);
"""


# ---------------------------------------------------------------------------
# Connection helper
# ---------------------------------------------------------------------------

@contextmanager
def _conn():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    finally:
        con.close()


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Speech:
    id: int
    alignment: str
    topic: str
    sentiment: str
    sample_text: str
    source: Optional[str] = None


def _row_to_speech(row: sqlite3.Row) -> Speech:
    return Speech(
        id=row["id"],
        alignment=row["alignment"],
        topic=row["topic"],
        sentiment=row["sentiment"],
        sample_text=row["sample_text"],
        source=row["source"],
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def init_db() -> None:
    """Create tables and indexes if they don't exist."""
    with _conn() as con:
        con.executescript(DDL)


def add_speech(alignment: str, topic: str, sentiment: str, sample_text: str, source: Optional[str] = None) -> int:
    """Insert a speech and return its new id."""
    if alignment not in VALID_ALIGNMENTS:
        raise ValueError(f"alignment must be one of {VALID_ALIGNMENTS}")
    if sentiment not in VALID_SENTIMENTS:
        raise ValueError(f"sentiment must be one of {VALID_SENTIMENTS}")
    with _conn() as con:
        cur = con.execute(
            "INSERT INTO speeches (alignment, topic, sentiment, sample_text, source) VALUES (?,?,?,?,?)",
            (alignment, topic, sentiment, sample_text, source),
        )
        return cur.lastrowid


def query_speeches(
    topic: Optional[str] = None,
    alignment: Optional[str] = None,
    sentiment: Optional[str] = None,
    limit: int = 5,
) -> list[Speech]:
    """
    Retrieve speeches filtered by topic (substring), alignment, and/or sentiment.
    Returns up to `limit` results ordered by relevance (topic match first).
    """
    clauses, params = [], []
    if topic:
        clauses.append("topic LIKE ?")
        params.append(f"%{topic}%")
    if alignment:
        clauses.append("alignment = ?")
        params.append(alignment)
    if sentiment:
        clauses.append("sentiment = ?")
        params.append(sentiment)

    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    sql = f"SELECT * FROM speeches {where} ORDER BY id DESC LIMIT ?"
    params.append(limit)

    with _conn() as con:
        rows = con.execute(sql, params).fetchall()
    return [_row_to_speech(r) for r in rows]


def get_speech(speech_id: int) -> Optional[Speech]:
    with _conn() as con:
        row = con.execute("SELECT * FROM speeches WHERE id=?", (speech_id,)).fetchone()
    return _row_to_speech(row) if row else None


def delete_speech(speech_id: int) -> bool:
    with _conn() as con:
        cur = con.execute("DELETE FROM speeches WHERE id=?", (speech_id,))
    return cur.rowcount > 0


# ---------------------------------------------------------------------------
# Seed data
# ---------------------------------------------------------------------------

SEED_SPEECHES = [
    {
        "alignment": "left",
        "topic": "healthcare",
        "sentiment": "inspirational",
        "sample_text": (
            "Healthcare is not a privilege reserved for the wealthy — it is a fundamental human right. "
            "When a mother cannot afford insulin for her child, that is a moral failure of our society. "
            "We must build a system where every person, regardless of income or zip code, can walk into "
            "a doctor's office and receive the care they deserve. The richest nation on earth can afford "
            "to guarantee that no one goes bankrupt because they got sick."
        ),
        "source": "Seed corpus v1",
    },
    {
        "alignment": "center-left",
        "topic": "healthcare",
        "sentiment": "positive",
        "sample_text": (
            "Expanding access to affordable healthcare strengthens our entire economy. "
            "When workers aren't locked into a job just to keep their insurance, they can start businesses, "
            "pursue better opportunities, and drive innovation. A public option alongside private coverage "
            "gives Americans real choice while preserving the competition that keeps costs in check. "
            "This is pragmatic progress, not ideology."
        ),
        "source": "Seed corpus v1",
    },
    {
        "alignment": "center",
        "topic": "healthcare",
        "sentiment": "neutral",
        "sample_text": (
            "The American healthcare system faces real challenges: rising premiums, surprise billing, "
            "and gaps in rural coverage. Any durable reform must control costs without sacrificing quality, "
            "preserve patient choice, and be fiscally sustainable. Both market-based solutions and targeted "
            "government programs have roles to play. The goal is a system that works for patients, "
            "providers, and taxpayers alike."
        ),
        "source": "Seed corpus v1",
    },
    {
        "alignment": "center-right",
        "topic": "healthcare",
        "sentiment": "cautionary",
        "sample_text": (
            "Government takeover of healthcare would eliminate the competition that drives medical innovation. "
            "The United States leads the world in new treatments and pharmaceuticals precisely because "
            "the private sector has incentive to invest. We should lower costs through transparency, "
            "tort reform, and allowing insurance to be sold across state lines — not through a bureaucratic "
            "system that rations care and stifles breakthroughs."
        ),
        "source": "Seed corpus v1",
    },
    {
        "alignment": "right",
        "topic": "healthcare",
        "sentiment": "cautionary",
        "sample_text": (
            "Every nation with government-run healthcare faces the same problems: long wait times, "
            "limited choice, and mediocre outcomes. The free market, empowered by health savings accounts "
            "and deregulation, will lower prices far more effectively than any Washington mandate. "
            "Our goal must be a system built on personal responsibility, competition, and individual freedom — "
            "not dependency on a one-size-fits-all federal program."
        ),
        "source": "Seed corpus v1",
    },
    {
        "alignment": "left",
        "topic": "education",
        "sentiment": "inspirational",
        "sample_text": (
            "Every child deserves a world-class education, not just those born into wealth. "
            "We must fully fund our public schools, cancel the burden of student debt, and make "
            "community college free. Education is the great equalizer — but only if we invest in it "
            "as a public good, not treat it as a commodity."
        ),
        "source": "Seed corpus v1",
    },
    {
        "alignment": "center",
        "topic": "education",
        "sentiment": "neutral",
        "sample_text": (
            "Improving educational outcomes requires honest data, accountability, and investment. "
            "We should reward excellent teachers, expand vocational training alongside college pathways, "
            "and ensure that funding follows students' needs. Neither blanket spending increases nor "
            "wholesale privatization has proven sufficient on its own — evidence-based reform is the way forward."
        ),
        "source": "Seed corpus v1",
    },
    {
        "alignment": "right",
        "topic": "education",
        "sentiment": "positive",
        "sample_text": (
            "School choice empowers parents and forces schools to compete for students by actually "
            "delivering results. Charter schools, voucher programs, and education savings accounts "
            "have lifted outcomes in community after community. When government monopolies are replaced "
            "by accountability to families, every school improves."
        ),
        "source": "Seed corpus v1",
    },
    {
        "alignment": "left",
        "topic": "climate",
        "sentiment": "cautionary",
        "sample_text": (
            "The climate crisis is not a future threat — it is here now, burning our forests, "
            "flooding our coastlines, and displacing millions. Fossil fuel corporations have spent "
            "decades funding denial while pocketing billions. A Green New Deal is not optional; "
            "it is the minimum response to an existential emergency that demands we transform our "
            "economy with the urgency of wartime mobilization."
        ),
        "source": "Seed corpus v1",
    },
    {
        "alignment": "center-right",
        "topic": "climate",
        "sentiment": "neutral",
        "sample_text": (
            "Climate change is a real challenge that demands market-driven solutions, not economy-killing "
            "mandates. A revenue-neutral carbon tax that returns money to citizens encourages innovation "
            "without picking winners and losers. American ingenuity in nuclear, natural gas, and "
            "renewables can lead the world — if we get government red tape out of the way."
        ),
        "source": "Seed corpus v1",
    },
    {
        "alignment": "center-left",
        "topic": "immigration",
        "sentiment": "positive",
        "sample_text": (
            "Immigrants have always been the engine of American renewal. They start businesses, "
            "fill critical roles in healthcare and agriculture, and enrich our culture. "
            "A fair immigration system secures our borders while creating a clear, humane pathway "
            "to legal status for the millions who have built their lives here and contribute every day."
        ),
        "source": "Seed corpus v1",
    },
    {
        "alignment": "right",
        "topic": "immigration",
        "sentiment": "cautionary",
        "sample_text": (
            "A nation without borders is not a nation. Every sovereign country has the right — "
            "and the duty — to control who enters. Unchecked illegal immigration strains public "
            "services, undermines wages for American workers, and creates security risks. "
            "We welcome legal immigrants who follow the rules; we must enforce the law for those who do not."
        ),
        "source": "Seed corpus v1",
    },
]


def seed_db() -> int:
    """Insert seed speeches, skipping any that already exist (by sample_text). Returns count inserted."""
    init_db()
    inserted = 0
    with _conn() as con:
        for s in SEED_SPEECHES:
            exists = con.execute(
                "SELECT 1 FROM speeches WHERE sample_text=?", (s["sample_text"],)
            ).fetchone()
            if not exists:
                con.execute(
                    "INSERT INTO speeches (alignment, topic, sentiment, sample_text, source) VALUES (?,?,?,?,?)",
                    (s["alignment"], s["topic"], s["sentiment"], s["sample_text"], s.get("source")),
                )
                inserted += 1
    return inserted


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    init_db()
    n = seed_db()
    print(f"Seeded {n} new speeches into {DB_PATH}\n")

    print("--- Healthcare speeches (any alignment) ---")
    for sp in query_speeches(topic="healthcare", limit=3):
        print(f"  [{sp.alignment:12}] [{sp.sentiment:13}] {sp.sample_text[:80]}...")

    print("\n--- Left-aligned speeches ---")
    for sp in query_speeches(alignment="left", limit=3):
        print(f"  [{sp.topic:12}] [{sp.sentiment:13}] {sp.sample_text[:80]}...")
