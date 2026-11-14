from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from rapidfuzz import fuzz

    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "db" / "study_bank.db"

VALID_ITEM_TYPES = {"content_note", "question"}
VALID_QUESTION_TYPES = {
    "true_false",
    "single_choice",
    "multiple_choice",
    "short_answer",
    "open_ended",
    "calculation",
}
VALID_RELEVANCY = {"core", "supplementary", "contextual"}


class StudyBankDB:
    """SQLite database manager for the Study System item bank and telemetry."""

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON;")
        self.conn.execute("PRAGMA journal_mode = WAL;")
        self._init_schema()

    def _init_schema(self) -> None:
        with self.conn:
            self.conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS items (
                    id TEXT PRIMARY KEY,
                    major TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    item_type TEXT NOT NULL CHECK(item_type IN ('content_note', 'question')),
                    question_type TEXT,
                    related_content_labels_json TEXT NOT NULL DEFAULT '[]',
                    relevancy TEXT NOT NULL CHECK(relevancy IN ('core', 'supplementary', 'contextual')),
                    difficulty INTEGER NOT NULL CHECK(difficulty BETWEEN 1 AND 5),
                    source_origin TEXT NOT NULL,
                    source_name TEXT,
                    source_location TEXT,
                    prompt TEXT NOT NULL,
                    prompt_hash TEXT NOT NULL UNIQUE,
                    options_json TEXT,
                    answer_json TEXT,
                    explanation TEXT,
                    times_seen INTEGER NOT NULL DEFAULT 0,
                    times_right INTEGER NOT NULL DEFAULT 0,
                    times_wrong INTEGER NOT NULL DEFAULT 0,
                    times_skipped INTEGER NOT NULL DEFAULT 0,
                    last_seen_at TEXT,
                    stability_days REAL NOT NULL DEFAULT 3.0,
                    familiarity_score REAL,
                    mastery_score REAL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS tags (
                    tag_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    major TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    name TEXT NOT NULL,
                    full_path TEXT NOT NULL UNIQUE,
                    parent_id INTEGER REFERENCES tags(tag_id) ON DELETE CASCADE,
                    level INTEGER NOT NULL DEFAULT 0,
                    question_count INTEGER NOT NULL DEFAULT 0,
                    rated_question_count INTEGER NOT NULL DEFAULT 0,
                    familiarity_score REAL,
                    mastery_score REAL
                );

                CREATE TABLE IF NOT EXISTS item_tags (
                    item_id TEXT NOT NULL REFERENCES items(id) ON DELETE CASCADE,
                    tag_id INTEGER NOT NULL REFERENCES tags(tag_id) ON DELETE CASCADE,
                    PRIMARY KEY (item_id, tag_id)
                );

                CREATE INDEX IF NOT EXISTS idx_items_major_subject ON items(major, subject);
                CREATE INDEX IF NOT EXISTS idx_tags_parent ON tags(parent_id);
                """
            )

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join(text.strip().lower().split())

    @classmethod
    def compute_prompt_hash(cls, major: str, subject: str, prompt: str) -> str:
        payload = f"{cls._normalize_text(major)}||{cls._normalize_text(subject)}||{cls._normalize_text(prompt)}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _find_semantic_match(
        prompt: str,
        existing_prompts: list[tuple[str, str]],
        threshold: float = 88.0,
    ) -> dict[str, Any] | None:
        if not HAS_RAPIDFUZZ:
            return None
        prompt_lower = prompt.lower()
        for existing_id, existing_text in existing_prompts:
            existing_lower = existing_text.lower()
            set_score = fuzz.token_set_ratio(prompt_lower, existing_lower)
            sort_score = fuzz.token_sort_ratio(prompt_lower, existing_lower)
            combined_score = (0.5 * set_score) + (0.5 * sort_score)
            if combined_score >= threshold:
                return {
                    "existing_id": existing_id,
                    "similarity": round(combined_score, 1),
                    "existing_prompt": existing_text,
                }
        return None

    @staticmethod
    def _validate_item(item: dict[str, Any]) -> None:
        required_fields = {
            "major",
            "subject",
            "item_type",
            "subject_tags",
            "relevancy",
            "difficulty",
            "source_reference",
            "prompt",
        }
        missing = required_fields - set(item.keys())
        if missing:
            raise ValueError(f"Missing required fields: {sorted(missing)}")

        item_type = item["item_type"]
        if item_type not in VALID_ITEM_TYPES:
            raise ValueError(f"Invalid item_type: {item_type}")

        q_type = item.get("question_type")
        if item_type == "content_note":
            if q_type is not None:
                raise ValueError("content_note items must have question_type=None")
        else:
            if q_type not in VALID_QUESTION_TYPES:
                raise ValueError(f"Invalid question_type for question: {q_type}")

        if item["relevancy"] not in VALID_RELEVANCY:
            raise ValueError(f"Invalid relevancy: {item['relevancy']}")

        diff = item["difficulty"]
        if not isinstance(diff, int) or not (1 <= diff <= 5):
            raise ValueError(
                f"difficulty must be an integer between 1 and 5, got {diff}"
            )

        if not isinstance(item["subject_tags"], list) or not item["subject_tags"]:
            raise ValueError(
                "subject_tags must be a non-empty list of hierarchical tag paths"
            )

        if item_type == "question":
            options = item.get("options")
            if q_type in ("true_false", "single_choice", "multiple_choice"):
                if not isinstance(options, list) or len(options) < 2:
                    raise ValueError(
                        f"Question type '{q_type}' requires an 'options' list with >= 2 choices."
                    )
                # Suporte ao novo esquema de tuplos: [["Texto da opção", True/False], ...]
                if all(
                    isinstance(opt, (list, tuple))
                    and len(opt) == 2
                    and isinstance(opt[0], str)
                    and isinstance(opt[1], bool)
                    for opt in options
                ):
                    correct_texts = [opt[0] for opt in options if opt[1]]
                    if (
                        q_type in ("true_false", "single_choice")
                        and len(correct_texts) != 1
                    ):
                        raise ValueError(
                            f"Question type '{q_type}' must have exactly 1 option marked True."
                        )
                    if q_type == "multiple_choice" and len(correct_texts) < 1:
                        raise ValueError(
                            "Question type 'multiple_choice' must have at least 1 option marked True."
                        )
                    item["answer"] = (
                        correct_texts[0]
                        if q_type != "multiple_choice"
                        else correct_texts
                    )
                else:
                    if item.get("answer") is None:
                        raise ValueError(
                            "Question items must have a non-null 'answer'."
                        )
                    if q_type == "multiple_choice" and not isinstance(
                        item["answer"], list
                    ):
                        raise ValueError("multiple_choice 'answer' must be a list.")
            else:
                if item.get("answer") is None:
                    raise ValueError("Question items must have a non-null 'answer'.")

    def _ensure_tag_hierarchy(
        self, major: str, subject: str, path_str: str
    ) -> list[int]:
        parts = [p.strip() for p in path_str.split(">") if p.strip()]
        tag_ids: list[int] = []
        parent_id: int | None = None
        current_path_parts: list[str] = []

        for level, part in enumerate(parts):
            current_path_parts.append(part)
            full_path = " > ".join(current_path_parts)
            row = self.conn.execute(
                "SELECT tag_id FROM tags WHERE full_path = ?", (full_path,)
            ).fetchone()
            if row:
                tag_id = int(row["tag_id"])
            else:
                cur = self.conn.execute(
                    """
                    INSERT INTO tags (major, subject, name, full_path, parent_id, level)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (major, subject, part, full_path, parent_id, level),
                )
                tag_id = int(cur.lastrowid)
            tag_ids.append(tag_id)
            parent_id = tag_id

        return tag_ids

    def add_item(
        self,
        item: dict[str, Any],
        *,
        force_similar: bool = False,
        semantic_threshold: float = 88.0,
    ) -> dict[str, Any]:
        self._validate_item(item)

        major = item["major"].strip()
        subject = item["subject"].strip()
        prompt = item["prompt"].strip()
        p_hash = self.compute_prompt_hash(major, subject, prompt)

        existing = self.conn.execute(
            "SELECT id FROM items WHERE prompt_hash = ?", (p_hash,)
        ).fetchone()
        if existing:
            return {"status": "exact_duplicate", "id": existing["id"]}

        if not force_similar and HAS_RAPIDFUZZ:
            rows = self.conn.execute(
                "SELECT id, prompt FROM items WHERE major = ? AND subject = ?",
                (major, subject),
            ).fetchall()
            existing_prompts = [(r["id"], r["prompt"]) for r in rows]
            match = self._find_semantic_match(
                prompt, existing_prompts, semantic_threshold
            )
            if match:
                return {"status": "similar_flagged", **match}

        item_id = item.get("id") or str(uuid.uuid4())
        src = item.get("source_reference") or {}
        now_iso = datetime.now(timezone.utc).isoformat()

        with self.conn:
            self.conn.execute(
                """
                INSERT INTO items (
                    id, major, subject, item_type, question_type,
                    related_content_labels_json, relevancy, difficulty,
                    source_origin, source_name, source_location,
                    prompt, prompt_hash, options_json, answer_json, explanation,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item_id,
                    major,
                    subject,
                    item["item_type"],
                    item.get("question_type"),
                    json.dumps(
                        item.get("related_content_labels", []), ensure_ascii=False
                    ),
                    item["relevancy"],
                    item["difficulty"],
                    src.get("origin", "synthesized"),
                    src.get("source_name"),
                    src.get("source_location"),
                    prompt,
                    p_hash,
                    json.dumps(item.get("options"), ensure_ascii=False)
                    if item.get("options") is not None
                    else None,
                    json.dumps(item.get("answer"), ensure_ascii=False)
                    if item.get("answer") is not None
                    else None,
                    item.get("explanation"),
                    now_iso,
                ),
            )

            all_tag_ids: set[int] = set()
            for tag_path in item["subject_tags"]:
                all_tag_ids.update(self._ensure_tag_hierarchy(major, subject, tag_path))

            for tid in all_tag_ids:
                self.conn.execute(
                    "INSERT OR IGNORE INTO item_tags (item_id, tag_id) VALUES (?, ?)",
                    (item_id, tid),
                )

        return {"status": "inserted", "id": item_id}

    def import_json_batch(
        self,
        json_path: str | Path,
        *,
        force_similar: bool = False,
        semantic_threshold: float = 88.0,
    ) -> dict[str, Any]:
        path = Path(json_path)
        items = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(items, list):
            raise ValueError("JSON batch file must contain a top-level list of items.")

        summary: dict[str, Any] = {
            "file": str(path),
            "total": len(items),
            "inserted": 0,
            "exact_duplicates": 0,
            "similar_flagged": [],
            "errors": [],
        }

        for idx, item in enumerate(items):
            try:
                res = self.add_item(
                    item,
                    force_similar=force_similar,
                    semantic_threshold=semantic_threshold,
                )
                if res["status"] == "inserted":
                    summary["inserted"] += 1
                elif res["status"] == "exact_duplicate":
                    summary["exact_duplicates"] += 1
                elif res["status"] == "similar_flagged":
                    summary["similar_flagged"].append({"index": idx, **res})
            except Exception as exc:
                summary["errors"].append({"index": idx, "error": str(exc)})

        self.refresh_tag_metrics()
        return summary

    def record_attempt(self, item_id: str, outcome: str) -> dict[str, Any]:
        if outcome not in ("right", "wrong", "skipped"):
            raise ValueError(f"Invalid outcome: {outcome}")

        row = self.conn.execute(
            "SELECT * FROM items WHERE id = ?", (item_id,)
        ).fetchone()
        if not row:
            raise KeyError(f"Item not found: {item_id}")

        times_seen = int(row["times_seen"]) + 1
        times_right = int(row["times_right"]) + (1 if outcome == "right" else 0)
        times_wrong = int(row["times_wrong"]) + (1 if outcome == "wrong" else 0)
        times_skipped = int(row["times_skipped"]) + (1 if outcome == "skipped" else 0)
        stability = float(row["stability_days"])

        if outcome == "right":
            stability = min(365.0, stability * 1.6)
        elif outcome == "wrong":
            stability = max(1.0, stability * 0.5)

        now_iso = datetime.now(timezone.utc).isoformat()
        familiarity = None
        mastery = None
        if times_seen >= 5:
            denom = max(1, times_right + times_wrong)
            acc = times_right / denom
            conf = 1.0 - math.exp(-times_seen / 10.0)
            familiarity = round(100.0 * acc * conf, 2)
            mastery = familiarity

        with self.conn:
            self.conn.execute(
                """
                UPDATE items
                SET times_seen = ?,
                    times_right = ?,
                    times_wrong = ?,
                    times_skipped = ?,
                    last_seen_at = ?,
                    stability_days = ?,
                    familiarity_score = ?,
                    mastery_score = ?
                WHERE id = ?
                """,
                (
                    times_seen,
                    times_right,
                    times_wrong,
                    times_skipped,
                    now_iso,
                    stability,
                    familiarity,
                    mastery,
                    item_id,
                ),
            )

        return {
            "id": item_id,
            "times_seen": times_seen,
            "familiarity_score": familiarity,
            "mastery_score": mastery,
        }

    def refresh_tag_metrics(self) -> None:
        tags = self.conn.execute("SELECT tag_id FROM tags").fetchall()
        with self.conn:
            for t in tags:
                tid = t["tag_id"]
                rows = self.conn.execute(
                    """
                    SELECT i.familiarity_score, i.mastery_score
                    FROM items i
                    JOIN item_tags it ON i.id = it.item_id
                    WHERE it.tag_id = ? AND i.item_type = 'question'
                    """,
                    (tid,),
                ).fetchall()
                q_count = len(rows)
                rated = [r for r in rows if r["mastery_score"] is not None]
                r_count = len(rated)
                if r_count > 0:
                    avg_fam = round(
                        sum(float(r["familiarity_score"]) for r in rated) / r_count, 2
                    )
                    avg_mas = round(
                        sum(float(r["mastery_score"]) for r in rated) / r_count, 2
                    )
                else:
                    avg_fam = None
                    avg_mas = None

                self.conn.execute(
                    """
                    UPDATE tags
                    SET question_count = ?,
                        rated_question_count = ?,
                        familiarity_score = ?,
                        mastery_score = ?
                    WHERE tag_id = ?
                    """,
                    (q_count, r_count, avg_fam, avg_mas, tid),
                )

    def close(self) -> None:
        self.conn.close()
