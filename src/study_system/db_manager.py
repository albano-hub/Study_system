import hashlib
import json
import math
import sqlite3
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from rapidfuzz import fuzz

    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False


def generate_uuid() -> str:
    """
    Generate a time-ordered UUIDv7 (Python 3.14+) or fallback to UUIDv4.
    """
    uuid7_fn = getattr(uuid, "uuid7", None)
    return str(uuid7_fn()) if callable(uuid7_fn) else str(uuid.uuid4())


def utc_now_iso() -> str:
    """Return current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


class StudyBankDB:
    """
    SQLite + JSON storage engine for study items, hierarchical tags,
    and mastery metrics.
    """

    def __init__(self, db_path: str | Path = "data/db/study_bank.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self._configure_pragmas()
        self._init_schema()

    def _configure_pragmas(self) -> None:
        self.conn.execute("PRAGMA foreign_keys = ON;")
        self.conn.execute("PRAGMA journal_mode = WAL;")
        self.conn.execute("PRAGMA synchronous = NORMAL;")
        self.conn.execute("PRAGMA busy_timeout = 5000;")

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "StudyBankDB":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def _init_schema(self) -> None:
        with self.conn:
            self.conn.executescript("""
                CREATE TABLE IF NOT EXISTS items (
                    id TEXT PRIMARY KEY,
                    content_hash TEXT UNIQUE NOT NULL,
                    item_type TEXT NOT NULL CHECK(item_type IN ('question',
                    'content_note')),
                    question_type TEXT CHECK(
                        (item_type = 'content_note' AND question_type IS NULL) OR
                        (item_type = 'question' AND question_type IN (
                            'true_false', 'single_choice', 'multiple_choice',
                            'short_answer', 'open_ended', 'calculation'
                        ))
                    ),
                    major TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    relevancy TEXT NOT NULL CHECK(relevancy IN ('core',
                    'supplementary')),
                    difficulty INTEGER NOT NULL CHECK(difficulty BETWEEN 1 AND 5),
                    source_origin TEXT NOT NULL,
                    source_name TEXT NOT NULL,
                    source_location TEXT,
                    prompt TEXT NOT NULL,
                    options_json TEXT,
                    answer_json TEXT,
                    explanation TEXT,
                    times_seen INTEGER NOT NULL DEFAULT 0,
                    times_right INTEGER NOT NULL DEFAULT 0,
                    times_wrong INTEGER NOT NULL DEFAULT 0,
                    times_skipped INTEGER NOT NULL DEFAULT 0,
                    stability_days REAL NOT NULL DEFAULT 7.0,
                    familiarity_score REAL DEFAULT NULL,
                    mastery_score REAL DEFAULT NULL,
                    last_reviewed TEXT DEFAULT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_items_subject ON items(subject);
                CREATE INDEX IF NOT EXISTS idx_items_major ON items(major);

                CREATE TABLE IF NOT EXISTS tags (
                    tag_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    subject TEXT NOT NULL,
                    name TEXT NOT NULL,
                    full_path TEXT NOT NULL,
                    parent_id INTEGER REFERENCES tags(tag_id) ON DELETE CASCADE,
                    question_count INTEGER NOT NULL DEFAULT 0,
                    tag_familiarity_score REAL DEFAULT NULL,
                    tag_mastery_score REAL DEFAULT NULL,
                    last_updated TEXT DEFAULT NULL,
                    UNIQUE(subject, full_path)
                );

                CREATE INDEX IF NOT EXISTS idx_tags_subject_path ON tags(subject,
                full_path);
                CREATE INDEX IF NOT EXISTS idx_tags_parent ON tags(parent_id);

                CREATE TABLE IF NOT EXISTS item_tags (
                    item_id TEXT NOT NULL REFERENCES items(id) ON DELETE CASCADE,
                    tag_id INTEGER NOT NULL REFERENCES tags(tag_id) ON DELETE CASCADE,
                    PRIMARY KEY (item_id, tag_id)
                );

                CREATE INDEX IF NOT EXISTS idx_item_tags_tag ON item_tags(tag_id);

                CREATE TABLE IF NOT EXISTS item_relations (
                    item_id TEXT NOT NULL REFERENCES items(id) ON DELETE CASCADE,
                    related_label TEXT NOT NULL,
                    PRIMARY KEY (item_id, related_label)
                );

                CREATE INDEX IF NOT EXISTS idx_relations_label ON item_relations
                (related_label);

                CREATE TABLE IF NOT EXISTS system_state (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                INSERT OR IGNORE INTO system_state (key, value)
                VALUES ('exercises_since_tag_sync', '0');
            """)

    @staticmethod
    def compute_hash(item_type: str, prompt: str, answer: Any = None) -> str:
        norm_prompt = " ".join(prompt.lower().split())
        norm_ans = (
            " ".join(json.dumps(answer, sort_keys=True).lower().split())
            if answer is not None
            else ""
        )
        raw = f"{item_type.lower()}::{norm_prompt}::{norm_ans}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def _get_or_create_tag_hierarchy(self, subject: str, full_path: str) -> int:
        parts = [p.strip() for p in full_path.split(">") if p.strip()]
        if not parts:
            raise ValueError(f"Invalid empty tag path: {full_path!r}")

        parent_id: int | None = None
        current_path = ""
        tag_id: int | None = None

        for part in parts:
            current_path = f"{current_path} > {part}" if current_path else part
            row = self.conn.execute(
                "SELECT tag_id FROM tags WHERE subject = ? AND full_path = ?",
                (subject, current_path),
            ).fetchone()

            if row:
                tag_id = int(row["tag_id"])
            else:
                cur = self.conn.execute(
                    """INSERT INTO tags (subject, name, full_path, parent_id)
                       VALUES (?, ?, ?, ?)""",
                    (subject, part, current_path, parent_id),
                )
                tag_id = int(cur.lastrowid)  # type: ignore
            parent_id = tag_id

        assert tag_id is not None
        return tag_id

    @staticmethod
    def _find_semantic_match(
        prompt: str,
        existing_prompts: list[tuple[str, str]],
        threshold: int = 85,
    ) -> dict[str, Any] | None:
        if not HAS_RAPIDFUZZ:
            return None
        prompt_lower = prompt.lower()
        for existing_id, existing_text in existing_prompts:
            score = fuzz.token_set_ratio(prompt_lower, existing_text.lower())
            if score >= threshold:
                return {
                    "existing_id": existing_id,
                    "similarity": round(score, 1),
                    "existing_prompt": existing_text,
                }
        return None

    def import_json_batch(
        self,
        json_path: str | Path,
        check_semantic: bool = True,
        force_similar: bool = False,
        semantic_threshold: int = 85,
    ) -> dict[str, Any]:
        data = json.loads(Path(json_path).read_text(encoding="utf-8"))
        items = data if isinstance(data, list) else [data]
        stats: dict[str, Any] = {
            "inserted": 0,
            "updated": 0,
            "exact_duplicates": 0,
            "similar_flagged": [],
        }

        # Cache existing prompts per subject once to avoid N+1 SELECT queries
        prompt_cache: dict[str, list[tuple[str, str]]] = {}

        with self.conn:
            for item in items:
                item_type = item.get("item_type", "question")
                subject = item["subject"].strip()
                prompt = item["prompt"].strip()
                answer = item.get("answer")
                c_hash = self.compute_hash(item_type, prompt, answer)
                src = item.get("source_reference", {})
                metrics = item.get("metrics", {})
                now = utc_now_iso()

                # 1. Explicit ID upsert (editing an exported item)
                if item.get("id"):
                    existing_by_id = self.conn.execute(
                        "SELECT id FROM items WHERE id = ?", (item["id"],)
                    ).fetchone()
                    if existing_by_id:
                        item_id = str(existing_by_id["id"])
                        self.conn.execute(
                            """UPDATE items SET
                                content_hash = ?, item_type = ?, question_type = ?,
                                major = ?, subject = ?, relevancy = ?, difficulty = ?,
                                source_origin = ?, source_name = ?, source_location = ?,
                                prompt = ?, options_json = ?, answer_json = ?,
                                explanation = ?, updated_at = ?
                               WHERE id = ?""",
                            (
                                c_hash,
                                item_type,
                                item.get("question_type"),
                                item["major"].strip(),
                                subject,
                                item.get("relevancy", "core"),
                                int(item["difficulty"]),
                                src.get("origin", "provided_material"),
                                src.get("source_name", "Unknown"),
                                src.get("location"),
                                prompt,
                                json.dumps(item["options"])
                                if item.get("options") is not None
                                else None,
                                json.dumps(answer) if answer is not None else None,
                                item.get("explanation"),
                                now,
                                item_id,
                            ),
                        )
                        self._sync_item_links(
                            item_id,
                            subject,
                            item.get("subject_tags", []),
                            item.get("related_content_labels", []),
                        )
                        stats["updated"] += 1
                        continue

                # 2. Exact hash duplicate check
                exists = self.conn.execute(
                    "SELECT id FROM items WHERE content_hash = ?", (c_hash,)
                ).fetchone()
                if exists:
                    stats["exact_duplicates"] += 1
                    continue

                # 3. Semantic duplicate check
                # (against DB + earlier items in this same batch)
                if check_semantic and not force_similar:
                    if subject not in prompt_cache:
                        rows = self.conn.execute(
                            "SELECT id, prompt FROM items WHERE subject = ?", (subject,)
                        ).fetchall()
                        prompt_cache[subject] = [(r["id"], r["prompt"]) for r in rows]

                    sim = self._find_semantic_match(
                        prompt, prompt_cache[subject], threshold=semantic_threshold
                    )
                    if sim:
                        stats["similar_flagged"].append({"new_prompt": prompt, **sim})
                        continue

                # 4. Insert new item
                item_id = item.get("id") or generate_uuid()
                seen = int(metrics.get("times_seen", 0))
                right = int(metrics.get("times_right", 0))
                wrong = int(metrics.get("times_wrong", 0))
                skipped = int(metrics.get("times_skipped", 0))
                difficulty = int(item["difficulty"])
                fam, mas = self._calculate_scores(seen, right, wrong, difficulty)

                self.conn.execute(
                    """INSERT INTO items (
                        id, content_hash, item_type, question_type, major, subject,
                        relevancy, difficulty, source_origin, source_name,
                        source_location,
                        prompt, options_json, answer_json, explanation,
                        times_seen, times_right, times_wrong, times_skipped,
                        stability_days, familiarity_score, mastery_score,
                        last_reviewed, created_at, updated_at
                    ) VALUES
                    (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        item_id,
                        c_hash,
                        item_type,
                        item.get("question_type"),
                        item["major"].strip(),
                        subject,
                        item.get("relevancy", "core"),
                        difficulty,
                        src.get("origin", "provided_material"),
                        src.get("source_name", "Unknown"),
                        src.get("location"),
                        prompt,
                        json.dumps(item["options"])
                        if item.get("options") is not None
                        else None,
                        json.dumps(answer) if answer is not None else None,
                        item.get("explanation"),
                        seen,
                        right,
                        wrong,
                        skipped,
                        float(metrics.get("stability_days", 7.0)),
                        fam,
                        mas,
                        metrics.get("last_reviewed"),
                        now,
                        now,
                    ),
                )
                self._sync_item_links(
                    item_id,
                    subject,
                    item.get("subject_tags", []),
                    item.get("related_content_labels", []),
                )
                if subject in prompt_cache:
                    prompt_cache[subject].append((item_id, prompt))
                stats["inserted"] += 1

        if stats["inserted"] > 0 or stats["updated"] > 0:
            self.refresh_tag_metrics(force=True)

        return stats

    def _sync_item_links(
        self,
        item_id: str,
        subject: str,
        subject_tags: list[str],
        related_labels: list[str],
    ) -> None:
        self.conn.execute("DELETE FROM item_tags WHERE item_id = ?", (item_id,))
        self.conn.execute("DELETE FROM item_relations WHERE item_id = ?", (item_id,))

        for tag_path in subject_tags:
            tag_id = self._get_or_create_tag_hierarchy(subject, tag_path)
            self.conn.execute(
                "INSERT OR IGNORE INTO item_tags (item_id, tag_id) VALUES (?, ?)",
                (item_id, tag_id),
            )

        for rel in related_labels:
            clean_rel = rel.strip()
            if clean_rel:
                self.conn.execute(
                    """INSERT OR IGNORE INTO item_relations
                    (item_id, related_label) VALUES (?, ?)""",
                    (item_id, clean_rel),
                )

    @staticmethod
    def _calculate_scores(
        seen: int, right: int, wrong: int, difficulty: int
    ) -> tuple[float | None, float | None]:
        """Compute familiarity and difficulty-weighted mastery once seen >= 5."""
        if seen < 5:
            return None, None
        effective_attempts = right + wrong
        familiarity = (
            round(right / effective_attempts, 4) if effective_attempts > 0 else 0.0
        )
        difficulty_weight = 0.6 + (0.4 * (difficulty / 5.0))
        mastery = round(familiarity * difficulty_weight, 4)
        return familiarity, mastery

    def record_attempt(self, item_id: str, outcome: str) -> None:
        """Record a study attempt ('right', 'wrong', or 'skipped')."""
        if outcome not in {"right", "wrong", "skipped"}:
            raise ValueError(f"Invalid outcome: {outcome!r}")

        row = self.conn.execute(
            "SELECT * FROM items WHERE id = ?", (item_id,)
        ).fetchone()
        if not row:
            raise ValueError(f"Item {item_id!r} not found.")

        seen = int(row["times_seen"]) + 1
        right = int(row["times_right"]) + (1 if outcome == "right" else 0)
        wrong = int(row["times_wrong"]) + (1 if outcome == "wrong" else 0)
        skipped = int(row["times_skipped"]) + (1 if outcome == "skipped" else 0)

        stability = float(row["stability_days"])
        if outcome == "right":
            stability = min(180.0, stability * 1.6)
        elif outcome == "wrong":
            stability = max(2.0, stability * 0.5)

        familiarity, mastery = self._calculate_scores(
            seen, right, wrong, int(row["difficulty"])
        )
        now = utc_now_iso()

        with self.conn:
            self.conn.execute(
                """UPDATE items SET
                    times_seen = ?, times_right = ?, times_wrong = ?,
                    times_skipped = ?,
                    stability_days = ?, familiarity_score = ?, mastery_score = ?,
                    last_reviewed = ?, updated_at = ?
                   WHERE id = ?""",
                (
                    seen,
                    right,
                    wrong,
                    skipped,
                    stability,
                    familiarity,
                    mastery,
                    now,
                    now,
                    item_id,
                ),
            )
            self.conn.execute(
                """UPDATE system_state
                   SET value = CAST(CAST(value AS INTEGER) + 1 AS TEXT)
                   WHERE key = 'exercises_since_tag_sync'"""
            )

    @staticmethod
    def effective_mastery_with_decay(
        row: sqlite3.Row, now_dt: datetime | None = None
    ) -> float | None:
        if row["mastery_score"] is None or not row["last_reviewed"]:
            return row["mastery_score"]
        ref_now = now_dt or datetime.now(timezone.utc)
        last_dt = datetime.fromisoformat(row["last_reviewed"])
        days_elapsed = max(0.0, (ref_now - last_dt).total_seconds() / 86400.0)
        decay_multiplier = math.exp(-days_elapsed / float(row["stability_days"]))
        return round(float(row["mastery_score"]) * decay_multiplier, 4)

    def refresh_tag_metrics(
        self, batch_threshold: int = 15, stale_hours: int = 24, force: bool = False
    ) -> bool:
        pending = int(
            self.conn.execute(
                "SELECT value FROM system_state WHERE key = 'exercises_since_tag_sync'"
            ).fetchone()["value"]
        )
        ref_now = datetime.now(timezone.utc)

        if not force and pending < batch_threshold:
            oldest = self.conn.execute(
                "SELECT MIN(last_updated) AS oldest FROM tags"
            ).fetchone()["oldest"]
            if oldest:
                age_hours = (
                    ref_now - datetime.fromisoformat(oldest)
                ).total_seconds() / 3600.0
                if age_hours < stale_hours:
                    return False

        now_iso = ref_now.isoformat()
        tags = self.conn.execute(
            "SELECT tag_id, subject, full_path FROM tags"
        ).fetchall()

        with self.conn:
            for tag in tags:
                items = self.conn.execute(
                    """SELECT DISTINCT i.* FROM items i
                       JOIN item_tags it ON i.id = it.item_id
                       JOIN tags t ON it.tag_id = t.tag_id
                       WHERE t.subject = ? AND (t.full_path = ?
                       OR t.full_path LIKE ?)""",
                    (tag["subject"], tag["full_path"], f"{tag['full_path']} > %"),
                ).fetchall()

                q_count = len(items)
                rated = [i for i in items if i["mastery_score"] is not None]

                if rated:
                    # Weight 'core' items 1.25x vs 'supplementary' 1.0x
                    weights = [1.25 if i["relevancy"] == "core" else 1.0 for i in rated]
                    total_weight = sum(weights)

                    avg_fam = round(
                        sum(
                            float(i["familiarity_score"]) * w
                            for i, w in zip(rated, weights)
                        )
                        / total_weight,
                        4,
                    )
                    avg_mas = round(
                        sum(
                            (self.effective_mastery_with_decay(i, ref_now) or 0.0) * w
                            for i, w in zip(rated, weights)
                        )
                        / total_weight,
                        4,
                    )
                else:
                    avg_fam = None
                    avg_mas = None

                self.conn.execute(
                    """UPDATE tags SET
                        question_count = ?, tag_familiarity_score = ?,
                        tag_mastery_score = ?, last_updated = ?
                       WHERE tag_id = ?""",
                    (q_count, avg_fam, avg_mas, now_iso, tag["tag_id"]),
                )

            # Prune orphaned tags that no longer have any associated questions
            self.conn.execute("DELETE FROM tags WHERE question_count = 0")

            self.conn.execute(
                """UPDATE system_state SET value = '0'
                WHERE key = 'exercises_since_tag_sync'"""
            )
        return True

    def export_to_json(
        self, output_path: str | Path, subject: str | None = None
    ) -> int:
        query = "SELECT * FROM items"
        params: tuple[Any, ...] = ()
        if subject:
            query += " WHERE subject = ?"
            params = (subject,)

        rows = self.conn.execute(query, params).fetchall()
        if not rows:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text("[]\n", encoding="utf-8")
            return 0

        # Bulk-fetch all tags and relations in 2 queries instead of 2*N queries
        tags_by_item: dict[str, list[str]] = defaultdict(list)
        for r in self.conn.execute(
            """SELECT it.item_id, t.full_path
               FROM item_tags it
               JOIN tags t ON t.tag_id = it.tag_id"""
        ):
            tags_by_item[r["item_id"]].append(r["full_path"])

        rels_by_item: dict[str, list[str]] = defaultdict(list)
        for r in self.conn.execute("SELECT item_id, related_label FROM item_relations"):
            rels_by_item[r["item_id"]].append(r["related_label"])

        ref_now = datetime.now(timezone.utc)
        exported: list[dict[str, Any]] = []

        for r in rows:
            item_id = r["id"]
            exported.append(
                {
                    "id": item_id,
                    "content_hash": r["content_hash"],
                    "major": r["major"],
                    "subject": r["subject"],
                    "item_type": r["item_type"],
                    "question_type": r["question_type"],
                    "subject_tags": tags_by_item[item_id],
                    "related_content_labels": rels_by_item[item_id],
                    "relevancy": r["relevancy"],
                    "difficulty": r["difficulty"],
                    "source_reference": {
                        "origin": r["source_origin"],
                        "source_name": r["source_name"],
                        "location": r["source_location"],
                    },
                    "prompt": r["prompt"],
                    "options": (
                        json.loads(r["options_json"])
                        if r["options_json"] is not None
                        else None
                    ),
                    "answer": (
                        json.loads(r["answer_json"])
                        if r["answer_json"] is not None
                        else None
                    ),
                    "explanation": r["explanation"],
                    "metrics": {
                        "times_seen": r["times_seen"],
                        "times_right": r["times_right"],
                        "times_wrong": r["times_wrong"],
                        "times_skipped": r["times_skipped"],
                        "stability_days": r["stability_days"],
                        "familiarity_score": r["familiarity_score"],
                        "mastery_score": self.effective_mastery_with_decay(r, ref_now),
                        "last_reviewed": r["last_reviewed"],
                    },
                }
            )

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8", newline="\n") as f:
            json.dump(exported, f, indent=2, ensure_ascii=False)
            f.write("\n")

        return len(exported)
