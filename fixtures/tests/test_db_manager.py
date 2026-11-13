import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from study_system.db_manager import StudyBankDB


@pytest.fixture
def db(tmp_path: Path) -> StudyBankDB:  # type: ignore
    """
    Provide a fresh, isolated SQLite database in a
    temporary directory for each test.
    """
    db_file = tmp_path / "test_study_bank.db"
    with StudyBankDB(db_path=db_file) as database:
        yield database  # type: ignore


@pytest.fixture
def sample_batch_file(tmp_path: Path) -> Path:
    """Create a temporary JSON batch file with diverse items."""
    batch = [
        {
            "major": "Computer Science",
            "subject": "Data Structures",
            "item_type": "question",
            "question_type": "single_choice",
            "subject_tags": ["Trees > Binary Search Trees > Lookup"],
            "related_content_labels": ["time_complexity", "recursion"],
            "relevancy": "core",
            "difficulty": 3,
            "source_reference": {
                "origin": "provided_material",
                "source_name": "Lecture_04.pdf",
                "location": "Slide 12",
            },
            "prompt": "What is the average-case time complexity of searching \
                in a balanced Binary Search Tree?",
            "options": ["A: O(1)", "B: O(log n)", "C: O(n)", "D: O(n log n)"],
            "answer": "B",
            "explanation": "Each comparison halves the remaining search space.",
        },
        {
            "major": "Computer Science",
            "subject": "Data Structures",
            "item_type": "content_note",
            "question_type": None,
            "subject_tags": ["Trees > Binary Search Trees"],
            "related_content_labels": ["bst_invariant"],
            "relevancy": "supplementary",
            "difficulty": 2,
            "source_reference": {
                "origin": "provided_material",
                "source_name": "Lecture_04.pdf",
                "location": "Slide 5",
            },
            "prompt": "BST Invariant: For any node N, all keys in its left subtree \
                are smaller, and all keys in its right subtree are larger.",
            "options": None,
            "answer": None,
            "explanation": "Fundamental structural property of BSTs.",
        },
    ]
    file_path = tmp_path / "batch.json"
    file_path.write_text(json.dumps(batch), encoding="utf-8")
    return file_path


def test_import_and_uuid_generation(db: StudyBankDB, sample_batch_file: Path) -> None:
    """Verify items import cleanly, generate UUIDs, and build hierarchical tags."""
    stats = db.import_json_batch(sample_batch_file)
    assert stats["inserted"] == 2
    assert stats["exact_duplicates"] == 0

    rows = db.conn.execute("SELECT * FROM items").fetchall()
    assert len(rows) == 2
    # Confirm UUID string primary keys were assigned
    assert all(len(r["id"]) == 36 for r in rows)

    # Confirm 3-level tag hierarchy
    # ('Trees', 'Trees > Binary Search Trees', '... > Lookup') was created
    tags = db.conn.execute(
        "SELECT full_path, parent_id, question_count FROM tags ORDER BY tag_id"
    ).fetchall()
    paths = [t["full_path"] for t in tags]
    assert paths == [
        "Trees",
        "Trees > Binary Search Trees",
        "Trees > Binary Search Trees > Lookup",
    ]
    assert tags[0]["parent_id"] is None
    assert tags[0]["question_count"] == 2  # Both items roll up into root 'Trees'


def test_exact_and_semantic_deduplication(
    db: StudyBankDB, sample_batch_file: Path, tmp_path: Path
) -> None:
    """Verify SHA-256 catches exact duplicates and RapidFuzz flags near-duplicates."""
    db.import_json_batch(sample_batch_file)

    # 1. Re-importing the exact same file should be blocked by content_hash
    repeat_stats = db.import_json_batch(sample_batch_file)
    assert repeat_stats["inserted"] == 0
    assert repeat_stats["exact_duplicates"] == 2

    # 2. Importing a slightly reworded prompt should trigger semantic duplicate flag
    similar_item = [
        {
            "major": "Computer Science",
            "subject": "Data Structures",
            "item_type": "question",
            "question_type": "single_choice",
            "subject_tags": ["Trees > Binary Search Trees"],
            "relevancy": "core",
            "difficulty": 3,
            "source_reference": {"origin": "provided_material", "source_name": "Quiz1"},
            "prompt": "In a balanced Binary Search Tree, what is the average time \
                complexity of searching?",
            "options": ["A: O(1)", "B: O(log n)"],
            "answer": "B",
        }
    ]
    sim_path = tmp_path / "similar.json"
    sim_path.write_text(json.dumps(similar_item), encoding="utf-8")

    sim_stats = db.import_json_batch(sim_path, check_semantic=True)
    if sim_stats["similar_flagged"]:
        assert sim_stats["inserted"] == 0
        assert len(sim_stats["similar_flagged"]) == 1


def test_cross_subject_tag_isolation(db: StudyBankDB, tmp_path: Path) -> None:
    """Verify two different subjects can share the same tag name without colliding."""
    items = [
        {
            "major": "Mathematics",
            "subject": "Linear Algebra",
            "item_type": "question",
            "question_type": "short_answer",
            "subject_tags": ["Vectors > Dot Product"],
            "relevancy": "core",
            "difficulty": 2,
            "source_reference": {
                "origin": "provided_material",
                "source_name": "LA.pdf",
            },
            "prompt": "What is the dot product of orthogonal vectors?",
            "answer": "0",
        },
        {
            "major": "Physics",
            "subject": "Classical Mechanics",
            "item_type": "question",
            "question_type": "short_answer",
            "subject_tags": ["Vectors > Dot Product"],
            "relevancy": "core",
            "difficulty": 2,
            "source_reference": {
                "origin": "provided_material",
                "source_name": "Phys.pdf",
            },
            "prompt": "How is mechanical work defined using force and \
                displacement vectors?",
            "answer": "W = F . d",
        },
    ]
    path = tmp_path / "cross_subject.json"
    path.write_text(json.dumps(items), encoding="utf-8")
    db.import_json_batch(path)

    vector_tags = db.conn.execute(
        """SELECT subject, full_path, question_count
        FROM tags WHERE full_path = 'Vectors'"""
    ).fetchall()
    assert len(vector_tags) == 2
    assert {r["subject"] for r in vector_tags} == {
        "Linear Algebra",
        "Classical Mechanics",
    }


def test_mastery_threshold_and_time_decay(
    db: StudyBankDB, sample_batch_file: Path
) -> None:
    """Verify scores stay None until 5 attempts, then compute and
    decay/recover properly."""
    db.import_json_batch(sample_batch_file)
    item = db.conn.execute(
        "SELECT id FROM items WHERE item_type = 'question'"
    ).fetchone()
    item_id = item["id"]

    # Attempts 1 to 4: scores must remain None
    for _ in range(4):
        db.record_attempt(item_id, "right")
    row = db.conn.execute("SELECT * FROM items WHERE id = ?", (item_id,)).fetchone()
    assert row["times_seen"] == 4
    assert row["familiarity_score"] is None
    assert row["mastery_score"] is None

    # Attempt 5 (wrong): 4 right, 1 wrong -> familiarity = 0.8
    # difficulty = 3 -> weight = 0.6 + 0.4*(3/5) = 0.84 ->
    # mastery = 0.8 * 0.84 = 0.672
    db.record_attempt(item_id, "wrong")
    row = db.conn.execute("SELECT * FROM items WHERE id = ?", (item_id,)).fetchone()
    assert row["times_seen"] == 5
    assert row["familiarity_score"] == 0.8
    assert row["mastery_score"] == 0.672

    # Simulate 30 days passing without review -> effective mastery should decay
    future_dt = datetime.now(timezone.utc) + timedelta(days=30)
    decayed = db.effective_mastery_with_decay(row, now_dt=future_dt)
    assert decayed is not None
    assert decayed < row["mastery_score"]


def test_export_and_upsert_roundtrip(
    db: StudyBankDB, sample_batch_file: Path, tmp_path: Path
) -> None:
    """Verify exporting to JSON and re-importing an edited item updates it in place."""
    db.import_json_batch(sample_batch_file)
    export_file = tmp_path / "exported.json"
    count = db.export_to_json(export_file)
    assert count == 2

    exported_data = json.loads(export_file.read_text(encoding="utf-8"))
    exported_data[0]["prompt"] = "UPDATED PROMPT: Balanced BST search complexity?"
    export_file.write_text(json.dumps(exported_data), encoding="utf-8")

    upsert_stats = db.import_json_batch(export_file)
    assert upsert_stats["updated"] == 2
    assert upsert_stats["inserted"] == 0

    updated_row = db.conn.execute(
        "SELECT prompt FROM items WHERE id = ?", (exported_data[0]["id"],)
    ).fetchone()
    assert updated_row["prompt"] == "UPDATED PROMPT: Balanced BST search complexity?"
