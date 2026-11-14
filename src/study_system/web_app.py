from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from study_system.db_manager import StudyBankDB

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "db" / "study_bank.db"
STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="StudyBank API", version="0.1.0")


def get_db() -> StudyBankDB:
    return StudyBankDB(db_path=DEFAULT_DB_PATH)


class SessionRequest(BaseModel):
    major: str
    subject: str
    modules: list[str] = Field(default_factory=list)
    item_type: str = "question"  # "question" or "content_note"
    question_types: list[str] = Field(default_factory=list)
    size: int | str = 28  # 13, 28, 44, or "infinite"
    previous_wrong_ids: list[str] = Field(default_factory=list)
    previous_right_ids: list[str] = Field(default_factory=list)


class AttemptRecord(BaseModel):
    item_id: str
    outcome: str  # "right", "wrong", "skipped"


class TelemetryBatchRequest(BaseModel):
    attempts: list[AttemptRecord]


def _parse_item_row(row: Any, tags_by_item: dict[str, list[str]]) -> dict[str, Any]:
    full_tags = tags_by_item.get(row["id"], [])
    # Extract individual unordered leaf/branch tag labels without full path arrows
    bubble_set: set[str] = set()
    modules: set[str] = set()
    for path in full_tags:
        parts = [p.strip() for p in path.split(">") if p.strip()]
        if parts:
            modules.add(parts[0])
            for p in parts[1:]:
                bubble_set.add(p)
            if len(parts) == 1:
                bubble_set.add(parts[0])

    bubbles = list(bubble_set)
    random.shuffle(bubbles)

    options = json.loads(row["options_json"]) if row["options_json"] else None
    answer = json.loads(row["answer_json"]) if row["answer_json"] else None

    return {
        "id": row["id"],
        "major": row["major"],
        "subject": row["subject"],
        "item_type": row["item_type"],
        "question_type": row["question_type"],
        "relevancy": row["relevancy"],
        "difficulty": row["difficulty"],
        "prompt": row["prompt"],
        "options": options,
        "answer": answer,
        "explanation": row["explanation"],
        "times_seen": row["times_seen"],
        "modules": sorted(modules),
        "tag_bubbles": bubbles[:4],
    }


def _load_all_items_with_tags(db: StudyBankDB) -> list[dict[str, Any]]:
    tag_rows = db.conn.execute(
        """
        SELECT it.item_id, t.full_path
        FROM item_tags it
        JOIN tags t ON it.tag_id = t.tag_id
        """
    ).fetchall()
    tags_by_item: dict[str, list[str]] = {}
    for tr in tag_rows:
        tags_by_item.setdefault(tr["item_id"], []).append(tr["full_path"])

    rows = db.conn.execute("SELECT * FROM items").fetchall()
    return [_parse_item_row(r, tags_by_item) for r in rows]


@app.get("/api/catalog")
def get_catalog() -> dict[str, Any]:
    db = get_db()
    try:
        items = _load_all_items_with_tags(db)
        majors_map: dict[str, dict[str, Any]] = {}

        for item in items:
            major = item["major"]
            subject = item["subject"]
            m_entry = majors_map.setdefault(major, {"major": major, "courses": {}})
            c_entry = m_entry["courses"].setdefault(
                subject,
                {
                    "subject": subject,
                    "total_items": 0,
                    "question_count": 0,
                    "flashcard_count": 0,
                    "modules": {},
                    "question_types": {},
                },
            )
            c_entry["total_items"] += 1
            if item["item_type"] == "question":
                c_entry["question_count"] += 1
                qt = item["question_type"] or "other"
                c_entry["question_types"][qt] = c_entry["question_types"].get(qt, 0) + 1
            else:
                c_entry["flashcard_count"] += 1

            for mod in item["modules"]:
                mod_info = c_entry["modules"].setdefault(
                    mod, {"name": mod, "questions": 0, "flashcards": 0, "total": 0}
                )
                mod_info["total"] += 1
                if item["item_type"] == "question":
                    mod_info["questions"] += 1
                else:
                    mod_info["flashcards"] += 1

        catalog_list = []
        for major_name, m_data in majors_map.items():
            courses_list = []
            for subj_name, c_data in m_data["courses"].items():
                c_data["modules"] = list(c_data["modules"].values())
                courses_list.append(c_data)
            catalog_list.append({"major": major_name, "courses": courses_list})

        return {"majors": catalog_list, "all_items": items}
    finally:
        db.close()


@app.post("/api/session")
def build_session(req: SessionRequest) -> dict[str, Any]:
    db = get_db()
    try:
        items = _load_all_items_with_tags(db)
        pool = [
            it
            for it in items
            if it["major"] == req.major
            and it["subject"] == req.subject
            and it["item_type"] == req.item_type
        ]

        if req.modules:
            mod_set = set(req.modules)
            pool = [it for it in pool if mod_set.intersection(it["modules"])]

        if req.item_type == "question" and req.question_types:
            qt_set = set(req.question_types)
            pool = [it for it in pool if it["question_type"] in qt_set]

        if not pool:
            raise HTTPException(
                status_code=404,
                detail="No items match the selected filters.",
            )

        wrong_set = set(req.previous_wrong_ids)
        right_set = set(req.previous_right_ids)

        # Weighted sampling without replacement:
        # Unseen / other pool items = 3.0 | Previous wrong = 1.5 | Previous right = 0.5
        def item_weight(it: dict[str, Any]) -> float:
            iid = it["id"]
            if iid in wrong_set:
                base = 1.5
            elif iid in right_set:
                base = 0.5
            else:
                base = 3.0
            if it.get("relevancy") == "core":
                base *= 1.15
            return base

        if req.size == "infinite":
            target_count = len(pool)
        else:
            target_count = min(int(req.size), len(pool))

        selected: list[dict[str, Any]] = []
        remaining = list(pool)
        while len(selected) < target_count and remaining:
            weights = [item_weight(x) for x in remaining]
            choice = random.choices(remaining, weights=weights, k=1)[0]
            selected.append(choice)
            remaining.remove(choice)

        return {
            "session_items": selected,
            "pool_size": len(pool),
            "infinite": req.size == "infinite",
        }
    finally:
        db.close()


@app.post("/api/telemetry/batch")
def record_telemetry_batch(payload: TelemetryBatchRequest) -> dict[str, Any]:
    if not payload.attempts:
        return {"recorded": 0}

    db = get_db()
    recorded = 0
    try:
        for att in payload.attempts:
            outcome = (
                att.outcome
                if att.outcome in ("right", "wrong", "skipped")
                else "skipped"
            )
            try:
                db.record_attempt(att.item_id, outcome=outcome)
                recorded += 1
            except KeyError:
                continue
        db.refresh_tag_metrics()
        return {"recorded": recorded, "status": "ok"}
    finally:
        db.close()


STATIC_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/")
def serve_index() -> FileResponse:
    return FileResponse(
        STATIC_DIR / "index.html",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
    )


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
