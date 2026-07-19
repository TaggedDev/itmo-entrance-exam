import json
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class JsonlStore:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.audit_path = self.data_dir / "audit_log.jsonl"
        self.pending_path = self.data_dir / "pending_tickets.jsonl"

    def append_audit(self, event: dict[str, Any]) -> None:
        self._append(self.audit_path, self._with_time(event))

    async def append_audit_async(self, event: dict[str, Any]) -> None:
        await asyncio.to_thread(self.append_audit, event)

    def append_pending(self, ticket: dict[str, Any]) -> None:
        self._append(self.pending_path, self._with_time(ticket))

    async def append_pending_async(self, ticket: dict[str, Any]) -> None:
        await asyncio.to_thread(self.append_pending, ticket)

    def list_pending(self) -> list[dict[str, Any]]:
        if not self.pending_path.exists():
            return []
        with self.pending_path.open("r", encoding="utf-8") as file:
            return [self._normalize_pending(json.loads(line)) for line in file if line.strip()]

    async def list_pending_async(self) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self.list_pending)

    def moderate(self, ticket_id: str, action: str, operator_note: str) -> str:
        tickets = self.list_pending()
        kept: list[dict[str, Any]] = []
        found = False
        for ticket in tickets:
            if ticket["ticket_id"] == ticket_id:
                found = True
                self.append_audit(
                    {
                        "event": "moderated",
                        "ticket_id": ticket_id,
                        "action": action,
                        "operator_note": operator_note,
                    }
                )
            else:
                kept.append(ticket)
        self._rewrite(self.pending_path, kept)
        return "moderated" if found else "not_found"

    async def moderate_async(self, ticket_id: str, action: str, operator_note: str) -> str:
        return await asyncio.to_thread(self.moderate, ticket_id, action, operator_note)

    def _append(self, path: Path, event: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(event, ensure_ascii=False) + "\n")

    def _rewrite(self, path: Path, events: list[dict[str, Any]]) -> None:
        with path.open("w", encoding="utf-8") as file:
            for event in events:
                file.write(json.dumps(event, ensure_ascii=False) + "\n")

    def _with_time(self, event: dict[str, Any]) -> dict[str, Any]:
        return {"created_at": datetime.now(timezone.utc).isoformat(), **event}

    def _normalize_pending(self, ticket: dict[str, Any]) -> dict[str, Any]:
        if "answer" in ticket and "original_text" in ticket:
            return ticket
        return {
            **ticket,
            "original_text": ticket.get("original_text", ticket.get("text", "")),
            "redacted_text": ticket.get("redacted_text", ticket.get("text", "")),
            "requires_human_review": ticket.get("requires_human_review", True),
            "answer": ticket.get("answer", ticket.get("draft_response", "")),
        }
