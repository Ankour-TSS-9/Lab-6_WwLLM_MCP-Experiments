import json
import os
from datetime import datetime, timezone
from mcp.server.fastmcp import FastMCP

NOTES_FILE = os.path.join(os.path.dirname(__file__), "notes.json")

mcp = FastMCP("personal-assistant-memory")


def _load_notes():
    if not os.path.exists(NOTES_FILE):
        return []
    with open(NOTES_FILE, "r", encoding="utf-8") as f:
        content = f.read().strip()
        return json.loads(content) if content else []


def _save_notes(notes):
    with open(NOTES_FILE, "w", encoding="utf-8") as f:
        json.dump(notes, f, indent=2)


@mcp.tool()
def save_note(content: str, tags: list[str] = []) -> str:
    """Save a note with optional tags for later retrieval.

    Args:
        content: The text content of the note to remember.
        tags: Optional list of keyword tags to categorize the note.
    """
    notes = _load_notes()
    new_note = {
        "id": len(notes) + 1,
        "content": content,
        "tags": tags,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    notes.append(new_note)
    _save_notes(notes)
    return f"Note saved with id {new_note['id']}."


@mcp.tool()
def search_notes(query: str) -> str:
    """Search saved notes by matching the query against note content and tags.

    Args:
        query: The search term to look for in notes.
    """
    notes = _load_notes()
    query_lower = query.lower()
    matches = [
        n for n in notes
        if query_lower in n["content"].lower()
        or any(query_lower in tag.lower() for tag in n["tags"])
    ]
    if not matches:
        return "No matching notes found."
    return json.dumps(matches, indent=2)


if __name__ == "__main__":
    mcp.run(transport="stdio")