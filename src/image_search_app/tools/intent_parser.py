from __future__ import annotations

import re
from dataclasses import dataclass, field

from image_search_app.tools.time_parser import TimeParser, TimeRange

# Keywords that indicate the user wants location/GPS filtering
_GPS_KEYWORDS = {"beach", "location", "near", "at", "place", "where", "outdoor", "park", "city", "mountain", "lake"}


@dataclass
class QueryIntent:
    mode: str
    people: list[str] = field(default_factory=list)
    needs_gps: bool = False
    time_range: TimeRange | None = None


class IntentParser:
    """Deterministic intent parser that extracts people, time, and location intent.

    Person names are matched against known names in the database.
    """

    def __init__(self) -> None:
        self.time_parser = TimeParser()
        self._known_names: set[str] | None = None

    def _load_known_names(self) -> set[str]:
        """Load all distinct person names from the database (cached)."""
        if self._known_names is not None:
            return self._known_names

        from sqlalchemy import select

        from image_search_app.db import PersonRecord, get_session

        names: set[str] = set()
        with get_session() as session:
            rows = session.scalars(
                select(PersonRecord.name).where(PersonRecord.name.isnot(None)).distinct()
            ).all()
            for name in rows:
                if name:
                    names.add(name.lower())
        self._known_names = names
        return names

    def invalidate_cache(self) -> None:
        """Clear the cached names so they're reloaded on next parse."""
        self._known_names = None

    def _extract_people(self, query: str) -> list[str]:
        """Find known person names mentioned in the query.

        Matches full names and first names from the database.
        E.g., if "Michael Powell" is in DB, both "Michael Powell" and "michael" match.
        """
        known = self._load_known_names()
        if not known:
            return []

        lowered = query.lower()
        found: list[str] = []
        found_lower: set[str] = set()

        # First try full name matches (longer matches first to avoid partial conflicts)
        for name in sorted(known, key=len, reverse=True):
            if name in lowered and name not in found_lower:
                found.append(name)
                found_lower.add(name)

        # Also try matching by first name or last name for multi-word names
        for name in known:
            if name in found_lower:
                continue
            parts = name.split()
            if len(parts) > 1:
                for part in parts:
                    if re.search(rf"\b{re.escape(part)}\b", lowered):
                        found.append(name)
                        found_lower.add(name)
                        break

        return found

    def _detect_gps_intent(self, query: str) -> bool:
        """Check if the query implies location-based filtering."""
        words = set(re.findall(r"[a-z]+", query.lower()))
        return bool(words & _GPS_KEYWORDS)

    def parse_text_query(self, query: str) -> QueryIntent:
        return QueryIntent(
            mode="text",
            people=self._extract_people(query),
            needs_gps=self._detect_gps_intent(query),
            time_range=self.time_parser.parse(query),
        )

    def parse_image_query(self, query: str | None) -> QueryIntent:
        if query:
            intent = self.parse_text_query(query)
            intent.mode = "image+text"
            return intent
        return QueryIntent(mode="image")
