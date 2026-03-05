from __future__ import annotations

from dataclasses import dataclass, field

from image_search_app.tools.time_parser import TimeParser, TimeRange


@dataclass
class QueryIntent:
    mode: str
    people: list[str] = field(default_factory=list)
    needs_gps: bool = False
    time_range: TimeRange | None = None


class IntentParser:
    """Deterministic intent parser starter.

    Replace or augment with LLM-assisted parser in agent planning stage.
    """

    def __init__(self) -> None:
        self.time_parser = TimeParser()

    def parse_text_query(self, query: str) -> QueryIntent:
        lowered = query.lower()
        people = []
        if "tom" in lowered:
            people.append("tom")

        return QueryIntent(
            mode="text",
            people=people,
            needs_gps=("beach" in lowered or "location" in lowered),
            time_range=self.time_parser.parse(query),
        )

    def parse_image_query(self, query: str | None) -> QueryIntent:
        if query:
            intent = self.parse_text_query(query)
            intent.mode = "image+text"
            return intent
        return QueryIntent(mode="image")
