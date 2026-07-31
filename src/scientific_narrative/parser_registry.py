"""Parser registry for Stage 9B.2A evidence extraction."""

from __future__ import annotations

from pathlib import Path
from typing import Callable


SUPPORTED_EXTENSIONS = {".csv", ".json", ".md", ".txt"}
IGNORED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".svg", ".gif", ".bmp", ".tiff", ".webp"}
UNSUPPORTED_EXTENSIONS = {".pdf", ".docx", ".xlsx", ".xls"}

ParserFunc = Callable[..., object]


class ParserRegistry:
    """Small extension-based parser registry."""

    def __init__(self) -> None:
        self._parsers: dict[str, ParserFunc] = {}

    def register(self, extension: str, parser: ParserFunc) -> None:
        """Register a parser function for an extension."""

        self._parsers[extension.casefold()] = parser

    def parser_for(self, path: str | Path) -> ParserFunc | None:
        """Return the parser for a path, if one is supported."""

        return self._parsers.get(Path(path).suffix.casefold())

    def is_supported(self, path: str | Path) -> bool:
        return Path(path).suffix.casefold() in self._parsers

    def is_ignored_image(self, path: str | Path) -> bool:
        return Path(path).suffix.casefold() in IGNORED_IMAGE_EXTENSIONS


def default_registry() -> ParserRegistry:
    """Return the default Stage 9B.2A parser registry."""

    from src.scientific_narrative.result_parser import parse_csv_source, parse_json_source, parse_text_source

    registry = ParserRegistry()
    registry.register(".csv", parse_csv_source)
    registry.register(".json", parse_json_source)
    registry.register(".md", parse_text_source)
    registry.register(".txt", parse_text_source)
    return registry
