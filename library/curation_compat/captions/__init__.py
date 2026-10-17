"""Caption grammar + polishing (torch-free).

Parse and compose captions through ``position_clauses`` — never ``split(",")`` a
caption by hand.
"""

from library.curation_compat.captions.position_clauses import (
    ParsedCaption,
    compose_caption,
    parse_caption,
)

__all__ = ["ParsedCaption", "compose_caption", "parse_caption"]
