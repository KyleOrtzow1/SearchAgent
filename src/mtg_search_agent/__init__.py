"""MTG Search Agent package public API.

Keep imports lightweight to avoid side effects when importing submodules,
e.g., mtg_search_agent.tools.scryfall_api.
"""

from typing import TYPE_CHECKING

__all__ = ["SearchOrchestrator"]

if TYPE_CHECKING:
	# For type checkers only; avoids runtime side effects
	from .orchestrator import SearchOrchestrator as SearchOrchestrator


def __getattr__(name: str):
	if name == "SearchOrchestrator":
		# Lazy import to prevent heavy dependencies during simple submodule imports
		from .orchestrator import SearchOrchestrator
		return SearchOrchestrator
	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")