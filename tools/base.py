from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ToolContract:
    """Declares which context keys a tool reads/writes — executor uses this for pre-flight checks."""
    reads:          list[str]
    writes:         list[str]
    optional_reads: list[str] = field(default_factory=list)


class ToolError(RuntimeError):
    pass


class BaseTool(ABC):
    name:         str
    description:  str
    input_schema: dict
    contract:     ToolContract

    @abstractmethod
    async def run(self, ctx: dict) -> dict: ...

    def check_ctx(self, ctx: dict) -> None:
        """Call at the top of run() — raises ToolError on missing required keys."""
        missing = [k for k in self.contract.reads if k not in ctx]
        if missing:
            raise ToolError(f"[{self.name}] Missing required context keys: {missing}")
