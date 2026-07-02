"""
BRAIN/tools/__init__.py

Exports the tool infrastructure and provides a `@tool` decorator for
declaring tools inline — the alternative to a register(registry) function.

Two registration patterns are supported by _auto_register_tools:

  Pattern A — register() function (for complex tools, e.g. closures):
      def register(registry):
          registry.register(ToolEntry(name="my_tool", ...))

  Pattern B — @tool decorator (for simple stateless tools):
      from BRAIN.tools import tool

      @tool(
          name="my_tool",
          description="What it does.",
          parameters={"type": "object", "properties": {...}, "required": [...]},
      )
      async def my_tool(arg: str) -> str:
          return ...

Both patterns are discovered automatically — adding a file is enough.
"""

from BRAIN.tools.registry import ToolCall, ToolEntry, ToolRegistry, ToolResult


def tool(
    name: str,
    description: str,
    parameters: dict,
    *,
    category: str = "general",
    needs_confirmation: bool = False,
    background: bool = False,
    timeout: float = 60.0,
    check_fn=None,
    capability_name: str | None = None,
    capability_description: str | None = None,
    capability_refusal: str | None = None,
):
    """
    Decorator that marks an async function as a SOFi tool.

    Attaches a ToolEntry as `fn._tool_entry` so _auto_register_tools picks
    it up at import time — no register(registry) function needed.

    Usage::

        @tool(
            name="read_file",
            description="Read a file from disk and return its contents.",
            parameters={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
            category="filesystem",
        )
        async def read_file(path: str) -> str:
            ...
    """
    def decorator(fn):
        fn._tool_entry = ToolEntry(
            name=name,
            description=description,
            schema=parameters,
            handler=fn,
            check_fn=check_fn,
            needs_confirmation=needs_confirmation,
            background=background,
            timeout=timeout,
            category=category,
            capability_name=capability_name,
            capability_description=capability_description,
            capability_refusal=capability_refusal,
        )
        return fn
    return decorator


__all__ = ["ToolRegistry", "ToolEntry", "ToolResult", "ToolCall", "tool"]
