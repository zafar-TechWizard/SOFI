"""
BRAIN/state/self_model.py — SOFi's runtime capability registry

What it is
==========
A runtime-mutable record of what SOFi *can* and *cannot* do right now. The
persona's "What I can do" / "What I can't do" sections are rendered from
this, not hardcoded into `personality.json` directly.

Why it exists
=============
Today (V1), SOFi has a fixed set of baseline capabilities defined in
`personality.json` → `current_truth.can_do` / `current_truth.cannot_do`.
These are immutable identity-level facts: she has memory, she can reason,
she has no internet, etc.

Tomorrow, when tools get added (file read, calendar, web), each tool
registers itself here. The persona block updates AUTOMATICALLY — SOFi's
"what I can do" answer changes truthfully without anyone editing
`personality.json`.

The model — Option B (two flags)
================================
Each registered Capability carries TWO booleans:

  installed  — is this capability present in this build of SOFi at all?
               (e.g., the file_read tool was bundled in v0.2 but not v0.1)
  available  — is it usable RIGHT NOW?
               (e.g., file_read is installed but the workspace permission
                is missing, or Docker is down so memory can't reach Neo4j)

Four combinations:

  installed=True,  available=True   →  appears in "What I can do"
  installed=True,  available=False  →  appears in "What I can't do" with
                                       refusal_offline (e.g., "memory is
                                       offline right now — Docker isn't
                                       running")
  installed=False, available=False  →  appears in "What I can't do" with
                                       refusal_not_built (e.g., "I don't
                                       have file access in this build")
  installed=False, available=True   →  nonsense; treated as not_installed

The two-flag design lets SOFi distinguish "this is temporarily down" from
"this was never built" — a real-person distinction that matters in
honesty-of-limits.

Public API
==========
    sm = SelfModel.from_personality(personality_dict)   # builds baseline
    sm.register(Capability(...))                        # add a tool
    sm.set_state("file_read", available=False)          # toggle at runtime
    can_do_lines, cannot_do_lines = sm.render_for_prompt()
    snapshot_dict = sm.snapshot()                       # for diagnostics

How it plugs in
===============
At Brain.setup():
    sm = SelfModel.from_personality(get_personality_dict())
    set_self_model(sm)                  # tells persona.py to render from sm
    warm_cache()                        # rebuild cached blocks

When tools land later:
    sm.register(FileReadCapability(installed=True, available=True))
    set_self_model(sm)                  # invalidates cache; rebuilt next turn
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ============================================================================
# Capability
# ============================================================================

@dataclass
class Capability:
    """
    One thing SOFi may or may not be able to do, with state.

    The strings here are FIRST-PERSON phrases that drop into the persona
    block verbatim. Phrase them the way SOFi would speak them.

    Examples (for a hypothetical file-read tool):
        Capability(
            name="file_read",
            description="Open files he asks about — code, notes, drafts.",
            refusal_offline="I can't reach the file system right now.",
            refusal_not_built="Reading files isn't built into me yet.",
            installed=True,
            available=True,
        )
    """
    name: str
    description: str
    refusal_offline: str = ""
    refusal_not_built: str = ""
    installed: bool = True
    available: bool = True

    @property
    def is_working(self) -> bool:
        """True iff this capability is currently usable."""
        return self.installed and self.available

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "installed": self.installed,
            "available": self.available,
            "is_working": self.is_working,
        }


# ============================================================================
# SelfModel
# ============================================================================

class SelfModel:
    """
    SOFi's live capability registry.

    Holds two layers:
      - baseline_*: immutable identity-level lines from personality.json
                    (these don't get toggled; they're always-true facts)
      - registered: named Capability objects with installed/available flags
                    that CAN be toggled at runtime
    """

    def __init__(
        self,
        baseline_can_do: List[str],
        baseline_cannot_do: List[str],
    ) -> None:
        self._baseline_can_do: List[str] = list(baseline_can_do)
        self._baseline_cannot_do: List[str] = list(baseline_cannot_do)
        self._registered: Dict[str, Capability] = {}

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_personality(cls, personality: Dict[str, Any]) -> "SelfModel":
        """
        Build a SelfModel from a loaded personality.json dict.

        Reads `current_truth.can_do` and `current_truth.cannot_do` as the
        baseline. No tools are registered by default — those get added
        later via register().
        """
        truth = personality.get("current_truth") or {}
        return cls(
            baseline_can_do=list(truth.get("can_do") or []),
            baseline_cannot_do=list(truth.get("cannot_do") or []),
        )

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, capability: Capability) -> None:
        """
        Register or replace a capability by name.

        If a capability with the same name already exists, it is overwritten
        — useful for hot-swapping (e.g., file_read installed but now
        toggled offline).
        """
        if not capability.name:
            raise ValueError("Capability.name must be a non-empty string")
        self._registered[capability.name] = capability

    def unregister(self, name: str) -> None:
        """Remove a capability by name. No-op if absent."""
        self._registered.pop(name, None)

    def set_state(
        self,
        name: str,
        *,
        installed: Optional[bool] = None,
        available: Optional[bool] = None,
    ) -> None:
        """
        Toggle flags on an already-registered capability.

        Args:
            name:      capability name (must already be registered)
            installed: if not None, set installed to this value
            available: if not None, set available to this value

        Raises:
            KeyError if the capability isn't registered.
        """
        cap = self._registered.get(name)
        if cap is None:
            raise KeyError(f"capability not registered: {name!r}")
        if installed is not None:
            cap.installed = bool(installed)
        if available is not None:
            cap.available = bool(available)

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def get(self, name: str) -> Optional[Capability]:
        """Return the registered Capability by name, or None."""
        return self._registered.get(name)

    def all_capabilities(self) -> List[Capability]:
        """All registered capabilities (does not include baseline lines)."""
        return list(self._registered.values())

    def snapshot(self) -> Dict[str, Any]:
        """A diagnostic snapshot of everything I currently model about myself."""
        return {
            "baseline_can_do":    list(self._baseline_can_do),
            "baseline_cannot_do": list(self._baseline_cannot_do),
            "registered": {
                name: cap.to_dict()
                for name, cap in self._registered.items()
            },
        }

    # ------------------------------------------------------------------
    # Render — for the prompt builder
    # ------------------------------------------------------------------

    def render_for_prompt(self) -> Tuple[List[str], List[str]]:
        """
        Produce the lines that drop into the persona block's
        "What I can do" and "What I can't do" sections.

        Returns:
            (can_do_lines, cannot_do_lines)

        Composition:
            can_do      = baseline_can_do
                        + each registered Capability where installed AND available
                          (rendered as its `description`)
            cannot_do   = baseline_cannot_do
                        + each registered Capability where installed BUT NOT available
                          (rendered as its `refusal_offline`, if set)
                        + each registered Capability where NOT installed
                          (rendered as its `refusal_not_built`, if set)

        Capabilities without the appropriate refusal text are silently
        omitted from cannot_do — silence is better than empty strings.
        """
        can_do: List[str] = list(self._baseline_can_do)
        cannot_do: List[str] = list(self._baseline_cannot_do)

        for cap in self._registered.values():
            if cap.installed and cap.available:
                can_do.append(cap.description)
            elif cap.installed and not cap.available:
                if cap.refusal_offline:
                    cannot_do.append(cap.refusal_offline)
            elif not cap.installed:
                if cap.refusal_not_built:
                    cannot_do.append(cap.refusal_not_built)
            # installed=False + available=True is treated as not_installed (above)

        return can_do, cannot_do
