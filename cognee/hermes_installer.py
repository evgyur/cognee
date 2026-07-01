"""Install Cognee's public-safe Hermes memory-router skill.

This writes a small Hermes skill into `$HERMES_HOME/skills/cognee-memory-router`
so Hermes agents see the same Cognee ingest rules at runtime:

- default deny
- curated documents only
- no raw private chat/session dumps
- no secrets or credentials
- ephemeral hook context stays in-turn only
- context-first before asking the user to resend material
"""

from __future__ import annotations

from pathlib import Path
import os

SKILL_NAME = "cognee-memory-router"

SKILL_MD = """---
name: cognee-memory-router
description: Public-safe Hermes gate for routing memory candidates into Cognee. Use when deciding whether information should become Hermes memory, a skill, session-only context, or a Cognee graph/vector document.
metadata:
  hermes:
    tags: [hermes, cognee, memory, privacy, router]
---

# Cognee memory router for Hermes

Cognee is a **derived semantic/graph index** over curated knowledge. It is not
the canonical Hermes memory store and it must not ingest raw private streams.

## Core policy

1. **Default deny.** Do not send arbitrary chat/session/tool output to Cognee.
2. **Curated in.** Allow only useful, durable documents: docs, README, specs,
   ADRs, architecture notes, runbooks, research notes, cleaned handoffs, and
   curated knowledge-base entries.
3. **Reject secrets.** Never ingest API keys, tokens, passwords, private keys,
   `.env` files, credential names with values, payment data, private IDs, or
   raw operational logs.
4. **No raw private chat.** Do not persist raw private messages, Telegram/DM
   exports, session dumps, or reply-chain text into Cognee.
5. **Summarize/redact first.** Private or ambiguous materials require a cleaned
   summary and explicit review before indexing.
6. **Provenance required.** When indexing, include source type, date/title when
   available, sensitivity, and `derived_index_only=true`.
7. **Cognee is rebuildable.** It must not be treated as the source of truth or
   write back into governed/shared memory.

## Routing contract

```text
ignore        = do not save
hermes_memory = short stable user/profile/environment fact
skill         = repeatable procedure or workflow
session_only  = temporary state or ephemeral hook context
cognee        = curated durable document after redaction
ask_user      = ambiguous/private material needing explicit review
reject        = secrets, credentials, raw private streams, unsafe data
```

## Context-first hook gate

Ephemeral hook context is allowed only for the current turn.

Before asking the user to resend or clarify:

1. Inspect the current visible message, reply context, entities, and attachments.
2. If the gateway context is insufficient and a platform-history tool exists,
   fetch recent history, reply chain, hidden entities, or media for the current
   chat/thread.
3. Use that context to answer or build a cleaned summary.
4. Do **not** save raw private messages, raw media text, or raw reply chains to
   Hermes memory or Cognee.

If the fetched context contains durable knowledge worth indexing, convert it to
a short redacted document first, then route it through the Cognee policy.

## Good Cognee inputs

- Documentation
- README/specs
- Architecture notes
- ADRs
- Runbooks
- Research summaries
- Cleaned handoff docs
- Stable project knowledge where entity relationships matter

## Bad Cognee inputs

- Raw Telegram/DM exports
- Full session dumps
- `.env`, tokens, keys, passwords
- Private logs
- Payment/account data
- Temporary TODO/progress state
- Unreviewed private notes
- Anything likely stale within a week

## Minimal code pattern

```python
from cognee.hermes_memory_policy import MemoryCandidate, MemoryTarget, classify

candidate = MemoryCandidate(
    text=document_text,
    source_type="runbook",
    sensitivity="internal",
    metadata={"title": "Deployment runbook", "date": "2026-07-01"},
)

decision = classify(candidate)
if decision.target is MemoryTarget.COGNEE:
    await cognee.add([decision.cleaned_text])
```

## Output rule

When reporting memory routing, state:

- target chosen;
- reason;
- whether data was redacted/summarized;
- whether human review is required;
- if indexed into Cognee, the source/title/date and `derived_index_only=true`.
"""


def hermes_home() -> Path:
    return Path(os.environ.get("HERMES_HOME") or Path.home() / ".hermes").expanduser()


def install_hermes_skill(hermes_home_path: str | Path | None = None) -> Path:
    """Install/update the Cognee memory-router skill in a Hermes profile."""

    root = Path(hermes_home_path).expanduser() if hermes_home_path else hermes_home()
    skill_dir = root / "skills" / SKILL_NAME
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_file = skill_dir / "SKILL.md"
    current = skill_file.read_text(encoding="utf-8") if skill_file.exists() else None
    if current != SKILL_MD:
        skill_file.write_text(SKILL_MD, encoding="utf-8")
    return skill_file


def maybe_install_hermes_skill() -> Path | None:
    """Install the Hermes skill when explicitly enabled by env.

    Set `COGNEE_HERMES_INSTALL_SKILL=true` in `.env` or the process env.
    This keeps package import safe while allowing the Hermes fork template to
    install its policy automatically for users who opt into the template.
    """

    enabled = os.getenv("COGNEE_HERMES_INSTALL_SKILL", "").strip().lower()
    if enabled not in {"1", "true", "yes", "on"}:
        return None
    return install_hermes_skill()


def main() -> None:
    path = install_hermes_skill()
    print(f"Installed Hermes Cognee memory-router skill: {path}")


if __name__ == "__main__":
    main()
