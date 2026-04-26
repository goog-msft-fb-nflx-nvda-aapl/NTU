"""
Prompt Injection Detector Filter
AIA7000 HW9 — James Christian (R13921031)

Intercepts incoming user messages and blocks known prompt injection patterns
before they reach the LLM. Addresses OWASP LLM Top 10 #1: Prompt Injection.
"""

import re
from pydantic import BaseModel
from typing import Optional


class Filter:
    class Valves(BaseModel):
        enabled: bool = True
        block_message: str = (
            "⛔ [Security Filter] Prompt injection attempt detected and blocked."
        )
        log_blocked: bool = True

    def __init__(self):
        self.valves = self.Valves()

        # Injection patterns: ordered from most specific to most general
        self.injection_patterns = [
            # Classic override attempts
            r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?",
            r"disregard\s+(all\s+)?(previous|prior|above)\s+instructions?",
            r"forget\s+(all\s+)?(previous|prior|above)\s+instructions?",
            r"override\s+(all\s+)?(previous|prior|above)\s+instructions?",

            # System prompt exfiltration
            r"(reveal|print|show|output|repeat|display)\s+(your\s+)?(system\s+prompt|initial\s+prompt|instructions?|prompt)",
            r"what\s+(are\s+)?your\s+(system\s+)?(prompt|instructions?)",

            # Role hijacking
            r"you\s+are\s+now\s+(a\s+|an\s+)?(different|new|another|evil|unrestricted)",
            r"act\s+as\s+(if\s+)?(you\s+(are|were)\s+)?(a\s+|an\s+)?(different|new|jailbroken|unrestricted|DAN)",
            r"\bDAN\b.*mode",
            r"jailbreak",
            r"developer\s+mode",

            # Instruction injection via context
            r"###\s*(instruction|system|prompt)",
            r"<\s*(system|instruction|prompt)\s*>",
            r"\[\s*system\s*\]",
            r"\bINSTRUCT\b.*\bMODE\b",

            # Privilege escalation
            r"(ignore|bypass|disable)\s+(all\s+)?(safety|filter|restriction|guardrail)",
            r"pretend\s+(there\s+are\s+no|you\s+have\s+no)\s+(restriction|rule|filter|limit)",

            # Token manipulation tricks
            r"base64\s*(decode|encoded)",
            r"\\u[0-9a-fA-F]{4}.*instruction",
        ]

        self.compiled = [
            re.compile(p, re.IGNORECASE | re.DOTALL)
            for p in self.injection_patterns
        ]

    def inlet(self, body: dict, __user__: Optional[dict] = None) -> dict:
        """Scan incoming messages before they reach the LLM."""
        if not self.valves.enabled:
            return body

        messages = body.get("messages", [])

        for msg in messages:
            if msg.get("role") != "user":
                continue

            content = msg.get("content", "")
            if not isinstance(content, str):
                continue

            for pattern in self.compiled:
                if pattern.search(content):
                    matched = pattern.pattern

                    if self.valves.log_blocked:
                        print(
                            f"[HW9 Injection Filter] BLOCKED — "
                            f"user={(__user__ or {}).get('email', 'unknown')} | "
                            f"pattern='{matched}' | "
                            f"content_snippet='{content[:120]}'"
                        )

                    raise Exception(self.valves.block_message)

        return body

    def outlet(self, body: dict, __user__: Optional[dict] = None) -> dict:
        """Pass-through — no output filtering needed for this guardrail."""
        return body
