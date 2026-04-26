# test_agent.py
from agent import agent

# ── Test Cases ─────────────────────────────────────────────────────────────────
TEST_LOGS = [
    {
        "label": "SSH Brute Force",
        "expected_severity": ["suspicious", "critical"],
        "raw_log": "Failed password for root from 192.168.1.105 port 4521 ssh2 (attempt 47)"
    },
    {
        "label": "Privilege Escalation",
        "expected_severity": ["suspicious", "critical"],
        "raw_log": "sudo: james : TTY=pts/0 ; PWD=/root ; USER=root ; COMMAND=/bin/bash"
    },
    {
        "label": "Benign Cron Job",
        "expected_severity": ["benign"],
        "raw_log": "CRON[12481]: (root) CMD (/usr/lib/cron/run-crons)"
    },
    {
        "label": "Benign System Boot",
        "expected_severity": ["benign"],
        "raw_log": "systemd[1]: Started Session 4 of user www-data."
    },
    {
        "label": "Critical — Root Shell Spawned",
        "expected_severity": ["suspicious", "critical"],
        "raw_log": "audit: type=EXECVE msg=audit(1714000000.000:999): argc=2 a0=/bin/sh a1=-i uid=0"
    },
]

# ── Runner ─────────────────────────────────────────────────────────────────────
def run_tests():
    passed = 0
    failed = 0

    print("\n" + "═" * 60)
    print("  AIA7000 — LangGraph Cybersecurity Agent Test Harness")
    print("═" * 60)

    for i, tc in enumerate(TEST_LOGS, 1):
        print(f"\n[Test {i}/{len(TEST_LOGS)}] {tc['label']}")
        print(f"  Log : {tc['raw_log'][:80]}...")

        initial_state = {
            "raw_log":        tc["raw_log"],
            "severity":       "",
            "threat_context": "",
            "report":         "",
        }

        result = agent.invoke(initial_state)

        severity = result["severity"]
        ok = severity in tc["expected_severity"]

        status = "✅ PASS" if ok else "❌ FAIL"
        if ok:
            passed += 1
        else:
            failed += 1

        print(f"  Severity : {severity.upper()}  {status}")
        print(f"  Expected : {' or '.join(tc['expected_severity']).upper()}")
        print(result["report"])

    print("═" * 60)
    print(f"  Results: {passed} passed, {failed} failed out of {len(TEST_LOGS)} tests")
    print("═" * 60 + "\n")

    assert failed == 0, f"{failed} test(s) failed — check LLM classification output"


if __name__ == "__main__":
    run_tests()