"""
Autonomous Cybersecurity Compliance Agent
AIA7000 HW10 — James Christian (R13921031)

Closed-loop LangGraph agent for cybersecurity compliance monitoring.
Feedback loop: validate_report → retry enrich_threat if quality check fails.
"""

import json
import re
import time
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
import ollama

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL = "llama3.2:latest"
MAX_RETRIES = 3

# ── State ──────────────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    log_entry: str
    classification: dict
    enrichment: dict
    report: dict
    validation: dict
    retry_count: int
    retry_feedback: str
    history: list[str]

# ── Helpers ────────────────────────────────────────────────────────────────────
def llm(prompt: str, system: str = "") -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = ollama.chat(model=MODEL, messages=messages)
    return resp["message"]["content"].strip()

def extract_json(text: str) -> dict:
    """Extract first JSON object from LLM output."""
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {}

# ── Node 1: classify_log ───────────────────────────────────────────────────────
def classify_log(state: AgentState) -> AgentState:
    print(f"\n[Node 1] classify_log")
    log = state["log_entry"]

    prompt = f"""Analyze this security log entry and classify it.
Log: {log}

Respond ONLY with a JSON object with these exact keys:
{{
  "event_type": "<one of: auth_failure, data_exfiltration, malware, policy_violation, reconnaissance, unknown>",
  "severity": "<one of: critical, high, medium, low>",
  "source_ip": "<extracted IP or null>",
  "user": "<extracted username or null>",
  "confidence": <float 0.0-1.0>
}}"""

    raw = llm(prompt)
    result = extract_json(raw)
    if not result:
        result = {"event_type": "unknown", "severity": "low", "source_ip": None, "user": None, "confidence": 0.3}

    print(f"  → event_type={result.get('event_type')} severity={result.get('severity')} confidence={result.get('confidence')}")
    state["classification"] = result
    state["history"].append(f"classify_log: {result.get('event_type')} ({result.get('severity')})")
    return state

# ── Node 2: enrich_threat ──────────────────────────────────────────────────────
# def enrich_threat(state: AgentState) -> AgentState:
#     retry = state["retry_count"]
#     feedback = state.get("retry_feedback", "")
#     print(f"\n[Node 2] enrich_threat (attempt {retry + 1}/{MAX_RETRIES + 1})")
def enrich_threat(state: AgentState) -> AgentState:
    # Increment retry counter here (not in the edge function — state mutations must happen in nodes)
    if state["enrichment"]:  # not first run
        state["retry_count"] += 1
    retry = state["retry_count"]
    feedback = state.get("retry_feedback", "")
    print(f"\n[Node 2] enrich_threat (attempt {retry + 1}/{MAX_RETRIES + 1})")
    
    cls = state["classification"]
    log = state["log_entry"]

    feedback_section = ""
    if feedback:
        feedback_section = f"\nPrevious report was rejected. Reason: {feedback}\nPlease address this in your enrichment.\n"

    prompt = f"""You are a cybersecurity analyst. Enrich this threat based on the classification.
{feedback_section}
Log: {log}
Classification: {json.dumps(cls)}

Respond ONLY with a JSON object with these exact keys:
{{
  "threat_actor_profile": "<description of likely attacker type>",
  "attack_vector": "<method used>",
  "affected_assets": ["<asset1>", "<asset2>"],
  "compliance_violations": ["<standard:rule>"],
  "ioc_list": ["<indicator1>", "<indicator2>"],
  "risk_score": <integer 1-10>,
  "recommended_action": "<specific remediation step>"
}}"""

    raw = llm(prompt)
    result = extract_json(raw)
    if not result:
        result = {
            "threat_actor_profile": "Unknown",
            "attack_vector": "Unknown",
            "affected_assets": [],
            "compliance_violations": [],
            "ioc_list": [],
            "risk_score": 5,
            "recommended_action": "Investigate further"
        }

    print(f"  → risk_score={result.get('risk_score')} violations={result.get('compliance_violations')}")
    state["enrichment"] = result
    state["history"].append(f"enrich_threat attempt {retry + 1}: risk_score={result.get('risk_score')}")
    return state

# ── Node 3: generate_report ────────────────────────────────────────────────────
def generate_report(state: AgentState) -> AgentState:
    print(f"\n[Node 3] generate_report")
    cls = state["classification"]
    enr = state["enrichment"]

    prompt = f"""Generate a structured cybersecurity compliance incident report.

Classification: {json.dumps(cls)}
Enrichment: {json.dumps(enr)}

Respond ONLY with a JSON object with these exact keys:
{{
  "incident_id": "<generate a short ID like INC-XXXX>",
  "summary": "<2-3 sentence incident summary>",
  "severity": "<critical/high/medium/low>",
  "compliance_impact": "<which regulations are affected and how>",
  "recommendation": "<specific actionable recommendation>",
  "escalate": <true or false>,
  "confidence_score": <float 0.0-1.0>
}}"""

    raw = llm(prompt)
    result = extract_json(raw)
    if not result:
        result = {
            "incident_id": "INC-0000",
            "summary": "Unable to generate report.",
            "severity": cls.get("severity", "unknown"),
            "compliance_impact": "Unknown",
            "recommendation": "",
            "escalate": False,
            "confidence_score": 0.1
        }

    print(f"  → incident_id={result.get('incident_id')} confidence={result.get('confidence_score')} escalate={result.get('escalate')}")
    state["report"] = result
    state["history"].append(f"generate_report: {result.get('incident_id')} confidence={result.get('confidence_score')}")
    return state

# ── Node 4: validate_report ────────────────────────────────────────────────────
def validate_report(state: AgentState) -> AgentState:
    print(f"\n[Node 4] validate_report")
    report = state["report"]
    enr = state["enrichment"]
    issues = []

    # Quality gate checks
    if not report.get("recommendation") or len(report.get("recommendation", "")) < 15:
        issues.append("recommendation is missing or too vague")

    if not report.get("severity") or report["severity"] not in ["critical", "high", "medium", "low"]:
        issues.append("severity is not properly assigned")

    if not report.get("compliance_impact") or report["compliance_impact"] in ["Unknown", ""]:
        issues.append("compliance_impact is missing — must reference a specific standard (e.g. NIST, PCI-DSS, ISO 27001)")

    if (report.get("confidence_score") or 0) < 0.5:
        issues.append(f"confidence_score {report.get('confidence_score')} is below threshold 0.5")

    if not enr.get("compliance_violations") or len(enr["compliance_violations"]) == 0:
        issues.append("no compliance violations identified — re-examine the log for policy breaches")

    passed = len(issues) == 0
    feedback = "; ".join(issues) if issues else ""

    print(f"  → {'PASSED' if passed else 'FAILED'} | issues: {issues}")
    state["validation"] = {"passed": passed, "issues": issues}
    state["retry_feedback"] = feedback
    state["history"].append(f"validate_report: {'PASSED' if passed else 'FAILED'} issues={issues}")
    return state

# ── Conditional edge: retry or end ────────────────────────────────────────────
# def should_retry(state: AgentState) -> str:
#     passed = state["validation"].get("passed", False)
#     retry_count = state["retry_count"]

#     if passed:
#         print(f"\n[Router] Validation PASSED → END")
#         return "end"
#     elif retry_count >= MAX_RETRIES:
#         print(f"\n[Router] Max retries ({MAX_RETRIES}) reached → END (forced)")
#         return "end"
#     else:
#         print(f"\n[Router] Validation FAILED → retry enrich_threat (retry {retry_count + 1}/{MAX_RETRIES})")
#         state["retry_count"] += 1
#         return "retry"

# ── Conditional edge: retry or end ────────────────────────────────────────────
def should_retry(state: AgentState) -> str:
    passed = state["validation"].get("passed", False)
    retry_count = state["retry_count"]

    if passed:
        print(f"\n[Router] Validation PASSED → END")
        return "end"
    elif retry_count >= MAX_RETRIES:
        print(f"\n[Router] Max retries ({MAX_RETRIES}) reached → END (forced)")
        return "end"
    else:
        print(f"\n[Router] Validation FAILED → retry enrich_threat (retry {retry_count + 1}/{MAX_RETRIES})")
        return "retry"
        
# ── Build graph ────────────────────────────────────────────────────────────────
def build_graph():
    g = StateGraph(AgentState)
    g.add_node("classify_log", classify_log)
    g.add_node("enrich_threat", enrich_threat)
    g.add_node("generate_report", generate_report)
    g.add_node("validate_report", validate_report)

    g.set_entry_point("classify_log")
    g.add_edge("classify_log", "enrich_threat")
    g.add_edge("enrich_threat", "generate_report")
    g.add_edge("generate_report", "validate_report")
    g.add_conditional_edges(
        "validate_report",
        should_retry,
        {"end": END, "retry": "enrich_threat"}
    )
    return g.compile()

# ── Test cases ─────────────────────────────────────────────────────────────────
TEST_LOGS = [
    {
        "id": "TC-01",
        "description": "Brute force + data exfiltration",
        "log": (
            "2024-11-14T03:22:11Z WARN auth_service: 47 failed login attempts from 198.51.100.42 "
            "for user admin@corp.com within 60 seconds. Account locked. "
            "Subsequent successful login from same IP detected. "
            "Large outbound transfer (2.3GB) to 203.0.113.99 observed post-login."
        )
    },
    {
        "id": "TC-02",
        "description": "Ransomware lateral movement",
        "log": (
            "2024-11-14T11:45:03Z ERROR endpoint_protection: Suspicious process svchost_update.exe "
            "launched on WORKSTATION-047 by user jsmith. "
            "Process began encrypting files in \\\\FILESERVER\\HR_Docs and \\\\FILESERVER\\Finance. "
            "SMB lateral movement detected to 10.0.1.15, 10.0.1.22, 10.0.1.31. "
            "C2 beacon to 185.220.101.5:4444 detected."
        )
    },
    {
        "id": "TC-03",
        "description": "Compliance policy violation",
        "log": (
            "2024-11-14T14:10:55Z INFO dlp_agent: User contractor_bob downloaded 14,500 customer PII "
            "records (names, SSNs, credit card numbers) to personal USB device on KIOSK-003. "
            "DLP policy CF-104 triggered. User is not authorized for bulk PII export. "
            "Action occurred outside business hours."
        )
    },
]

# ── Run ────────────────────────────────────────────────────────────────────────
def run_all():
    graph = build_graph()
    results = []

    for tc in TEST_LOGS:
        print(f"\n{'='*60}")
        print(f"TEST CASE {tc['id']}: {tc['description']}")
        print(f"{'='*60}")

        initial_state: AgentState = {
            "log_entry": tc["log"],
            "classification": {},
            "enrichment": {},
            "report": {},
            "validation": {},
            "retry_count": 0,
            "retry_feedback": "",
            "history": []
        }

        start = time.time()
        final = graph.invoke(initial_state)
        elapsed = time.time() - start

        results.append({
            "id": tc["id"],
            "description": tc["description"],
            "log": tc["log"],
            "classification": final["classification"],
            "enrichment": final["enrichment"],
            "report": final["report"],
            "validation": final["validation"],
            "retry_count": final["retry_count"],
            "history": final["history"],
            "elapsed": round(elapsed, 2)
        })

        print(f"\n── FINAL REPORT ──")
        print(json.dumps(final["report"], indent=2))
        print(f"\n── VALIDATION ──")
        print(json.dumps(final["validation"], indent=2))
        print(f"\n── HISTORY ──")
        for h in final["history"]:
            print(f"  {h}")
        print(f"\n  Total time: {elapsed:.1f}s | Retries: {final['retry_count']}")

    # Save results for report generation
    with open("hw10_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n\nResults saved to hw10_results.json")
    return results

if __name__ == "__main__":
    run_all()