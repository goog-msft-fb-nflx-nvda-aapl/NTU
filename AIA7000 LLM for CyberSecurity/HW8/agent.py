# agent.py
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama

# ── 1. State Schema ────────────────────────────────────────────────────────────
class LogState(TypedDict):
    raw_log:        str
    severity:       str   # "benign" | "suspicious" | "critical"
    threat_context: str
    report:         str

# ── 2. LLM ─────────────────────────────────────────────────────────────────────
llm = ChatOllama(model="llama3.2", base_url="http://localhost:11434", temperature=0)

# ── 3. Nodes ───────────────────────────────────────────────────────────────────
# def classify_log(state: LogState) -> LogState:
#     prompt = f"""You are a cybersecurity analyst. Classify this log line.
# Reply with EXACTLY one word: benign, suspicious, or critical.

# Log: {state['raw_log']}

# Classification:"""
#     result = llm.invoke(prompt)
#     severity = result.content.strip().lower().split()[0]
#     if severity not in ("benign", "suspicious", "critical"):
#         severity = "suspicious"
#     return {**state, "severity": severity}

def classify_log(state: LogState) -> LogState:
    prompt = f"""You are a cybersecurity analyst. Classify this log line as benign, suspicious, or critical.

Rules:
- benign   : routine system events (cron, systemd, normal logins, scheduled tasks)
- suspicious: unusual but not confirmed malicious (brute force attempts, sudo to shell, unexpected commands)
- critical : confirmed attack indicators (root shell spawned, exploit executed, uid=0 shell, mass failures)

Red flags that are ALWAYS suspicious or critical:
- sudo used to launch /bin/bash or /bin/sh (interactive shell escalation)
- repeated failed passwords (brute force)
- uid=0 shell execution
- EXECVE with /bin/sh or /bin/bash

Reply with EXACTLY one word: benign, suspicious, or critical.

Log: {state['raw_log']}

Classification:"""
    result = llm.invoke(prompt)
    severity = result.content.strip().lower().split()[0]
    if severity not in ("benign", "suspicious", "critical"):
        severity = "suspicious"
    return {**state, "severity": severity}

def enrich_threat(state: LogState) -> LogState:
    prompt = f"""You are a threat intelligence analyst.
Given this log line, provide a 2-3 sentence threat context: what attack is likely occurring,
what the attacker's goal is, and what immediate action should be taken.

Log: {state['raw_log']}
Severity: {state['severity']}

Threat context:"""
    result = llm.invoke(prompt)
    return {**state, "threat_context": result.content.strip()}


def generate_report(state: LogState) -> LogState:
    threat_section = (
        f"Threat Context : {state['threat_context']}"
        if state["threat_context"]
        else "Threat Context : N/A — log classified as benign"
    )
    report = f"""
╔══════════════════════════════════════════════════════╗
║              INCIDENT REPORT                         ║
╚══════════════════════════════════════════════════════╝
Raw Log      : {state['raw_log']}
Severity     : {state['severity'].upper()}
{threat_section}
═══════════════════════════════════════════════════════
"""
    return {**state, "report": report}


# ── 4. Routing Logic ───────────────────────────────────────────────────────────
def route_by_severity(state: LogState) -> Literal["enrich_threat", "generate_report"]:
    if state["severity"] in ("suspicious", "critical"):
        return "enrich_threat"
    return "generate_report"


# ── 5. Build Graph ─────────────────────────────────────────────────────────────
def build_agent():
    graph = StateGraph(LogState)

    graph.add_node("classify_log",   classify_log)
    graph.add_node("enrich_threat",  enrich_threat)
    graph.add_node("generate_report", generate_report)

    graph.set_entry_point("classify_log")

    graph.add_conditional_edges(
        "classify_log",
        route_by_severity,
        {
            "enrich_threat":   "enrich_threat",
            "generate_report": "generate_report",
        }
    )

    graph.add_edge("enrich_threat", "generate_report")
    graph.add_edge("generate_report", END)

    return graph.compile()


agent = build_agent()