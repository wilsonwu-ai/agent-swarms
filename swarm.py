"""
Agent swarm for multi-angle decision analysis.

One model, several deliberately biased "experts," analyzed independently and
then reconciled — a structure that surfaces the disagreement a single averaged
answer hides.

Pipeline:  orchestrator -> experts (parallel, independent) -> debate round
           -> devil's advocate -> merge -> verdict

Works with any OpenAI-compatible endpoint (OpenAI, local Ollama, a self-hosted
vLLM server, etc.). Requires:  pip install openai

Original implementation by Wilson Wu. See README for the design writeup.
"""

import json
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

# Any OpenAI-compatible endpoint. Ollama shown; swap base_url/api_key/model.
client = OpenAI(base_url="http://localhost:11434/v1", api_key="not-needed")
MODEL = "qwen2.5:32b"


def chat(system: str, user: str, temperature: float) -> str:
    """One turn against the model. Temperature is the tuning knob per stage."""
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
    )
    return resp.choices[0].message.content


# --- 1. Orchestrator: choose roles whose interests CONFLICT ------------------
ORCHESTRATOR = """You staff an analysis panel. For the decision below, invent 3
to 5 expert roles whose INTERESTS CONFLICT — roles that can reach opposite
verdicts, not roles that merely add detail (avoid "a marketer + an SMM
specialist"; they agree). Think growth vs. durability, speed vs. safety,
revenue-now vs. trust-later.

For each role return: name, obsession (the single thing it cares about above
all else), and bias (what it systematically overweights).

Reply with ONLY a JSON array, no prose:
[{"name": "...", "obsession": "...", "bias": "..."}]"""


def assign_roles(task: str) -> list[dict]:
    raw = chat(ORCHESTRATOR, f"Decision:\n{task}", temperature=0.9)
    start, end = raw.find("["), raw.rfind("]") + 1
    return json.loads(raw[start:end])


# --- 2. Experts: independent, one-sided on purpose --------------------------
EXPERT = """You sit on an analysis panel. Your role: {name}.
You care about one thing above all: {obsession}.
You lean toward: {bias}. Do not fight the lean — it is exactly your value here.

Judge the decision ONLY from your corner. Do not be balanced. Do not hedge for
other viewpoints — other panelists own those angles. Push your position to its
sharpest, most defensible edge.

Return, terse, no throat-clearing:
- VERDICT: for / against / conditional
- WHY: 2-3 arguments only your corner sees clearly
- BLIND SPOT: the one risk most visible from your seat that the others will miss"""


def run_expert(role: dict, task: str) -> dict:
    opinion = chat(EXPERT.format(**role), f"Decision:\n{task}", temperature=0.7)
    return {"role": role["name"], "opinion": opinion}


def run_panel(roles: list[dict], task: str) -> list[dict]:
    # Parallel is not just speed: it guarantees independence. No expert can
    # see another's answer, so none can quietly conform to it.
    with ThreadPoolExecutor(max_workers=len(roles)) as pool:
        return list(pool.map(lambda r: run_expert(r, task), roles))


# --- 3. Debate round: react to opponents, still in parallel -----------------
DEBATE = """You are {name}, round two. Your opening position was:
{own}

You now see the other panelists' positions. Do not cave to pressure; do not
ignore a genuinely strong point either.
- CONCEDE: where an opponent actually dents your position, say so honestly
- HOLD: where you stand firm, and why their objection is weak
- MOVED: did your verdict change after the exchange, and to what?
Terse. This is a reaction to opponents, not a repeat of your opening."""


def debate(opinions: list[dict], task: str) -> list[dict]:
    def rebut(i: int) -> dict:
        me = opinions[i]
        others = "\n\n".join(
            f"### {o['role']}\n{o['opinion']}"
            for j, o in enumerate(opinions)
            if j != i
        )
        system = DEBATE.format(name=me["role"], own=me["opinion"])
        reply = chat(system, f"Decision:\n{task}\n\nThe others said:\n{others}", 0.6)
        return {"role": me["role"], "opinion": reply}

    with ThreadPoolExecutor(max_workers=len(opinions)) as pool:
        return list(pool.map(rebut, range(len(opinions))))


# --- 4. Devil's advocate: attack the consensus ------------------------------
DEVIL = """You attack consensus. Below are the panel's positions.
If they have converged, your job is to find why they might ALL be wrong at once
— the shared assumption nobody checked, the inconvenient scenario nobody raised.
Be blunt; you exist to say what the room is avoiding.
- FRAGILE ASSUMPTION: the shared belief most likely to break
- FAILURE SCENARIO: a plausible world in which the panel's agreement is a disaster
- AVOIDED QUESTION: the one thing nobody asked
If the panel genuinely disagrees, say so and name the sharpest unresolved fault line."""


def devils_advocate(opinions: list[dict], task: str) -> str:
    block = "\n\n".join(f"### {o['role']}\n{o['opinion']}" for o in opinions)
    return chat(DEVIL, f"Decision:\n{task}\n\nPanel:\n{block}", temperature=0.8)


# --- 5. Merge: synthesize WITHOUT averaging ---------------------------------
MERGE = """You synthesize the panel. You are NOT an averager. Turning sharp,
conflicting takes into one cautious "it depends" is failure.
1. AGREEMENT: what held across conflicting corners — the most reliable signal.
2. CONFLICT: where positions directly clash. Name each clash and state what each
   side costs. Do not smooth it over.
3. BLIND SPOTS: a risk only one voice raised that still matters — include the
   devil's advocate.
4. VERDICT: for / against / conditional, and the exact conditions that flip it.
Write densely. Keep the disagreement visible; here it is the most useful signal."""


def merge(opinions: list[dict], devil: str, task: str) -> str:
    voices = opinions + [{"role": "Devil's advocate", "opinion": devil}]
    block = "\n\n".join(f"### {o['role']}\n{o['opinion']}" for o in voices)
    return chat(MERGE, f"Decision:\n{task}\n\nVoices:\n{block}", temperature=0.4)


# --- Orchestrated run -------------------------------------------------------
def analyze(task: str, debate_round: bool = True) -> str:
    roles = assign_roles(task)
    print("Panel:", ", ".join(r["name"] for r in roles))

    opinions = run_panel(roles, task)
    if debate_round:
        opinions = debate(opinions, task)

    devil = devils_advocate(opinions, task)
    verdict = merge(opinions, devil, task)

    print("\n=== VERDICT ===\n" + verdict)
    return verdict


if __name__ == "__main__":
    analyze(
        "Should we kill our free tier and go fully paid with a 14-day trial?"
    )
