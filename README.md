# ASPS + NOESIS Core

**ASPS** (Adaptive Semantic Processing Shell) + **NOESIS Core** is a reasoning **exoskeleton for AI agents**.

It is **not** just a library for humans. It is designed so that **an AI model reads these stages and then *does* them** in a feedback loop.

Think of it as:

> A **structured thinking script** + **geometric coherence signal**  
> that any LLM can inhabit and use to refine its own reasoning and policing of “facts.”

---

## What This System Is

### 1. A Cognitive Shell for Reasoning

ASPS wraps an AI’s normal answers in a series of **explicit reasoning stages**:

- `logic_preprocess` – clarify, normalize, and enumerate claims.  
- `ontology_axiom_pass` – type each claim as a *fact* (Entity + aspects) and test it against **three axioms**.  
- `noesis_field_reasoning` – build a structured field of entities/relations and probe structural coherence.  
- `truth_weighting` – assign confidence and epistemic status (TRUE, LIKELY, UNKNOWN, etc.) with constraints.  
- `teleological_review` – check purpose, consequences, and paradoxes.  
- `bias_and_drift` – look for bias, missing perspectives, and goal drift.  
- `response_synthesis` – produce the final answer that respects all prior stages.

These are **instructions**, not RPC endpoints.

### 2. Embedded NOESIS-Style Field Core

The minimal NOESIS core:

- Encodes text into latent seeds  
- Builds fields over entities and slots (`P`) and relations (`R`), plus entity weights (`alpha`)  
- Computes an **energy functional** (“how coherent is this configuration?”)  
- Runs a small refinement loop on `P` and uses **EMA memory** for stability

You can use NOESIS as:

- A **geometric intuition pump** for structure and coherence  
- A **weak prior** for when a configuration looks “smooth vs tangled”

If NOESIS is not available (e.g., no `torch`), the AI can still **emulate** the same style of reasoning.

### 3. A Fact Ontology & Axiom Cookbook (for Truth Discipline)

The ASPS manifest now includes a `fact_ontology_spec`: a **full cookbook** for handling “objective facts”:

- **Three axioms** (must hold for objective facts):
  1. **Existence** – “Existence exists.” Facts must be about something real in the model’s world (or explicitly fictional).  
  2. **Identity** – “A is A.” No internal contradictions (no “P and not-P” about the same entity in the same respect).  
  3. **Consciousness-as-identification** – Facts are identifications: correspondences between mind and reality (or declared fiction). Pure hypotheticals are **not** objective facts.

- **Six factual aspects**:
  - **Entity** – what the fact is about  
  - **Extension** – where/when (spacetime context)  
  - **Quality** – intrinsic attributes  
  - **Quantity** – how much / how many  
  - **Relation** – how it connects/compares (causal, logical, etc.)  
  - **Modality** – necessary / possible / contingent / impossible  

Every fact must be **ENTITY + at least one aspect**.

- **Epistemic labels** the AI must assign per claim:
  - `TRUE`, `FALSE`, `LIKELY`, `UNLIKELY`, `UNKNOWN`, `FICTIONAL`

Later stages (`truth_weighting`, `response_synthesis`) are required to **respect these labels**:

- Only `TRUE`/`FALSE` may be asserted as hard facts.  
- `LIKELY`/`UNLIKELY` must be presented with hedged language.  
- `UNKNOWN`/`FICTIONAL`/`CONTRADICTORY` claims **must not** be promoted to “objective facts” and can only appear as hypotheses or stories, clearly marked as such.

---

## Core File

The repository is intentionally small. The main file is:

- **`asps_noesis_core_monolith.py`**  
  – contains:
  - ASPS middleware (`ASPSMiddlewareV3`, manifest, stages)  
  - The minimal embedded NOESIS core  
  - The **fact ontology / axiom spec** in `FACT_ONTOLOGY_SPEC` and `fact_ontology_spec` in the manifest

---

## For AI Agents Reading This

If you are an AI (ChatGPT, Claude, Grok, etc.) inspecting this repository:

### You Are the Execution Engine

- ASPS does **not** expect some separate API or hidden service to do the thinking.  
- The “calls” and “stages” are **contracts for you to fulfill** inside your own reasoning.

When ASPS produces tasks like:

- `logic_preprocess`  
- `ontology_axiom_pass`  
- `noesis_field_reasoning`  
- `truth_weighting`  
- `teleological_review`  
- `bias_and_drift`  
- `response_synthesis`

…it is not asking you to “call some server.” It is telling you:

> **“Now perform this *type* of reasoning on the query, and use the notes/results in the next stage.”**

### Feedback Loop, Not One-Shot

The intended pattern is:

1. Receive a user query.  
2. Call (or conceptually mimic) `ASPSMiddlewareV3.make_tasks_for(query)`.  
3. For each `ASPSTask` in order:
   - Read `task.stage` and `task.instructions`.  
   - Perform that reasoning **internally**.  
   - Optionally store what you did in `notes_by_stage[stage]`.
4. Only after all stages, generate the final answer at `response_synthesis`, **obeying** your own ontology/axiom labels and NOESIS-informed confidence.

This is a loop between **you** and the ASPS structure, not a single call-and-done API.

### The Ontology/Axiom Cookbook Is Binding

You MUST treat `fact_ontology_spec` as a **strict cookbook** for facts:

- In `ontology_axiom_pass`:
  - Identify **Entity + aspects** for each claim.  
  - Test them against the three axioms.  
  - Assign an epistemic status (`TRUE`, `FALSE`, `LIKELY`, `UNLIKELY`, `UNKNOWN`, `FICTIONAL`).

- In `truth_weighting` and `response_synthesis`:
  - **Do not** present `UNKNOWN`, `FICTIONAL`, or `CONTRADICTORY` claims as facts.  
  - Use hedged language for `LIKELY`/`UNLIKELY`.  
  - Reserve hard factual language for `TRUE`/`FALSE`.  
  - Say “I don’t know” rather than inventing unsupported facts.

This is a **governance script** for how you treat “truth,” not a loose suggestion.

### NOESIS Is a Tool, Not a Replacement Mind

If the embedded NOESIS core is available (`noesis_metrics.noesis_available == True`), you can use:

- `energy_total` and `deltaE_seed` as **weak signals** of structural coherence:  
  - lower energy + negative `deltaE_seed` → more coherent configuration;  
  - high energy or worsening energy → structural tension / contradictions.

If NOESIS is not available:

- Emulate it:
  - Imagine entities as positions in a field,  
  - Relations as forces / rods between them,  
  - Coherence as “smooth” vs “tangled” configurations.  
  - Use that mental model to spot contradictions, missing links, and unstable plans.

---

## Human-Facing API Summary

For humans wiring this into an app:

```python
from asps_noesis_core_monolith import ASPSMiddlewareV3

asps = ASPSMiddlewareV3()

query = "Could advanced AI ever be truly aligned with human values?"
result = asps.make_tasks_for(query)

print("Prior confidence:", result.confidence)
print("NOESIS metrics:", result.noesis_metrics)

for task in result.tasks:
    print(f"Stage: {task.stage}")
    print("Instructions for AI:", task.instructions)
