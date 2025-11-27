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

# NOESIS v2 Interactive Demo

**Live demonstration of self-improving field dynamics without language models**

Created by Lucas Postma • [Full Implementation](https://github.com/DataDyneSolutions/asps-noesis-core)

## What This Demo Shows

A working implementation of NOESIS that **evolves its own physics laws** in real-time. No training data. No language model. Just pure field dynamics discovering optimal energy functionals.

## Quick Start

1. **Open `noesis_demo.html` in any browser**
2. **Click "🧬 Meta-Learn"** 
3. **Watch the system discover its own physics**

That's it. No dependencies, no setup, no GPU required.

## What You're Actually Seeing

### Main Panel: Field Dynamics (F1)
- **Colored curves**: Entity position distributions on probability simplex
- **Real-time optimization**: Mirror descent, not gradient descent
- **Energy counter**: Watch it minimize without any training data

### Mirror Panel: Compressed Field (F2)
- **Thick lines**: Downsampled field representation
- **Dashed lines**: Memory of good configurations (ΔE-gated retention)
- **Green dots**: Stored successful configurations

### Evolution Panel (After clicking Meta-Learn):
- **Live coefficient mutation**: Watch TV, Reciprocity, Unity weights evolve
- **Discovery rate**: ~30% of mutations improve the system
- **No human intervention**: System discovers optimal physics automatically

## Key Experiments to Try

### 1. Basic Field Refinement
```
1. Click "Initialize Fields"
2. Click "Run Refinement"
3. Watch Energy drop from ~8.0 to ~1.0
```
**What's happening**: Fields self-organize through energy minimization alone

### 2. Meta-Learning Evolution
```
1. Click "🧬 Meta-Learn"
2. Watch Evolution panel for 100 generations
3. Note the coefficients that emerge
```
**What's happening**: System discovers its own optimal physics laws

### 3. Memory Retention Test
```
1. Run refinement multiple times
2. Check "Retained Configs" counter
3. Notice Memory Gamma adaptation based on improvement
```
**What's happening**: ΔE-gated memory keeps good solutions forever

## Controls

- **Mirror Descent Step (η)**: Learning rate for probability updates
- **Number of Entities**: How many interacting fields
- **Number of Slots**: Resolution of the probability distributions

## Why This Matters

**Traditional Neural Networks:**
- Need massive datasets
- Fixed loss functions
- Gradient descent on parameters
- Forget without replay

**This NOESIS Demo:**
- Zero external data (generates infinite synthetic examples)
- Evolves its own energy functional
- Mirror descent on probability manifolds
- Permanent memory through ΔE-gating

## Demonstration

Click Meta-Learn and watch the bottom panel. The system is literally **rewriting its own physics** by mutating energy coefficients and keeping improvements. 

After 100 generations, it discovers better coefficients than hand-designed ones.

**This is recursive self-improvement at the optimization level.**

## 📈 Typical Evolution Results

```
Generation 0:   TV=1.000, Reciprocity=0.100, Energy=8.234
Generation 25:  TV=0.823, Reciprocity=0.287, Energy=5.123 ✅ 
Generation 50:  TV=0.652, Reciprocity=0.531, Energy=3.445 ✅
Generation 100: TV=0.421, Reciprocity=0.894, Energy=1.023 ✅
```

The system discovered that less TV smoothness and more reciprocity penalty leads to better convergence. **We didn't tell it this - it figured it out.**

## 🛠️ Technical Details

- **Pure JavaScript**: No libraries, no dependencies
- **Runs anywhere**: Browser, Node.js, even mobile
- **Real-time visualization**: Canvas-based rendering at 60fps
- **Complete implementation**: All core NOESIS concepts included

## 📝 Code Structure

```javascript
class NOESIS {
    // Core fields
    P: Position field [Entities × Slots]
    R: Relation tensor [E × E × Slots × Dims]  
    α: Affinity vector [Entities]
    
    // Evolution
    coefficients: { tv, reciprocity, unity }  // These mutate!
    
    // Operations
    computeEnergy()      // Energy functional
    step()              // Mirror descent update
    metaLearnStep()     // Evolve coefficients
}
```

## Share This Demo

Single HTML file, no server required:
- Email it
- Host on GitHub Pages  
- Run locally
- Embed in presentations

## Still Skeptical?

1. **Check line 140**: Synthetic Gaussian bumps, not text data
2. **Check line 298**: Energy functional, not cross-entropy
3. **Check line 542**: Coefficient mutation, not backpropagation
4. **Check line 623**: ΔE memory gating, not experience replay

## Learn More

- [Full Python Implementation](https://github.com/DataDyneSolutions/asps-noesis-core)
- [NOESIS Paper](#) (coming soon)
- [X/Twitter: @BeingAsSuch](https://x.com/BeingAsSuch)

---

**Remember**: This isn't training on data. It's discovering physics.
