# ASPS-NOESIS Core

### A Complete Framework for Structured AI Reasoning & Epistemic Auditing

**Author:** Lucas Postma ([@BeingAsSuch](https://x.com/BeingAsSuch))  
**Organization:** DataDyne Solutions LLC  
**License:** Apache 2.0  
**Version:** 3.0

---

## Foreword

This framework represents years of thinking about how to make AI reasoning auditable, structured, and transparent. I'm releasing it because I believe these tools should exist in the world — not locked away in a private repo.

**I encourage you to:**
- Experiment freely with ASPS, NOESIS, and AIAT
- Build on these ideas, extend them, break them, improve them
- Use them in your research, projects, or products

**I ask in return:**
- If this work helps you, please attribute it. A citation or acknowledgment goes a long way.
- If you build something interesting with it, reach out — I'd love to see what you create.
- If you're working on related problems, let's collaborate.

There is significantly more work not yet on this repo — theoretical foundations, additional implementations, and ideas I need help bringing to life for the greater good of the AI safety and alignment community. If you're interested in contributing or collaborating, DM me on X [@BeingAsSuch](https://x.com/BeingAsSuch).

Let's build something meaningful together.

— Lucas Postma

---

## Overview

This repository contains three interconnected systems for structured AI reasoning:

| Component | Description | Language |
|-----------|-------------|----------|
| **ASPS** | Axiomatic Structured Processing Shell — middleware that enforces rigorous epistemic reasoning | Python |
| **NOESIS** | Neural field dynamics engine — triadic representations with energy-based refinement | PyTorch |
| **AIAT** | AI Audit Trail — visual physics simulation for epistemic coherence | React |

Together, they address the **black box problem** in AI: we can't see *why* AI systems believe what they believe.

---

## ASPS: Axiomatic Structured Processing Shell

**File:** `asps_noesis_core_monolith.py`

ASPS is cognitive middleware that structures and audits AI reasoning through mandatory stages:

### Processing Pipeline

1. **Logic Preprocess** — Clarify terms, resolve ambiguities, extract explicit claims
2. **Ontology/Axiom Pass** — Apply three axioms (Existence, Identity, Consciousness) and six aspects (Entity, Extension, Quality, Quantity, Relation, Modality)
3. **NOESIS Field Reasoning** — Build structured field of entities and relations
4. **Truth Weighting** — Estimate claim likelihood respecting ontology classifications
5. **Teleological Review** — Examine purpose, consequences, potential paradoxes
6. **Bias & Drift Check** — Flag biases, missing perspectives, goal drift
7. **Response Synthesis** — Produce final answer integrating all stages

### Three Axioms (Must Hold for Objective Facts)

```
Axiom 1 – Existence: "Existence exists."
Axiom 2 – Identity: "A is A." (No contradictions)
Axiom 3 – Consciousness: Facts are successful identifications of reality.
```

### Epistemic Status Labels

Claims must be classified as: `TRUE`, `FALSE`, `LIKELY`, `UNLIKELY`, `UNKNOWN`, or `FICTIONAL`

```python
from asps_noesis_core_monolith import ASPSMiddlewareV3

asps = ASPSMiddlewareV3()
result = asps.make_tasks_for("What are the core challenges in building safe AGI?")
print(result.to_json())
```

---

## NOESIS: Neural Field Dynamics Engine

**File:** `noesis_v2_complete.py`

NOESIS models reasoning as **triadic field dynamics** — entities, relations, and attention weights that evolve toward coherent configurations.

### Core Architecture

```
Text → Encoder → Seed Fields (P, R, α) → Refinement Loop → Energy Minimization
         ↓                                      ↑
    Rod Attention ←──── Mirror Field (F2) ──────┘
         ↓
    Field Memory (EMA)
```

### Key Components

| Component | Function |
|-----------|----------|
| **FieldShape** | Defines triadic representation dimensions |
| **FieldOps** | Energy functional + totality/reciprocity operators |
| **DynamicRodAttention** | O(N×M) attention with token↔rod interaction |
| **FieldMemory** | EMA retention gated by energy improvement |
| **BasisExpander** | Coefficients → slot distributions |
| **FieldPyramid** | Multi-scale mirror field (F2) for coarse guidance |

### Energy Function

```python
E = TV_smoothness(P) + λ_rel * reciprocity(R) + λ_aff * ||α||² + supervision
```

Where:
- `P[B,E,S]` — Entity distributions over slots
- `R[B,E,E,S,D]` — Pairwise relation tensors
- `α[B,E]` — Entity attention weights

### Refinement via Mirror Descent

```python
# P update (entropic)
log_P = log(P) - η * ∇_P(E)
P_new = softmax(log_P)

# R update (momentum)
v_R = β * v_R + (1-β) * ∇_R(E)
R_new = R - η_R * v_R
```

### Usage

```python
from noesis_v2_complete import TriadicNoesisParallelMemFast, FieldShape, noesis_reflect

# Quick reflection
result = noesis_reflect("Is AGI coming before 2030?", steps=2)
print(f"Energy improvement: {result['dE']:.4f}")

# Full model
shape = FieldShape(n_entities=8, n_basis=16, d_relation=8, d_latent=64)
model = TriadicNoesisParallelMemFast(
    vocab=256, shape=shape, n_slots=64,
    use_dynamic_rod_attn=True, n_rods=64
)
```

---

## AIAT: AI Audit Trail (Epistemic Coherence Field)

**File:** `AIAT-Epistemic-Coherence-Field.jsx` | **Demo:** `noesis_demo.html`

AIAT is a visual physics simulation that transforms abstract reasoning into an interactive force-directed graph.

### The ECF Model

Claims are **charged particles** where:
- **Contradictions repel** (like negative charges)
- **Supporting claims attract** (like positive charges)  
- **System energy = incoherence** (lower = more coherent)

### Energy Function

```
E = Σ_supports (d_ij / 120) · s · c_ij · k_s · 0.5
  + Σ_contradicts (300 / d_ij) · s · c_ij · k_c
  + w_c · N_contested + w_u · N_ungrounded
```

### Dual Epistemic Stance Framework

| Stance | Penalizes | Weights |
|--------|-----------|---------|
| **Coherentist** | Conflict | w_c=0.6, w_u=0.15 |
| **Foundationalist** | Ungrounded claims | w_c=0.2, w_u=0.9 |

This operationalizes a 300-year philosophical debate into a practical toggle.

### Features

- ✅ Knowledge-backed vs. heuristic relation detection
- ✅ Overconfidence warnings for post-hoc rationalization
- ✅ Counterfactual analysis ("What if we remove this claim?")
- ✅ Tunable priors with sensitivity analysis
- ✅ LessWrong AI safety benchmarks

---

## How They Connect

```
┌─────────────────────────────────────────────────────────────┐
│                      USER QUERY                             │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                         ASPS                                │
│  • Extract claims                                           │
│  • Apply axioms (Existence, Identity, Consciousness)        │
│  • Classify epistemic status                                │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                        NOESIS                               │
│  • Encode text → latent seed                                │
│  • Build triadic field (P, R, α)                            │
│  • Refine via energy minimization                           │
│  • Update field memory                                      │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                         AIAT                                │
│  • Visualize claims as particles                            │
│  • Show contradictions/support as forces                    │
│  • Resolve to equilibrium                                   │
│  • Display coherence metrics                                │
└─────────────────────────────────────────────────────────────┘
```

---

## Use Cases

- **AI Alignment Research** — Audit reasoning chains for hidden contradictions
- **LLM Cognitive Shell** — Enforce structured reasoning in language models
- **Debate Analysis** — Visualize logical structure of complex arguments
- **Red-Teaming** — Stress-test belief systems under different assumptions
- **Education** — Teach epistemology through interactive simulation

---

## Citation & Attribution

### Academic Citation

If you use ASPS, NOESIS, or the Epistemic Coherence Field concept in academic work, please cite:

```bibtex
@software{asps_noesis2025,
  author = {Lucas Postma},
  title = {ASPS-NOESIS Core: Structured AI Reasoning & Epistemic Coherence Fields},
  year = {2025},
  url = {https://github.com/DataDyneSolutions/asps-noesis-core},
  note = {A framework for auditable AI reasoning with physics-based epistemic visualization}
}
```

### Derivative Works

**If you build upon, extend, or adapt these concepts, please provide attribution:**

> "Based on the ASPS-NOESIS framework by Lucas Postma (@BeingAsSuch), 2025"

This includes:
- Research papers using the ECF energy function, NOESIS field dynamics, or ASPS axiom system
- Software tools implementing similar epistemic visualization or structured reasoning
- Educational materials explaining these concepts
- Commercial products derived from this work

---

## License

**Licensed under the Apache License, Version 2.0**

Copyright 2025 Lucas Postma (@BeingAsSuch)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

**Attribution is requested (not required by license) for these concepts in derivative works.**

---

## Contributing

Interested in extending the framework? Ideas welcome:

- Expanding the AIAT knowledge base
- Adding new epistemic stances (Pragmatist? Bayesian?)
- Improving NOESIS rod attention efficiency
- Building API endpoints for programmatic access
- Integrating with existing LLM frameworks

Open an issue or DM [@BeingAsSuch](https://x.com/BeingAsSuch).

---

## Links

- **Live Demo**: [Claude Artifact](https://claude.ai) *(publish link here)*
- **Author**: [@BeingAsSuch](https://x.com/BeingAsSuch)
- **Organization**: [DataDyneSolutions](https://github.com/DataDyneSolutions)

---

*"The order and connection of ideas is the same as the order and connection of things."*  
— Spinoza, Ethics IIP7

---

**Built by [Lucas Postma (@BeingAsSuch)](https://x.com/BeingAsSuch)**
