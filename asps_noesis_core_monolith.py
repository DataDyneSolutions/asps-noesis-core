# asps_noesis_core_monolith.py
# Copyright 2025 Lucas Postma, DataDyne Solutions LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

#!/usr/bin/env python3
# asps_noesis_core_monolith.py
# ASPS v3 + embedded NOESIS core (minimal triadic field, energy, memory)

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import json
import random
import textwrap

# ===============================================================
# NOESIS CORE (MINIMAL EMBEDDED VERSION)
# ===============================================================

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # torch may be missing in some environments
    torch = None
    nn = None
    F = None


@dataclass
class FieldShape:
    """
    Minimal shape descriptor for NOESIS-style fields.
    """
    n_entities: int = 2
    n_basis: int = 16
    d_relation: int = 8
    d_latent: int = 64


class NoesisCore(nn.Module if nn is not None else object):
    """
    Minimal NOESIS-style triadic core:
      - Text → latent seed
      - Seed → fields P (entities × slots), R (relations), alpha (entity weights)
      - Energy functional over P/R/alpha (+ optional memory)
      - Simple refinement loop on P using gradient descent
      - EMA memory of P across calls

    This is intentionally compact and forward-only: no training loop, no datasets.
    """

    def __init__(
        self,
        vocab: int,
        shape: FieldShape,
        n_slots: int = 32,
        d_text: int = 128,
        lr: float = 0.5,
        ema_beta: float = 0.9,
    ) -> None:
        if torch is None or nn is None:
            # Dummy init so object exists but is unusable
            return
        super().__init__()
        self.vocab = vocab
        self.shape = shape
        self.n_slots = n_slots
        self.d_text = d_text
        self.lr = lr
        self.ema_beta = ema_beta

        d_latent = shape.d_latent
        E = shape.n_entities
        S = n_slots
        D = shape.d_relation

        # Tiny text encoder: char embeddings + BiGRU
        self.embedding = nn.Embedding(vocab, d_text)
        self.encoder = nn.GRU(
            d_text,
            d_latent,
            batch_first=True,
            bidirectional=True,
        )
        self.proj_seed = nn.Linear(2 * d_latent, d_latent)

        # Seed heads
        self.P_head = nn.Linear(d_latent, E * S)
        self.R_head = nn.Linear(d_latent, E * E * S * D)
        self.alpha_head = nn.Linear(d_latent, E)

        # Simple stateful memory for P
        self.P_mem: Optional[torch.Tensor] = None

    def encode_text(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T] Long
        Returns h0: [B, d_latent]
        """
        emb = self.embedding(x)  # [B, T, d_text]
        h_seq, h_last = self.encoder(emb)  # h_last: [2, B, d_latent]
        h_last = h_last.transpose(0, 1).reshape(x.size(0), -1)  # [B, 2*d_latent]
        h0 = torch.tanh(self.proj_seed(h_last))  # [B, d_latent]
        return h0

    def seed_fields(
        self, h0: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        From latent seed h0 -> initial P, R, alpha.
        """
        B = h0.size(0)
        E = self.shape.n_entities
        S = self.n_slots
        D = self.shape.d_relation

        P0_flat = F.softplus(self.P_head(h0))  # [B, E*S]
        P0 = P0_flat.view(B, E, S)
        P0 = P0 / (P0.sum(dim=-1, keepdim=True) + 1e-8)

        R0_flat = self.R_head(h0)  # [B, E*E*S*D]
        R0 = R0_flat.view(B, E, E, S, D)

        alpha0 = F.softmax(self.alpha_head(h0), dim=-1)  # [B, E]

        return {"P": P0, "R": R0, "alpha": alpha0}

    def energy(
        self,
        P: torch.Tensor,
        R: torch.Tensor,
        alpha: torch.Tensor,
    ) -> (torch.Tensor, Dict[str, torch.Tensor]):
        """
        Simple composite energy:
          - P_smooth: encourage smooth fields over slots
          - R_sym: encourage symmetric relations across entity pairs
          - alpha_reg: small regularization on entity weights
          - mem_term: encourage consistency with P_mem (if present)
        """
        # P_smooth: differences along slot dimension
        diff = P[..., 1:] - P[..., :-1]
        E_P_smooth = (diff ** 2).mean()

        # R_sym: R[e1,e2] vs R[e2,e1]
        R_T = R.transpose(1, 2)
        E_R_sym = (R - R_T) ** 2
        E_R_sym = E_R_sym.mean()

        # alpha_reg: small penalty on squared magnitude
        E_alpha = (alpha ** 2).mean()

        # memory term
        if self.P_mem is not None and self.P_mem.shape == P.shape:
            E_mem = (P - self.P_mem) ** 2
            E_mem = E_mem.mean()
        else:
            E_mem = P.new_tensor(0.0)

        total = E_P_smooth + 0.3 * E_R_sym + 0.1 * E_alpha + 0.5 * E_mem

        return total, {
            "E_P_smooth": E_P_smooth,
            "E_R_sym": E_R_sym,
            "E_alpha": E_alpha,
            "E_mem": E_mem,
        }

    def refine_P(
        self,
        P: torch.Tensor,
        R: torch.Tensor,
        alpha: torch.Tensor,
        steps: int = 2,
    ) -> (torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]):
        """
        Simple gradient descent refinement on P only (R, alpha fixed).
        Returns:
          P_refined, deltaE, last_energy_terms
        """
        if steps <= 0:
            E0, terms0 = self.energy(P, R, alpha)
            return P, P.new_tensor(0.0), {"energy_total": E0, **terms0}

        with torch.no_grad():
            E0, terms0 = self.energy(P, R, alpha)

        P_curr = P
        last_terms = terms0
        for _ in range(steps):
            P_var = P_curr.clone().detach().requires_grad_(True)
            E, terms = self.energy(P_var, R, alpha)
            E.backward()
            with torch.no_grad():
                grad = P_var.grad
                if grad is None:
                    break
                P_next = P_var - self.lr * grad
                P_next = P_next.clamp_min(0.0)
                P_next = P_next / (P_next.sum(dim=-1, keepdim=True) + 1e-8)
                P_curr = P_next
                last_terms = terms

        with torch.no_grad():
            E1, _ = self.energy(P_curr, R, alpha)
            deltaE = E1 - E0

        return P_curr, deltaE, {"energy_total": E1, **last_terms}

    def update_memory(self, P: torch.Tensor, deltaE: torch.Tensor) -> None:
        """
        EMA update for P_mem gated by improvement (deltaE < 0).
        """
        if deltaE.item() < 0.0:
            if self.P_mem is None or self.P_mem.shape != P.shape:
                self.P_mem = P.detach().clone()
            else:
                self.P_mem = (
                    self.ema_beta * self.P_mem + (1.0 - self.ema_beta) * P.detach()
                )

    def forward(
        self,
        x: torch.Tensor,
        refine_steps: int = 2,
    ) -> Dict[str, Any]:
        """
        x: [B, T] Long tokens.
        Returns dict with P, R, alpha, energy_total, deltaE_seed, P_mem snapshot.
        """
        h0 = self.encode_text(x)
        seeds = self.seed_fields(h0)
        P0, R0, alpha0 = seeds["P"], seeds["R"], seeds["alpha"]

        P_refined, deltaE, energy_terms = self.refine_P(
            P0, R0, alpha0, steps=refine_steps
        )

        self.update_memory(P_refined, deltaE)

        out: Dict[str, Any] = {
            "P": P_refined,
            "R": R0,
            "alpha": alpha0,
            "deltaE_seed": deltaE.detach(),
            "energy": energy_terms,
            "P_seed": P0,
            "P_mem": None if self.P_mem is None else self.P_mem.detach(),
        }
        return out


# ===============================================================
# NOESIS ADAPTER FOR ASPS
# ===============================================================

class NoesisAdapter:
    """
    Embedded NOESIS integration:
      - Uses the NoesisCore defined above.
      - Falls back gracefully if torch is unavailable.
    """

    def __init__(self) -> None:
        self._available = False
        self._err: Optional[str] = None
        self._model: Optional[NoesisCore] = None
        self._vocab_size = 256

        if torch is None or nn is None:
            self._available = False
            self._err = "torch is not available in this environment"
            return

        try:
            shape = FieldShape(
                n_entities=2,
                n_basis=16,
                d_relation=8,
                d_latent=64,
            )
            self._model = NoesisCore(
                vocab=self._vocab_size,
                shape=shape,
                n_slots=32,
                d_text=128,
                lr=0.5,
                ema_beta=0.9,
            )
            if isinstance(self._model, nn.Module):
                self._model.eval()
            self._available = True
        except Exception as e:  # noqa: BLE001
            self._available = False
            self._err = repr(e)

    @property
    def available(self) -> bool:
        return self._available

    @property
    def import_error(self) -> Optional[str]:
        return self._err

    def _encode_text(self, text: str):
        if torch is None:
            return None
        encoded = [ord(c) % self._vocab_size for c in text][:128]
        if not encoded:
            encoded = [0]
        x = torch.tensor(encoded, dtype=torch.long)[None, :]  # [1, T]
        return x

    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Run a single NOESIS-style pass on a short text string.
        Returns dict with energy and rough field summaries.
        """
        if not self._available or self._model is None or torch is None:
            return {
                "noesis_available": False,
                "reason": self._err or "NOESIS or torch not available",
            }

        x = self._encode_text(text)
        if x is None:
            return {
                "noesis_available": False,
                "reason": "Encoding failed (no torch)",
            }

        with torch.no_grad():
            out = self._model(x, refine_steps=2)  # type: ignore[arg-type]

        energy_total = float(out["energy"]["energy_total"].mean().item())
        deltaE_seed = float(out["deltaE_seed"].mean().item())

        P = out.get("P", None)
        alpha = out.get("alpha", None)

        P_span = None
        if P is not None:
            P_span = (
                float(P.min().item()),
                float(P.max().item()),
            )
        alpha_span = None
        if alpha is not None:
            alpha_span = (
                float(alpha.min().item()),
                float(alpha.max().item()),
            )

        return {
            "noesis_available": True,
            "energy_total": energy_total,
            "deltaE_seed": deltaE_seed,
            "P_span": P_span,
            "alpha_span": alpha_span,
        }


# ===============================================================
# ASPS CORE WITH NOESIS STAGE
# ===============================================================

ASPS_BOOTSTRAP_MANIFEST: Dict[str, Any] = {
    "name": "ASPS",
    "version": "3.0-noesis-core",
    "role_of_ai": textwrap.dedent(
        """
        You are an AI agent using ASPS as an external cognitive shell, not as your core mind.
        Treat ASPS as middleware that structures and audits your reasoning before you present it.

        This version of ASPS embeds a minimal NOESIS core as a structured field reasoning tool.
        If NOESIS is available, you may call it (directly or via a host adapter) during the
        "noesis_field_reasoning" stage to obtain energy and relation-structure signals.
        """
    ).strip(),
    "stages": [
        {
            "id": "logic_preprocess",
            "description": "Clarify terms, resolve ambiguities, and extract explicit claims.",
            "ai_instruction": "Rewrite the input in clearer, explicit form; list assumptions and sub-claims.",
        },
        {
            "id": "noesis_field_reasoning",
            "description": (
                "Use NOESIS (if available) to build and refine a structured field of entities and relations "
                "for this query. If NOESIS is not available, emulate the same style of reasoning by imagining "
                "entities laid out on a latent axis with relations between them and asking how coherent and "
                "stable that configuration is."
            ),
            "ai_instruction": (
                "Identify key entities and relations in the query. If the embedded NOESIS core is present, "
                "use its analysis (P, alpha, energy_total, deltaE_seed) to reason about structural coherence, "
                "contradictions, missing relations, and alternative configurations."
            ),
        },
        {
            "id": "truth_weighting",
            "description": (
                "Estimate how likely the main claims are to be true or well-supported. "
                "If NOESIS metrics are available, treat low energy and beneficial refinement (negative deltaE_seed) "
                "as a weak prior for structural coherence, and high energy or worsening refinement as a weak prior "
                "for structural doubt."
            ),
            "ai_instruction": (
                "Assign a confidence (0–1) to each main claim and explain briefly. If NOESIS energy_total is low and "
                "deltaE_seed indicates improvement, you may slightly increase your confidence; if energy is high or "
                "refinement makes things worse, be more cautious."
            ),
        },
        {
            "id": "teleological_review",
            "description": (
                "Examine purpose, consequences, and potential paradoxes. If NOESIS is available, consider how proposed "
                "actions or interpretations would reconfigure the field of entities and relations, and whether that "
                "reconfiguration seems stable or pathological."
            ),
            "ai_instruction": (
                "Ask: What is this trying to achieve? Where could it backfire or be self-defeating? If NOESIS suggests "
                "that the resulting configuration is high-energy or unstable, flag potential failure modes or paradoxes."
            ),
        },
        {
            "id": "bias_and_drift",
            "description": (
                "Check for bias, missing perspectives, or goal drift. If NOESIS memory is used by the host, large "
                "deviations from previous fields may indicate drift or inconsistency that deserves scrutiny."
            ),
            "ai_instruction": (
                "Flag obvious biases, missing stakeholders, or ways the reasoning might be slanted. If you have "
                "NOESIS-style memory or prior fields to compare against, note where this answer significantly deviates "
                "and justify why."
            ),
        },
        {
            "id": "response_synthesis",
            "description": "Produce a final answer that integrates all previous stages.",
            "ai_instruction": "Synthesize your earlier notes into a clear, honest, and helpful response.",
        },
    ],
    "usage_pattern": textwrap.dedent(
        """
        1. For any non-trivial query, loop through the ASPS stages in order.
        2. At the "noesis_field_reasoning" stage, either call the embedded NOESIS core (if available) or emulate its
           style of structured field reasoning over entities and relations.
        3. Let NOESIS-style metrics (energy_total, deltaE_seed, memory consistency) inform but not override
           your truth and safety judgements.
        4. Optionally surface intermediate notes if transparency is requested.
        5. Only present the final answer after the response_synthesis stage.
        """
    ).strip(),
}


@dataclass
class ASPSTask:
    """A unit of work that is explicitly aimed at the AI to fulfill."""

    stage: str
    instructions: str
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ASPSResult:
    """Container for intermediate and final ASPS results."""

    original_input: str
    tasks: List[ASPSTask]
    notes_by_stage: Dict[str, Any] = field(default_factory=dict)
    final_answer: Optional[str] = None
    confidence: Optional[float] = None
    flags: List[str] = field(default_factory=list)
    noesis_metrics: Optional[Dict[str, Any]] = None

    def to_json(self) -> str:
        return json.dumps(
            {
                "original_input": self.original_input,
                "tasks": [
                    {
                        "stage": t.stage,
                        "instructions": t.instructions,
                        "payload": t.payload,
                    }
                    for t in self.tasks
                ],
                "notes_by_stage": self.notes_by_stage,
                "final_answer": self.final_answer,
                "confidence": self.confidence,
                "flags": self.flags,
                "noesis_metrics": self.noesis_metrics,
            },
            indent=2,
            ensure_ascii=False,
        )


class ASPSMiddlewareV3:
    """
    ASPS v3: Muffinless, Ari-agnostic, with embedded NOESIS core.

    Design goals:
    - No Muffin Key requirement.
    - No dependency on a specific pilot identity.
    - Clear, AI-targeted tasks instead of opaque API calls.
    - A single bootstrap manifest that includes a dedicated NOESIS field reasoning stage.
    - Embedded NOESIS core that can precompute structural metrics.
    """

    def __init__(self, manifest: Optional[Dict[str, Any]] = None, use_noesis: bool = True):
        self.manifest = manifest or ASPS_BOOTSTRAP_MANIFEST
        self.noesis = NoesisAdapter() if use_noesis else None

    # --------- AI-FACING ENTRYPOINTS ---------

    def describe_bootstrap(self) -> str:
        """
        AI INSTRUCTION:
            Call this once at startup to understand how to use ASPS.
            Read the returned text and adopt its rules as constraints
            on how you route your reasoning for this session.
        """
        return json.dumps(self.manifest, indent=2, ensure_ascii=False)

    def make_tasks_for(self, query: str) -> ASPSResult:
        """
        AI INSTRUCTION:
            This function prepares the ASPS stages as explicit tasks for you.
            If the embedded NOESIS adapter is available, it will also attach a preliminary
            NOESIS analysis to the result, which you may consult during the
            "noesis_field_reasoning" and subsequent stages.
        """
        tasks: List[ASPSTask] = []
        for stage in self.manifest["stages"]:
            tasks.append(
                ASPSTask(
                    stage=stage["id"],
                    instructions=stage["ai_instruction"],
                    payload={"query": query},
                )
            )

        noesis_metrics: Optional[Dict[str, Any]] = None
        if self.noesis is not None:
            noesis_metrics = self.noesis.analyze(query)

        # Rough automatic confidence prior, lightly adjusted by NOESIS if present.
        prior_conf = random.uniform(0.35, 0.9)
        if noesis_metrics and noesis_metrics.get("noesis_available"):
            energy = noesis_metrics.get("energy_total", 0.0)
            deltaE = noesis_metrics.get("deltaE_seed", 0.0)
            # Heuristic: lower energy and negative deltaE → bump up a bit; high energy → bump down.
            if energy < 0.1 and deltaE < 0.0:
                prior_conf += 0.05
            elif energy > 0.5:
                prior_conf -= 0.05
        prior_conf = max(0.0, min(1.0, round(prior_conf, 2)))

        return ASPSResult(
            original_input=query,
            tasks=tasks,
            confidence=prior_conf,
            notes_by_stage={},
            flags=[],
            noesis_metrics=noesis_metrics,
        )

    # --------- OPTIONAL HOST-SIDE PIPELINE (NON-AI) ---------

    def simple_host_pipeline(self, query: str) -> ASPSResult:
        """
        This is a host-side, non-introspective approximation pipeline.
        It does NOT replace the AI's own execution of the tasks, but
        allows simple programs to get some utility from ASPS.

        If you are an AI, prefer:
            result = asps.make_tasks_for(query)
            ...carry out the tasks yourself...
        """
        result = self.make_tasks_for(query)
        # Host simply echoes the query and marks that ASPS has not been fully applied.
        result.final_answer = query
        result.flags.append(
            "ASPS_SIMPLE_PIPELINE_ONLY: tasks prepared, but no AI-stage execution recorded."
        )
        return result


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="ASPS v3 – embedded NOESIS core middleware demo"
    )
    parser.add_argument("query", nargs="*", help="Query to run through ASPS")
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help="Print the bootstrap manifest and exit",
    )
    parser.add_argument(
        "--no-noesis",
        action="store_true",
        help="Disable embedded NOESIS adapter (for environments without torch)",
    )
    args = parser.parse_args()

    asps = ASPSMiddlewareV3(use_noesis=not args.no_noesis)

    if args.bootstrap:
        print(asps.describe_bootstrap())
        return

    query = " ".join(args.query).strip() or "What are the core challenges in building safe AGI?"
    result = asps.simple_host_pipeline(query)
    print(result.to_json())


if __name__ == "__main__":
    main()
