````text
noesis_v2_complete.py 
Created by: Lucas Postma (DataDyne Solutions LLC)
GitHub: https://github.com/DataDyneSolutions/asps-noesis-core
X: @BeingAsSuch
Drop a star to support the work!

#!/usr/bin/env python3
# noesis_v2_complete.py
# NOESIS v2: Complete Implementation
# (A) Stabilized Triadic + Dynamic Rod Attention (O(N*M)) + Mirror Field (F2)
# (B) FieldMemory (EMA over P/R/alpha with dE & gate-weighted retention)
# (+) ParallelFieldOps (entity-axis sharding when torch.distributed is initialized)
# (+) Streaming/Flash-style Rod Attention for memory efficiency
# PyTorch >= 2.0
#
# PERFORMANCE TIPS:
# 1. Use torch.compile for 2-3x speedup (PyTorch 2.0+):
#    model = torch.compile(model, mode="reduce-overhead")
# 2. Enable mixed precision training:
#    from torch.cuda.amp import autocast, GradScaler
#    scaler = GradScaler()
#    with autocast():
#        outputs = model(inputs)
# 3. For distributed training:
#    torchrun --nproc_per_node=4 train.py
#    (ParallelFieldOps will automatically shard entity axis)

from __future__ import annotations
import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

# ==============================================================================
# Configuration
# ==============================================================================
class Cfg:
    # refinement (A) – inner loop
    md_step   = 0.20   # mirror-descent step for P (entropic update)
    mom       = 0.80   # momentum for R, alpha
    eta_R     = 0.10   # step size for R
    eta_A     = 0.10   # step size for alpha
    gate_tau  = 0.50   # refinement gate threshold

    # loss weights
    w_field     = 1.00
    w_deltaE    = 0.20
    w_cat       = 0.10
    w_disc      = 1.00
    w_nce       = 0.20
    w_couple    = 0.10
    w_mg        = 0.10
    w_energy2   = 0.05
    w_mem       = 0.10    # memory consistency (P vs P_mem)
    w_alpha_ent = 0.05    # alpha entropy (mild sparsity prior)

    # bias basis (F2 -> rods)
    bias_k      = 8

    # memory (B) – EMA retention for FieldMemory
    mem_gamma_base = 0.85   # base EMA factor
    mem_gamma_gain = 0.10   # extra retention when dE improves (E_seed - E_refined)
    mem_init_scale = 0.00   # seed memory from first P/R/alpha (0..1)

# ==============================================================================
# Utilities
# ==============================================================================
def mlp(d_in: int, d_hidden: int, d_out: int, n_layers: int = 2, act=nn.SiLU) -> nn.Module:
    layers: list[nn.Module] = []
    dims = [d_in] + [d_hidden] * (n_layers - 1) + [d_out]
    for i in range(len(dims) - 2):
        layers += [nn.Linear(dims[i], dims[i + 1]), act()]
    layers += [nn.Linear(dims[-2], dims[-1])]
    return nn.Sequential(*layers)

# ==============================================================================
# Core Field Operations
# ==============================================================================
class FieldShape:
    """Defines the shape of the triadic field representation."""
    def __init__(self, n_entities: int, n_basis: int, d_relation: int, d_latent: int):
        self.n_entities = n_entities
        self.n_basis    = n_basis
        self.d_relation = d_relation
        self.d_latent   = d_latent

class FieldOps(nn.Module):
    """NOESIS operators + Energy functional (minimal)."""
    def __init__(self, shape: FieldShape, n_slots: int):
        super().__init__()
        self.shape = shape
        self.n_slots = n_slots
        self.lambda_rel = 0.10
        self.lambda_aff = 0.01
        self.temp = 0.20

    def totality(self, P: torch.Tensor) -> torch.Tensor:
        # Normalize per-sample across entities to keep a valid simplex per slot-slice.
        # P: [B,E,S] -> normalize over E.
        return P / (P.sum(dim=1, keepdim=True) + 1e-8)

    def reciprocity_penalty(self, R: torch.Tensor) -> torch.Tensor:
        # R: [B,E,E,S,d]
        return ((R + R.transpose(1, 2))**2).mean()

    def leash_constraint(self, P0: torch.Tensor, P1: torch.Tensor, leash_max: float = 0.3) -> torch.Tensor:
        # Encourage two entities to remain close in slot-space (for ToyWorld sanity)
        B, S = P0.shape
        idx  = torch.linspace(0, 1, S, device=P0.device)
        mu0  = (P0 * idx).sum(-1) / (P0.sum(-1) + 1e-8)
        mu1  = (P1 * idx).sum(-1) / (P1.sum(-1) + 1e-8)
        return torch.relu((mu0 - mu1).abs() - leash_max).mean()

    def energy_terms(self, P: torch.Tensor, R: torch.Tensor, A: torch.Tensor,
                     targets: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        # TV smoothness + relation reciprocity + mild alpha quadratic + optional supervision
        tv    = (P[..., 1:] - P[..., :-1]).abs().mean()            # TV over slots
        recip = self.reciprocity_penalty(R)
        aff   = (A**2).mean()
        sup   = torch.tensor(0.0, device=P.device)
        if targets is not None:
            if "P" in targets:     sup = sup + F.mse_loss(P, targets["P"])
            if "alpha" in targets: sup = sup + F.mse_loss(A, targets["alpha"])
        total = tv + self.lambda_rel * recip + self.lambda_aff * aff + sup
        return {"total": total, "tv": tv, "recip": recip, "aff": aff, "sup": sup}

# ==============================================================================
# Parallel Field Operations
# ==============================================================================
class ParallelFieldOps(nn.Module):
    """Wrap FieldOps to shard the entity axis across distributed ranks when dist.is_initialized()."""
    def __init__(self, base_ops: nn.Module, axis: str = "E", sync_avg: bool = False):
        super().__init__()
        self.base = base_ops
        self.axis = axis.lower()
        self.sync_avg = sync_avg
        self._dist = dist.is_available() and dist.is_initialized()
        self.rank = dist.get_rank() if self._dist else 0
        self.world_size = dist.get_world_size() if self._dist else 1

    @property
    def shape(self):   return self.base.shape

    @property
    def n_slots(self): return self.base.n_slots

    def totality(self, P):               return self.base.totality(P)
    def leash_constraint(self, *a, **k): return self.base.leash_constraint(*a, **k)

    def energy_terms(self, P: torch.Tensor, R: torch.Tensor, A: torch.Tensor,
                     targets: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        if (not self._dist) or self.world_size == 1 or self.axis != "e":
            return self.base.energy_terms(P, R, A, targets)

        B, E, S = P.shape
        idx_slices = torch.chunk(torch.arange(E, device=P.device), self.world_size)
        idx = idx_slices[self.rank]

        P_loc = P.index_select(1, idx)                  # [B, E_loc, S]
        A_loc = A.index_select(1, idx)                  # [B, E_loc]
        R_loc = R.index_select(1, idx).index_select(2, idx)  # [B,E_loc,E_loc,S,d]

        t_loc = None
        if targets is not None:
            tP = targets.get("P"); tA = targets.get("alpha")
            t_loc = {}
            if tP is not None: t_loc["P"]     = tP.index_select(1, idx)
            if tA is not None: t_loc["alpha"] = tA.index_select(1, idx)

        terms = self.base.energy_terms(P_loc, R_loc, A_loc, t_loc)
        terms["total"] = terms["total"] / float(self.world_size)

        if self.sync_avg:
            tot = terms["total"].detach().clone()
            dist.all_reduce(tot, op=dist.ReduceOp.SUM)
            terms["total_global"] = tot
        return terms

# ==============================================================================
# Text Encoding
# ==============================================================================
class TinyTextEncoder(nn.Module):
    def __init__(self, n_chars=256, d_model=256, d_latent=128):
        super().__init__()
        self.char = nn.Embedding(n_chars, d_model)
        self.enc  = nn.GRU(d_model, d_model//2, bidirectional=True, batch_first=True)
        self.proj = nn.Linear(d_model, d_latent)
    
    def forward(self, x, return_tokens=False):
        z,_ = self.enc(self.char(x))        # [B,T,d_model]
        h = self.proj(z.mean(1))            # [B,d_latent]
        return (h, z) if return_tokens else h

# ==============================================================================
# Dynamic Rod Attention
# ==============================================================================
class DynamicRodAttention(nn.Module):
    """Tokens -> Rods -> Tokens, O(N*M), with optional bias [B,N,M]."""
    def __init__(self, d_model=256, n_rods=64, iters=2, use_rod_self_attn=True):
        super().__init__()
        self.d = d_model
        self.m = n_rods
        self.iters = iters
        
        # Token projections
        self.qx = nn.Linear(d_model, d_model)
        self.kx = nn.Linear(d_model, d_model)
        self.vx = nn.Linear(d_model, d_model)
        
        # Rod projections
        self.kr = nn.Linear(d_model, d_model)
        self.vr = nn.Linear(d_model, d_model)
        
        # Rod initialization and update
        self.rod_init = nn.Parameter(torch.randn(n_rods, d_model) * 0.02)
        self.gru = nn.GRUCell(d_model, d_model)
        
        # Optional rod self-attention
        self.sa = nn.MultiheadAttention(d_model, max(1, d_model//64), batch_first=True) if use_rod_self_attn else None
        
        # Normalization and output
        self.nx = nn.LayerNorm(d_model)
        self.nr = nn.LayerNorm(d_model)
        self.po = nn.Linear(d_model, d_model)
        self.on = nn.LayerNorm(d_model)
    
    def forward(self, x, rod_bias: Optional[torch.Tensor]=None):
        B, N, d = x.shape
        x = self.nx(x)
        Qx, Kx, Vx = self.qx(x), self.kx(x), self.vx(x)
        
        # Initialize rods
        R = self.nr(self.rod_init.unsqueeze(0).expand(B, -1, -1).contiguous())
        
        # Iterative rod refinement
        for _ in range(self.iters):
            # Tokens attend to rods
            KR = self.kr(R)
            logits_tr = (Qx @ KR.transpose(1,2)) / (d**0.5)
            if rod_bias is not None: 
                logits_tr = logits_tr + rod_bias
            attn_tr = torch.softmax(logits_tr, dim=-1)
            
            # Aggregate information from tokens
            rin = attn_tr.transpose(1,2) @ Vx
            
            # Update rods with GRU
            R = self.gru(rin.reshape(B*self.m, d), R.reshape(B*self.m, d)).reshape(B, self.m, d)
            R = self.nr(R)
            
            # Optional rod self-attention
            if self.sa is not None:
                R2, _ = self.sa(R, R, R)
                R = self.nr(R + R2)
        
        # Tokens attend back to refined rods
        KR = self.kr(R)
        VR = self.vr(R)
        logits_rt = (Qx @ KR.transpose(1,2)) / (d**0.5)
        attn_rt = torch.softmax(logits_rt, dim=-1)
        Y = attn_rt @ VR
        
        return self.on(x + self.po(Y))

# ==============================================================================
# Streaming Rod Attention (Memory-Efficient)
# ==============================================================================
class StreamingDynamicRodAttention(DynamicRodAttention):
    """Flash-style tiled computation for rod attention."""
    def __init__(self, *args, tile_tokens: int = 64, tile_rods: int = None, 
                 return_mode: str = "rods", **kwargs):
        super().__init__(*args, **kwargs)
        self.tile_tokens = int(tile_tokens)
        self.tile_rods = int(tile_rods) if tile_rods is not None else int(tile_tokens)
        assert return_mode in ("rods", "tokens", "both"), "return_mode must be 'rods' | 'tokens' | 'both'"
        self.return_mode = return_mode

    def forward(self, x, rod_bias=None):
        B, N, d = x.shape
        M = self.m
        x = self.nx(x)
        Qx, Kx, Vx = self.qx(x), self.kx(x), self.vx(x)

        # Initialize rods
        R = self.nr(self.rod_init.unsqueeze(0).expand(B, -1, -1).contiguous())
        inv_sqrt = 1.0 / math.sqrt(d)
        tsize = self.tile_tokens

        # Iterative refinement with streaming
        for _ in range(self.iters):
            # Stream tokens -> rods
            m = torch.full((B, M), float("-inf"), device=x.device, dtype=x.dtype)
            l = torch.zeros((B, M), device=x.device, dtype=x.dtype)
            num = torch.zeros((B, M, d), device=x.device, dtype=x.dtype)
            
            for t0 in range(0, N, tsize):
                t1 = min(N, t0 + tsize)
                Kt = Kx[:, t0:t1, :]
                Vt = Vx[:, t0:t1, :]
                scores = torch.einsum("bmd,btd->bmt", R, Kt) * inv_sqrt
                
                if rod_bias is not None:
                    rb_tile = rod_bias[:, t0:t1, :].permute(0, 2, 1)
                    scores = scores + rb_tile
                
                smax = scores.max(-1).values
                m_new = torch.maximum(m, smax)
                exp_m = torch.exp(m - m_new)
                exp_scores = torch.exp(scores - m_new.unsqueeze(-1))
                l = l * exp_m + exp_scores.sum(-1)
                num = num * exp_m.unsqueeze(-1) + torch.einsum("bmt,btd->bmd", exp_scores, Vt)
                m = m_new

            attn_out_rods = num / (l.unsqueeze(-1) + 1e-8)

            # Optional rod self-attention
            if self.sa is not None:
                o = self.on(attn_out_rods)
                attn_out_rods = self.sa(o, o, o, need_weights=False)[0]

            # GRU update
            R_flat = R.reshape(B * M, d)
            U_flat = attn_out_rods.reshape(B * M, d)
            R = self.gru(U_flat, R_flat).reshape(B, M, d)
            R = self.nr(R)

        if self.return_mode == "rods":
            return R

        # Stream rods -> tokens
        Kr = self.kr(R)
        Vr = self.vr(R)
        m_t = torch.full((B, N), float("-inf"), device=x.device, dtype=x.dtype)
        l_t = torch.zeros((B, N), device=x.device, dtype=x.dtype)
        num_t = torch.zeros((B, N, d), device=x.device, dtype=x.dtype)
        rsize = self.tile_rods

        for r0 in range(0, M, rsize):
            r1 = min(M, r0 + rsize)
            Kr_tile = Kr[:, r0:r1, :]
            Vr_tile = Vr[:, r0:r1, :]
            scores_rt = torch.einsum("bnd,bmd->bnm", Qx, Kr_tile) * inv_sqrt
            
            if rod_bias is not None:
                scores_rt = scores_rt + rod_bias[:, :, r0:r1]
            
            smax = scores_rt.max(-1).values
            m_new = torch.maximum(m_t, smax)
            exp_m = torch.exp(m_t - m_new)
            exp_scores = torch.exp(scores_rt - m_new.unsqueeze(-1))
            l_t = l_t * exp_m + exp_scores.sum(-1)
            num_t = num_t * exp_m.unsqueeze(-1) + torch.einsum("bnm,bmd->bnd", exp_scores, Vr_tile)
            m_t = m_new

        Y_tokens = num_t / (l_t.unsqueeze(-1) + 1e-8)
        
        if self.return_mode == "tokens":
            return Y_tokens
        else:
            return R, Y_tokens

# ==============================================================================
# Bridge Module
# ==============================================================================
class Bridge(nn.Module):
    """Upward: tokens->field seeds; Downward: field->logits; Gate & InfoNCE."""
    def __init__(self, d_in, d_br, vocab, nE, nB, S, dR):
        super().__init__()
        self.up   = mlp(d_in, 256, d_br, 2)
        self.down = mlp(d_br, 256, vocab, 2)
        self.tC   = nn.Linear(d_br, nE*nB)
        self.tA   = nn.Linear(d_br, nE)
        self.tR   = nn.Linear(d_br, nE*nE*S*dR)
        self.g    = mlp(d_in + d_br, 128, 1, 2)
    
    def upward(self, h): 
        u = self.up(h)
        return u, {
            "coeff": self.tC(u), 
            "alpha": self.tA(u), 
            "relation": self.tR(u)
        }
    
    def downward(self, s): 
        return self.down(s)
    
    def gate_refine(self, h, u): 
        return torch.sigmoid(self.g(torch.cat([u, h], -1)))
    
    @staticmethod
    def info_nce(u, s, temp=0.05):
        u = F.normalize(u, -1)
        s = F.normalize(s, -1)
        L = (u @ s.t()) / temp
        y = torch.arange(u.size(0), device=u.device)
        return F.cross_entropy(L, y)

# ==============================================================================
# Basis Expansion & Field Pyramid
# ==============================================================================
class BasisExpander(nn.Module):
    def __init__(self, nB, S, learned=True): 
        super().__init__()
        self.nB = nB
        self.S = S
        self.learned = learned
        self.net = mlp(nB, 128, S, 3, nn.SiLU) if learned else None
    
    def forward(self, C):
        B, E, K = C.shape
        if self.learned:
            return F.softplus(self.net(C.reshape(B*E, K))).view(B, E, self.S)
        
        Wk = torch.linspace(-1, 1, K, device=C.device)
        cos = torch.cos(torch.einsum("bek,k->bek", C, Wk))
        slot = torch.linspace(0, 1, self.S, device=C.device)[None, None, :].expand(B, E, -1)
        return F.relu(torch.einsum("bek,bes->bes", cos, slot))

class FieldPyramid(nn.Module):
    """Mirror field (F2) operations."""
    def __init__(self, shape_L, S1, S2): 
        super().__init__()
        self.S1 = S1
        self.S2 = S2
        self.ops_L = FieldOps(shape_L, S2)
    
    def compress(self, P1, A1):
        B, E, S1 = P1.shape
        S2 = self.S2
        P2 = F.interpolate(P1.reshape(B*E, 1, S1), size=S2, mode="linear", align_corners=False)
        P2 = P2.reshape(B, E, S2).clamp_min(0.0)
        return P2, F.softplus(A1)
    
    def prolongate(self, P2):
        B, E, S2 = P2.shape
        S1 = self.S1
        return F.interpolate(P2.reshape(B*E, 1, S2), size=S1, mode="linear", align_corners=False).reshape(B, E, S1)
    
    def downsample_R(self, R1):
        B, E, E2, S1, d = R1.shape
        S2 = self.S2
        Rv = R1.permute(0, 1, 2, 4, 3).reshape(B*E*E2*d, 1, S1)
        Rv = F.interpolate(Rv, size=S2, mode="linear", align_corners=False)
        return Rv.reshape(B, E, E2, d, S2).permute(0, 1, 2, 4, 3)

# ==============================================================================
# Field Memory
# ==============================================================================
class FieldMemory(nn.Module):
    """EMA memory for (P,R,alpha) with dE & gate-weighted retention."""
    def __init__(self, shape: FieldShape, n_slots: int):
        super().__init__()
        self.shape = shape
        self.n_slots = n_slots
        E, dR = shape.n_entities, shape.d_relation
        self.register_buffer("P_mem", torch.zeros(1, E, n_slots))
        self.register_buffer("R_mem", torch.zeros(1, E, E, n_slots, dR))
        self.register_buffer("A_mem", torch.zeros(1, E))
        self.initialized = False
    
    def ensure(self, B: int, device, P_seed=None, R_seed=None, A_seed=None):
        if (not self.initialized) or (self.P_mem.shape[0] != B) or (self.P_mem.device != device):
            E = self.shape.n_entities
            S = self.n_slots
            dR = self.shape.d_relation
            self.P_mem = torch.zeros(B, E, S, device=device)
            self.R_mem = torch.zeros(B, E, E, S, dR, device=device)
            self.A_mem = torch.zeros(B, E, device=device)
            
            s = float(getattr(Cfg, "mem_init_scale", 0.0))
            if s > 0 and P_seed is not None:
                self.P_mem.copy_(s * P_seed.detach())
                if R_seed is not None: 
                    self.R_mem.copy_(s * R_seed.detach())
                if A_seed is not None: 
                    self.A_mem.copy_(s * A_seed.detach())
            self.initialized = True
    
    def update(self, P_new, R_new, A_new, dE: torch.Tensor, gate: torch.Tensor):
        B = P_new.shape[0]
        de = torch.clamp(dE, min=0.0)
        if de.ndim == 0: 
            de = de.view(1).expand(B)
        g = gate.view(B)
        
        base = float(getattr(Cfg, "mem_gamma_base", 0.85))
        gain = float(getattr(Cfg, "mem_gamma_gain", 0.10))
        gamma = torch.clamp(base + gain * (1.0 - torch.tanh(de)) * g, 0.0, 0.999)
        
        gP = gamma.view(B, 1, 1)
        gR = gamma.view(B, 1, 1, 1, 1)
        gA = gamma.view(B, 1)
        
        self.P_mem = gP * self.P_mem + (1.0 - gP) * P_new.detach()
        self.R_mem = gR * self.R_mem + (1.0 - gR) * R_new.detach()
        self.A_mem = gA * self.A_mem + (1.0 - gA) * A_new.detach()
        
        return self.P_mem, self.R_mem, self.A_mem

# ==============================================================================
# Main NOESIS Model
# ==============================================================================
class TriadicNoesisParallelMem(nn.Module):
    """Complete NOESIS v2 model with triadic fields, rod attention, and memory."""
    
    def __init__(self, vocab: int, shape: FieldShape, n_slots: int,
                 d_text: int = 128, learned_basis: bool = True,
                 use_dynamic_rod_attn: bool = True, n_rods: int = 64, rod_iters: int = 2,
                 use_parallel_ops: bool = True, parallel_axis: str = "E", sync_avg: bool = False):
        super().__init__()
        self.shape = shape
        self.n_slots = n_slots
        self.vocab = vocab

        # Text encoder
        self.enc = TinyTextEncoder(256, 256, d_text)
        
        # Rod attention setup
        self.use_rods = use_dynamic_rod_attn
        if self.use_rods:
            self.rod = DynamicRodAttention(256, n_rods, rod_iters, True)
            self.pre = nn.Sequential(
                nn.LayerNorm(d_text + 256),
                mlp(d_text + 256, max(128, d_text), d_text, 2),
                nn.LayerNorm(d_text)
            )
            self.bias_tok = nn.Linear(256, Cfg.bias_k)
            self.bias_rod = nn.Linear(self.rod.d, Cfg.bias_k)
            self.bias_proj = nn.Parameter(torch.randn(Cfg.bias_k, max(8, n_slots//4)) / math.sqrt(Cfg.bias_k))
            self.gamma = nn.Parameter(torch.tensor(0.5))
        else:
            self.rod = None

        # Bridge for field seeding
        self.bridge = Bridge(d_text, shape.d_latent, vocab, shape.n_entities, 
                           shape.n_basis, n_slots, shape.d_relation)

        # Field operations
        base_ops = FieldOps(shape, n_slots)
        if use_parallel_ops and dist.is_available() and dist.is_initialized() and parallel_axis.lower() == "e":
            self.ops = ParallelFieldOps(base_ops, axis="e", sync_avg=sync_avg)
        else:
            self.ops = base_ops

        # Basis expansion and pyramid
        self.basis = BasisExpander(shape.n_basis, n_slots, learned=learned_basis)
        S2 = max(8, n_slots//4)
        self.S2 = S2
        self.pyr = FieldPyramid(
            FieldShape(shape.n_entities, max(1, shape.n_basis//2), shape.d_relation, shape.d_latent),
            n_slots, S2
        )

        # Field memory
        self.memory = FieldMemory(shape, n_slots)

        # Refinement parameters
        self.md_step = Cfg.md_step
        self.mom = Cfg.mom
        self.eta_R = Cfg.eta_R
        self.eta_A = Cfg.eta_A

    def seed(self, seeds: Dict[str, torch.Tensor]):
        """Initialize fields from seed values."""
        B = seeds["alpha"].shape[0]
        E = self.shape.n_entities
        K = self.shape.n_basis
        S = self.n_slots
        dR = self.shape.d_relation
        
        C = seeds["coeff"].view(B, E, K)
        A = torch.nn.functional.softplus(seeds["alpha"]) + 1e-8
        R = seeds["relation"].view(B, E, E, S, dR)
        P = self.basis(C).clamp_min(0.0)
        
        return P, R, A

    def sum_field(self, P, A):
        """Summarize field state."""
        B, E, S = P.shape
        pn = self.ops.totality(P)
        ent = -(pn.clamp_min(1e-8) * pn.clamp_min(1e-8).log()).sum(-1)
        h = torch.stack([pn.mean(-1), ent, A], -1).view(B, -1)
        return mlp(h.shape[-1], 128, self.shape.d_latent, 2)(h)

    def bias_from_F2(self, P2, z_tokens):
        """Compute rod bias from mirror field F2."""
        if not self.use_rods or self.rod is None: 
            return None
        
        B, N, d = z_tokens.shape
        M = self.rod.m
        S2 = self.S2
        
        R0 = self.rod.rod_init.unsqueeze(0).expand(B, M, -1)
        K = Cfg.bias_k
        
        t = self.bias_tok(z_tokens)  # [B,N,K]
        r = self.bias_rod(R0)        # [B,M,K]
        
        W = self.bias_proj[:, :S2]   # [K,S2]
        ts = torch.softmax(t @ W, dim=-1)   # [B,N,S2]
        rs = torch.softmax(r @ W, dim=-1)   # [B,M,S2]
        
        sal = self.pyr.ops_L.totality(P2).sum(1, keepdim=True)  # [B,1,S2]
        sal = sal / (sal.sum(-1, keepdim=True) + 1e-8)
        
        mid = torch.einsum("bns,bks->bnk", ts, sal)            # [B,N,1]
        bias = torch.einsum("bnk,bmk->bnm", mid, rs)           # [B,N,M]
        
        return self.gamma * bias

    def refine(self, P, R, A, steps=2, targets=None):
        """Iterative refinement via mirror descent."""
        vR = torch.zeros_like(R)
        vA = torch.zeros_like(A)
        
        for _ in range(steps):
            terms = self.ops.energy_terms(P, R, A, targets)
            
            # Add leash constraint if applicable
            leash = torch.tensor(0.0, device=P.device)
            if self.shape.n_entities >= 2:
                T = self.ops.totality(P)
                leash = self.ops.leash_constraint(T[:, 0], T[:, 1])
            
            total = terms['total'] + leash
            
            # Compute gradients
            gP, gR, gA = torch.autograd.grad(total, [P, R, A], retain_graph=True, allow_unused=True)
            
            # Update P with mirror descent
            if gP is not None:
                Pn = self.ops.totality(P).clamp_min(1e-8)
                logP = Pn.log() - self.md_step * gP.detach()
                P = torch.softmax(logP, dim=-1)
            
            # Update R with momentum
            if gR is not None:
                vR = self.mom * vR + (1 - self.mom) * gR
                R = R - self.eta_R * vR
                R = 0.5 * (R - R.transpose(1, 2))  # Enforce anti-symmetry
            
            # Update A with momentum
            if gA is not None:
                vA = self.mom * vA + (1 - self.mom) * gA
                A = A - self.eta_A * vA
        
        return P, R, A, self.ops.energy_terms(P, R, A, targets)

    def forward(self, x, targets: Optional[Dict[str, torch.Tensor]] = None, refine_max_steps=3):
        """Forward pass through NOESIS."""
        B = x.size(0)
        
        # Encode text
        h0, z = self.enc(x, return_tokens=True)

        # Initial seed F1 & mirror F2
        u0, sd0 = self.bridge.upward(h0)
        P10, R10, A10 = self.seed(sd0)
        P2, A2 = self.pyr.compress(P10, A10)
        R2 = self.pyr.downsample_R(R10)

        # Rod attention guided by F2
        if self.use_rods and self.rod is not None:
            rb = self.bias_from_F2(P2, z)
            y = self.rod(z, rb)
            ypool = y.mean(1)
            h_in = self.pre(torch.cat([h0, ypool], -1))
        else:
            h_in = h0

        # Final seed & baseline energy
        u, sd = self.bridge.upward(h_in)
        P1, R1, A1 = self.seed(sd)
        seed_terms = self.ops.energy_terms(P1, R1, A1, targets)
        E_seed = seed_terms['total'].detach()

        # Gated refinement
        g = self.bridge.gate_refine(h_in, u)
        steps = refine_max_steps if (g.mean() >= Cfg.gate_tau) else 0
        
        if steps > 0:
            P1.requires_grad_(True)
            R1.requires_grad_(True)
            A1.requires_grad_(True)
            P1, R1, A1, eng1 = self.refine(P1, R1, A1, steps, targets)
        else:
            eng1 = seed_terms

        # Memory update
        self.memory.ensure(B, x.device, P_seed=P1, R_seed=R1, A_seed=A1)
        dE = torch.clamp(E_seed - eng1['total'].detach(), min=0.0)
        Pm, Rm, Am = self.memory.update(P1, R1, A1, dE, gate=g)

        # Mirror diagnostics
        P2_hat, _ = self.pyr.compress(P1, A1)
        P1_up = self.pyr.prolongate(P2)
        eng2 = self.pyr.ops_L.energy_terms(P2, R2, A2)

        # Output head
        s = self.sum_field(P1, A1)
        logits = self.bridge.downward(s)
        nce = self.bridge.info_nce(u, s)

        return {
            "logits": logits, 
            "P": P1, 
            "R": R1, 
            "alpha": A1, 
            "energy": eng1, 
            "nce": nce, 
            "gate": g.detach(),
            "P2": P2, 
            "P2_hat": P2_hat, 
            "P1_up": P1_up, 
            "energy2": eng2["total"].detach(),
            "P_mem": Pm, 
            "R_mem": Rm, 
            "alpha_mem": Am, 
            "deltaE_seed": dE.detach()
        }

# ==============================================================================
# Dynamic & Fast Variants
# ==============================================================================
class TriadicNoesisParallelMemDyn(TriadicNoesisParallelMem):
    """Dynamic variant with adaptive computation budget."""
    def __init__(self, *args, use_dynamic_griddle: bool = True, griddle_budget_ms: float = 1e9, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_dynamic_griddle = use_dynamic_griddle
        self._budget_left_ms = griddle_budget_ms
        self._last_deltaE = 0.0

    def forward(self, x, targets=None, refine_max_steps=None, **kw):
        # Dynamically decide refinement steps
        if refine_max_steps is None and self.use_dynamic_griddle:
            P_probe = targets.get("P", None) if targets else None
            if P_probe is None:
                B = x.size(0)
                E = self.shape.n_entities
                S = self.n_slots
                P_probe = torch.full((B, E, S), 1.0 / float(S), device=x.device, dtype=x.dtype)
            
            H_mean = float(_entropy_prob(P_probe).mean().item())
            dE = float(getattr(self, "_last_deltaE", 0.0))
            _, _, refine_steps = _choose_mode(H_mean, dE, float(self._budget_left_ms))
            refine_max_steps = int(refine_steps)
        elif refine_max_steps is None:
            refine_max_steps = 0

        out = super().forward(x, targets=targets, refine_max_steps=refine_max_steps, **kw)
        
        # Track last ΔE
        if isinstance(out, dict) and "deltaE_seed" in out:
            self._last_deltaE = float(out["deltaE_seed"].detach().mean().item())
        
        return out

class TriadicNoesisParallelMemFast(TriadicNoesisParallelMemDyn):
    """Fast variant with streaming rod attention."""
    def __init__(self, *args, use_streaming_rods: bool = True, 
                 tile_tokens: int = 64, tile_rods: int = None,
                 return_mode: str = "rods", **kwargs):
        super().__init__(*args, **kwargs)
        
        if self.use_rods and self.rod is not None and use_streaming_rods:
            old = self.rod
            use_rod_self = (getattr(old, "sa", None) is not None)
            self.rod = StreamingDynamicRodAttention(
                d_model=old.d, n_rods=old.m, iters=old.iters, 
                use_rod_self_attn=use_rod_self,
                tile_tokens=tile_tokens, tile_rods=tile_rods, 
                return_mode=return_mode
            )

# ==============================================================================
# Loss Function
# ==============================================================================
class TriadicLossMem(nn.Module):
    """Complete loss function with memory consistency and entropy regularization."""
    def forward(self, model: nn.Module, out: Dict[str, torch.Tensor],
                targets: Optional[Dict[str, torch.Tensor]] = None,
                labels: Optional[torch.Tensor] = None,
                energy_before: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:

        P, A = out["P"], out["alpha"]
        eng1 = out["energy"]
        nce = out["nce"]

        # Field supervision loss
        l_field = torch.tensor(0.0, device=P.device)
        if targets and "P" in targets:     
            l_field += F.mse_loss(P, targets["P"])
        if targets and "alpha" in targets: 
            l_field += F.mse_loss(A, targets["alpha"])

        # Unity constraint
        Pn = model.ops.totality(P)
        l_unity = (Pn.sum(1) - 1.0).pow(2).mean()

        # Energy improvement loss
        l_deltaE = torch.tensor(0.0, device=P.device)
        if energy_before is not None:
            l_deltaE = torch.clamp(eng1["total"] - energy_before, min=0.0)

        # Discrimination loss
        l_disc = torch.tensor(0.0, device=P.device)
        if "logits" in out and labels is not None:
            l_disc = F.cross_entropy(out["logits"], labels)

        # Mirror field losses
        l_couple = F.mse_loss(out["P2_hat"], out["P2"].detach())
        l_mg = F.mse_loss(model.ops.totality(P), model.ops.totality(out["P1_up"]))
        l_e2 = out["energy2"]

        # Memory consistency
        l_mem = F.mse_loss(out["P"], out["P_mem"].detach())

        # Alpha entropy regularization
        An = A / (A.sum(dim=1, keepdim=True) + 1e-8)
        l_alpha_ent = -(An.clamp_min(1e-8) * An.clamp_min(1e-8).log()).sum(dim=1).mean()

        # Total loss
        total = (Cfg.w_field * l_field + 
                Cfg.w_deltaE * l_deltaE + 
                Cfg.w_cat * l_unity + 
                Cfg.w_disc * l_disc + 
                Cfg.w_nce * nce +
                Cfg.w_couple * l_couple + 
                Cfg.w_mg * l_mg + 
                Cfg.w_energy2 * l_e2 +
                Cfg.w_mem * l_mem + 
                Cfg.w_alpha_ent * l_alpha_ent)

        return {
            "total": total,
            "l_field": l_field.detach(),
            "l_deltaE": l_deltaE.detach(),
            "l_unity": l_unity.detach(),
            "l_disc": l_disc.detach(),
            "l_nce": nce.detach(),
            "l_couple": l_couple.detach(),
            "l_mg": l_mg.detach(),
            "l_e2": l_e2.detach(),
            "l_mem": l_mem.detach(),
            "l_alpha_ent": l_alpha_ent.detach(),
            "energy_total": eng1["total"].detach()
        }

# ==============================================================================
# Training
# ==============================================================================
class Trainer(nn.Module):
    def __init__(self, model: nn.Module, lr=1e-3):
        super().__init__()
        self.model = model
        self.opt = torch.optim.AdamW(model.parameters(), lr=lr)
        self.crit = TriadicLossMem()
    
    def step(self, x, batch, labels: Optional[torch.Tensor] = None):
        self.model.train()
        
        # Get baseline energy
        with torch.no_grad():
            o0 = self.model(x, targets=batch, refine_max_steps=0)
            E0 = o0["energy"]["total"].detach()
        
        # Forward with refinement
        o = self.model(x, targets=batch, refine_max_steps=2)
        
        # Compute loss
        loss = self.crit(self.model, o, targets=batch, labels=labels, energy_before=E0)
        
        # Optimize
        self.opt.zero_grad()
        if not torch.isfinite(loss["total"]).all():
            print("[WARN] Non-finite loss – skipping step")
            self.opt.zero_grad(set_to_none=True)
            return

    # Skip finite but excessively large losses to avoid toxic updates
    if float(loss["total"].detach().cpu()) > 1e5:
        print("[WARN] Outlier loss > 1e5 — skipping step")
        self.opt.zero_grad(set_to_none=True)
        return
        loss["total"].backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        nn.utils.clip_grad_value_(self.model.parameters(), 5.0)
        self.opt.step()
        
        # Return metrics
        rep = {k: float(v.detach().cpu()) for k, v in loss.items()}
        rep["gate_mean"] = float(o["gate"].mean().cpu())
        return rep

# ==============================================================================
# Utilities & Testing
# ==============================================================================
class ToyWorld:
    """Debug dataset for testing."""
    def __init__(self, n_entities=2, n_slots=32, leash_sigma=0.02):
        self.n_entities = n_entities
        self.n_slots = n_slots
        self.slots = torch.linspace(0, 1, n_slots)
        self.sig = leash_sigma
    
    def _bump(self, mu): 
        return torch.exp(-((self.slots.unsqueeze(0) - mu.unsqueeze(1))**2) / (2 * self.sig**2))
    
    def sample_batch(self, B, device="cpu"):
        mu0 = torch.rand(B, device=device) * 0.6 + 0.2
        P = [self._bump(mu0)]
        
        for e in range(1, self.n_entities):
            mu = (mu0 + torch.randn(B, device=device) * 0.03).clamp(0, 1)
            P.append(self._bump(mu))
        
        P = torch.stack(P, dim=1).to(device)
        A = torch.stack([0.7 * torch.ones(B, device=device) for _ in range(self.n_entities)], dim=1)
        return {"P": P, "alpha": A}

# ==============================================================================
# Helper Functions
# ==============================================================================
def _entropy_prob(_P, _eps: float = 1e-8):
    """Compute entropy of probability distributions."""
    _P = _P.clamp_min(_eps)
    return -(_P * _P.log()).sum(dim=-1)

def _choose_mode(_H_mean: float, _dE_last: float, _budget_left: float, rmin: int = 8, rmax: int = 64):
    """Choose computation mode based on entropy and energy improvement."""
    need_precision = (_H_mean > 0.6) or (_dE_last > 1e-3)
    if _budget_left < 0.0:
        need_precision = False
    mode = 1 if need_precision else 0
    refine = 1 if need_precision else 0
    Hc = 0.0 if _H_mean < 0.0 else (1.0 if _H_mean > 1.0 else _H_mean)
    r = int(rmin + (rmax - rmin) * Hc)
    return mode, r, refine

def _simple_char_tokens(s: str, vocab: int = 256, max_len: int = 160):
    """Map string to integer tokens [0..vocab-1]. Char-level; stable & deterministic."""
    arr = [min(vocab-1, ord(c)) for c in (s or "")[:max_len]]
    if not arr: 
        arr = [0]
    return torch.tensor([arr], dtype=torch.long)

@torch.no_grad()
def noesis_reflect(text: str,
                  model: Optional[nn.Module] = None,
                  shape: Optional[FieldShape] = None,
                  steps: int = 1,
                  device: Optional[str] = None) -> Dict[str, float]:
    """
    Lightweight interface to probe NOESIS energy improvement on a text stimulus.
    
    Returns: dict with keys:
      - E0: baseline energy at 0 refinement steps
      - E1: energy after `steps` refinement steps
      - dE: max(0, E0 - E1) (positive means refinement helped)
      - gate: mean gate value used to decide refinement (if available)
    """
    dev = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model is None:
        shape = shape or FieldShape(n_entities=8, n_basis=16, d_relation=8, d_latent=64)
        model = TriadicNoesisParallelMem(vocab=256, shape=shape, n_slots=64, d_text=128, use_dynamic_rod_attn=False)
    
    model = model.to(dev).eval()
    
    x = _simple_char_tokens(text, vocab=getattr(model, "vocab", 256)).to(dev)
    out0 = model(x, targets=None, refine_max_steps=0)
    out1 = model(x, targets=None, refine_max_steps=int(max(0, steps)))
    
    def _get_total(o):
        try:
            e = o["energy"]["total"]
            return float(e.detach().item())
        except Exception:
            return float("nan")
    
    E0 = _get_total(out0)
    E1 = _get_total(out1)
    dE = max(0.0, E0 - E1) if (math.isfinite(E0) and math.isfinite(E1)) else 0.0
    
    gate = 0.0
    try:
        g = out1.get("gate", None)
        if g is not None and hasattr(g, "detach"):
            gate = float(g.detach().mean().item())
    except Exception:
        pass
    
    return {"E0": E0, "E1": E1, "dE": dE, "gate": gate}

# ==============================================================================
# Main Test
# ==============================================================================
if __name__ == "__main__":
    torch.manual_seed(42)
    
    # Create model
    shape = FieldShape(n_entities=2, n_basis=8, d_relation=4, d_latent=32)
    model = TriadicNoesisParallelMemFast(
        vocab=4096, 
        shape=shape, 
        n_slots=24, 
        use_dynamic_rod_attn=True, 
        n_rods=12, 
        rod_iters=1,
        use_streaming_rods=True
    )
    
    # Test forward pass
    B, T = 2, 96
    x = torch.randint(0, 256, (B, T))
    tw = ToyWorld(2, 24)
    batch = tw.sample_batch(B)
    
    out0 = model(x, targets=batch, refine_max_steps=0)
    out2 = model(x, targets=batch, refine_max_steps=2)
    
    print("NOESIS v2 Complete - Test Results:")
    print("-" * 40)
    print(f"Logits shape:    {list(out2['logits'].shape)}")
    print(f"P shape:         {list(out2['P'].shape)}")
    print(f"P_mem shape:     {list(out2['P_mem'].shape)}")
    print(f"ΔE_seed mean:    {float(out2['deltaE_seed'].mean()):.4f}")
    print(f"Gate mean:       {float(out2['gate'].mean()):.4f}")
    print(f"Energy (no ref): {float(out0['energy']['total']):.4f}")
    print(f"Energy (ref=2):  {float(out2['energy']['total']):.4f}")
    
    # Test reflection interface
    print("\nReflection test:")
    result = noesis_reflect("Hello NOESIS", model=model, steps=2, device="cpu")
    print(f"Text: 'Hello NOESIS'")
    print(f"  E0: {result['E0']:.4f}")
    print(f"  E1: {result['E1']:.4f}")
    print(f"  ΔE: {result['dE']:.4f}")
    print(f"  Gate: {result['gate']:.4f}")
