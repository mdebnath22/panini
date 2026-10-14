"""
TODO #2 — Neural Realization Map for Body Derivations (Φ_body,θ)
====================================================================

The manipulation-side compositional-generalization experiment (Section 4.3
of the paper) trains a transformer to predict rule twists and recover rule
identity from a derivation sequence, then tests generalization to held-out
rule compositions. This is the missing morphology-side mirror: same
architecture family (context encoder, latent predictor, decoder, inverse
readout head), applied to BODY rule tokens instead of manipulation rule
tokens, predicting each rule's LOCAL PLACEMENT TRANSFORM (translation +
axis-angle rotation, the body-grammar analogue of a manipulation twist)
instead of an SE(3) twist.

Held-out generalization split: train on legged bodies with leg count in
{3,4,5,6} and segments-per-leg in {1,2}; test on leg counts {7,8} (never
seen) and segments-per-leg=3 (never seen) -- i.e. held-out STRUCTURAL
compositions of the same six primitive rules, exactly analogous to the
manipulation experiment's held-out rule-ORDER compositions.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from ptg.body_grammar import BodyGrammar, legged_plan
from ptg.body_sutra import seed_body_rules
from gmm.phi_body import PhiBody
from gmm.se3 import geodesic_distance  # reused for endpoint-style error if needed

OPS = ["ADD_TORSO", "ADD_LIMB", "ARTICULATE", "EXTEND", "TERMINATE", "SYMMETRIZE"]
OP2I = {o: i for i, o in enumerate(OPS)}
PARAM_KEYS = ["seg_len", "seg_rad", "mount_angle", "joint_range", "torso_height"]
TOKEN_DIM = len(OPS) + len(PARAM_KEYS) + 1  # +1 vibhasa flag placeholder


def log_so3(R: np.ndarray) -> np.ndarray:
    cos_theta = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    if theta < 1e-8:
        return np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0],
                         R[1, 0] - R[0, 1]]) / 2.0
    return theta / (2 * np.sin(theta)) * np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


def placement_6d(local_transform: np.ndarray) -> np.ndarray:
    """4x4 local placement -> 6D (translation, axis-angle rotation), the
    body-grammar analogue of a manipulation twist target."""
    t = local_transform[:3, 3]
    w = log_so3(local_transform[:3, :3])
    return np.concatenate([t, w]).astype(np.float32)


def tokenize_applied(applied) -> np.ndarray:
    v = np.zeros(TOKEN_DIM, dtype=np.float32)
    v[OP2I[applied.sutra.op]] = 1.0
    off = len(OPS)
    for k, key in enumerate(PARAM_KEYS):
        v[off + k] = applied.params.get(key, 0.0)
    v[-1] = 0.0  # vibhasa flag (unused in this template)
    return v


def tokenize_derivation(derivation) -> np.ndarray:
    return np.stack([tokenize_applied(a) for a in derivation])


def sample_body_derivations(n: int, leg_counts: List[int], seg_counts: List[int],
                            seed: int) -> List:
    rng = np.random.default_rng(seed)
    g = BodyGrammar(seed_body_rules(), seed=seed)
    out = []
    for _ in range(n):
        n_legs = int(rng.choice(leg_counts))
        segs = int(rng.choice(seg_counts))
        deriv = g.derive(legged_plan(n_legs=n_legs, segments_per_leg=segs,
                                     seg_len=float(rng.uniform(0.08, 0.18)),
                                     seg_rad=float(rng.uniform(0.015, 0.03))))
        out.append(deriv)
    return out


class PhiBodyNeural(nn.Module):
    """Same architecture family as the manipulation-side PhiNeural: a
    causal transformer context encoder, a latent predictor, a placement
    decoder (regression head, replacing the twist decoder), and an inverse
    readout head for op identity."""

    def __init__(self, d_model: int = 96, n_heads: int = 4, n_layers: int = 2,
                max_len: int = 64):
        super().__init__()
        self.in_proj = nn.Linear(TOKEN_DIM, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=4 * d_model,
            batch_first=True, dropout=0.0)
        self.encoder = nn.TransformerEncoder(enc_layer, n_layers)
        self.decoder = nn.Sequential(
            nn.Linear(d_model, 2 * d_model), nn.GELU(),
            nn.Linear(2 * d_model, 6))  # 6D placement target
        self.inv_head = nn.Linear(d_model, len(OPS))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.in_proj(x) + self.pos[:, : x.shape[1]]
        mask = nn.Transformer.generate_square_subsequent_mask(
            x.shape[1], device=x.device)
        z = self.encoder(h, mask=mask, is_causal=True)
        return {"placement": self.decoder(z), "op_logits": self.inv_head(z)}


def build_batch(derivations: List, phi: PhiBody):
    xs, targets, ids, = [], [], []
    for deriv in derivations:
        x = tokenize_derivation(deriv)
        tgt = np.stack([placement_6d(phi.rule_placement(a)["local"])
                        for a in deriv])
        rid = np.array([OP2I[a.sutra.op] for a in deriv])
        xs.append(x); targets.append(tgt); ids.append(rid)
    L = max(len(x) for x in xs)
    B = len(derivations)
    X = torch.zeros(B, L, TOKEN_DIM)
    Y = torch.zeros(B, L, 6)
    ID = torch.zeros(B, L, dtype=torch.long)
    for i, (x, y, rid) in enumerate(zip(xs, targets, ids)):
        l = len(x)
        X[i, :l] = torch.tensor(x)
        Y[i, :l] = torch.tensor(y)
        ID[i, :l] = torch.tensor(rid)
    return X, Y, ID


def evaluate(model: PhiBodyNeural, derivations: List, phi: PhiBody) -> Dict:
    model.eval()
    placement_errs, op_accs = [], []
    with torch.no_grad():
        for deriv in derivations:
            x = torch.tensor(tokenize_derivation(deriv)).unsqueeze(0)
            out = model(x)
            pred = out["placement"][0].numpy()
            true = np.stack([placement_6d(phi.rule_placement(a)["local"])
                             for a in deriv])
            placement_errs.append(float(np.mean((pred - true) ** 2)))
            pred_ops = out["op_logits"][0].argmax(-1).numpy()
            true_ops = np.array([OP2I[a.sutra.op] for a in deriv])
            op_accs.append(float((pred_ops == true_ops).mean()))
    return {"placement_mse": float(np.mean(placement_errs)),
           "op_id_acc": float(np.mean(op_accs))}


def run(n_train: int = 600, n_test: int = 100, epochs: int = 60,
       seed: int = 0, verbose: bool = True) -> Dict:
    torch.manual_seed(seed)
    phi = PhiBody()

    # in-distribution: leg counts {3,4,5,6}, segs {1,2} -- the paper's
    # original body sizes live inside this range
    train_derivs = sample_body_derivations(n_train, [3, 4, 5, 6], [1, 2], seed)
    test_id = sample_body_derivations(n_test, [3, 4, 5, 6], [1, 2], seed + 1)
    # held-out structural compositions: leg counts and a segment count
    # never seen during training
    test_ood = sample_body_derivations(n_test, [7, 8], [3], seed + 2)

    model = PhiBodyNeural()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    bs = 64
    for ep in range(epochs):
        perm = np.random.permutation(len(train_derivs))
        total = 0.0
        for i in range(0, len(train_derivs), bs):
            batch = [train_derivs[j] for j in perm[i:i + bs]]
            X, Y, ID = build_batch(batch, phi)
            out = model(X)
            loss_place = ((out["placement"] - Y) ** 2).mean()
            loss_op = nn.functional.cross_entropy(
                out["op_logits"].reshape(-1, len(OPS)), ID.reshape(-1))
            loss = loss_place + loss_op
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
        if verbose and (ep + 1) % 10 == 0:
            print(f"epoch {ep+1:3d}  loss={total/max(1,len(train_derivs)//bs):.4f}")

    id_metrics = evaluate(model, test_id, phi)
    ood_metrics = evaluate(model, test_ood, phi)
    if verbose:
        print("\nMorphology-side compositional generalization")
        print(f"{'split':<28}{'placement MSE':>15}{'op-ID acc':>12}")
        print(f"{'in-distribution':<28}{id_metrics['placement_mse']:>15.5f}"
             f"{id_metrics['op_id_acc']:>12.3f}")
        print(f"{'held-out (7-8 legs, 3 seg)':<28}{ood_metrics['placement_mse']:>15.5f}"
             f"{ood_metrics['op_id_acc']:>12.3f}")
    return {"in_distribution": id_metrics, "held_out": ood_metrics}


if __name__ == "__main__":
    run()
