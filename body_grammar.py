from __future__ import annotations
import random
from typing import Dict, List, Optional, Sequence
import numpy as np

from .body_karaka import BodyEntity, BodyFrame, BodyKaraka
from .body_sutra import BodySutra, BodyState
from .paribhasa import ParibhasaResolver


class BodyGrammarError(RuntimeError): pass


class BodyGrammar:
    def __init__(self, rules: Sequence[BodySutra], resolver=None, seed=None):
        self.rules = list(rules)
        self.resolver = resolver or ParibhasaResolver(self.rules)
        self.rng = random.Random(seed)

    def applicable(self, state, frame):
        return [s for s in self.rules if s.applicable(state, frame)]

    def derive(self, plan, *, sample_vibhasa=False, init_markers=None, log=None):
        state = BodyState()
        if init_markers: state.markers |= set(init_markers)
        derivation = []
        for step in plan:
            op = step["op"]
            frame = step.get("frame", BodyFrame())
            cands = [s for s in self.applicable(state, frame) if s.op == op]
            if not cands:
                raise BodyGrammarError(f"No applicable body-sūtra for op '{op}' "
                                       f"(markers={state.markers})")
            chosen = self.resolver.pick(cands, sample_vibhasa=sample_vibhasa)
            if chosen.vibhasa and sample_vibhasa:
                ranked = self.resolver.resolve(cands, sample_vibhasa=True)
                chosen = self.rng.choice(ranked[: min(2, len(ranked))])
            if log is not None:
                merged = frame.merge_inherited(state.inherited)
                log.append({"candidates": list(cands), "state_markers": set(state.markers),
                           "frame": merged, "chosen": chosen})
            applied = chosen.apply(state, frame, overrides=step.get("params"))
            derivation.append(applied)
            state = applied.next_state
        return derivation


def _ent(name): return BodyEntity(name, np.eye(4))
def torso_frame(): return BodyFrame().bind(BodyKaraka.MULA, _ent("torso"))
def _limb_role_frame():
    return (BodyFrame().bind(BodyKaraka.ANGA, _ent("limb"))
            .bind(BodyKaraka.DANDA, _ent("seg")).bind(BodyKaraka.SANDHI, _ent("joint"))
            .bind(BodyKaraka.PADA, _ent("foot")))


def legged_plan(n_legs: int, segments_per_leg: int = 2, mount_angles=None,
               seg_len: float = 0.12, seg_rad: float = 0.02) -> List[Dict]:
    if mount_angles is None:
        mount_angles = list(np.linspace(0, 2 * np.pi, n_legs, endpoint=False))
    rf = _limb_role_frame()
    plan: List[Dict] = [{"op": "ADD_TORSO", "frame": torso_frame()}]
    for i, ang in enumerate(mount_angles):
        first = (i == 0)
        plan.append({"op": "ADD_LIMB", "frame": rf,
                    "params": ({"mount_angle": float(ang)} if not first
                              else {"mount_angle": float(ang),
                                    "seg_len": seg_len, "seg_rad": seg_rad})})
        for s in range(1, segments_per_leg):
            plan.append({"op": "ARTICULATE", "frame": rf})
            plan.append({"op": "EXTEND", "frame": rf})
        plan.append({"op": "TERMINATE", "frame": rf})
    return plan
