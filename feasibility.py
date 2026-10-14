from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np


@dataclass
class FeasibilitySpec:
    required_reach: float = 0.30
    min_clearance: float = 0.05
    mass_budget: float = 6.0
    require_symmetry: bool = False
    min_support_area: float = 0.02
    min_tip_separation: float = 0.03


@dataclass
class FeasibilityReport:
    ok: bool
    checks: Dict[str, bool]
    margins: Dict[str, float]


def _tips(tree):
    pts = [n.world[:3, 3] for n in tree.walk() if n.kind == "foot"]
    if not pts:
        pts = [n.world[:3, 3] for n in tree.walk() if n.kind == "link"]
    return np.array(pts) if pts else np.zeros((0, 3))


def _total_mass(tree):
    return float(sum(n.geom.get("mass", 0.0) for n in tree.walk()))


def _support_area(tips):
    if len(tips) < 3: return 0.0
    xy = tips[:, :2]
    try:
        from scipy.spatial import ConvexHull
        return float(ConvexHull(xy).volume)
    except Exception:
        c = xy.mean(0)
        ang = np.arctan2(xy[:,1]-c[1], xy[:,0]-c[0])
        q = xy[np.argsort(ang)]
        x, y = q[:,0], q[:,1]
        return float(0.5*abs(np.dot(x, np.roll(y,-1)) - np.dot(y, np.roll(x,-1))))


def check_feasibility(tree, spec: Optional[FeasibilitySpec] = None) -> FeasibilityReport:
    spec = spec or FeasibilitySpec()
    tips = _tips(tree); torso_c = tree.world[:3, 3]
    checks, margins = {}, {}
    reach = float(np.max(np.linalg.norm(tips[:,:2]-torso_c[:2], axis=1))) if len(tips) else 0.0
    margins["reach"] = reach - spec.required_reach; checks["reach"] = margins["reach"] >= 0
    margins["clearance"] = torso_c[2] - spec.min_clearance
    checks["clearance"] = margins["clearance"] >= 0
    m = _total_mass(tree)
    margins["mass_budget"] = spec.mass_budget - m; checks["mass_budget"] = margins["mass_budget"] >= 0
    area = _support_area(tips)
    margins["support"] = area - spec.min_support_area; checks["support"] = margins["support"] >= 0
    if len(tips) >= 2:
        d = np.linalg.norm(tips[:,None,:]-tips[None,:,:], axis=-1)
        d = d + np.eye(len(tips))*1e9
        sep = float(d.min())
    else:
        sep = 1e9
    margins["self_clear"] = sep - spec.min_tip_separation
    checks["self_clear"] = margins["self_clear"] >= 0
    ok = all(checks.values())
    return FeasibilityReport(ok=ok, checks=checks, margins=margins)
