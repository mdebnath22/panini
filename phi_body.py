from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np
from ptg.body_karaka import TreeNode

_JOINT_NAME = {0.0: "hinge", 1.0: "slide", 2.0: "ball", 3.0: "free"}

def _trans(v):
    T = np.eye(4); T[:3, 3] = v; return T

def _rot_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    T = np.eye(4); T[:3, :3] = np.array([[c,-s,0],[s,c,0],[0,0,1]]); return T


class PhiBody:
    def rule_placement(self, applied) -> Dict:
        op = applied.sutra.op; p = applied.params
        if op == "ADD_TORSO":
            return {"kind": "torso", "local": _trans([0,0,p.get("torso_height",0.30)]),
                   "geom": {"length": p.get("torso_len",0.30), "radius": p.get("torso_rad",0.08),
                            "mass": p.get("torso_mass",2.0)}, "joint": {"type": "free"}}
        if op == "ADD_LIMB":
            ang = p.get("mount_angle", 0.0); r_torso = p.get("torso_rad", 0.08)
            local = _rot_z(ang) @ _trans([r_torso, 0, 0])
            jt = _JOINT_NAME.get(p.get("joint_type", 0.0), "hinge")
            return {"kind": "link", "local": local,
                   "geom": {"length": p.get("seg_len",0.12), "radius": p.get("seg_rad",0.02),
                            "mass": p.get("seg_mass",0.15)},
                   "joint": {"type": jt, "axis": (0,1,0), "range": p.get("joint_range",0.9),
                            "actuated": True}}
        if op == "ARTICULATE":
            jt = _JOINT_NAME.get(p.get("joint_type", 0.0), "hinge")
            return {"kind": "joint_marker", "local": _trans([p.get("_parent_len",0.12),0,0]),
                   "geom": {}, "joint": {"type": jt, "axis": (0,1,0),
                            "range": p.get("joint_range",0.9),
                            "actuated": bool(p.get("actuated",1.0))}}
        if op == "EXTEND":
            taper = p.get("taper", 0.85); base_len = p.get("seg_len", 0.12)
            base_rad = p.get("seg_rad", 0.02)
            return {"kind": "link", "local": _trans([base_len,0,0]),
                   "geom": {"length": base_len, "radius": base_rad*taper,
                            "mass": p.get("seg_mass",0.15)*taper}, "joint": None}
        if op == "TERMINATE":
            base_len = p.get("seg_len", 0.12)
            return {"kind": "foot", "local": _trans([base_len,0,0]),
                   "geom": {"radius": p.get("foot_rad",0.025), "mass": p.get("foot_mass",0.05)},
                   "joint": None}
        if op == "SYMMETRIZE":
            M = np.eye(4); M[1,1] = -1.0
            return {"kind": "mirror", "local": M, "geom": {}, "joint": None}
        raise KeyError(f"Φ_body has no placement for op '{op}'")

    def build_tree(self, derivation) -> TreeNode:
        root = None; current_tip = None; current_limb_root = None
        last_seg_len = 0.12; node_i = 0
        for applied in derivation:
            place = self.rule_placement(applied); op = applied.sutra.op
            if op == "ADD_TORSO":
                root = TreeNode("torso", "torso", place["local"], place["geom"], place["joint"])
                current_tip = root; continue
            if op == "ADD_LIMB":
                node = TreeNode(f"limb{node_i}_root", "link", place["local"],
                               place["geom"], place["joint"]); node_i += 1
                root.children.append(node)
                current_limb_root = node; current_tip = node
                last_seg_len = place["geom"]["length"]; continue
            if op == "ARTICULATE":
                place2 = dict(place); place2["local"] = _trans([last_seg_len,0,0])
                node = TreeNode(f"joint{node_i}", "joint_marker", place2["local"], {}, place2["joint"])
                node_i += 1; current_tip.children.append(node); current_tip = node; continue
            if op == "EXTEND":
                node = TreeNode(f"seg{node_i}", "link", place["local"], place["geom"], place["joint"])
                node_i += 1; current_tip.children.append(node); current_tip = node
                last_seg_len = place["geom"]["length"]; continue
            if op == "TERMINATE":
                node = TreeNode(f"foot{node_i}", "foot", place["local"], place["geom"], None)
                node_i += 1; current_tip.children.append(node)
                current_tip = current_limb_root; continue
            if op == "SYMMETRIZE":
                if current_limb_root is not None:
                    mirrored = _mirror_subtree(current_limb_root, node_i)
                    node_i += 1 + sum(1 for _ in current_limb_root.walk())
                    root.children.append(mirrored)
                continue
        _fill_world(root, np.eye(4))
        return root


def _mirror_subtree(node, offset):
    M = np.eye(4); M[1,1] = -1.0
    new_local = M @ node.local @ M
    clone = TreeNode(f"mir{offset}_{node.node_id}", node.kind, new_local,
                     dict(node.geom), dict(node.joint) if node.joint else None)
    for c in node.children:
        clone.children.append(_mirror_subtree(c, offset))
    return clone


def _fill_world(node, parent_world):
    node.world = parent_world @ node.local
    for c in node.children:
        _fill_world(c, node.world)
