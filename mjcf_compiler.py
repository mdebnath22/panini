from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np

_MJ_JOINT = {"hinge": "hinge", "slide": "slide", "ball": "ball", "free": "free"}


def _mat_to_quat(R) -> Tuple[float, float, float, float]:
    t = np.trace(R)
    if t > 0:
        s = 0.5 / np.sqrt(t + 1.0); w = 0.25 / s
        x = (R[2,1]-R[1,2])*s; y = (R[0,2]-R[2,0])*s; z = (R[1,0]-R[0,1])*s
    else:
        i = int(np.argmax([R[0,0], R[1,1], R[2,2]]))
        if i == 0:
            s = 2.0*np.sqrt(1.0+R[0,0]-R[1,1]-R[2,2])
            w=(R[2,1]-R[1,2])/s; x=0.25*s; y=(R[0,1]+R[1,0])/s; z=(R[0,2]+R[2,0])/s
        elif i == 1:
            s = 2.0*np.sqrt(1.0+R[1,1]-R[0,0]-R[2,2])
            w=(R[0,2]-R[2,0])/s; x=(R[0,1]+R[1,0])/s; y=0.25*s; z=(R[1,2]+R[2,1])/s
        else:
            s = 2.0*np.sqrt(1.0+R[2,2]-R[0,0]-R[1,1])
            w=(R[1,0]-R[0,1])/s; x=(R[0,2]+R[2,0])/s; y=(R[1,2]+R[2,1])/s; z=0.25*s
    q = np.array([w,x,y,z]); return tuple(q/(np.linalg.norm(q)+1e-12))


def _clamp(v, lo, hi): return float(max(lo, min(hi, v)))


class MJCFCompiler:
    def __init__(self):
        self.actuators: List[str] = []

    def _body_xml(self, node, indent=3) -> str:
        pad = "  "*indent
        pos = node.local[:3, 3]; quat = _mat_to_quat(node.local[:3,:3])
        lines = [f'{pad}<body name="{node.node_id}" pos="{pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}" '
                f'quat="{quat[0]:.4f} {quat[1]:.4f} {quat[2]:.4f} {quat[3]:.4f}">']
        if node.joint and node.joint.get("type") != "free":
            jt = _MJ_JOINT.get(node.joint["type"], "hinge")
            axis = node.joint.get("axis", (0,1,0))
            rng = _clamp(node.joint.get("range", 0.9), 0.1, 3.0)
            if jt in ("hinge", "slide"):
                lines.append(f'{pad}  <joint name="{node.node_id}_j" type="{jt}" '
                            f'axis="{axis[0]} {axis[1]} {axis[2]}" range="{-rng:.3f} {rng:.3f}"/>')
                if node.joint.get("actuated"): self.actuators.append(f"{node.node_id}_j")
            elif jt == "ball":
                lines.append(f'{pad}  <joint name="{node.node_id}_j" type="ball"/>')
        if node.kind == "torso":
            L=_clamp(node.geom.get("length",0.30),0.05,1.0); r=_clamp(node.geom.get("radius",0.08),0.02,0.3)
            m=_clamp(node.geom.get("mass",2.0),0.1,20.0)
            lines.append(f'{pad}  <geom name="{node.node_id}_g" type="capsule" '
                        f'fromto="{-L/2:.3f} 0 0 {L/2:.3f} 0 0" size="{r:.3f}" mass="{m:.3f}"/>')
        elif node.kind == "link":
            L=_clamp(node.geom.get("length",0.12),0.02,0.6); r=_clamp(node.geom.get("radius",0.02),0.005,0.1)
            m=_clamp(node.geom.get("mass",0.15),0.01,3.0)
            lines.append(f'{pad}  <geom name="{node.node_id}_g" type="capsule" '
                        f'fromto="0 0 0 {L:.3f} 0 0" size="{r:.3f}" mass="{m:.3f}"/>')
        elif node.kind == "foot":
            r=_clamp(node.geom.get("radius",0.025),0.005,0.08); m=_clamp(node.geom.get("mass",0.05),0.01,1.0)
            lines.append(f'{pad}  <geom name="{node.node_id}_g" type="sphere" size="{r:.3f}" mass="{m:.3f}"/>')
        for c in node.children:
            lines.append(self._body_xml(c, indent+1))
        lines.append(f'{pad}</body>')
        return "\n".join(lines)

    def compile(self, tree, *, name="panini_body") -> str:
        self.actuators = []
        body_block = self._body_xml(tree, indent=3)
        act_lines = "\n".join(f'    <position name="{a}_act" joint="{a}" kp="18" '
                              f'kv="1.5" ctrlrange="-1 1"/>' for a in self.actuators)
        return f"""<mujoco model="{name}">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.005" gravity="0 0 -9.81"/>
  <default><geom rgba="0.7 0.6 0.5 1" friction="1 0.1 0.1"/></default>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1" rgba="0.3 0.3 0.3 1"/>
    <light pos="0 0 3"/>
{body_block}
  </worldbody>
  <actuator>
{act_lines}
  </actuator>
</mujoco>
"""


def compile_and_validate(tree, *, name="panini_body", settle_steps=200) -> Dict:
    comp = MJCFCompiler()
    xml = comp.compile(tree, name=name)
    report = {"ok": False, "loaded": False, "stepped": False,
             "n_actuators": len(comp.actuators), "error": None, "xml": xml}
    try:
        import mujoco
        model = mujoco.MjModel.from_xml_string(xml)
        report["loaded"] = True
        data = mujoco.MjData(model)
        for _ in range(settle_steps):
            mujoco.mj_step(model, data)
        if np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel)):
            report["stepped"] = True
        report["ok"] = report["loaded"] and report["stepped"]
    except Exception as e:
        report["error"] = f"{type(e).__name__}: {e}"
    return report
