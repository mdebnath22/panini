# panini-morphology — Grammar-Driven Robot Body Generation

Phase M0–M1 implementation: a Pāṇinian body grammar generates MuJoCo robot
morphologies, a Śulba-style geometric layer gates feasibility, and Φ_body
maps derivations to kinematic trees via SE(3) composition. Built on top of
the panini-motion manipulation grammar — the Pāṇinian mechanics (anuvṛtti,
it-tags, paribhāṣā, vibhāṣā) and SE(3) math are shared, unchanged.

## What runs today (all tested, MuJoCo 3.10)

```bash
python3 demo_morphology.py                 # full pipeline walkthrough
python3 -m experiments.mh5_body_homomorphism   # M-H5 go/no-go
python3 -m experiments.mh6_feasibility         # M-H6 gate + M-H3 compactness
python3 -m tests.test_morphology               # 6 smoke tests
```

## Files (new)

```
ptg/
  body_karaka.py    BodyKaraka roles (mūla/aṅga/daṇḍa/sandhi/pāda/preraka),
                    BodyEntity (geometry + parent-relative pose), TreeNode
  body_sutra.py     body sūtras: ADD_TORSO/ADD_LIMB/ARTICULATE/EXTEND/
                    TERMINATE/SYMMETRIZE + reach apavāda; reuses anuvṛtti,
                    it-tags, vibhāṣā, specificity
  body_grammar.py   body-derivation engine + legged_plan() template
gmm/
  phi_body.py       Φ_body: derivation -> kinematic tree via SE(3) placement
                    composition; chain_compose() for M-H5
  mjcf_compiler.py  kinematic tree -> MuJoCo MJCF; compile_and_validate()
                    loads+steps the model (SP-style physics gate)
sulba/
  feasibility.py    Śulba geometric gate: reach, clearance, mass, support
                    polygon, self-collision, symmetry — each with a margin
experiments/
  mh5_body_homomorphism.py   M-H5 defect (GO/NO-GO)
  mh6_feasibility.py         M-H6 gate efficiency + M-H3 compactness
tests/test_morphology.py     smoke tests
demo_morphology.py           end-to-end demo
```

## Results (this exact code)

- **M-H5 (go/no-go): GO.** Body-Φ homomorphism defect is exactly 0.0000 on
  inheritance-inactive rule pairs (ADD_LIMB→ARTICULATE) and small/bounded
  (0.038 ± 0.024) on inheritance-active pairs (ARTICULATE→EXTEND,
  ADD_LIMB→EXTEND). Composing body-rules == composing SE(3) part placements
  off-inheritance ⇒ the generated kinematic tree is a valid transform
  composition by construction. This is the same structured-defect result as
  the manipulation H5, reproduced for morphology.
- **M-H6 (Śulba gate efficiency):** on a demanding spec (≈5% of random bodies
  feasible), the gate reaches a task-satisfying body in 1.0 compile vs.
  9.6 ± 6.2 physics evaluations without it — ≈90% fewer physics evals. The
  gate rejects geometrically-infeasible bodies before paying for MuJoCo.
- **M-H3 (compactness):** anuvṛtti gives a fixed 6-rule grammar regardless of
  body size; a flat body grammar needs 17 / 37 / 49 ground rules for
  4-leg×2-seg / 6-leg×3-seg / 8-leg×3-seg bodies (2.8× / 6.2× / 8.2×).
- **Physics:** generated quadruped loads in MuJoCo, settles under gravity at
  0.076 m with 8 position actuators; 3/4/6-leg bodies all compile+step.

## Wiring to the rest of the roadmap

- **Phase M2 (JEPA fitness):** tokenize a body derivation exactly as
  panini-motion tokenizes rule sequences; predict task return from body
  structure + cheap rollout stats. Reuse phi_neural's transformer + EMA.
- **Phase M3 (outer loop):** mutate *derivations* (edit a sūtra, swap the
  limb template, toggle SYMMETRIZE) — anuvṛtti keeps mutations compact and
  legal-by-construction. Compare against raw-XML mutation (RoboMorph/LASeR).
- **Phase M4:** M-H1 (validity), M-H2 (task fit vs baselines), M-H4
  (JEPA-surrogate rank correlation), M-H7 (Ramanujan design-space count).

## Design notes

- ADD_LIMB mounts a limb by rotating about the torso z-axis (mount_angle)
  then translating to the torso surface — a pure SE(3) placement, so limb
  count/spacing is a grammar parameter, not a code change.
- SYMMETRIZE mirrors the just-built limb about the torso sagittal plane (an
  improper isometry handled explicitly in phi_body._mirror_subtree).
- The compiler clamps degenerate geometry and guarantees unique names — the
  auto-patcher role from the CoRL MuJoCo-XML validator, at the body level.
