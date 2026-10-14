# Typed Case-Role Production Grammars for Verifiable Robot Skill and Morphology Synthesis

Code, figures and LaTeX source for the paper. A typed case-role production
grammar (manipulation roles Agent/Patient/Instrument/Location/Source/Goal;
morphology roles Root/Limb/Segment/Joint/Terminal) with parameter inheritance
and specificity arbitration, a geometric realization map Φ to SE(3), a
feasibility gate, and evolutionary search over MuJoCo robot bodies.

## Layout

```
ptg/          grammar: roles, rules, arbitration (body_karaka, body_sutra, body_grammar, paribhasa, karaka)
gmm/          Φ_body (phi_body), SE(3) math (se3), MJCF compiler, CPG locomotion fitness (task_fitness)
sulba/        geometric feasibility gate (reach, clearance, mass, support polygon, self-collision)
experiments/  every experiment script (table below)
figures/      make_figures.py, schematic_robots.py, fig1-fig6 (.pdf/.png)
paper/        main.tex (TMLR format), tmlr.bib, figures/
tests/        smoke tests
docs/         earlier runbook / roadmap / per-module notes
```

## Setup

```bash
pip install numpy scipy matplotlib mujoco==3.13.0     # pin: results change across MuJoCo versions
pip install torch                                     # only for compgen / learned-box scripts
pip install cma scikit-posthocs imageio               # optional (some baselines, post-hoc tests, renders)
```

Run everything **from the repository root** as modules (`python3 -m experiments.<name>`).
Do not redirect stderr to /dev/null; it hides import errors.

MuJoCo 3.8.1 vs 3.13.0 gives different search results on identical code and
seeds (paper, simulator-sensitivity section). Pin the version.

## Experiments

Status: **T** = tested and reproduced by the author; **R** = run on the author's cluster (needs MuJoCo/torch);
**S** = syntax-checked only, not yet run.

| Script | What it measures | Status |
|---|---|---|
| `reviewer2_defect_decomposition` | Φ homomorphism defect: rotation vs translation parts (inactive 0.0; active ≈0.037) | T |
| `reviewer_q1q5_longer_chains` | Defect over chains k=2..6 (bounded, max 0.081) | T |
| `reviewer3_compactness_curve` | Flat vs typed rule count over legs × segments (max 13.5×) | T |
| `todo1_typing_vs_inheritance` | Compactness ablation: typing alone vs + inheritance | T |
| `todo5_cardinality_bound` | Derivation cardinality bound (21/21 match) | T |
| `reviewer_q6_runtime_cost` | Runtime cost of inheritance (3.4–3.7%) | T |
| `todo3_box_extended`, `reviewer_q2_region_sensitivity`, `reviewerC_specificity_sensitivity` | Specificity as box containment; width-floor sweep; order-embedding and Euclidean controls | T |
| `reviewerDE_noise_and_conjunction` | Bit-flip perception noise; box-intersection conjunction | T |
| `todo2_morphology_compgen` | Morphology compositional generalization (needs torch) | R |
| `main_baseline_table` | 6-method search comparison | R |
| `expanded_baseline_table` | 21-method search comparison, Friedman + Nemenyi | R |
| `expanded_validity_comparison` | Grammar mutation vs 14 direct-edit alternatives (15 conditions) | R |
| `todo4a_scaled_benchmark`, `m_search_loop` | Scaled multi-seed search vs random | R |
| `mh1_xml_baseline`, `mh6b_gate_ablation` | Raw-XML mutation validity; gate on/off under a fixed physics budget | R |
| `reviewerFGH_cmaes_budget_controller` | CMA-ES lite vs standard, budget sweep, controller study. F and G run; **H is blocked** until `gmm/task_fitness.evaluate_fitness` accepts `freq`/`amp`/`phase_offset` | R (F, G) |
| `diagnostic_noise_sweep`, `_v2` | Fitness noise vs gait seeds | R |
| `reviewer_q3_ood_diagnostic` | Manipulation compgen anomaly (OOD error < ID); needs the manipulation repo + torch | S |
| `reviewer_q8_goal_aligned_gate` | Goal-aligned feasibility gate | S |
| `reviewer_q7_cmaes_baseline` | Superseded by the baseline tables | S |
| `render_real_robots` | Real MuJoCo renders, grammar vs direct edit | S (see below) |

Helper: `box_embed_np.py` (NumPy box embeddings, required by the region scripts).

Example:

```bash
python3 -m experiments.reviewer3_compactness_curve
python3 -m experiments.expanded_baseline_table
python3 -m experiments.expanded_validity_comparison
python3 -m tests.test_search_loop && python3 -m tests.test_new_experiments
```

### Rendering

```bash
export MUJOCO_GL=egl        # or osmesa; needs a node with a GL backend (not a bare login node)
python3 -m experiments.render_real_robots --seed 0 --settle_steps 300
```
A previous run produced black images, and a no-DISPLAY node crashed in GLFW. The renderer is
unverified until you get a non-black image; if it stays black, check lighting.

### Figures

```bash
python3 figures/make_figures.py
```
`figures/schematic_robots.py` has two known cosmetic bugs (an escaped underscore in a title; collinear
segments hide the two-segment structure) and its output is not used in the paper.

## Key results (as reported in the paper)

- Homomorphism defect: exactly 0 on inheritance-inactive pairs; ≈0.037 when active (0° rotation, 3.75 ± 2.08 cm). Chains k=2–6 stay ≤ 0.081.
- Compactness: 6 typed rules vs 17/37/49 flat for 4×2, 6×3, 8×3 bodies; up to 13.5× in the sweep.
- Containment: 1.000 across width floors and transitive pairs; held-out conflict agreement box 0.812, order 0.906, Euclidean 0.719.
- Search: 21 methods, Friedman χ² = 33.46, p = 0.030; ours has the best average rank (6.25) but only coordinate descent is significantly different after Nemenyi.
- Gate: finds a feasible body in 1.0 vs 9.6 ± 6.2 evaluations; task fitness 1.137 vs 1.672 (p = 0.254). It buys constraint satisfaction, not task optimality.
- Validity: grammar 100%. Direct edits range from 0% to 100% depending on how tame the mutator is. The validity criterion (loads and steps without non-finite state) is permissive.
