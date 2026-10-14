# MASTER RUN-BOOK — Everything Built So Far

One page covering every experiment across the whole project: what it tests,
the exact command, and the result last time it ran. Organized in the order
a paper would present them.

---

## Part 1 — Manipulation grammar (needs the ORIGINAL panini-motion repo,
## not this `pm/` package — these files aren't in this zip)

```bash
cd /path/to/panini-motion
python3 -m experiments.h5_homomorphism      # H5: GO, defect ≡0 off-inheritance
python3 -m experiments.h6_compactness       # H6: 16x fewer rules than flat grammar
python3 -m experiments.h1_compgen           # H1: Φ⁻¹ 100% on OOD compositions (needs torch)
python3 -m tests.test_morphology            # smoke tests
```

## Part 2 — Morphology grammar + Śulba feasibility (this package: `pm/`)

```bash
cd pm/
python3 -m experiments.mh5_body_homomorphism   # M-H5: GO (rerun to confirm on your build)
python3 -m experiments.mh6_feasibility         # M-H6: ~90% fewer wasted physics evals
python3 -m tests.test_morphology               # if present — smoke tests
```
(`mh5_body_homomorphism.py` / `mh6_feasibility.py` were built two sessions
ago — if not in your current tree, ask and I'll regenerate them; the logic
is unchanged from what's documented in README_MORPHOLOGY.md.)

## Part 3 — Region embeddings for paribhāṣā (H-BOX)

Lives in a separate zip (`panini-hbox.zip`) — needs `torch`.

```bash
python3 -m experiments.h_box_paribhasa      # GO, 1.000 agreement both grammars
python3 -m tests.test_hbox
```

## Part 4 — Task optimization: the search loop (this package)

**This is the core "AI optimizes morphology for a task" result.**

```bash
cd pm/
python3 -m experiments.m_search_loop 2>/dev/null
# → single-seed illustration (seed=0), then the honest 8-seed benchmark:
#   evolutionary: 1.998 ± 0.634, wins 7/8 seeds, p=0.082 vs random search

python3 -m experiments.mh6b_gate_ablation 2>/dev/null
# → gate-on vs gate-off under a FIXED PHYSICS BUDGET (fair accounting)
#   finding: gate-off can win on raw task fitness (1.672 vs 1.137, gate-on
#   wins 2/8) — the Śulba gate optimizes for CONSTRAINT SATISFACTION, not
#   raw task score; these are different objectives, report both

python3 -m experiments.mh1_xml_baseline 2>/dev/null
# → grammar mutation: 100/100 valid, every seed
#   raw-XML mutation (realistic adversarial mutator): 76-82% valid
#   this is the RoboMorph/LASeR-style comparison the roadmap promised

python3 -m tests.test_search_loop
python3 -m tests.test_new_experiments
```

**Pin your environment before trusting any number** — a MuJoCo `3.8.1` vs
`3.13.0` mismatch previously produced completely different search outcomes
on the identical code (confirmed, bit-for-bit reproducible once matched):

```bash
pip freeze | grep -iE "mujoco|numpy|scipy|torch" > requirements.txt
```

---

## Full results table (paper-ready)

| Experiment | Claim | Result |
|---|---|---|
| H5 | Manipulation Φ is a homomorphism | GO — defect ≡0 off-inheritance, 0.03–0.05 on-inheritance |
| H6 | Manipulation grammar is compact | 16× fewer rules than flat grammar |
| H1 | Manipulation grammar composes | Φ⁻¹ 100% symbolic readout on OOD rule sequences |
| M-H5 | Morphology Φ is a homomorphism | GO — same structure as H5 |
| M-H6 | Śulba gate saves compute (any feasible body) | ~90% fewer wasted physics evals |
| H-BOX | Paribhāṣā specificity is geometric containment | GO — 1.000 agreement both grammars, untrained control 0.000 |
| **M2/M3** | Grammar-guided search beats random on a real task | **evo 1.998±0.634 vs rnd 1.543±0.552, wins 7/8 seeds, p=0.082** |
| **M-H6b** | Śulba gate improves TASK OUTCOME (not just efficiency) | **NOT confirmed — gate-off wins on raw task score (1.672 vs 1.137); gate optimizes constraint-satisfaction, a different objective** |
| **M-H1** | Grammar mutation stays valid; raw-XML mutation doesn't | **100% vs 76–82%, consistent across 3 seeds** |

Bold rows are new this session.

---

## Two bugs found and fixed while building this — keep both in your methods section

1. **CPG actuator gain**: `kp=8` was too weak to drive real locomotion —
   fitness was measuring "how the torso topples," not morphology. Fixed to
   `kp=18` with a blowup guard (`gmm/task_fitness.py`). Caught by noticing
   identical displacement across different leg counts, which is physically
   impossible for a real leg-count-sensitive gait.
2. **`Genome.angle_jitter` clamp**: allowed negative values, which crashed
   `rng.uniform(-neg, neg)` (`low > high`) roughly 2% of the time during
   mutation. Fixed the clamp to `[0, 0.3]` (it's a magnitude, not a signed
   value). Caught while building the XML-mutation baseline.

Both are the kind of "does my proxy metric/search actually measure what I
think" check that's often skipped in morphology-search papers and worth
stating explicitly as a methods contribution, not hiding as a footnote.

---

## What's still missing for a complete paper (updated priority)

1. **Scale M2/M3** to significance — more seeds (20+) and/or larger budget.
   Direction is right (7/8, p=0.082); this needs compute, not a new method.
2. **M-H1 compositional generalization** (train search on one leg/segment
   range, test transfer to held-out combinations) — mirrors manipulation H1,
   not yet built for morphology.
3. **Swarm-JEPA fitness surrogate** replacing the CPG proxy (Phase M2 proper)
   — the largest remaining piece, deserves its own session: needs a
   derivation tokenizer for bodies (mirrors `phi_neural.tokenize_sequence`),
   a training set of (genome, true fitness) pairs, and a rank-correlation
   validation against the CPG proxy.
4. **H-BOX rank correlation** — add 2–3 more apavāda pairs so volume-
   ordering is N>5, not N=2.
5. Ramanujan design-space cardinality bound (M-H7) — one counting-argument
   section, cheap once you're ready to write.
