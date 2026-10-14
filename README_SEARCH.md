# M2/M3 — The Missing Piece: Optimizing Morphology for a Task

Everything before this experiment (H5/M-H5 homomorphism, H6/M-H3
compactness, M-H6 feasibility, H-BOX region embeddings) proved the grammar
is *structurally sound* — valid, compact, compositional, and its specificity
logic is geometrically recoverable. **None of it showed the grammar could be
searched to find a body that is actually good at a task.** That's the
thesis's central claim, and this is the first experiment that tests it.

## Run

```bash
python3 -m experiments.m_search_loop      # single-seed illustration + honest multi-seed benchmark
python3 -m tests.test_search_loop         # smoke tests
```

## What it does

- **Genome = grammar parameters**, not raw XML: leg count, segments per
  limb, seg_len, seg_rad, mount-angle jitter. Mutating a genome re-derives
  the body through the *same grammar*, so every mutation is
  legal-by-construction (the direct comparison point against RoboMorph/
  LASeR-style raw-XML mutation, which can produce invalid trees).
- **Fitness = CPG-driven locomotion proxy** (`gmm/task_fitness.py`): a fixed
  sinusoidal gait (not a trained policy) drives the compiled MuJoCo body;
  fitness is forward CoM displacement minus an energy penalty and a soft
  fall penalty. This is deliberately a cheap, coarse proxy — standard
  practice in morphology search (Sims-style) so morphology search isn't
  confounded with control-policy search in this first pass. A trained-policy
  fitness (Swarm-JEPA per candidate body) is the natural, more expensive
  Phase M2 refinement, not what's tested here.
- **Śulba gate active by default**: infeasible genomes are rejected before
  any physics call (ties directly to M-H6's efficiency result).
- **Three search conditions**: grammar-guided (mu+lambda) evolutionary
  search, random search over the same genome space, and the gate on/off
  comparison is available via `use_gate=False`.

## Result (honest, multi-seed)

Single-seed illustration (seed=0, budget=60 physics evals each):

| | best fitness |
|---|---|
| Evolutionary search | **+2.461** |
| Random search | +1.477 |

**But single seeds are not the number to trust** — the fitness proxy is
noisy (a lucky/unlucky CPG phase can dominate a short rollout), so the
real result is the 8-seed benchmark:

| | mean ± std | wins |
|---|---|---|
| Evolutionary search | **1.998 ± 0.634** | **7/8 seeds** |
| Random search | 1.543 ± 0.552 | 1/8 seeds |

Paired t-test: t=2.032, **p=0.082** — trending clearly in the right
direction, not yet significant at n=8. This is reported as-is, not dressed
up: **more seeds and/or larger budget is the direct next step**, not a
different method.

## Known limitation, found and fixed during development — worth keeping in the writeup

The first version of the CPG fitness gave *identical* displacement (to 4
decimal places) across bodies with different leg counts, which is
impossible for a genuinely leg-count-sensitive gait — it turned out actuator
gain (`kp=8`) was too weak to drive locomotion at all; the signal was
dominated by how the torso happened to topple and slide, which is
morphology-independent. Raising `kp` to 40 caused physics blowups instead
(confirmed by MuJoCo's own instability warnings); `kp=18` with an explicit
implausible-displacement guard (`>3m` in the horizon ⇒ rejected, not scored)
is the settled value. This kind of proxy-reward sanity check — does fitness
actually vary with the thing you're claiming to optimize? — is worth stating
explicitly in the paper's methods section, not hidden as an implementation
detail: it's exactly the kind of failure mode ("reward doesn't measure what
you think") that morphology-search papers should be checking for and often
don't report.

## Files

```
gmm/task_fitness.py          CPG locomotion fitness evaluator + the blowup guard
experiments/m_search_loop.py Genome, mutation/crossover, evolutionary_search(),
                              random_search(), benchmark() (the honest multi-seed
                              report), run() (single-seed illustration)
tests/test_search_loop.py    monotonicity of evolutionary best-so-far, gate
                              rejection path, benchmark() smoke test
```

## Next steps toward a full paper (priority order)

1. **Scale this experiment**: larger budget (200+ evals) and more seeds
   (20+) to get past p=0.082 into clear significance — the direction is
   already right, this needs compute, not a new method.
2. **M-H1 (morphology compositional generalization)**: train the search on
   one leg-count/segment range, test transfer to held-out combinations
   (mirrors the manipulation H1 result).
3. **Ablate the Śulba gate inside this exact search loop** (`use_gate=False`
   vs `True`, same budget) — ties M-H6's efficiency claim directly to task
   outcome, not just "fewer wasted physics calls."
4. **Replace the CPG with Swarm-JEPA per-candidate fitness** (Phase M2) —
   compare rank correlation between the cheap CPG proxy and true trained-
   policy fitness; if correlated, the CPG is a valid cheap filter (the
   efficiency argument H-BOX and M-H6 both anticipate).
5. **Raw-XML-mutation baseline**: mutate the compiled MJCF directly (no
   grammar) and measure the fraction of resulting bodies that are even
   loadable — this is the concrete validity-rate comparison the roadmap's
   M-H1 promised and hasn't been run yet.
