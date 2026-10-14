# H-BOX — Paribhāṣā as Learned Region Containment

Tests whether Pāṇini's specificity hierarchy (apavāda overrides utsarga) is
recoverable as a **geometric containment structure** in a learned embedding
space, rather than only existing as the hand-coded `specificity` integer the
symbolic `ParibhasaResolver` uses.

## Run

```bash
python3 -m experiments.h_box_paribhasa   # full experiment + negative control
python3 -m tests.test_hbox               # smoke test
```

## The idea, precisely

- Each rule's **applicability condition** (its required kāraka roles +
  positive markers it consumes) becomes a learnable axis-aligned **box** in
  R^d (center + softplus half-width, with a width floor to prevent the
  well-documented box-embedding collapse failure mode).
- Each **derivation state** becomes a deterministic one-hot **point** in the
  same space (roles present, active markers, task family) — points are not
  learned, only rule boxes are, so the geometric claim isn't confounded with
  ordinary representation-learning noise.
- Training signal: (a) a rule's box must contain every state where it is
  symbolically known to fire, and (b) box(apavāda) ⊂ box(utsarga) for every
  known specific/general pair — the direct geometric translation of I.4.2
  ("in conflict, the more specific rule wins").

## Three falsifiable checks (all GO, both grammars)

| Check | Manipulation (APPROACH vs APPROACH_SIDE) | Morphology (ADD_LIMB vs ADD_LIMB_REACH) |
|---|---|---|
| Containment holds post-training | True | True |
| vol(specific) < vol(general) | True | True |
| Geometric pick == symbolic paribhāṣā pick | **1.000** (188/188) | **1.000** (207/207) |

**Negative control (mandatory, run every time):** an *untrained* bank —
boxes initialized at each rule's own feature prototype, identical default
width — scores 0.000 agreement and fails both containment and volume checks
on both grammars. The jump from 0.000 → 1.000 is what training does; the
eval is not trivially satisfied by initialization.

## What's deliberately NOT here

- **No union.** Only containment (the generality relation) and intersection
  (demonstrated once, see below) are exercised. Box-union is not closed and
  isn't attempted.
- **No complement/negation.** `blocks_markers` (e.g. "NOT holding") stays
  symbolic. Box complement is non-convex and only heuristically
  approximable — forcing it into geometry here would repeat the kind of
  overreach this project has been explicitly avoiding (see the Langlands
  note in the main roadmap).
- **No learned encoder for states.** States are fixed one-hot features. The
  only thing gradient descent touches is box geometry — this isolates
  "can regions hold Pāṇini's ordering" from "can a network learn a good
  embedding," which is a different (and separately interesting) question.

## Compositional bonus: intersection as logical AND

`box(APPROACH_SIDE) ∩ box(ROTATE_TOOL)` is computed directly (both require
KARANA) and is non-empty — the geometric analogue of "find a state
satisfying both a tool-approach condition AND a tool-rotation condition,"
answered by box intersection instead of a symbolic query, mirroring the
KG multi-hop reasoning use case in the box-embedding literature.

## Files

```
regions/
  box_embed.py       BoxBank (learnable center+width per rule), containment
                      margin, point-containment, intersection, volume-floor
                      regularizer — all the geometric primitives
  rule_features.py   deterministic one-hot feature space shared by rule
                      condition-boxes and state-points, for both grammars
experiments/
  h_box_paribhasa.py supervision generation (instrumented derive() calls
                      logging which rule fired per candidate set), training,
                      the three evaluation checks, negative control,
                      intersection demo
tests/test_hbox.py    smoke test asserting GO + agreement > 0.95 + correct
                      volume ordering, at half the sample count (robustness)
```

## Honest limitations / next steps

- Only one apavāda/utsarga pair exists per grammar in the current rule
  catalogue (APPROACH/APPROACH_SIDE; ADD_LIMB/ADD_LIMB_REACH), so the
  volume-ordering claim currently has N=2 data points, not a distribution.
  Adding more apavāda rules (e.g. a fragile-object GRASP exception, a
  low-clearance TERMINATE variant) would let this become an actual rank
  correlation instead of a two-point comparison — natural next step before
  writing this up as its own section.
- The agreement metric is evaluated on the same rule catalogue used for
  training (no held-out *rule*, only held-out *states*). A stronger version
  trains on a subset of scenes/markers and evaluates generalization to novel
  marker combinations not seen during training.
- `ROTATE_TOOL`'s box in the intersection demo was trained only from its own
  firing states (not jointly optimized against APPROACH_SIDE for
  intersection quality), so the intersection number is illustrative, not
  yet a trained-and-validated compositional claim — that would need its own
  supervision (states requiring both conditions simultaneously).
