# Roadmap — Grammar-Driven Robot Morphology Optimization

**Final aim:** AI generates/optimizes MuJoCo robot *morphologies* fit to a
given task, using a Pāṇinian body grammar to propose bodies, a Śulba-style
geometric layer to enforce feasibility, and a JEPA (or LLM) evaluator to
score and refine them.

This is a redirection, not a restart. You reuse: JEPA-ProPS (the fitness
optimizer), panini-motion (the grammar+SE(3) machinery, re-pointed from
trajectories to bodies), and the CoRL MuJoCo-XML validator (the SP1–SP5
feasibility gate).

---

## The single clarifying reframe

panini-motion currently generates **motion**: kāraka roles bind to task
objects, Φ emits end-effector twists. For the morphology goal you re-point
the exact same apparatus one level down:

| panini-motion (now)            | panini-morphology (target)                    |
|--------------------------------|-----------------------------------------------|
| kāraka roles bind task objects | roles bind **body parts** (torso, limb, joint, actuator, foot) |
| sūtra = an action step         | sūtra = a **body-construction rule**          |
| anuvṛtti inherits motion params| anuvṛtti inherits a **limb template** across repeated limbs |
| paribhāṣā resolves action conflicts | paribhāṣā resolves **body-rule conflicts** (general vs task-specific limb) |
| Φ: rule → SE(3) twist          | Φ: rule → **link geometry + joint frame** (SE(3) placement of a body element) |
| H5 homomorphism defect         | still holds: composing body-rules ↦ composing SE(3) placements of parts |

The homomorphism story survives verbatim: a body is a kinematic tree, each
part placed by an SE(3) transform relative to its parent, and rule
composition ↦ transform composition. Your H5 result already proves Φ is a
clean monoid homomorphism off-inheritance — that becomes the *correctness
guarantee that the generated XML is a valid kinematic tree*.

---

## Source-to-contribution mapping (be disciplined)

- **Aṣṭādhyāyī (load-bearing):** the generative body grammar. anuvṛtti =
  don't respecify each of 6 hexapod legs, inherit a leg-schema and vary the
  delta. it-tags = a joint can't attach until its parent link is declared.
  paribhāṣā = task-specific limb rule overrides the generic one.
- **Kātyāyana Śulba Sūtra (the novel identity):** constraint-satisfying
  *construction* under geometric budgets — reach envelope, ground clearance,
  mass, symmetry, self-collision. The Śulba transforms (square↔circle of
  equal area, combining/differencing areas) are the historical precedent for
  "construct a shape meeting a geometric spec," which is exactly morphology
  feasibility. **Build the paper's identity here — nobody in morphology ML
  has this framing.**
- **Ramanujan (one section, no more):** highly-composite-number / partition
  counting to *bound the cardinality* of the body-plan design space at
  derivation depth d. A rigorous complexity characterization. Do NOT make it
  structural — that would repeat the "Langlands" overreach.

---

## Phase M0 — Body Grammar Formalization (Weeks 1–3, no training)

**Goal:** specify the morphology grammar and the body-Φ before any search.

1. **Body kāraka schema.** Redefine roles for *anatomy*:
   `aṅga` (limb/appendage), `sandhi` (joint), `daṇḍa` (link/segment),
   `mūla` (root/torso), `pāda` (foot/contact), `preraka` (actuator).
   Keep the enum in `ptg/karaka.py`; add a `BodyEntity` with link geometry
   (length, radius, mass) alongside the SE(3) pose.
2. **Body sūtras.** Rules that *emit body elements*:
   - `ADD_LIMB` (attach an aṅga to mūla at an inherited mounting frame),
   - `EXTEND` (append a daṇḍa to the current limb — anuvṛtti carries segment
     length/taper),
   - `ARTICULATE` (insert a sandhi with a joint type + axis),
   - `TERMINATE` (cap with a pāda contact geom),
   - `SYMMETRIZE` (vibhāṣā: mirror the current limb — optional, multimodal).
   it-tags gate legality: `ARTICULATE` requires a `has_link` marker;
   `TERMINATE` requires `open_chain`.
3. **Body-Φ (`gmm/phi_body.py`).** Each applied rule → the SE(3) placement of
   its part relative to the parent, plus the part's geometric attributes.
   The composed transform chain *is* the kinematic tree. Reuse `gmm/se3.py`
   unchanged.
4. **Deliverable:** BNF-style body-grammar spec + a hand-worked proof that Φ
   preserves composition for a 2-part chain (torso→limb→segment). You've
   already got the H5 harness that measures the defect; re-run it on body
   pairs.

**Reuses:** all of `ptg/`, `gmm/se3.py`, the H5 experiment.

---

## Phase M1 — Grammar → MuJoCo XML Compiler (Weeks 3–6)

**Goal:** turn a derivation into a runnable MuJoCo model, gated by feasibility.

1. **XML emitter.** Walk the body-Φ kinematic tree → `<body>/<geom>/<joint>`
   MuJoCo XML. This is where your existing **MuJoCo-XML grammar and V-JEPA2
   auto-patcher** from the morphology work plug in directly — repurpose the
   patcher to repair malformed generated bodies.
2. **Śulba feasibility layer (`baselines/`→`sulba/feasibility.py`).** Before
   compiling, run geometric constraint checks framed as Śulba constructions:
   - reach envelope ⊇ required workspace (area/volume construction),
   - ground clearance ≥ threshold (a "combining areas" style budget),
   - total mass ≤ payload budget,
   - bilateral symmetry where the task template demands it,
   - no self-intersection at rest (the SP1–SP5 validator from CoRL).
   A body that fails is rejected *before* any expensive rollout — this is the
   sample-efficiency win.
3. **Sanity target:** procedurally generate 200 bodies, compile, and confirm
   MuJoCo loads them and they stand/settle without exploding. Report
   first-pass compile rate and post-patch recovery rate (mirrors your CoRL
   58.7% / 84% numbers).

**Reuses:** MuJoCo-XML grammar, V-JEPA2 patcher, SP1–SP5 validator.

---

## Phase M2 — Fitness Evaluation via JEPA (Weeks 6–10)

**Goal:** score each generated body on the task, cheaply.

The key insight connecting your two papers: **JEPA-ProPS optimizes a policy
for a *fixed* body. Morphology search needs a policy score for *each candidate*
body.** Two evaluation modes, cheap→expensive:

1. **Fast surrogate (primary novelty).** Train a JEPA that predicts a body's
   achievable return *from its structure* — tokenize the derivation (you
   already have `tokenize_sequence`) + a few cheap rollout statistics, predict
   task fitness. This is a *morphology-fitness world model*: it lets the
   search loop reject bad bodies without full policy training. This is the
   direct descendant of "JEPA as optimizer," now optimizing over bodies not
   policies.
2. **Ground-truth eval.** For bodies that pass the surrogate, run JEPA-ProPS
   (Swarm-JEPA) to actually train a control policy and get a real return.
   Reuse the checkpointed pipeline as-is.
3. **Calibration experiment:** surrogate-predicted fitness vs. true
   Swarm-JEPA return across N bodies — report rank correlation. High
   correlation = the surrogate is a valid cheap filter (the paper's efficiency
   claim).

**Reuses:** JEPA-ProPS end-to-end, `phi_neural.tokenize_sequence`.

---

## Phase M3 — The Outer Search Loop (Weeks 10–13)

**Goal:** close the loop — propose bodies, evaluate, refine.

Three proposer variants (this is your ablation axis and the honest baseline
set):

1. **Grammar + evolutionary search.** Mutate/recombine *derivations* (edit a
   sūtra, swap a limb template, toggle a vibhāṣā branch). Anuvṛtti makes
   mutations compact and legal-by-construction — a mutation is a rule edit,
   not a raw XML edit, so most mutations stay valid. Compare against raw-XML
   mutation (the RoboMorph/LASeR-style LLM baseline) on validity rate.
2. **LLM proposer.** Few-shot Claude/Qwen proposes derivations in your JSON
   plan format (drops into the existing `PTG.derive` interface). Compares
   "LLM writes grammar derivations" vs. "LLM writes raw XML" (your CoRL
   result says raw XML hallucinates predictably — grammar should fix that).
3. **JEPA-guided search.** Use the Phase-M2 surrogate as the acquisition
   function: the fitness world-model proposes which grammar edit to try next,
   the same way JEPA-ProPS proposes the next policy vector. **This is the
   cleanest through-line to your JEPA paper and the strongest headline.**

**Reuses:** Swarm-JEPA search structure, LLM parser interface.

---

## Phase M4 — Experiments (Weeks 13–18)

| Hyp | Claim | Design | Metric |
|-----|-------|--------|--------|
| M-H1 | Grammar bodies are valid-by-construction | grammar-mutation vs raw-XML-mutation | MuJoCo compile + feasibility pass rate |
| M-H2 | Task-fit morphologies emerge | run full loop on locomotion + reach + payload tasks | best task return vs RoboMorph/LASeR baseline |
| M-H3 | Anuvṛtti gives compact, transferable bodies | ablate anuvṛtti (per-limb respecify) | rules-per-body + cross-task transfer |
| M-H4 | JEPA surrogate is a valid cheap filter | surrogate fitness vs true Swarm-JEPA return | rank correlation; rollouts saved |
| M-H5 | Body-Φ is a homomorphism (correctness) | H5 defect on body-rule pairs | geodesic defect, ≡0 off-inheritance |
| M-H6 | Śulba layer improves sample efficiency | with/without feasibility gate | rollouts to first task-solving body |
| M-H7 | Design space cardinality bound is tight | Ramanujan-style count vs enumerated bodies at depth d | predicted vs actual #distinct bodies |

Run **M-H5 first** (you already have the harness) as the go/no-go: if body-Φ
isn't a clean homomorphism, the "generated XML is a guaranteed-valid
kinematic tree" claim needs revision before investing in the loop.

---

## Phase M5 — Analysis & Paper (Weeks 18–22)

- **Directional-decodability probe** (reuse your causal-compression / RPLM
  methodology): are body-part roles more recoverable from the JEPA fitness
  latent than reconstructible? Gives a mechanistic story for *why* the
  surrogate works — the analogy to your protein-representation work.
- **Ramanujan section:** partition/highly-composite counting bounds the
  number of distinct depth-d bodies; validate against enumeration on a small
  grammar. One rigorous section, clearly scoped.
- **Positioning:** lead with the Śulba feasibility framing (novel) +
  JEPA-guided morphology search (through-line to your prior work). Frame
  Aṣṭādhyāyī as the generative backbone, not the headline.

---

## What you can start today (concrete first commits)

1. Add `BodyEntity` + body-kāraka roles to `ptg/karaka.py`.
2. Write 5 body sūtras in `ptg/sutra.py` (ADD_LIMB, EXTEND, ARTICULATE,
   TERMINATE, SYMMETRIZE) — mirror the existing seed_rules structure.
3. Write `gmm/phi_body.py` (part → SE(3) placement) — copy phi_seed.py's
   shape, swap twist emission for parent-relative part placement.
4. Re-run the H5 harness on body-rule pairs (M-H5 go/no-go) — this validates
   the whole approach in an afternoon, on Sol, with code you already have.
5. Only then build the XML emitter (Phase M1).

The ordering is deliberate: steps 1–4 reuse ~90% of panini-motion and give
you the go/no-go result before you commit to the MuJoCo compiler.
