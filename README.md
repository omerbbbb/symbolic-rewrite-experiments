# Symbolic Rewrite Experiments

Can a learning system that uses **no numbers at all** discover compositional
structure in language — and does a conventional neural network do any better on
exactly the same data?

This repository contains a series of controlled experiments answering that
question. Every result here is reproducible from a fixed seed with the Python
standard library (the neural baseline additionally needs PyTorch).

**Short answer: no, and no.** Both approaches fail the compositional test, in
different and informative ways. The experiments are designed so that a negative
result is a real result — most of the code is the machinery that makes failure
trustworthy rather than an artifact.

---

## 1. The question

A hidden artificial language is generated with a deliberately compositional
rule. Sentences look like this, with **no spaces and no word boundaries**:

```
bufopaforotibuburor#
```

Behind the scenes that is `bufo` + `pafo` + `rotibu` + `buror`
(subject + filler + verb + object), but the learner never sees the split.

The hidden grammar is a **Latin square**: which family of objects is allowed
depends on the *combination* of the subject's hidden family and the verb —
neither alone is sufficient.

| | verb 1 | verb 2 | verb 3 |
|---|---|---|---|
| subject family A | objects X | objects Y | objects Z |
| subject family B | objects Y | objects Z | objects X |
| subject family C | objects Z | objects X | objects Y |

Then specific **subject+verb combinations are held out entirely** — they never
appear in training with any object. To answer correctly on a held-out pair the
learner must combine what it learned about that subject from *other* verbs with
what it learned about that verb from *other* subjects. That is compositional
generalization, and it is the only thing being measured.

## 2. The two learners

### The symbolic learner (`symbolic_rewrite_experiment_v4.py`)

No tokenizer, no embeddings, no neural network, no gradients, no probabilities,
no similarity scores, no learned numbers of any kind. Its entire state is:

- a 2D grid of discrete symbol cells, including one reserved `OUT` register,
- symbolic previous/next links recording arrival order,
- a set of discrete rewrite rules,
- a provenance table making every internal symbol explainable back to raw characters.

It learns by **target-blind random search**: it proposes a random rewrite rule
without being told which character is required, executes it, reads whatever
lands in `OUT`, and only then compares that to the target. Success stores the
rule; failure discards it and leaves no trace. If the search budget runs out the
training step is left `UNSOLVED` rather than patched.

Rules generalize only after proving reusable, and a proposed generalization is
vetoed if it contradicts any actually observed training decision.

### The neural baseline (`transformer_baseline_v4.py`)

An ordinary decoder-only Transformer trained from scratch on the identical raw
character strings: learned character embeddings, causal self-attention,
softmax, cross-entropy, AdamW. No pretraining, no external tokenizer. Three
sizes (29K / 107K / 807K parameters).

## 3. What was found

**Both learners fit the training data and neither generalizes compositionally.**

| | symbolic V4 | Transformer (best size) |
|---|---|---|
| held-out precision | 0.291 *(chance 0.333)* | — |
| held-out exact-set accuracy | 0.000 | — |
| held-out Recall@4 | — | 0.292 *(chance 0.333)* |
| held-out pairwise accuracy | — | 0.498 *(chance 0.500)* |

The controls are what make this meaningful:

- **The Transformer genuinely fits the corpus.** Character-level accuracy cannot
  reach 1.0 here because the corpus is ambiguous, so the *empirical ceiling* is
  computed (0.898); the model reaches 0.896 — 99.9% of the achievable maximum.
- **The measurement works.** On subject+verb combinations that *were* in
  training, the same ranking procedure gives pairwise accuracy 0.70 and 0.76
  probability mass on the right family (rising to 1.00 and 0.99 in the
  fixed-world experiment, where each relation is seen many times). It collapses
  to chance only on unseen combinations.
- **Scale does not help.** In a fixed-world experiment where the hidden language
  is frozen and only the number of training examples grows 100× (nested
  datasets, 100 → 10,000), the symbolic learner's precision stays flat at
  0.297 → 0.318 and exact-set accuracy is 0.000 in all 25 runs.

### The one real difference between them

A post-hoc probe asks whether each learner internally represents the hidden
subject families, without ever giving it the labels.

- The **symbolic learner recovers nothing**: zero family-exclusive rules or
  internal symbols at every scale. Its apparent "grouping" is degenerate — the
  same-family and different-family grouping rates rise together (0.778 vs
  0.763), meaning it merges every subject into one blob.
- The **Transformer builds a clean internal representation**: false grouping of
  different families drops to **0.000** while correct grouping rises to 0.467,
  and it recovers whole hidden families (0 → 0.6 per seed as data grows). Its
  per-subject behaviour is nearly one-hot and often matches the true Latin-square
  row exactly.

So the Transformer learns *what the families are* and still cannot *use* that
knowledge on an unseen subject+verb pair.

### V5: does adding memory traces help?

`symbolic_rewrite_v5.py` tests one idea: when a rule succeeds, it leaves a
symbolic **residue** in the workspace. Residues become permanent after the rule
succeeds in K distinct contexts, can be required by later rules, and can be
rewritten into higher-order structures — all discovered by the same target-blind
process, never by comparing words or merging states.

The mechanism operates (294 persistent residues, 64 rules conditioned on them,
higher-order combinations discovered). It does not produce convergence:
same-family residue overlap 0.0088 vs different-family 0.0059. An orthographic
control shows why that 1.5× ratio is an illusion:

```
residue overlap, pairs with HIGH character overlap : 0.0184
residue overlap, pairs with LOW  character overlap : 0.0000
```

Residues are shared **only** between words that happen to look alike. Since
residues record which rules fired, and which rules fire depends on surface
characters, family members with unrelated spellings leave disjoint traces.

## 4. Honest-measurement machinery

A large part of this code exists to make the negative result trustworthy:

- **Target blindness is proven, not asserted.** The proposal function's
  signature is inspected, its source is scanned for any mention of the target,
  and it is called twice with the RNG state restored under two different
  hypothetical targets — the proposed rule must be byte-identical.
- **Cheat controls** assert on every run: held-out strings absent from training,
  no whitespace or boundary characters in any input, the learner's vocabulary
  contains only observed characters, no rule created after freezing, the
  training archive locked and unreachable at inference, and the learner's source
  scanned for any reference to hidden generator metadata.
- **Chance baselines are stated next to every metric**, and controls separate
  "cannot fit" from "fit but cannot generalize".
- **Ablations** (10 seeds each): removing reuse or shuffling character order
  collapses the symbolic learner to zero, confirming it depends on both.

## 5. Repository layout

```
symbolic_rewrite_experiment_v3.py   earlier version, kept for reference
symbolic_rewrite_experiment_v4.py   the symbolic learner + ablations + probe
symbolic_rewrite_v5.py              V5: the residue mechanism
transformer_baseline_v4.py          from-scratch Transformer baseline
fixed_world_scaling.py              fixed-world scaling experiment (both learners)
report_fixed_world.py               prints the scaling report from saved results
results/                            all saved outputs from the runs above
```

## 6. Running it

The symbolic experiments need only Python 3 (standard library):

```bash
python3 symbolic_rewrite_experiment_v4.py     # ~5 min, 6 ablations x 10 seeds
python3 symbolic_rewrite_v5.py                # ~5 min
```

The Transformer baseline needs PyTorch:

```bash
pip install torch
python3 transformer_baseline_v4.py            # ~1 hour on CPU
```

The fixed-world scaling experiment runs both learners on identical data:

```bash
python3 fixed_world_scaling.py                # several hours; see note below
python3 report_fixed_world.py                 # prints the report from saved results
```

Each script writes its results as JSON next to itself; the committed copies of
those outputs are in `results/`.

**A note on runtime.** The symbolic learner's search cost is not uniform across
random seeds. In the scaling experiment most runs at the
largest scale took ~20 minutes while one particular seed took 4.5 hours — a 10×
blow-up caused by the rule system saturating, not by dataset size (that same
seed cost 4.2 hours on a dataset 3.3× smaller). This is a property of the architecture and is
reported rather than hidden.

## 7. Interpretation

The headline is not "symbolic bad, neural good". Both fail the compositional
test at chance level. What differs is *where* they fail:

- The symbolic learner never forms any representation of the latent categories.
  It memorizes surface rules, reuses them heavily (400,000+ reuse events) and
  licenses nearly every object — high recall, chance precision.
- The Transformer forms a clean latent representation of exactly the right
  categories and masters every combination it has seen, then fails to compose
  that knowledge across an unseen pairing.

A plausible reading is that the hard part is not *discovering* the latent
structure but *composing* two independently learned factors at inference —
and that neither mechanism here does the second thing.

---

*Licensed under the MIT License (see `LICENSE`).*
