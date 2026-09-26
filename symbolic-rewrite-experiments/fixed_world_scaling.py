#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fixed_world_scaling.py

ONE clean fixed-world scaling experiment.

The hidden world (9 subjects / 3 subject families / 3 shared verbs / 12 objects /
3 object families / the Latin-square subject-family x verb -> object-family map /
the held-out subject+verb combinations) is generated ONCE per seed and is
IDENTICAL at every dataset size.  The only thing that changes with scale is the
NUMBER OF TRAINING EXAMPLES: more surface strings for the same fixed relations,
obtained by varying neutral filler material only.

Neither architecture is modified:
  * the symbolic learner is symbolic_rewrite_experiment_v4.Learner, unchanged,
    in its full_system configuration (V4's default Config);
  * the neural baseline is transformer_baseline_v4's SMALL causal Transformer,
    unchanged, from scratch, same optimization settings.

Both learners receive the EXACT SAME raw training strings.

Run:   python3 fixed_world_scaling.py       (needs the torch venv)
Saves: fixed_world_scaling_results.json
"""

import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import symbolic_rewrite_experiment_v4 as V4          # symbolic learner + eval
import transformer_baseline_v4 as TB                 # neural baseline + eval

EOS = V4.EOS
SEEDS = [1, 2, 3, 4, 5]
SCALES = [100, 300, 1000, 3000, 10000]
N_FILLERS = 150            # fixed filler pool, identical at every scale
N_VAL = 200                # fixed validation set, identical at every scale


# =============================================================================
# 1-3.  THE FROZEN WORLD AND THE NESTED DATASETS
# =============================================================================


class FixedWorld(object):
    """Exposes exactly the interface of V4.HiddenCorpus, so every V4 and
    Transformer evaluation function can be reused verbatim.

    The world is built once.  `train_at(N)` returns the first N strings of a
    single master ordering, so the training sets are strictly nested:
        train_100 subset train_300 subset ... subset train_10000
    """

    def __init__(self, seed, n_fillers=N_FILLERS, max_train=max(SCALES),
                 n_val=N_VAL):
        rng = random.Random(seed * 7919 + 13)
        self.seed = seed
        self._access_count = 0
        words = set()

        def fresh(family_words):
            for _ in range(200000):
                nsyl = rng.choice([2, 2, 3])
                coda = rng.random() < 0.4 and nsyl == 2
                w = V4._make_word(rng, nsyl, coda)
                if w in words:
                    continue
                if any(u.startswith(w) or w.startswith(u) for u in words):
                    continue
                if any(u[:2] == w[:2] or u[-2:] == w[-2:] for u in family_words):
                    continue
                words.add(w)
                return w
            raise RuntimeError("lexicon generation failed")

        def fam(n):
            f = []
            for _ in range(n):
                f.append(fresh(f))
            return f

        # ---- the FIXED hidden world -------------------------------------
        self.subject_families = [fam(3) for _ in range(3)]        # 9 subjects
        self.verbs = fam(3)                                       # 3 shared verbs
        self.object_families = [fam(4) for _ in range(3)]         # 12 objects
        self.objfam_table = [[(fi + vi) % 3 for vi in range(3)] for fi in range(3)]
        self.subj_family = {s: fi for fi, f in enumerate(self.subject_families)
                            for s in f}
        self.obj_family = {o: fi for fi, f in enumerate(self.object_families)
                           for o in f}
        self.verb_index = {v: i for i, v in enumerate(self.verbs)}

        # ---- neutral filler pool: fixed size, fixed distribution ---------
        # generated independently of subject family, verb and object family
        self.fillers = [fresh([]) for _ in range(n_fillers)]
        self.probe_filler = self.fillers[0]

        # ---- the FIXED held-out subject+verb combinations ----------------
        perms = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]]
        good = [p for p in perms
                if len(set(self.objfam_table[fi][p[fi]] for fi in range(3))) == 3]
        vperm = good[rng.randrange(len(good))]
        self.heldout_pairs = [(self.subject_families[fi][rng.randrange(3)],
                               self.verbs[vperm[fi]]) for fi in range(3)]
        held = set(self.heldout_pairs)

        self.heldout_compatible, self.heldout_incompatible = {}, {}
        for (s, v) in self.heldout_pairs:
            fi, vi = self.subj_family[s], self.verb_index[v]
            gf = self.objfam_table[fi][vi]
            pfx = s + self.probe_filler + v
            self.heldout_compatible[pfx] = set(
                pfx + o + EOS for o in self.object_families[gf])
            self.heldout_incompatible[pfx] = set(
                pfx + o + EOS for gi in range(3) if gi != gf
                for o in self.object_families[gi])
        self.heldout_prefixes = sorted(self.heldout_compatible)

        # EVERY surface string of a held-out pair, for EVERY filler, is banned
        self.heldout_strings = set()
        for (s, v) in self.heldout_pairs:
            fi, vi = self.subj_family[s], self.verb_index[v]
            for f in self.fillers:
                for o in self.object_families[self.objfam_table[fi][vi]]:
                    self.heldout_strings.add(s + f + v + o + EOS)

        # ---- one master ordering -> nested training sets ------------------
        triples = []
        for fi in range(3):
            for s in self.subject_families[fi]:
                for vi, v in enumerate(self.verbs):
                    if (s, v) in held:
                        continue
                    for o in self.object_families[self.objfam_table[fi][vi]]:
                        triples.append((s, v, o))
        self.n_triples = len(triples)
        pool = [s + f + v + o + EOS for (s, v, o) in triples for f in self.fillers]
        pool = sorted(set(pool))
        rng.shuffle(pool)
        assert len(pool) >= max_train + n_val, "filler pool too small"
        self._master = pool[:max_train]
        self.val_strings = sorted(pool[max_train:max_train + n_val])
        self.train_strings = list(self._master)      # default = the largest set

        for t in self._master + self.val_strings:
            assert t not in self.heldout_strings
            assert " " not in t and t.endswith(EOS) and t.count(EOS) == 1

    # ---- nested training sets ------------------------------------------
    def train_at(self, n):
        return list(self._master[:n])

    # ---- V4.HiddenCorpus interface --------------------------------------
    def touch(self):
        self._access_count += 1

    @property
    def access_count(self):
        return self._access_count

    def decompose(self, s):
        self.touch()
        if not s.endswith(EOS):
            return None
        body = s[:-1]
        for subj in self.subj_family:
            if not body.startswith(subj):
                continue
            r1 = body[len(subj):]
            for f in self.fillers:
                if not r1.startswith(f):
                    continue
                r2 = r1[len(f):]
                for v in self.verbs:
                    if r2.startswith(v):
                        o = r2[len(v):]
                        if o in self.obj_family:
                            return (subj, f, v, o)
        return None

    def is_grammatical(self, s):
        d = self.decompose(s)
        if d is None:
            return False
        subj, _f, v, o = d
        return self.obj_family[o] == \
            self.objfam_table[self.subj_family[subj]][self.verb_index[v]]

    def lexicon(self):
        self.touch()
        out = []
        for f in self.subject_families:
            out.extend(f)
        out.extend(self.verbs)
        out.extend(self.fillers)
        for f in self.object_families:
            out.extend(f)
        return out

    def world_signature(self):
        """Everything that must stay identical across scales."""
        return {
            "subject_families": [list(f) for f in self.subject_families],
            "verbs": list(self.verbs),
            "object_families": [list(f) for f in self.object_families],
            "objfam_table": [list(r) for r in self.objfam_table],
            "heldout_pairs": [list(p) for p in self.heldout_pairs],
            "heldout_prefixes": list(self.heldout_prefixes),
            "probe_filler": self.probe_filler,
            "n_fillers": len(self.fillers),
        }


class FixedDataset(object):
    """The Transformer's view of one (world, N).  Same attributes as
    transformer_baseline_v4.Dataset, so TB.run_one works unchanged."""

    def __init__(self, world, train_strings):
        self.seed = world.seed
        self._corpus = world
        self.train_strings = list(train_strings)
        self.val_strings = list(world.val_strings)
        self.heldout_prefixes = list(world.heldout_prefixes)
        self.heldout_compatible = {p: set(world.heldout_compatible[p])
                                   for p in world.heldout_prefixes}
        self.heldout_incompatible = {p: set(world.heldout_incompatible[p])
                                     for p in world.heldout_prefixes}
        self.object_words = [o for f in world.object_families for o in f]
        self.object_families = [list(f) for f in world.object_families]
        self.obj_family = dict(world.obj_family)
        self.subject_families = [list(f) for f in world.subject_families]
        self.subj_family = dict(world.subj_family)
        self.verbs = list(world.verbs)
        self.verb_index = dict(world.verb_index)
        self.objfam_table = [list(r) for r in world.objfam_table]
        self.probe_filler = world.probe_filler
        self.heldout_pairs = [tuple(p) for p in world.heldout_pairs]
        self.heldout_strings = set(world.heldout_strings)
        chars = sorted(set("".join(self.train_strings)))
        self.itos = [TB.PAD_TOK, TB.BOS_TOK] + chars
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        self.pad_id = self.stoi[TB.PAD_TOK]
        self.bos_id = self.stoi[TB.BOS_TOK]
        self.chars = chars
        self.max_len = 64
        assert self.max_len > 12 + max(len(s) for s in
                                       self.train_strings + self.val_strings)

    def encode(self, s):
        return [self.bos_id] + [self.stoi[c] for c in s]


# =============================================================================
# 5.  SYMBOLIC V4 RUNNER  (full_system only, learner unchanged)
# =============================================================================


def probe_summary(probe, world):
    groups = probe["behavioral_groups"]
    cmp_ = probe["behavioral_comparison"]
    fams = [sorted(f) for f in world.subject_families]
    recovered = len([g for g in groups if sorted(g) in fams])
    excl = {}
    for k in ["rules_applied", "generalized_rules_applied", "internal_rules_applied",
              "internal_symbols_in_views", "internal_symbols_active_at_verb"]:
        excl[k] = sum(len(row[k + "_exclusive_to_family"])
                      for row in probe["family_common_structures"])
    return {
        "groups": groups,
        "same_family_pairs_grouped": cmp_["same_family_pairs_grouped"],
        "same_family_pairs_total": cmp_["same_family_pairs_total"],
        "different_family_pairs_grouped": cmp_["different_family_pairs_grouped"],
        "different_family_pairs_total": cmp_["different_family_pairs_total"],
        "families_fully_recovered": recovered,
        "exact_partition_recovered": cmp_["groups_equal_hidden_families"],
        "family_exclusive_structures": excl,
    }


def run_symbolic(world, train_strings, seed):
    cfg = V4.Config()                     # full_system defaults, unchanged
    learner = V4.Learner(cfg, seed)
    problems = V4.cheat_controls(learner, world, train_strings,
                                 world.val_strings, "pre-train")
    t0 = time.time()
    learner.train(train_strings)
    learner.freeze()
    train_secs = time.time() - t0

    heldout = []
    for prefix in world.heldout_prefixes:
        gen = learner.possible_completions(prefix, cfg.max_completion_len)
        cls = V4.classify_completions(world, prefix, set(gen))
        met = V4.metrics_for(prefix, world, cls)
        heldout.append({"prefix": prefix, "metrics": met,
                        "n_generated": len(gen),
                        "compatible": cls["compatible"][:8]})
    problems += V4.cheat_controls(learner, world, train_strings,
                                  world.val_strings, "post-inference")
    probe = V4.subject_family_probe(learner, world, do_substitution=False)
    diag = V4.rule_diagnostics(learner)
    dyn = V4.learning_dynamics(learner)
    audit = V4.final_contradiction_audit(learner)

    mean = lambda k: sum(h["metrics"][k] for h in heldout) / float(len(heldout))
    st = learner.stats
    return {
        "train_seconds": train_secs,
        "precision": mean("precision"),
        "recall": mean("recall"),
        "exact_set_accuracy": mean("exact_set_accuracy"),
        "incompatible_generation_rate": mean("incompatible_generation_rate"),
        "malformed_generation_rate": mean("malformed_generation_rate"),
        "mean_completion_set_size": mean("generated"),
        "decisions": st["decisions"],
        "new_rule_search_fraction": (st["required_new_rule_search"] /
                                     float(max(1, st["decisions"]))),
        "unsolved": st["unsolved"],
        "reuse_events": st["reuse_events"],
        "generalization_events": st["generalization_events"],
        "generalization_rejected_counterexample":
            st["generalization_rejected_counterexample"],
        "active_rules": diag["active_rules"],
        "total_rules": diag["total_rules"],
        "internal_symbols": diag["internal_symbols"],
        "reused_rules": diag["reused_rules"],
        "new_rule_rate_first_decile": dyn[0]["new_rule_rate"] if dyn else None,
        "new_rule_rate_last_decile": dyn[-1]["new_rule_rate"] if dyn else None,
        "active_generalized_with_contradictions": len(audit),
        "probe": probe_summary(probe, world),
        "heldout": heldout,
        "cheat_control_problems": problems,
    }


# =============================================================================
# 6.  TRANSFORMER RUNNER  (SMALL only, architecture + settings unchanged)
# =============================================================================


def run_transformer(world, train_strings, seed):
    ds = FixedDataset(world, train_strings)
    cfg = dict(TB.CONFIGS)["SMALL"]
    t0 = time.time()
    r = TB.run_one(ds, "SMALL", cfg, seed, verbose=False)
    b = r["best_val"]
    p = r["subject_family_probe"]
    fams = [sorted(f) for f in world.subject_families]
    recovered = len([g for g in p["behavioral_groups"] if sorted(g) in fams])
    ceil = r["train_ceiling"]["accuracy_ceiling"]
    return {
        "train_seconds": time.time() - t0,
        "parameters": r["parameters"],
        "best_val_step": r["best_val_step"],
        "train_next_char_accuracy": b["train_next_char_accuracy"],
        "final_train_next_char_accuracy": r["final"]["train_next_char_accuracy"],
        "train_ceiling": ceil,
        "fit_fraction_of_ceiling": r["final"]["train_next_char_accuracy"] / ceil,
        "val_loss": b["val_loss"],
        "val_next_char_accuracy": b["val_next_char_accuracy"],
        "top4_exact_family_accuracy": b["top4_exact_family_accuracy"],
        "recall_at_4": b["recall_at_4"],
        "pairwise_family_accuracy": b["pairwise_family_accuracy"],
        "probability_mass_correct_family": b["probability_mass_correct_family"],
        "mean_rank_correct": b["mean_rank_correct"],
        "probe": {
            "groups": p["behavioral_groups"],
            "same_family_pairs_grouped": p["same_family_pairs_grouped"],
            "same_family_pairs_total": p["same_family_pairs_total"],
            "different_family_pairs_grouped": p["different_family_pairs_grouped"],
            "different_family_pairs_total": p["different_family_pairs_total"],
            "families_fully_recovered": recovered,
            "exact_partition_recovered": p["groups_equal_hidden_families"],
            "seen_combo_argmax_correct": p["seen_combo_argmax_correct"],
            "seen_combo_total": p["seen_combo_total"],
            "heldout_combo_argmax_correct": p["heldout_combo_argmax_correct"],
            "heldout_combo_total": p["heldout_combo_total"],
            "combination_control": p["combination_control"],
        },
        "free_generation_classes": [g["greedy_class"]
                                    for g in b["free_generation"]],
    }


# =============================================================================
# INTEGRITY CHECKS
# =============================================================================


def integrity_checks(worlds, datasets_by_seed):
    out = {"fixed_lexicon": True, "fixed_grammar": True, "fixed_heldout": True,
           "nested": True, "data_identity": True, "violations": []}
    for seed, world in worlds.items():
        sigs = []
        prev = None
        for N in SCALES:
            tr = datasets_by_seed[seed][N]
            sigs.append(world.world_signature())
            if prev is not None and not set(prev).issubset(set(tr)):
                out["nested"] = False
                out["violations"].append(
                    "seed %d: train_%d is not a superset of the smaller set" % (seed, N))
            prev = tr
            for s in tr:
                if s in world.heldout_strings:
                    out["violations"].append(
                        "seed %d N=%d: held-out string present" % (seed, N))
                    break
                for p in world.heldout_prefixes:
                    if s.startswith(p):
                        out["violations"].append(
                            "seed %d N=%d: held-out prefix present" % (seed, N))
                        break
                if any(c in s for c in " \t|_-"):
                    out["violations"].append(
                        "seed %d N=%d: boundary character in input" % (seed, N))
                    break
        base = sigs[0]
        for s in sigs[1:]:
            if s["subject_families"] != base["subject_families"] or \
               s["object_families"] != base["object_families"] or \
               s["verbs"] != base["verbs"]:
                out["fixed_lexicon"] = False
            if s["objfam_table"] != base["objfam_table"]:
                out["fixed_grammar"] = False
            if s["heldout_pairs"] != base["heldout_pairs"] or \
               s["heldout_prefixes"] != base["heldout_prefixes"]:
                out["fixed_heldout"] = False
        # validation set fixed and disjoint from the largest training set
        if set(world.val_strings) & set(datasets_by_seed[seed][SCALES[-1]]):
            out["violations"].append("seed %d: validation overlaps training" % seed)
    if out["violations"]:
        pass
    return out


# =============================================================================
# MAIN
# =============================================================================


def main():
    outpath = os.path.join(HERE, "fixed_world_scaling_results.json")
    # be a good citizen on the user's machine: cap the neural baseline's thread
    # use.  This affects wall-clock only, never the experiment.
    TB.torch.set_num_threads(4)
    print("=" * 78)
    print("FIXED-WORLD SCALING EXPERIMENT")
    print("=" * 78)
    print("seeds %s | scales %s | filler pool %d | validation %d (fixed)"
          % (SEEDS, SCALES, N_FILLERS, N_VAL))

    worlds, datasets = {}, {}
    for seed in SEEDS:
        w = FixedWorld(seed)
        worlds[seed] = w
        datasets[seed] = {N: w.train_at(N) for N in SCALES}
    integ = integrity_checks(worlds, datasets)
    w1 = worlds[SEEDS[0]]
    print("seed %d world: subjects %s" % (SEEDS[0], w1.subject_families))
    print("  verbs %s | objects %s" % (w1.verbs, w1.object_families))
    print("  latin square %s | held-out pairs %s"
          % (w1.objfam_table, w1.heldout_pairs))
    print("  distinct (subject,verb,object) relations in training: %d"
          % w1.n_triples)
    print("  example strings: %s" % datasets[SEEDS[0]][100][:3])
    print("")

    results = {"scales": SCALES, "seeds": SEEDS, "integrity": integ,
               "world_signatures": {str(s): worlds[s].world_signature()
                                    for s in SEEDS},
               "symbolic": {}, "transformer": {}, "data_identity": True}

    for N in SCALES:
        results["symbolic"][str(N)] = []
        results["transformer"][str(N)] = []
        for seed in SEEDS:
            world = worlds[seed]
            train = datasets[seed][N]

            # ---- fairness: both learners get the identical list -----------
            sym_train = list(train)
            tf_train = list(train)
            assert sym_train == tf_train, "training strings differ"

            t0 = time.time()
            sym = run_symbolic(world, sym_train, seed)
            tf = run_transformer(world, tf_train, seed)
            sym["seed"] = seed
            sym["n_train"] = len(sym_train)
            tf["seed"] = seed
            tf["n_train"] = len(tf_train)
            results["symbolic"][str(N)].append(sym)
            results["transformer"][str(N)].append(tf)
            print("  N=%-6d seed %d  [V4] P=%.2f R=%.2f exact=%.2f rules=%-5d "
                  "same=%d/%d diff=%d/%d | [TF] top4=%.2f R@4=%.2f pair=%.2f fam=%d "
                  "fit=%.3f  (%.0fs)"
                  % (N, seed, sym["precision"], sym["recall"],
                     sym["exact_set_accuracy"], sym["active_rules"],
                     sym["probe"]["same_family_pairs_grouped"],
                     sym["probe"]["same_family_pairs_total"],
                     sym["probe"]["different_family_pairs_grouped"],
                     sym["probe"]["different_family_pairs_total"],
                     tf["top4_exact_family_accuracy"], tf["recall_at_4"],
                     tf["pairwise_family_accuracy"],
                     tf["probe"]["families_fully_recovered"],
                     tf["fit_fraction_of_ceiling"], time.time() - t0))
            sys.stdout.flush()
            with open(outpath, "w") as f:
                json.dump(results, f, indent=1, ensure_ascii=False)
        print("")

    # ---------------------------------------------------------------- report
    avg = lambda rows, k: sum(r[k] for r in rows) / float(len(rows))
    pavg = lambda rows, k: sum(r["probe"][k] for r in rows) / float(len(rows))
    psum = lambda rows, k: sum(r["probe"][k] for r in rows)

    print("================ FIXED-WORLD SCALING RESULTS ================")
    print("")
    print("FIXED LEXICON ACROSS SCALES: %s" % ("PASS" if integ["fixed_lexicon"] else "FAIL"))
    print("FIXED GRAMMAR ACROSS SCALES: %s" % ("PASS" if integ["fixed_grammar"] else "FAIL"))
    print("FIXED HELD-OUT PAIRS ACROSS SCALES: %s" % ("PASS" if integ["fixed_heldout"] else "FAIL"))
    print("NESTED TRAINING SETS: %s" % ("PASS" if integ["nested"] else "FAIL"))
    print("SYMBOLIC/TRANSFORMER DATA IDENTITY: PASS")
    print("")
    print("SYMBOLIC V4:")
    for N in SCALES:
        rows = results["symbolic"][str(N)]
        print("N=%d:" % N)
        print("  precision                      %.4f" % avg(rows, "precision"))
        print("  recall                         %.4f" % avg(rows, "recall"))
        print("  exact_set_accuracy             %.4f" % avg(rows, "exact_set_accuracy"))
        print("  incompatible_generation_rate   %.4f" % avg(rows, "incompatible_generation_rate"))
        print("  malformed_generation_rate      %.4f" % avg(rows, "malformed_generation_rate"))
        print("  new-rule-search fraction       %.4f  (first decile %.3f, last %.3f)"
              % (avg(rows, "new_rule_search_fraction"),
                 avg(rows, "new_rule_rate_first_decile"),
                 avg(rows, "new_rule_rate_last_decile")))
        print("  reuse events                   %.0f" % avg(rows, "reuse_events"))
        print("  active rules                   %.0f   internal symbols %.0f"
              % (avg(rows, "active_rules"), avg(rows, "internal_symbols")))
        print("  family-exclusive rules/symbols  %s"
              % {k: sum(r["probe"]["family_exclusive_structures"][k] for r in rows)
                 for k in rows[0]["probe"]["family_exclusive_structures"]})
    print("")
    print("TRANSFORMER SMALL:")
    for N in SCALES:
        rows = results["transformer"][str(N)]
        print("N=%d:" % N)
        print("  train fit / empirical ceiling  %.4f  (acc %.4f, ceiling %.4f)"
              % (avg(rows, "fit_fraction_of_ceiling"),
                 avg(rows, "final_train_next_char_accuracy"),
                 avg(rows, "train_ceiling")))
        print("  validation loss                %.4f" % avg(rows, "val_loss"))
        print("  top-4 exact-family accuracy    %.4f" % avg(rows, "top4_exact_family_accuracy"))
        print("  Recall@4                       %.4f   [chance 0.333]" % avg(rows, "recall_at_4"))
        print("  pairwise family accuracy       %.4f   [chance 0.500]" % avg(rows, "pairwise_family_accuracy"))
        print("  probability mass correct fam.  %.4f   [chance 0.333]" % avg(rows, "probability_mass_correct_family"))
        print("  mean rank of correct objects   %.2f    [chance 6.50]" % avg(rows, "mean_rank_correct"))
    print("")
    print("SUBJECT FAMILY PROBE BY SCALE:")
    print("  %-7s | %-34s | %-34s" % ("N", "SYMBOLIC V4", "TRANSFORMER SMALL"))
    print("  %-7s | %-34s | %-34s" % ("", "same%  diff%  fullfam  exact",
                                      "same%  diff%  fullfam  exact"))
    for N in SCALES:
        sr = results["symbolic"][str(N)]
        tr = results["transformer"][str(N)]
        f = lambda rows: "%.3f  %.3f  %4.1f     %d/%d" % (
            psum(rows, "same_family_pairs_grouped") /
            float(psum(rows, "same_family_pairs_total")),
            psum(rows, "different_family_pairs_grouped") /
            float(psum(rows, "different_family_pairs_total")),
            pavg(rows, "families_fully_recovered"),
            sum(1 for r in rows if r["probe"]["exact_partition_recovered"]),
            len(rows))
        print("  %-7d | %-34s | %-34s" % (N, f(sr), f(tr)))
    print("")
    print("HELD-OUT GENERALIZATION BY SCALE:")
    print("  %-7s | %-28s | %-40s" % ("N", "SYMBOLIC V4", "TRANSFORMER SMALL"))
    print("  %-7s | %-28s | %-40s" % ("", "prec   recall  exact", 
                                      "top4    R@4     pairwise  mass"))
    for N in SCALES:
        sr = results["symbolic"][str(N)]
        tr = results["transformer"][str(N)]
        print("  %-7d | %.3f  %.3f   %.3f     | %.3f   %.3f   %.3f     %.3f"
              % (N, avg(sr, "precision"), avg(sr, "recall"),
                 avg(sr, "exact_set_accuracy"),
                 avg(tr, "top4_exact_family_accuracy"), avg(tr, "recall_at_4"),
                 avg(tr, "pairwise_family_accuracy"),
                 avg(tr, "probability_mass_correct_family")))
    print("")
    print("TRANSFORMER seen vs held-out (subject,verb) combinations by scale:")
    for N in SCALES:
        tr = results["transformer"][str(N)]
        c = lambda k: sum(r["probe"]["combination_control"][k] for r in tr) / float(len(tr))
        print("  N=%-6d seen: R@4 %.3f pair %.3f mass %.3f | held-out: R@4 %.3f "
              "pair %.3f mass %.3f"
              % (N, c("seen_recall_at_4"), c("seen_pairwise_family_accuracy"),
                 c("seen_probability_mass_correct_family"),
                 c("heldout_recall_at_4"), c("heldout_pairwise_family_accuracy"),
                 c("heldout_probability_mass_correct_family")))
    print("")

    # ------------------------------------------------------- interpretation
    def trend(rows_by_N, key, getter=avg):
        return [getter(rows_by_N[str(N)], key) for N in SCALES]
    sym_recall = trend(results["symbolic"], "recall")
    sym_exact = trend(results["symbolic"], "exact_set_accuracy")
    tf_pair = trend(results["transformer"], "pairwise_family_accuracy")
    tf_r4 = trend(results["transformer"], "recall_at_4")
    sym_same = [psum(results["symbolic"][str(N)], "same_family_pairs_grouped") /
                float(psum(results["symbolic"][str(N)], "same_family_pairs_total"))
                for N in SCALES]
    tf_same = [psum(results["transformer"][str(N)], "same_family_pairs_grouped") /
               float(psum(results["transformer"][str(N)], "same_family_pairs_total"))
               for N in SCALES]
    tf_fam = [pavg(results["transformer"][str(N)], "families_fully_recovered")
              for N in SCALES]
    sym_fam = [pavg(results["symbolic"][str(N)], "families_fully_recovered")
               for N in SCALES]

    sym_gen = sym_exact[-1] > 0.25 or (sym_recall[-1] - sym_recall[0]) > 0.25
    tf_gen = (tf_pair[-1] > 0.65) or (tf_r4[-1] > 0.55)
    sym_fam_up = sym_fam[-1] >= sym_fam[0] + 1.0 or sym_same[-1] >= sym_same[0] + 0.25
    tf_fam_up = tf_fam[-1] >= tf_fam[0] + 1.0 or tf_same[-1] >= tf_same[0] + 0.25

    print("INTERPRETATION:")
    print("  symbolic recall by N:            %s" % ["%.3f" % x for x in sym_recall])
    print("  symbolic exact-set by N:         %s" % ["%.3f" % x for x in sym_exact])
    print("  symbolic same-family grouping:   %s" % ["%.3f" % x for x in sym_same])
    print("  transformer pairwise by N:       %s" % ["%.3f" % x for x in tf_pair])
    print("  transformer Recall@4 by N:       %s" % ["%.3f" % x for x in tf_r4])
    print("  transformer same-family grouping:%s" % ["%.3f" % x for x in tf_same])
    print("  transformer full families rec.:  %s" % ["%.1f" % x for x in tf_fam])
    if sym_gen and tf_gen:
        verdict = "D. Both improve with scale."
    elif tf_gen and not sym_gen:
        verdict = ("B. Transformer begins recovering generalization with scale, "
                   "symbolic V4 does not.")
    elif sym_gen and not tf_gen:
        verdict = ("C. Symbolic V4 begins recovering generalization with scale, "
                   "Transformer does not.")
    elif sym_fam_up or tf_fam_up:
        verdict = ("E. Subject families are recovered internally (at least "
                   "partly, and increasingly with scale) but the held-out "
                   "subject+verb relation still fails.")
    else:
        verdict = "A. Both models remain flat at chance even at N=%d." % SCALES[-1]
    print("  %s" % verdict)
    print("")
    viol = integ["violations"]
    for N in SCALES:
        for r in results["symbolic"][str(N)]:
            viol += r["cheat_control_problems"]
    print("CHEAT-CONTROL VIOLATIONS: %d" % len(viol))
    for v in viol[:10]:
        print("   !! %s" % v)
    print("  checks: fixed world across scales; nested training sets; held-out")
    print("  pairs absent at every scale for every filler; validation fixed and")
    print("  disjoint from training; identical raw strings for both learners;")
    print("  no boundaries; V4 archive locked at inference; no rules after freeze.")
    print("=============================================================")
    with open(outpath, "w") as f:
        json.dump(results, f, indent=1, ensure_ascii=False)
    print("files written: fixed_world_scaling_results.json")


if __name__ == "__main__":
    main()
