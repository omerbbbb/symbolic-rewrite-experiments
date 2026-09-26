#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
transformer_baseline_v4.py

A conventional causal Transformer baseline, trained FROM SCRATCH on exactly the
datasets produced by symbolic_rewrite_experiment_v4.py (V4).

V4 is NOT modified and NOT re-run: this file imports V4's HiddenCorpus and uses
it as the single source of the data, so every seed gives byte-for-byte the same
training strings, validation strings, held-out subject+verb combinations and
hidden compatible object sets that V4 received.

The learner here is deliberately ordinary: learned character embeddings, learned
positional embeddings, causal self-attention, feed-forward layers, softmax,
cross-entropy, AdamW.  No pretraining, no pretrained embeddings, no external
tokenizer, no external corpus, no transfer learning, no hidden labels, no word
boundaries.

Run:   python3 transformer_baseline_v4.py
Saves: transformer_baseline_results.json
"""

import copy
import inspect
import json
import math
import os
import random
import statistics
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import symbolic_rewrite_experiment_v4 as V4      # DATASET ONLY

EOS = V4.EOS
SEEDS = list(range(1, 11))                       # the same 10 seeds as V4

# special tokens: standard LM machinery, not boundary information.  BOS marks
# the start of the whole string (V4 uses one too); PAD is only ever masked out.
PAD_TOK = "<pad>"
BOS_TOK = "<bos>"


# =============================================================================
# 1.  THE EXACT V4 DATASET
# =============================================================================


class Dataset(object):
    """Everything the Transformer is allowed to see, plus (kept separate) the
    hidden material that only the evaluation harness may touch."""

    def __init__(self, seed):
        self.seed = seed
        corpus = V4.HiddenCorpus(seed)           # exactly V4's generator
        self._corpus = corpus

        # ---- visible to training ----------------------------------------
        self.train_strings = list(corpus.train_strings)
        self.val_strings = list(corpus.val_strings)

        # ---- evaluation-only (never passed to train_model) ---------------
        self.heldout_prefixes = list(corpus.heldout_prefixes)
        self.heldout_compatible = {p: set(corpus.heldout_compatible[p])
                                   for p in corpus.heldout_prefixes}
        self.heldout_incompatible = {p: set(corpus.heldout_incompatible[p])
                                     for p in corpus.heldout_prefixes}
        self.object_words = []
        for fam in corpus.object_families:
            self.object_words.extend(fam)
        self.object_families = [list(f) for f in corpus.object_families]
        self.obj_family = dict(corpus.obj_family)
        self.subject_families = [list(f) for f in corpus.subject_families]
        self.subj_family = dict(corpus.subj_family)
        self.verbs = list(corpus.verbs)
        self.verb_index = dict(corpus.verb_index)
        self.objfam_table = [list(r) for r in corpus.objfam_table]
        self.probe_filler = corpus.probe_filler
        self.heldout_pairs = [tuple(p) for p in corpus.heldout_pairs]
        self.heldout_strings = set(corpus.heldout_strings)

        # ---- vocabulary: ONLY from the training corpus -------------------
        chars = sorted(set("".join(self.train_strings)))
        self.itos = [PAD_TOK, BOS_TOK] + chars
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        self.pad_id = self.stoi[PAD_TOK]
        self.bos_id = self.stoi[BOS_TOK]
        self.chars = chars
        # fixed positional budget: longer than any string the model can ever
        # be given (longest sentence + free-generation slack).  A buffer size,
        # not information: the model never sees a longer sequence.
        self.max_len = 64
        assert self.max_len > 12 + max(len(s) for s in
                                       self.train_strings + self.val_strings)

    def encode(self, s):
        return [self.bos_id] + [self.stoi[c] for c in s]


def dataset_identity_checks(ds, results_v4_path):
    """Assert the data is byte-for-byte what V4 used."""
    problems = []
    # (a) the generator is literally V4's class
    if type(ds._corpus).__module__ != "symbolic_rewrite_experiment_v4":
        problems.append("corpus does not come from symbolic_rewrite_experiment_v4")
    # (b) reconstructing it gives identical strings
    again = V4.HiddenCorpus(ds.seed)
    if list(again.train_strings) != ds.train_strings:
        problems.append("seed %d: train strings not reproducible" % ds.seed)
    if list(again.val_strings) != ds.val_strings:
        problems.append("seed %d: val strings not reproducible" % ds.seed)
    # (c) for the seed stored in results_v4.json, compare byte-for-byte with the
    #     dataset V4 actually ran on
    if os.path.exists(results_v4_path):
        blob = json.load(open(results_v4_path))
        for key, d in blob.get("dataset_examples", {}).items():
            if d["seed"] != ds.seed:
                continue
            if d["train_strings"] != ds.train_strings:
                problems.append("seed %d: train strings differ from results_v4.json"
                                % ds.seed)
            if d["val_strings"] != ds.val_strings:
                problems.append("seed %d: val strings differ from results_v4.json"
                                % ds.seed)
            if d["heldout_prefixes"] != ds.heldout_prefixes:
                problems.append("seed %d: held-out prefixes differ" % ds.seed)
            mine = {p: sorted(ds.heldout_compatible[p]) for p in ds.heldout_prefixes}
            if d["heldout_compatible_strings"] != mine:
                problems.append("seed %d: compatible sets differ" % ds.seed)
    return problems


def cheat_controls(ds, model_fn_names):
    """Section 12 assertions."""
    problems = []
    # held-out strings absent from train and validation
    for s in ds.train_strings + ds.val_strings:
        if s in ds.heldout_strings:
            problems.append("held-out string in train/val: %r" % s)
        for p in ds.heldout_prefixes:
            if s.startswith(p):
                problems.append("held-out prefix in train/val: %r" % s)
    # no boundaries / spaces anywhere in the model's input
    for s in ds.train_strings + ds.val_strings:
        if any(c in s for c in " \t|_-"):
            problems.append("boundary/separator character in input: %r" % s)
        if s.count(EOS) != 1 or not s.endswith(EOS):
            problems.append("malformed input: %r" % s)
    # vocabulary built only from the training corpus
    train_chars = set("".join(ds.train_strings))
    if set(ds.chars) != train_chars:
        problems.append("vocabulary is not exactly the training characters")
    for t in ds.itos[2:]:
        if len(t) != 1:
            problems.append("non-character token in vocabulary: %r" % t)
    # the training function must not be able to reach hidden labels
    sig = inspect.signature(train_model)
    for name in sig.parameters:
        if name in ("corpus", "dataset", "ds", "labels", "families"):
            problems.append("train_model receives %s" % name)
    src = inspect.getsource(train_model) + inspect.getsource(CausalTransformer)
    for bad in ["subj_family", "obj_family", "objfam_table", "object_families",
                "subject_families", "heldout_compatible", "heldout_incompatible",
                "heldout_pairs", "object_words", "load_state_dict_from_url",
                "from_pretrained", "pretrained"]:
        if bad in src:
            problems.append("training code references %r" % bad)
    return problems


# =============================================================================
# 2-3.  A SMALL, ORDINARY CAUSAL TRANSFORMER (RANDOM INIT)
# =============================================================================


class Block(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                          batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(),
                                nn.Linear(d_ff, d_model), nn.Dropout(dropout))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + self.drop(a)
        x = x + self.ff(self.ln2(x))
        return x


class CausalTransformer(nn.Module):
    """Standard decoder-only Transformer.  All parameters are randomly
    initialized for every run; nothing is ever loaded from disk or the net."""

    def __init__(self, vocab_size, d_model, n_layers, n_heads, max_len,
                 dropout=0.1, pad_id=0):
        super().__init__()
        d_ff = 4 * d_model
        self.tok = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.max_len = max_len

    def forward(self, idx):
        B, L = idx.shape
        pos = torch.arange(L, device=idx.device).unsqueeze(0)
        x = self.drop(self.tok(idx) + self.pos(pos))
        mask = torch.full((L, L), float("-inf"), device=idx.device).triu(1)
        for b in self.blocks:
            x = b(x, mask)
        return self.head(self.ln_f(x))


CONFIGS = [
    ("TINY",   dict(d_model=32,  n_layers=2, n_heads=2)),
    ("SMALL",  dict(d_model=64,  n_layers=2, n_heads=4)),
    ("MEDIUM", dict(d_model=128, n_layers=4, n_heads=4)),
]


def n_params(model):
    return sum(p.numel() for p in model.parameters())


# =============================================================================
# 4.  TRAINING  (sees only token ids; no corpus object, no labels)
# =============================================================================


def make_batch_tensors(strings, ds):
    ids = [ds.encode(s) for s in strings]
    L = max(len(i) for i in ids)
    x = torch.full((len(ids), L - 1), ds.pad_id, dtype=torch.long)
    y = torch.full((len(ids), L - 1), ds.pad_id, dtype=torch.long)
    for r, seq in enumerate(ids):
        x[r, :len(seq) - 1] = torch.tensor(seq[:-1])
        y[r, :len(seq) - 1] = torch.tensor(seq[1:])
    return x, y


def evaluate_teacher_forced(model, x, y, pad_id):
    model.eval()
    with torch.no_grad():
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               y.reshape(-1), ignore_index=pad_id)
        pred = logits.argmax(-1)
        m = (y != pad_id)
        acc = ((pred == y) & m).sum().item() / float(m.sum().item())
    return loss.item(), acc


def train_model(vocab_size, pad_id, max_len, train_xy, val_xy, cfg, seed,
                max_steps=3000, batch_size=32, lr=1e-3, warmup=200,
                eval_every=100, log=None):
    """Ordinary from-scratch training.  Receives ONLY tensors of token ids and
    the model hyper-parameters: it cannot reach the corpus or any label.

    Validation loss is used only to pick the best checkpoint.  Training always
    runs the full budget so that undertraining cannot explain a failure.
    """
    torch.manual_seed(seed * 1000 + cfg["d_model"])
    model = CausalTransformer(vocab_size, cfg["d_model"], cfg["n_layers"],
                              cfg["n_heads"], max_len, pad_id=pad_id)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    xtr, ytr = train_xy
    xva, yva = val_xy
    g = torch.Generator().manual_seed(seed * 7 + 1)
    n = xtr.size(0)
    best = {"val_loss": float("inf"), "step": 0, "state": None}
    history = []
    for step in range(1, max_steps + 1):
        model.train()
        idx = torch.randperm(n, generator=g)[:batch_size]
        xb, yb = xtr[idx], ytr[idx]
        logits = model(xb)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               yb.reshape(-1), ignore_index=pad_id)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for gp in opt.param_groups:
            gp["lr"] = lr * min(1.0, step / float(warmup))
        opt.step()
        if step % eval_every == 0 or step == max_steps:
            vl, va = evaluate_teacher_forced(model, xva, yva, pad_id)
            tl, ta = evaluate_teacher_forced(model, xtr, ytr, pad_id)
            history.append({"step": step, "train_loss": tl, "train_acc": ta,
                            "val_loss": vl, "val_acc": va})
            if vl < best["val_loss"]:
                best = {"val_loss": vl, "step": step,
                        "state": copy.deepcopy(model.state_dict())}
            if log:
                log(step, tl, ta, vl, va)
    final_state = copy.deepcopy(model.state_dict())
    return model, best, final_state, history


# =============================================================================
# 5-6.  EVALUATION AS A PROBABILISTIC LM  (harness only, after training)
# =============================================================================


@torch.no_grad()
def continuation_logprob(model, ds, prefix, cont):
    """log P(cont | prefix), summed autoregressive character log-probabilities."""
    model.eval()
    ids = ds.encode(prefix + cont)
    x = torch.tensor(ids[:-1]).unsqueeze(0)
    tgt = ids[1:]
    logp = F.log_softmax(model(x), dim=-1)[0]
    total = 0.0
    for i in range(len(prefix), len(tgt)):
        total += logp[i, tgt[i]].item()
    return total


def rank_objects(model, ds, prefix):
    """Rank ALL object words of the lexicon by sequence log-probability.
    The model is never told which family is correct."""
    scored = []
    for o in ds.object_words:
        lp = continuation_logprob(model, ds, prefix, o + EOS)
        scored.append((o, lp, lp / float(len(o) + 1)))
    scored.sort(key=lambda t: -t[1])
    return scored


def empirical_ceiling(strings):
    """The best next-character accuracy / cross-entropy any model could reach on
    these strings, given that the corpus is genuinely ambiguous (the same prefix
    is followed by different characters in different strings).  Computed from
    the empirical conditional distribution; used only to interpret whether the
    Transformer FIT the corpus."""
    from collections import defaultdict
    cnt = defaultdict(lambda: defaultdict(int))
    for s in strings:
        for i in range(len(s)):
            cnt[s[:i]][s[i]] += 1
    tot = hit = 0
    ent = 0.0
    for s in strings:
        for i in range(len(s)):
            d = cnt[s[:i]]
            n = sum(d.values())
            tot += 1
            if s[i] == max(d, key=d.get):
                hit += 1
            ent += -math.log(d[s[i]] / float(n))
    return {"accuracy_ceiling": hit / float(tot),
            "cross_entropy_floor": ent / float(tot),
            "branching_decisions": sum(
                1 for s in strings for i in range(len(s)) if len(cnt[s[:i]]) > 1),
            "decisions": tot}


def metrics_from_ranking(scored, correct, all_objects):
    """Turn a ranking of the object lexicon into the section-6 metrics."""
    order = [o for o, _lp, _n in scored]
    top4 = set(order[:4])
    ranks = {o: i + 1 for i, o in enumerate(order)}
    lp = {o: l for o, l, _n in scored}
    wins = tot = 0
    for c in correct:
        for w in all_objects:
            if w in correct:
                continue
            tot += 1
            if lp[c] > lp[w]:
                wins += 1
    mx = max(lp.values())
    ex = {o: math.exp(lp[o] - mx) for o in lp}
    Z = sum(ex.values())
    norm_order = [o for o, _l, _n in sorted(scored, key=lambda t: -t[2])]
    return {
        "top4": sorted(top4),
        "top4_exact_family": 1.0 if top4 == set(correct) else 0.0,
        "recall_at_4": len(top4 & set(correct)) / float(len(correct)),
        "precision_at_4": len(top4 & set(correct)) / 4.0,
        "pairwise_family_accuracy": wins / float(tot) if tot else 0.0,
        "probability_mass_correct_family": sum(ex[o] for o in correct) / Z,
        "mean_rank_correct": statistics.mean([ranks[o] for o in correct]),
        "median_rank_correct": statistics.median([ranks[o] for o in correct]),
        "lengthnorm_top4_exact_family":
            1.0 if set(norm_order[:4]) == set(correct) else 0.0,
        "lengthnorm_recall_at_4":
            len(set(norm_order[:4]) & set(correct)) / float(len(correct)),
    }


def heldout_metrics(model, ds, prefix):
    correct = sorted(set(s[len(prefix):-1] for s in ds.heldout_compatible[prefix]))
    scored = rank_objects(model, ds, prefix)
    m = metrics_from_ranking(scored, correct, ds.object_words)
    m["prefix"] = prefix
    m["correct_objects"] = correct
    m["ranking"] = [[o, round(l, 4)] for o, l, _n in scored]
    return m


# =============================================================================
# 7.  FREE GENERATION (secondary)
# =============================================================================


@torch.no_grad()
def greedy_generate(model, ds, prefix, max_new=10):
    model.eval()
    ids = ds.encode(prefix)
    out = ""
    for _ in range(max_new):
        logits = model(torch.tensor(ids).unsqueeze(0))[0, -1]
        nxt = int(logits.argmax().item())
        ch = ds.itos[nxt]
        if len(ch) != 1:
            break
        out += ch
        ids.append(nxt)
        if ch == EOS:
            break
    return prefix + out


@torch.no_grad()
def beam_generate(model, ds, prefix, width=10, max_new=10):
    model.eval()
    beams = [(0.0, ds.encode(prefix), "", False)]
    done = []
    for _ in range(max_new):
        cand = []
        for score, ids, txt, fin in beams:
            if fin:
                done.append((score, txt))
                continue
            logp = F.log_softmax(model(torch.tensor(ids).unsqueeze(0))[0, -1], -1)
            top = torch.topk(logp, min(width, logp.numel()))
            for lp, ix in zip(top.values.tolist(), top.indices.tolist()):
                ch = ds.itos[ix]
                if len(ch) != 1:
                    continue
                cand.append((score + lp, ids + [ix], txt + ch, ch == EOS))
        if not cand:
            break
        cand.sort(key=lambda t: -t[0])
        beams = cand[:width]
        if all(b[3] for b in beams):
            break
    for score, ids, txt, fin in beams:
        if fin:
            done.append((score, txt))
    done.sort(key=lambda t: -t[0])
    seen, out = set(), []
    for score, txt in done:
        if txt in seen:
            continue
        seen.add(txt)
        out.append(prefix + txt)
    return out[:width]


def classify_string(ds, prefix, s):
    if s in ds.heldout_compatible[prefix]:
        return "compatible"
    if s in ds.heldout_incompatible[prefix]:
        return "incompatible"
    if ds._corpus.decompose(s) is not None:
        return "incompatible"
    return "malformed"


# =============================================================================
# 8.  SUBJECT-FAMILY PROBE (no classifier is trained on hidden labels)
# =============================================================================


def subject_family_probe(model, ds):
    subjects = []
    for fam in ds.subject_families:
        subjects.extend(fam)
    subjects = sorted(subjects)
    patterns = {}
    signatures = {}
    for s in subjects:
        per_verb = {}
        sig = []
        for v in ds.verbs:
            pfx = s + ds.probe_filler + v
            scored = rank_objects(model, ds, pfx)
            lp = {o: l for o, l, _n in scored}
            mx = max(lp.values())
            ex = {o: math.exp(lp[o] - mx) for o in lp}
            Z = sum(ex.values())
            famp = [0.0, 0.0, 0.0]
            for o in ds.object_words:
                famp[ds.obj_family[o]] += ex[o] / Z
            truth_fam = ds.objfam_table[ds.subj_family[s]][ds.verb_index[v]]
            correct = ds.object_families[truth_fam]
            mm = metrics_from_ranking(scored, correct, ds.object_words)
            per_verb[v] = {"family_probabilities": [round(p, 4) for p in famp],
                           "argmax_family": famp.index(max(famp)),
                           "true_family": truth_fam,
                           "top_object": scored[0][0],
                           "metrics": mm,
                           "heldout_combination": (s, v) in ds.heldout_pairs}
            sig.append(famp.index(max(famp)))
        patterns[s] = per_verb
        signatures[s] = tuple(sig)

    groups = {}
    for s in subjects:
        groups.setdefault(signatures[s], []).append(s)
    groups = [sorted(g) for g in groups.values()]

    fam_of = ds.subj_family
    grp_of = {}
    for gi, g in enumerate(groups):
        for s in g:
            grp_of[s] = gi
    same_ok = same_tot = diff_bad = diff_tot = 0
    for i, a in enumerate(subjects):
        for b in subjects[i + 1:]:
            tog = grp_of[a] == grp_of[b]
            if fam_of[a] == fam_of[b]:
                same_tot += 1
                same_ok += 1 if tog else 0
            else:
                diff_tot += 1
                diff_bad += 1 if tog else 0
    # does each subject's behavioural signature equal its family's true row?
    true_rows = {s: tuple(ds.objfam_table[fam_of[s]][ds.verb_index[v]]
                          for v in ds.verbs) for s in subjects}
    correct_rows = [s for s in subjects if signatures[s] == true_rows[s]]
    # per (subject, verb) argmax correctness, split by seen / held-out combo
    seen_ok = seen_tot = held_ok = held_tot = 0
    for s in subjects:
        for v in ds.verbs:
            truth = ds.objfam_table[fam_of[s]][ds.verb_index[v]]
            hit = patterns[s][v]["argmax_family"] == truth
            if (s, v) in ds.heldout_pairs:
                held_tot += 1
                held_ok += 1 if hit else 0
            else:
                seen_tot += 1
                seen_ok += 1 if hit else 0
    # section-6 metrics split by whether the (subject, verb) COMBINATION was
    # present in training - the control that separates "learned the training
    # relation" from "the ranking procedure is broken".
    buckets = {"seen": [], "heldout": []}
    for s in subjects:
        for v in ds.verbs:
            b = "heldout" if patterns[s][v]["heldout_combination"] else "seen"
            buckets[b].append(patterns[s][v]["metrics"])
    combo = {}
    for b, ms in buckets.items():
        for k in ("top4_exact_family", "recall_at_4", "pairwise_family_accuracy",
                  "probability_mass_correct_family"):
            combo["%s_%s" % (b, k)] = (sum(m[k] for m in ms) / float(len(ms))
                                       if ms else 0.0)
        combo["%s_n" % b] = len(ms)

    return {
        "subjects": subjects,
        "combination_control": combo,
        "true_hidden_families": ds.subject_families,
        "per_subject_object_family_probabilities": patterns,
        "signatures": {s: list(signatures[s]) for s in subjects},
        "true_signature_rows": {s: list(true_rows[s]) for s in subjects},
        "behavioral_groups": groups,
        "groups_equal_hidden_families":
            sorted([sorted(f) for f in ds.subject_families]) == sorted(groups),
        "same_family_pairs_grouped": same_ok, "same_family_pairs_total": same_tot,
        "different_family_pairs_grouped": diff_bad,
        "different_family_pairs_total": diff_tot,
        "subjects_with_correct_signature_row": correct_rows,
        "seen_combo_argmax_correct": seen_ok, "seen_combo_total": seen_tot,
        "heldout_combo_argmax_correct": held_ok, "heldout_combo_total": held_tot,
    }


# =============================================================================
# 9-10.  RUNNER
# =============================================================================


def run_one(ds, size_name, cfg, seed, max_steps=3000, verbose=True):
    xtr, ytr = make_batch_tensors(ds.train_strings, ds)
    xva, yva = make_batch_tensors(ds.val_strings, ds)
    model, best, final_state, history = train_model(
        len(ds.itos), ds.pad_id, ds.max_len, (xtr, ytr), (xva, yva), cfg, seed,
        max_steps=max_steps)

    out = {"seed": seed, "size": size_name, "config": cfg,
           "train_ceiling": empirical_ceiling(ds.train_strings),
           "val_ceiling": empirical_ceiling(ds.val_strings),
           "parameters": n_params(model), "steps": max_steps,
           "best_val_step": best["step"], "history": history[-1],
           "history_full": history}

    for tag, state in (("best_val", best["state"]), ("final", final_state)):
        model.load_state_dict(state)
        tl, ta = evaluate_teacher_forced(model, xtr, ytr, ds.pad_id)
        vl, va = evaluate_teacher_forced(model, xva, yva, ds.pad_id)
        per_prefix = [heldout_metrics(model, ds, p) for p in ds.heldout_prefixes]
        mean = lambda k: sum(m[k] for m in per_prefix) / float(len(per_prefix))
        gen = []
        if tag == "best_val":
            for p in ds.heldout_prefixes:
                g = greedy_generate(model, ds, p)
                bm = beam_generate(model, ds, p, width=10)
                gen.append({
                    "prefix": p,
                    "greedy": g, "greedy_class": classify_string(ds, p, g),
                    "beam": bm,
                    "beam_classes": [classify_string(ds, p, b) for b in bm],
                })
            out["subject_family_probe"] = subject_family_probe(model, ds)
        out[tag] = {
            "train_loss": tl, "train_next_char_accuracy": ta,
            "val_loss": vl, "val_next_char_accuracy": va,
            "top4_exact_family_accuracy": mean("top4_exact_family"),
            "recall_at_4": mean("recall_at_4"),
            "precision_at_4": mean("precision_at_4"),
            "pairwise_family_accuracy": mean("pairwise_family_accuracy"),
            "probability_mass_correct_family": mean("probability_mass_correct_family"),
            "mean_rank_correct": mean("mean_rank_correct"),
            "median_rank_correct": mean("median_rank_correct"),
            "lengthnorm_top4_exact_family_accuracy": mean("lengthnorm_top4_exact_family"),
            "lengthnorm_recall_at_4": mean("lengthnorm_recall_at_4"),
            "per_prefix": per_prefix,
            "free_generation": gen,
        }
    if verbose:
        b = out["best_val"]
        print("  [%-6s seed %2d] params=%-7d best_step=%-5d train_acc=%.3f "
              "val_loss=%.3f | top4exact=%.2f R@4=%.3f pair=%.3f mass=%.3f"
              % (size_name, seed, out["parameters"], out["best_val_step"],
                 b["train_next_char_accuracy"], b["val_loss"],
                 b["top4_exact_family_accuracy"], b["recall_at_4"],
                 b["pairwise_family_accuracy"],
                 b["probability_mass_correct_family"]))
        sys.stdout.flush()
    return out


def mean_of(rows, tag, key):
    xs = [r[tag][key] for r in rows]
    return sum(xs) / float(len(xs)) if xs else 0.0


def std_of(rows, tag, key):
    xs = [r[tag][key] for r in rows]
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / float(len(xs))
    return (sum((x - m) ** 2 for x in xs) / float(len(xs) - 1)) ** 0.5


# =============================================================================
# 11.  DATA-SCALE CURVE (same hidden grammar, more data) - run last
# =============================================================================


class ScaledCorpus(object):
    """The SAME hidden grammar as V4 - 3 subject families, verbs shared by all
    families, object family selected by the (subject-family, verb) Latin square,
    a neutral filler between subject and verb - with more lexical items and more
    sentences.  The rules of the language are unchanged.
    """

    def __init__(self, seed, n_subj, n_obj, n_fill, n_verbs=3):
        rng = random.Random(seed * 7919 + 977)
        words = set()

        def fresh(family_words):
            for _ in range(20000):
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

        self.subject_families = [fam(n_subj) for _ in range(3)]
        self.verbs = fam(n_verbs)
        self.object_families = [fam(n_obj) for _ in range(3)]
        self.fillers = fam(n_fill)
        self.objfam_table = [[(fi + vi) % 3 for vi in range(n_verbs)]
                             for fi in range(3)]
        self.subj_family = {s: fi for fi, f in enumerate(self.subject_families)
                            for s in f}
        self.obj_family = {o: fi for fi, f in enumerate(self.object_families)
                           for o in f}
        self.verb_index = {v: i for i, v in enumerate(self.verbs)}
        self.probe_filler = self.fillers[0]

        perms = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]]
        good = [p for p in perms
                if len(set(self.objfam_table[fi][p[fi]] for fi in range(3))) == 3]
        vperm = good[rng.randrange(len(good))]
        self.heldout_pairs = [(self.subject_families[fi][rng.randrange(n_subj)],
                               self.verbs[vperm[fi]]) for fi in range(3)]
        hp = set(self.heldout_pairs)

        self.heldout_compatible, self.heldout_incompatible = {}, {}
        for (s, v) in self.heldout_pairs:
            fi, vi = self.subj_family[s], self.verb_index[v]
            good_f = self.objfam_table[fi][vi]
            pfx = s + self.probe_filler + v
            self.heldout_compatible[pfx] = set(
                pfx + o + EOS for o in self.object_families[good_f])
            self.heldout_incompatible[pfx] = set(
                pfx + o + EOS for gi in range(3) if gi != good_f
                for o in self.object_families[gi])
        self.heldout_prefixes = sorted(self.heldout_compatible)
        self.heldout_strings = set()
        for (s, v) in self.heldout_pairs:
            fi, vi = self.subj_family[s], self.verb_index[v]
            for f in self.fillers:
                for o in self.object_families[self.objfam_table[fi][vi]]:
                    self.heldout_strings.add(s + f + v + o + EOS)

        rows = []
        for fi in range(3):
            for s in self.subject_families[fi]:
                for vi, v in enumerate(self.verbs):
                    if (s, v) in hp:
                        continue
                    for o in self.object_families[self.objfam_table[fi][vi]]:
                        for f in self.fillers:
                            rows.append(s + f + v + o + EOS)
        rows = sorted(set(rows))
        rng.shuffle(rows)
        cut = int(len(rows) * 0.8)
        self.train_strings = sorted(rows[:cut])
        self.val_strings = sorted(rows[cut:])
        for t in self.train_strings + self.val_strings:
            assert t not in self.heldout_strings
            for p in self.heldout_prefixes:
                assert not t.startswith(p)

    def decompose(self, s):
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


class ScaledDataset(Dataset):
    def __init__(self, seed, n_subj, n_obj, n_fill):
        corpus = ScaledCorpus(seed, n_subj, n_obj, n_fill)
        self.seed = seed
        self._corpus = corpus
        self.train_strings = list(corpus.train_strings)
        self.val_strings = list(corpus.val_strings)
        self.heldout_prefixes = list(corpus.heldout_prefixes)
        self.heldout_compatible = {p: set(corpus.heldout_compatible[p])
                                   for p in corpus.heldout_prefixes}
        self.heldout_incompatible = {p: set(corpus.heldout_incompatible[p])
                                     for p in corpus.heldout_prefixes}
        self.object_words = [o for f in corpus.object_families for o in f]
        self.object_families = [list(f) for f in corpus.object_families]
        self.obj_family = dict(corpus.obj_family)
        self.subject_families = [list(f) for f in corpus.subject_families]
        self.subj_family = dict(corpus.subj_family)
        self.verbs = list(corpus.verbs)
        self.verb_index = dict(corpus.verb_index)
        self.objfam_table = [list(r) for r in corpus.objfam_table]
        self.probe_filler = corpus.probe_filler
        self.heldout_pairs = [tuple(p) for p in corpus.heldout_pairs]
        self.heldout_strings = set(corpus.heldout_strings)
        chars = sorted(set("".join(self.train_strings)))
        self.itos = [PAD_TOK, BOS_TOK] + chars
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        self.pad_id = self.stoi[PAD_TOK]
        self.bos_id = self.stoi[BOS_TOK]
        self.chars = chars
        # fixed positional budget: longer than any string the model can ever
        # be given (longest sentence + free-generation slack).  A buffer size,
        # not information: the model never sees a longer sequence.
        self.max_len = 64
        assert self.max_len > 12 + max(len(s) for s in
                                       self.train_strings + self.val_strings)


SCALES = [("1x", 3, 4, 1), ("3x", 3, 4, 3), ("10x", 6, 7, 3), ("30x", 8, 10, 4)]
SCALE_SEEDS = [1, 2, 3]


# =============================================================================
# MAIN
# =============================================================================


def main():
    results_v4_path = os.path.join(HERE, "results_v4.json")
    print("=" * 78)
    print("TRANSFORMER BASELINE ON THE EXACT V4 DATASETS")
    print("=" * 78)

    identity_problems, cheat_problems = [], []
    datasets = {}
    for seed in SEEDS:
        ds = Dataset(seed)
        datasets[seed] = ds
        identity_problems += dataset_identity_checks(ds, results_v4_path)
        cheat_problems += cheat_controls(ds, None)
    print("dataset identity with V4: %s (%d problems)"
          % ("PASS" if not identity_problems else "FAIL", len(identity_problems)))
    d1 = datasets[SEEDS[0]]
    print("seed %d: train=%d val=%d vocab=%d chars=%s"
          % (SEEDS[0], len(d1.train_strings), len(d1.val_strings),
             len(d1.itos), "".join(d1.chars)))
    print("seed %d held-out prefixes: %s" % (SEEDS[0], d1.heldout_prefixes))
    print("")

    all_results = {}
    for size_name, cfg in CONFIGS:
        rows = []
        for seed in SEEDS:
            rows.append(run_one(datasets[seed], size_name, cfg, seed))
        all_results[size_name] = rows
        print("")

    # ---------------- data-scale curve (secondary, run last) --------------
    best_size = max(CONFIGS, key=lambda c: mean_of(all_results[c[0]], "best_val",
                                                   "pairwise_family_accuracy"))[0]
    best_cfg = dict(CONFIGS)[best_size]
    print("data-scale curve with the best-performing configuration: %s" % best_size)
    scale_rows = {}
    for tag, ns, no, nf in SCALES:
        rs = []
        for seed in SCALE_SEEDS:
            sds = ScaledDataset(seed, ns, no, nf)
            r = run_one(sds, "%s(%s)" % (best_size, tag), best_cfg, seed,
                        max_steps=3000)
            r["n_train"] = len(sds.train_strings)
            r["n_objects"] = len(sds.object_words)
            rs.append(r)
        scale_rows[tag] = rs
        print("")

    # ------------------------------------------------------------ files
    blob = {
        "note": "conventional causal Transformer baseline trained from scratch "
                "on the exact datasets of symbolic_rewrite_experiment_v4.py",
        "seeds": SEEDS,
        "dataset_identity_with_v4": "PASS" if not identity_problems else "FAIL",
        "dataset_identity_problems": identity_problems,
        "cheat_control_problems": cheat_problems,
        "pretraining_used": False,
        "configs": {n: c for n, c in CONFIGS},
        "results": {k: [{kk: vv for kk, vv in r.items() if kk != "history_full"}
                        for r in v] for k, v in all_results.items()},
        "scale_curve": {k: [{kk: vv for kk, vv in r.items()
                             if kk not in ("history_full", "subject_family_probe")}
                            for r in v] for k, v in scale_rows.items()},
        "scale_curve_config": best_size,
    }
    with open(os.path.join(HERE, "transformer_baseline_results.json"), "w") as f:
        json.dump(blob, f, indent=1, ensure_ascii=False)

    # ------------------------------------------------------ final report
    v4 = {"precision": 0.291, "recall": 0.808, "exact": 0.000}
    if os.path.exists(results_v4_path):
        a = json.load(open(results_v4_path))["ablations"]["full_system"]["aggregate"]
        v4 = {"precision": a["precision_mean"], "recall": a["recall_mean"],
              "exact": a["exact_set_accuracy_mean"]}

    print("================ TRANSFORMER BASELINE RESULTS ================")
    print("")
    print("DATASET IDENTITY WITH V4: %s"
          % ("PASS" if not identity_problems else "FAIL"))
    for p in identity_problems[:5]:
        print("   !! %s" % p)
    print("")
    print("model sizes / parameter counts:")
    for size_name, cfg in CONFIGS:
        print("  %-7s d_model=%-4d layers=%d heads=%d  params=%d"
              % (size_name, cfg["d_model"], cfg["n_layers"], cfg["n_heads"],
                 all_results[size_name][0]["parameters"]))
    print("")
    print("TRAIN FROM SCRATCH: PASS  (all parameters randomly initialized per seed)")
    print("PRETRAINING USED: NO")
    print("RAW CHARACTERS ONLY: %s  (vocab = %d training characters + <bos>/<pad>)"
          % ("PASS" if not cheat_problems else "FAIL", len(d1.chars)))
    print("")
    for size_name, _cfg in CONFIGS:
        rows = all_results[size_name]
        ceil = sum(r["train_ceiling"]["accuracy_ceiling"] for r in rows) / len(rows)
        floor = sum(r["train_ceiling"]["cross_entropy_floor"] for r in rows) / len(rows)
        vceil = sum(r["val_ceiling"]["accuracy_ceiling"] for r in rows) / len(rows)
        print("%s:" % size_name)
        print("  train next-char accuracy:              %.4f  (final-step %.4f)"
              % (mean_of(rows, "best_val", "train_next_char_accuracy"),
                 mean_of(rows, "final", "train_next_char_accuracy")))
        print("      empirical ceiling on this corpus:  %.4f   -> final-step model"
              " reaches %.1f%% of the achievable maximum"
              % (ceil, 100.0 * mean_of(rows, "final", "train_next_char_accuracy") / ceil))
        print("      train loss %.4f (final %.4f) vs entropy floor %.4f"
              % (mean_of(rows, "best_val", "train_loss"),
                 mean_of(rows, "final", "train_loss"), floor))
        print("  validation loss:                       %.4f"
              % mean_of(rows, "best_val", "val_loss"))
        print("  validation next-char accuracy:         %.4f  (val ceiling %.4f)"
              % (mean_of(rows, "best_val", "val_next_char_accuracy"), vceil))
        print("  held-out top-4 exact-family accuracy:  %.4f  (sd %.4f)"
              % (mean_of(rows, "best_val", "top4_exact_family_accuracy"),
                 std_of(rows, "best_val", "top4_exact_family_accuracy")))
        print("  Recall@4:                              %.4f  (sd %.4f)"
              % (mean_of(rows, "best_val", "recall_at_4"),
                 std_of(rows, "best_val", "recall_at_4")))
        print("  Precision@4:                           %.4f"
              % mean_of(rows, "best_val", "precision_at_4"))
        print("  pairwise family accuracy:              %.4f  (sd %.4f)"
              % (mean_of(rows, "best_val", "pairwise_family_accuracy"),
                 std_of(rows, "best_val", "pairwise_family_accuracy")))
        print("  probability mass on correct family:    %.4f   (chance = 0.3333)"
              % mean_of(rows, "best_val", "probability_mass_correct_family"))
        print("  mean / median rank of correct objects: %.2f / %.2f  (of 12)"
              % (mean_of(rows, "best_val", "mean_rank_correct"),
                 mean_of(rows, "best_val", "median_rank_correct")))
        print("  [secondary, length-normalized] top-4 exact %.4f  R@4 %.4f"
              % (mean_of(rows, "best_val", "lengthnorm_top4_exact_family_accuracy"),
                 mean_of(rows, "best_val", "lengthnorm_recall_at_4")))
        print("  [final-step checkpoint] top-4 exact %.4f  R@4 %.4f  pairwise %.4f"
              % (mean_of(rows, "final", "top4_exact_family_accuracy"),
                 mean_of(rows, "final", "recall_at_4"),
                 mean_of(rows, "final", "pairwise_family_accuracy")))
        print("")
    print("MEMORIZATION vs GENERALIZATION - the same object-ranking metrics on")
    print("(subject,verb) combinations that WERE in training vs the held-out ones:")
    print("  %-7s %-28s %-28s" % ("", "SEEN combinations", "HELD-OUT combinations"))
    for size_name, _cfg in CONFIGS:
        rows = all_results[size_name]
        c = lambda k: sum(r["subject_family_probe"]["combination_control"][k]
                          for r in rows) / float(len(rows))
        print("  %-7s top4exact %.3f R@4 %.3f pair %.3f mass %.3f | "
              "top4exact %.3f R@4 %.3f pair %.3f mass %.3f"
              % (size_name,
                 c("seen_top4_exact_family"), c("seen_recall_at_4"),
                 c("seen_pairwise_family_accuracy"),
                 c("seen_probability_mass_correct_family"),
                 c("heldout_top4_exact_family"), c("heldout_recall_at_4"),
                 c("heldout_pairwise_family_accuracy"),
                 c("heldout_probability_mass_correct_family")))
    print("  (chance: top4exact 0.002, R@4 0.333, pairwise 0.500, mass 0.333)")
    print("")
    print("per-seed held-out exact-family accuracy:")
    for size_name, _cfg in CONFIGS:
        print("  %-7s %s" % (size_name, " ".join(
            "%.2f" % r["best_val"]["top4_exact_family_accuracy"]
            for r in all_results[size_name])))
    print("per-seed Recall@4:")
    for size_name, _cfg in CONFIGS:
        print("  %-7s %s" % (size_name, " ".join(
            "%.2f" % r["best_val"]["recall_at_4"] for r in all_results[size_name])))
    print("per-seed pairwise family accuracy:")
    for size_name, _cfg in CONFIGS:
        print("  %-7s %s" % (size_name, " ".join(
            "%.2f" % r["best_val"]["pairwise_family_accuracy"]
            for r in all_results[size_name])))
    print("")
    print("FREE GENERATION (secondary; best_val checkpoint, %s, seed %d):"
          % (CONFIGS[-1][0], SEEDS[0]))
    for g in all_results[CONFIGS[-1][0]][0]["best_val"]["free_generation"]:
        print("  prefix %r" % g["prefix"])
        print("    greedy: %r  -> %s" % (g["greedy"], g["greedy_class"]))
        print("    beam10: %s" % ", ".join(
            "%s(%s)" % (b[len(g["prefix"]):], c[:6])
            for b, c in zip(g["beam"][:6], g["beam_classes"][:6])))
    for size_name, _cfg in CONFIGS:
        cc = {"compatible": 0, "incompatible": 0, "malformed": 0}
        bb = dict(cc)
        for r in all_results[size_name]:
            for g in r["best_val"]["free_generation"]:
                cc[g["greedy_class"]] += 1
                for c in g["beam_classes"]:
                    bb[c] += 1
        print("  %-7s greedy: %s | beam10: %s" % (size_name, cc, bb))
    print("")
    print("SUBJECT FAMILY PROBE (behavioural, no classifier trained on labels):")
    for size_name, _cfg in CONFIGS:
        rows = all_results[size_name]
        eq = sum(1 for r in rows if r["subject_family_probe"]["groups_equal_hidden_families"])
        so = sum(r["subject_family_probe"]["same_family_pairs_grouped"] for r in rows)
        st = sum(r["subject_family_probe"]["same_family_pairs_total"] for r in rows)
        db = sum(r["subject_family_probe"]["different_family_pairs_grouped"] for r in rows)
        dt = sum(r["subject_family_probe"]["different_family_pairs_total"] for r in rows)
        sk = sum(r["subject_family_probe"]["seen_combo_argmax_correct"] for r in rows)
        skt = sum(r["subject_family_probe"]["seen_combo_total"] for r in rows)
        hk = sum(r["subject_family_probe"]["heldout_combo_argmax_correct"] for r in rows)
        hkt = sum(r["subject_family_probe"]["heldout_combo_total"] for r in rows)
        cr = sum(len(r["subject_family_probe"]["subjects_with_correct_signature_row"])
                 for r in rows)
        print("  %-7s groups==families %d/%d seeds | same-family pairs grouped %d/%d"
              " | different-family grouped %d/%d" % (size_name, eq, len(rows), so, st, db, dt))
        print("          argmax object-family correct: seen combos %d/%d (%.2f), "
              "HELD-OUT combos %d/%d (%.2f)  [chance 0.33]"
              % (sk, skt, sk / float(skt), hk, hkt, hk / float(hkt)))
        print("          subjects whose full 3-verb signature equals their true "
              "family row: %d/%d" % (cr, 9 * len(rows)))
    p0 = all_results[CONFIGS[-1][0]][0]["subject_family_probe"]
    print("  per-subject object-family probability pattern (%s, seed %d):"
          % (CONFIGS[-1][0], SEEDS[0]))
    for s in p0["subjects"]:
        fam = [i for i, f in enumerate(p0["true_hidden_families"]) if s in f][0]
        pats = " | ".join("%s:%s" % (v[:4], p0["per_subject_object_family_probabilities"][s][v]["family_probabilities"])
                          for v in sorted(p0["per_subject_object_family_probabilities"][s]))
        print("    %-8s fam%d  sig=%s true=%s  %s"
              % (s, fam, p0["signatures"][s], p0["true_signature_rows"][s], pats))
    print("  behavioural groups: %s" % p0["behavioral_groups"])
    print("")
    print("COMPARISON TO V4:")
    print("  V4 full-system:")
    print("      precision          = %.3f" % v4["precision"])
    print("      recall             = %.3f" % v4["recall"])
    print("      exact-set accuracy = %.3f" % v4["exact"])
    print("  Transformer (best of the three sizes on each metric):")
    bt = max(CONFIGS, key=lambda c: mean_of(all_results[c[0]], "best_val",
                                            "top4_exact_family_accuracy"))[0]
    br = max(CONFIGS, key=lambda c: mean_of(all_results[c[0]], "best_val",
                                            "recall_at_4"))[0]
    bp = max(CONFIGS, key=lambda c: mean_of(all_results[c[0]], "best_val",
                                            "pairwise_family_accuracy"))[0]
    print("      exact-family accuracy = %.3f   (%s)"
          % (mean_of(all_results[bt], "best_val", "top4_exact_family_accuracy"), bt))
    print("      Recall@4              = %.3f   (%s)   [chance 0.333]"
          % (mean_of(all_results[br], "best_val", "recall_at_4"), br))
    print("      pairwise family acc.  = %.3f   (%s)   [chance 0.500]"
          % (mean_of(all_results[bp], "best_val", "pairwise_family_accuracy"), bp))
    print("")
    print("INTERPRETATION:")
    fits = all(mean_of(all_results[n], "final", "train_next_char_accuracy") >
               0.95 * (sum(r["train_ceiling"]["accuracy_ceiling"]
                           for r in all_results[n]) / float(len(all_results[n])))
               for n, _ in CONFIGS)
    best_pair = max(mean_of(all_results[n], "best_val", "pairwise_family_accuracy")
                    for n, _ in CONFIGS)
    best_exact = max(mean_of(all_results[n], "best_val", "top4_exact_family_accuracy")
                     for n, _ in CONFIGS)
    spread = max(mean_of(all_results[n], "best_val", "pairwise_family_accuracy")
                 for n, _ in CONFIGS) - \
        min(mean_of(all_results[n], "best_val", "pairwise_family_accuracy")
            for n, _ in CONFIGS)
    seen_pair = max(sum(r["subject_family_probe"]["combination_control"]
                        ["seen_pairwise_family_accuracy"] for r in all_results[n])
                    / float(len(all_results[n])) for n, _ in CONFIGS)
    if not fits:
        verdict = "C. Transformer cannot even adequately fit the small corpus."
    elif best_exact >= 0.5 or best_pair >= 0.8:
        verdict = ("A. Transformer fits training AND discovers the held-out "
                   "compositional relation.")
    else:
        verdict = ("B. Transformer fits training but ALSO fails the held-out "
                   "compositional relation.")
    print("  %s" % verdict)
    print("  (train next-char accuracy at the end of training, per size: %s; "
          "empirical ceiling %.3f)"
          % (", ".join("%s=%.3f" % (n, mean_of(all_results[n], "final",
                                               "train_next_char_accuracy"))
                       for n, _ in CONFIGS),
             sum(r["train_ceiling"]["accuracy_ceiling"]
                 for r in all_results[CONFIGS[0][0]]) / float(len(SEEDS))))
    print("  (seen-combination pairwise family accuracy = %.3f: the ranking "
          "procedure itself works)" % seen_pair)
    print("  (spread across model sizes in pairwise accuracy: %.3f -> %s)"
          % (spread, "D. results depend strongly on model size"
             if spread > 0.15 else "results do not depend strongly on model size"))
    print("")
    print("  data-scale curve (%s, %d seeds each, same hidden grammar):"
          % (best_size, len(SCALE_SEEDS)))
    for tag, _ns, _no, _nf in SCALES:
        rs = scale_rows[tag]
        print("    %-4s n_train=%-5d objects=%-3d  exact-family=%.3f  R@4=%.3f  "
              "pairwise=%.3f  train_acc=%.3f"
              % (tag, rs[0]["n_train"], rs[0]["n_objects"],
                 mean_of(rs, "best_val", "top4_exact_family_accuracy"),
                 mean_of(rs, "best_val", "recall_at_4"),
                 mean_of(rs, "best_val", "pairwise_family_accuracy"),
                 mean_of(rs, "final", "train_next_char_accuracy")))
    print("")
    print("CHEAT-CONTROL VIOLATIONS: %d" % len(cheat_problems))
    for p in cheat_problems[:10]:
        print("   !! %s" % p)
    print("  checks: held-out strings absent from train/val; no spaces or")
    print("  boundary characters; vocabulary == training characters only;")
    print("  train_model receives only token tensors (no corpus, no labels);")
    print("  no pretrained weights or embeddings are loaded anywhere;")
    print("  the object lexicon is used only by the evaluation harness.")
    print("==============================================================")


if __name__ == "__main__":
    main()
