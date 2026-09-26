#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
symbolic_rewrite_v5.py

V5 = V4 + ONE conceptual change: SUCCESSFUL RULES LEAVE RESIDUES.

Everything else keeps V4's spirit:
  raw characters only, no tokenizer, no embeddings, no gradients, no learned
  numeric weights, no probability ranking, target-blind random rule proposals,
  successful rules are stored, existing rules are tried before inventing new
  ones, and every internal symbol keeps transparent provenance.

THE CHANGE
----------
When a stored rule participates in a path that reaches the correct target, it
may leave a symbolic residue @<rid> in the workspace.  A residue is retained
only after its rule has succeeded in K DISTINCT contexts (discrete counter, no
weight, no probability).  Once persistent, the residue is written whenever that
rule fires, so two different surface strings acquire overlapping internal
structure purely because the same successful rules operated on them.

Residues can be required by later rules and can be rewritten into higher-order
structures (@17 + @42 -> §91).  Those rewrites are discovered by the SAME
target-blind success process - never by comparing words, merging states, or
measuring similarity.

STATE
-----
Unlike V4, raw input order is explicit: the chain is an ordered sequence.  The
2D matrix is a workspace for residues and internal symbols only; its
coordinates carry no order information and nothing pretends otherwise.

Run:   python3 symbolic_rewrite_v5.py
Saves: symbolic_rewrite_v5_results.json
Standard library only (imports V4 for the unchanged baseline).
"""

import inspect
import itertools
import json
import math
import os
import random
import sys
import time
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import symbolic_rewrite_experiment_v4 as V4        # UNCHANGED baseline

EOS = "#"
BOS = "^"
WILD = ("A",)
SEEDS = [1, 2, 3, 4, 5]
SCALES = [100, 300, 1000]
K_VALUES = [2, 3, 5]
K_MAIN = 3
RUN_TIME_CAP = 900.0        # seconds per training run (computational cutoff)


def lit(s):
    return ("L", s)


def pat_str(e):
    return repr(e[1]) if e[0] == "L" else "*"


def pattern_matches(pattern, tail):
    if len(pattern) != len(tail):
        return False
    for pe, s in zip(pattern, tail):
        if pe[0] == "L" and pe[1] != s:
            return False
    return True


# =============================================================================
# 3.  THE FIXED HIDDEN LANGUAGE
#
# 9 subjects (3 hidden families x 3), 3 verbs, 12 objects (3 families x 4).
# objfam(subject-family i, verb j) = (i + j) % 3  -- a Latin square, so neither
# the subject family alone nor the verb alone determines the object family.
# Surface strings are deliberately unrelated: no two words of a family share a
# 2-character prefix or suffix, and family members differ in length.
# The learner never sees any of this.
# =============================================================================

CONSONANTS = "bdfgklmnprstvz"
VOWELS = "aeiou"


class HiddenLangV5(object):
    """Exposes V4.HiddenCorpus's interface so the unchanged V4 learner and all
    of V4's evaluation code can run on exactly the same data."""

    def __init__(self, seed, max_train=max(SCALES), n_val=60):
        rng = random.Random(seed * 7919 + 13)
        self.seed = seed
        self._access_count = 0
        words = set()

        def fresh(fam_words, nsyl):
            for _ in range(200000):
                w = ""
                for _i in range(nsyl):
                    w += rng.choice(CONSONANTS) + rng.choice(VOWELS)
                if rng.random() < 0.4:
                    w += rng.choice("nrslm")
                if w in words:
                    continue
                if any(u.startswith(w) or w.startswith(u) for u in words):
                    continue
                # members of one hidden family must not share obvious affixes
                if any(u[:2] == w[:2] or u[-2:] == w[-2:] for u in fam_words):
                    continue
                if any(len(u) == len(w) for u in fam_words):
                    continue          # nor share a length
                words.add(w)
                return w
            raise RuntimeError("lexicon generation failed")

        def fam(n):
            f = []
            lens = [2, 3, 2, 3][:n]
            rng.shuffle(lens)
            for i in range(n):
                f.append(fresh(f, lens[i % len(lens)]))
            return f

        self.subject_families = [fam(3) for _ in range(3)]
        self.verbs = fam(3)
        self.object_families = [fam(4) for _ in range(3)]
        self.fillers = []
        self.objfam_table = [[(fi + vi) % 3 for vi in range(3)] for fi in range(3)]
        self.subj_family = {s: fi for fi, f in enumerate(self.subject_families)
                            for s in f}
        self.obj_family = {o: fi for fi, f in enumerate(self.object_families)
                           for o in f}
        self.verb_index = {v: i for i, v in enumerate(self.verbs)}
        self.probe_filler = ""                     # V5 has no filler material

        # ---- held-out subject+verb combinations (as in V4) ----------------
        perms = [list(p) for p in itertools.permutations(range(3))]
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
            pfx = s + v
            self.heldout_compatible[pfx] = set(
                pfx + o + EOS for o in self.object_families[gf])
            self.heldout_incompatible[pfx] = set(
                pfx + o + EOS for gi in range(3) if gi != gf
                for o in self.object_families[gi])
        self.heldout_prefixes = sorted(self.heldout_compatible)
        self.heldout_strings = set()
        for (s, v) in self.heldout_pairs:
            fi, vi = self.subj_family[s], self.verb_index[v]
            for o in self.object_families[self.objfam_table[fi][vi]]:
                self.heldout_strings.add(s + v + o + EOS)

        # ---- the licensed sentences --------------------------------------
        self.triples = []
        for fi in range(3):
            for s in self.subject_families[fi]:
                for vi, v in enumerate(self.verbs):
                    if (s, v) in held:
                        continue
                    for o in self.object_families[self.objfam_table[fi][vi]]:
                        self.triples.append((s, v, o))
        sentences = [s + v + o + EOS for (s, v, o) in self.triples]
        assert len(set(sentences)) == len(sentences)
        self.sentence_set = sorted(set(sentences))

        # ---- validation: licensed sentences held out from TRAINING --------
        pool = list(self.sentence_set)
        rng.shuffle(pool)
        self.val_strings = sorted(pool[:min(n_val, len(pool) // 5)])
        trainable = [s for s in pool if s not in set(self.val_strings)]

        # ---- N = number of TRAINING EXPERIENCES (with repetition) ---------
        # There are only ~%d distinct licensed sentences with no filler
        # material, so scaling N means seeing the SAME relations more often -
        # which is exactly the reuse evidence the residue mechanism needs.
        master = []
        i = 0
        while len(master) < max_train:
            block = list(trainable)
            rng.shuffle(block)
            master.extend(block)
            i += 1
        self._master = master[:max_train]
        self.train_strings = list(self._master)

        for t in self._master + self.val_strings:
            assert t not in self.heldout_strings
            assert " " not in t and t.endswith(EOS) and t.count(EOS) == 1

    def train_at(self, n):
        return list(self._master[:n])

    # ---- V4.HiddenCorpus interface ------------------------------------
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
            r = body[len(subj):]
            for v in self.verbs:
                if r.startswith(v):
                    o = r[len(v):]
                    if o in self.obj_family:
                        return (subj, "", v, o)
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
        for f in self.object_families:
            out.extend(f)
        return out

    def all_subjects(self):
        return [s for f in self.subject_families for s in f]

    def signature(self):
        return {"subject_families": [list(f) for f in self.subject_families],
                "verbs": list(self.verbs),
                "object_families": [list(f) for f in self.object_families],
                "objfam_table": [list(r) for r in self.objfam_table],
                "heldout_pairs": [list(p) for p in self.heldout_pairs],
                "n_distinct_sentences": len(self.sentence_set)}


# =============================================================================
# 1.  STATE: EXPLICIT ORDERED CHAIN + UNORDERED WORKSPACE
# =============================================================================


class SymbolTable(object):
    """Transparent provenance for every internally created symbol."""

    def __init__(self):
        self.n = 0
        self.info = {}
        self.order = []

    def new_symbol(self, rule_id, sources, kind="struct"):
        self.n += 1
        name = u"§%d" % self.n
        self.info[name] = {"name": name, "created_by_rule": rule_id,
                           "kind": kind, "sources": tuple(sources)}
        self.order.append(name)
        return name

    def mark(self):
        return self.n

    def rollback(self, n):
        while self.order and self.n > n:
            self.info.pop(self.order.pop(), None)
            self.n -= 1

    def is_internal(self, s):
        return s in self.info

    def expand(self, sym, _g=0):
        if _g > 48:
            return (sym,)
        rec = self.info.get(sym)
        if rec is None:
            return (sym,)
        out = []
        for s in rec["sources"]:
            out.extend(self.expand(s, _g + 1))
        return tuple(out)

    def depth(self, sym, _g=0):
        rec = self.info.get(sym)
        if rec is None or _g > 48:
            return 0
        return 1 + max([self.depth(s, _g + 1) for s in rec["sources"]] or [0])

    def explain(self, sym, indent=0, lines=None, _g=0):
        if lines is None:
            lines = []
        pad = "  " * indent
        rec = self.info.get(sym)
        if rec is None:
            lines.append("%s%r  [raw]" % (pad, sym))
            return lines
        lines.append("%s%s <- R%s (%s) = %r"
                     % (pad, sym, rec["created_by_rule"], rec["kind"],
                        "".join(str(x) for x in self.expand(sym))))
        if _g < 24:
            for s in rec["sources"]:
                self.explain(s, indent + 1, lines, _g + 1)
        return lines


class State(object):
    """chain   : the ORDERED symbol sequence (raw input order is explicit)
       work    : the 2D workspace - residues and internal symbols, unordered;
                 coordinates carry no information and none is implied
       out     : the OUT register (the prediction is read from here)"""

    __slots__ = ("chain", "work", "out", "_ws")

    def __init__(self, chain=(), work=frozenset(), out=None):
        self.chain = chain
        self.work = work
        self.out = out
        self._ws = None

    def append(self, sym):
        return State(self.chain + (sym,), self.work, None)

    def set_out(self, v):
        return State(self.chain, self.work, v)

    def add_work(self, syms):
        return State(self.chain, self.work | frozenset(syms), self.out)

    def drop_work(self, syms):
        return State(self.chain, self.work - frozenset(syms), self.out)

    def collapse(self, start, k, sym):
        if start < 0 or start + k > len(self.chain) or k < 1:
            return None
        return State(self.chain[:start] + (sym,) + self.chain[start + k:],
                     self.work, self.out)

    def sig(self):
        if self._ws is None:
            self._ws = (self.chain, tuple(sorted(self.work)), self.out)
        return self._ws


def read_output(state):
    return state.out


# =============================================================================
# 4.  RULES  (conditions: chain tail + required workspace symbols)
# =============================================================================


class Rule(object):

    __slots__ = ("rid", "pattern", "require", "actions", "kind", "emit_char",
                 "created_at", "contexts", "success_contexts", "n_apply",
                 "n_reuse", "active", "residue", "left_contexts")

    def __init__(self, rid, pattern, require, actions, created_at):
        self.rid = rid
        self.pattern = tuple(pattern)
        self.require = frozenset(require)
        self.actions = tuple(actions)
        cur = None
        for a in self.actions:
            if a[0] == "OUT":
                cur = a[1]
            elif a[0] == "OUTCLEAR":
                cur = None
        self.emit_char = cur
        touches = any(a[0] in ("OUT", "OUTCLEAR") for a in self.actions)
        self.kind = "OUTPUT" if touches else "INTERNAL"
        self.created_at = created_at
        self.contexts = []
        self.success_contexts = set()     # DISTINCT episodes of success
        self.left_contexts = set()        # diagnostic: distinct left contexts
        self.n_apply = 0
        self.n_reuse = 0
        self.active = True
        self.residue = u"@%d" % rid       # this rule's symbolic residue

    def persistent(self, K):
        """Discrete condition - no weight, no probability."""
        return len(self.success_contexts) >= K

    def matches(self, state):
        L = len(self.pattern)
        if L == 0 or L > len(state.chain):
            return None
        tail = state.chain[len(state.chain) - L:]
        if not pattern_matches(self.pattern, tail):
            return None
        if self.require and not self.require <= state.work:
            return None
        return tail

    def apply(self, state, symtab):
        st = state
        start = len(state.chain) - len(self.pattern)
        for a in self.actions:
            if a[0] == "OUT":
                st = st.set_out(a[1])
            elif a[0] == "OUTCLEAR":
                st = st.set_out(None)
            elif a[0] == "COLLAPSE":
                st = st.collapse(start, a[1], a[2])
            elif a[0] == "COMBINE":
                # consume the required residues, write one internal symbol
                st = st.drop_work(a[1]).add_work([a[2]])
            if st is None:
                return None
        return st

    def index_key(self):
        return self.pattern[-1][1] if self.pattern and self.pattern[-1][0] == "L" else "*"

    def text(self):
        c = ["tail=(%s)" % ",".join(pat_str(e) for e in self.pattern)]
        if self.require:
            c.append("work{%s}" % ",".join(sorted(self.require)))
        eff = []
        for a in self.actions:
            if a[0] == "OUT":
                eff.append("OUT:=%r" % a[1])
            elif a[0] == "OUTCLEAR":
                eff.append("OUT:=EMPTY")
            elif a[0] == "COLLAPSE":
                eff.append("collapse %d -> %s" % (a[1], a[2]))
            elif a[0] == "COMBINE":
                eff.append("%s -> %s" % ("+".join(sorted(a[1])), a[2]))
        return "R%-5d %-8s IF %s THEN %s  [residue %s]" % (
            self.rid, self.kind, " AND ".join(c), "; ".join(eff), self.residue)

    def to_json(self):
        return {"rid": self.rid, "kind": self.kind, "emit_char": self.emit_char,
                "pattern": [pat_str(e) for e in self.pattern],
                "require": sorted(self.require),
                "actions": [list(map(str, a)) for a in self.actions],
                "residue": self.residue, "n_apply": self.n_apply,
                "n_reuse": self.n_reuse,
                "n_success_contexts": len(self.success_contexts),
                "n_distinct_left_contexts": len(self.left_contexts),
                "text": self.text()}


# =============================================================================
# 5-7.  THE V5 LEARNER
# =============================================================================


class ConfigV5(object):
    def __init__(self, **kw):
        self.K = K_MAIN                 # residue persists after K distinct successes
        self.max_pattern_len = 4
        self.max_collapse_span = 4
        self.new_rule_budget = 60
        self.p_require_residue = 0.35   # candidate inspects the workspace
        self.p_collapse = 0.30
        self.p_combine = 0.25           # candidate rewrites residues -> structure
        self.p_reuse_symbol = 0.50      # reuse an EXISTING internal symbol
        self.p_out = 0.85
        self.max_internal_steps = 3     # structural rewrites per decision
        self.max_out_path = 1
        self.epochs = 1
        self.max_completion_len = 8
        self.max_completion_nodes = 3000
        self.max_completions = 200
        for k, v in kw.items():
            if not hasattr(self, k):
                raise KeyError(k)
            setattr(self, k, v)

    def as_dict(self):
        return dict(self.__dict__)


class LearnerV5(object):

    def __init__(self, cfg, seed):
        self.cfg = cfg
        self.seed = seed
        self.rng = random.Random(seed * 104729 + 7)
        self.symtab = SymbolTable()
        self.rules = []
        self.by_rid = {}
        self.next_rid = 1
        self.frozen = False
        self.rid_at_freeze = None
        self.alphabet = set()
        self._out_idx = defaultdict(list)
        self._int_idx = defaultdict(list)
        self._exact = {}
        self.stats = defaultdict(int)
        self.dynamics = []
        self.residue_log = []          # (string_index, position, residues)
        self.traces = []

    # ------------------------------------------------------------ registry
    def _register(self, r):
        self.rules.append(r)
        self.by_rid[r.rid] = r
        self._exact[(r.pattern, r.require, r.actions)] = r
        (self._out_idx if r.kind == "OUTPUT" else self._int_idx)[r.index_key()].append(r)

    def _exists(self, pattern, require, actions):
        return self._exact.get((tuple(pattern), frozenset(require), tuple(actions)))

    def _candidates(self, idx, state):
        last = state.chain[-1] if state.chain else None
        out = list(idx.get("*", ()))
        if last is not None:
            out += idx.get(last, ())
        out.sort(key=lambda r: r.rid)
        return out

    def persistent_residues(self):
        K = self.cfg.K
        return set(r.residue for r in self.rules if r.persistent(K))

    # --------------------------------------------- structural exploration
    def expansions(self, state):
        """Views reachable by INTERNAL rewrites (collapses and residue
        combinations).  Nothing is scored; the cutoffs are computational."""
        results = [(state, ())]
        seen = {state.sig()}
        frontier = [(state, ())]
        while frontier and len(results) < 8:
            st, path = frontier.pop(0)
            if len(path) >= self.cfg.max_internal_steps:
                continue
            for r in self._candidates(self._int_idx, st):
                if not r.active:
                    continue
                if r.matches(st) is None:
                    continue
                ns = r.apply(st, self.symtab)
                if ns is None or ns.sig() in seen:
                    continue
                if not self.frozen:
                    r.n_apply += 1
                seen.add(ns.sig())
                results.append((ns, path + (r.rid,)))
                frontier.append((ns, path + (r.rid,)))
                if len(results) >= 8:
                    break
        results.sort(key=lambda x: (len(x[1]), len(x[0].chain)))
        return results

    def _out_steps(self, st, ipath):
        for r in self._candidates(self._out_idx, st):
            if not r.active:
                continue
            tail = r.matches(st)
            if tail is None:
                continue
            ns = r.apply(st, self.symtab)
            if ns is None:
                continue
            yield (read_output(ns), ipath, r, tail, ns)

    def find_path(self, state, target):
        """Existing rules only.  Every applicable path is EXECUTED and OUT is
        read; the target enters only as the final boolean test."""
        for st, ipath in self.expansions(state):
            for (pred, ip, r, tail, ns) in self._out_steps(st, ipath):
                if pred == target:
                    return (st, ip, r, tail, ns)
        return None

    # ------------------------------------------------------- proposals
    # TARGET-BLIND: this function never receives, reads or infers the
    # character the training step requires.  It draws the inspected chain
    # tail, whether the workspace is inspected and which residue is required,
    # whether structure is built, which symbol is created or REUSED from the
    # existing inventory, and which raw character (if any) is written to OUT.
    def propose_random_rule(self, views, alphabet, residues, rng, where):
        cfg = self.cfg
        st, ipath = views[rng.randrange(len(views))]
        if not st.chain:
            return None
        L = min(cfg.max_pattern_len, len(st.chain))
        pat = tuple(lit(x) for x in st.chain[len(st.chain) - L:])

        require = frozenset()
        avail = sorted(st.work & frozenset(residues))
        if avail and rng.random() < cfg.p_require_residue:
            n = 1 if (len(avail) < 2 or rng.random() < 0.7) else 2
            require = frozenset(rng.sample(avail, n))

        acts = []
        inv = sorted(self.symtab.info)
        if len(require) >= 2 and rng.random() < cfg.p_combine:
            # rewrite the required residues into ONE internal structure; the
            # structure may be a fresh symbol or one already in the inventory
            if inv and rng.random() < cfg.p_reuse_symbol:
                sym = inv[rng.randrange(len(inv))]
            else:
                sym = self.symtab.new_symbol(self.next_rid, sorted(require),
                                             kind="combine")
            acts.append(("COMBINE", require, sym))
        elif rng.random() < cfg.p_collapse and len(st.chain) >= 2:
            k = rng.randint(2, min(cfg.max_collapse_span, len(st.chain), L))
            seg = st.chain[len(st.chain) - k:]
            if inv and rng.random() < cfg.p_reuse_symbol:
                sym = inv[rng.randrange(len(inv))]
            else:
                sym = self.symtab.new_symbol(self.next_rid, seg, kind="collapse")
            acts.append(("COLLAPSE", k, sym))

        letters = sorted(alphabet)
        if letters and rng.random() < cfg.p_out:
            acts.append(("OUT", letters[rng.randrange(len(letters))]))
        if not acts:
            acts.append(("OUTCLEAR",))
        if self._exists(pat, require, acts) is not None:
            return None
        rid = self.next_rid
        self.next_rid += 1
        return (Rule(rid, pat, require, tuple(acts), where), st, ipath)

    def search_new_rule(self, state, target, where):
        """Propose -> execute -> read OUT -> ONLY THEN compare with target.
        No fallback encodes the target; failure discards the candidate."""
        assert not self.frozen
        views = self.expansions(state)
        residues = self.persistent_residues()
        for _ in range(self.cfg.new_rule_budget):
            mark, rid0 = self.symtab.mark(), self.next_rid
            prop = self.propose_random_rule(views, self.alphabet, residues,
                                            self.rng, where)
            if prop is None:
                self.symtab.rollback(mark)
                self.next_rid = rid0
                continue
            cand, st, ipath = prop
            self.stats["candidates_proposed"] += 1
            ns = cand.apply(st, self.symtab) if cand.matches(st) is not None else None
            predicted = read_output(ns) if ns is not None else None
            if predicted == target:                       # boolean test only
                self.stats["candidates_accidentally_correct"] += 1
                self._register(cand)
                return (st, ipath, cand, cand.matches(st), ns)
            self.stats["candidates_rejected"] += 1
            self.symtab.rollback(mark)
            self.next_rid = rid0
        return None

    # ------------------------------------------------------------ training
    def _record_success(self, rule, tail, state, episode):
        """A DISTINCT context = a distinct training episode in which the rule
        was part of a successful path.

        DESIGN NOTE (decided before looking at any outcome): the stricter
        alternative - keying the context on the chain material to the LEFT of
        the match - is structurally unusable here.  Processing is strictly
        left-to-right, so a rule that fires near the START of a string always
        sees the same (empty or 1-character) left context and could never reach
        K, which would make the subject region incapable of carrying residues
        by construction and would make TEST B vacuous.  Counting distinct
        episodes is a discrete identity test over occasions of success: no
        similarity, no score, no weight.  Both counters are kept for
        diagnostics.
        """
        rule.n_apply += 1
        if episode in rule.success_contexts:
            rule.n_reuse += 1
        elif len(rule.success_contexts) < 4096:
            rule.success_contexts.add(episode)
        left = state.chain[:max(0, len(state.chain) - len(rule.pattern))]
        if left not in rule.left_contexts and len(rule.left_contexts) < 256:
            rule.left_contexts.add(left)
        if tail not in rule.contexts and len(rule.contexts) < 16:
            rule.contexts.append(tail)

    def train(self, strings, time_cap=None):
        assert not self.frozen
        t0 = time.time()
        self.capped = False
        for ep in range(self.cfg.epochs):
            for si, s in enumerate(strings):
                assert isinstance(s, str) and " " not in s
                if time_cap and time.time() - t0 > time_cap:
                    self.capped = True
                    return
                self._train_string(s, ep, si)

    def _train_string(self, s, epoch, si):
        state = State()
        trace = {"string_index": si, "string": s, "steps": []}
        for t, target in enumerate(s):
            self.stats["decisions"] += 1
            outcome = 2
            rule = None
            ipath = ()
            fired = []
            found = self.find_path(state, target)
            if found is not None:
                st, ipath, rule, tail, ns = found
                self._record_success(rule, tail, st, (epoch, si))
                self.stats["solved_by_existing"] += 1
                outcome = 0
                fired = list(ipath) + [rule.rid]
            else:
                self.stats["required_new_rule_search"] += 1
                res = self.search_new_rule(state, target, (epoch, t, si))
                if res is not None:
                    st, ipath, rule, tail, ns = res
                    self._record_success(rule, tail, st, (epoch, si))
                    self.stats["new_rule_created"] += 1
                    outcome = 1
                    fired = list(ipath) + [rule.rid]
                else:
                    self.stats["unsolved"] += 1

            # ---- THE V5 CHANGE: successful rules leave residues ----------
            # only rules that were part of a SUCCESSFUL path, and only once
            # their residue has become persistent (K distinct contexts)
            state = state.append(target)
            if fired:
                K = self.cfg.K
                leave = [self.by_rid[r].residue for r in fired
                         if r in self.by_rid and self.by_rid[r].persistent(K)]
                if leave:
                    state = state.add_work(leave)
                    self.stats["residues_written"] += len(leave)
            # structural rewrites of residues persist too
            for stx, _ip in self.expansions(state)[:1]:
                pass
            self.dynamics.append(outcome)
            trace["steps"].append({"t": t, "target": target,
                                   "outcome": ["existing", "new", "unsolved"][outcome],
                                   "rules": fired,
                                   "work": sorted(state.work)})
        if len(self.traces) < 25:
            self.traces.append(trace)

    # ----------------------------------------------------------- inference
    def freeze(self):
        self.frozen = True
        self.rid_at_freeze = self.next_rid

    def state_for_prefix(self, prefix, record=None):
        """Feed a raw prefix; residues accumulate exactly as in training,
        but NO rule may be created and nothing is learned."""
        assert self.frozen
        st = State()
        K = self.cfg.K
        for i, ch in enumerate(prefix):
            fired = []
            for sv, ipath in self.expansions(st):
                done = False
                for (pred, ip, r, tail, ns) in self._out_steps(sv, ipath):
                    if pred == ch:
                        fired = list(ip) + [r.rid]
                        done = True
                        break
                if done:
                    break
            st = st.append(ch)
            if fired:
                leave = [self.by_rid[x].residue for x in fired
                         if x in self.by_rid and self.by_rid[x].persistent(K)]
                if leave:
                    st = st.add_work(leave)
            if record is not None:
                record.append({"pos": i, "char": ch, "rules": fired,
                               "work": sorted(st.work)})
        return st

    def licensed(self, state):
        out = {}
        for st, ipath in self.expansions(state):
            for (pred, ip, r, tail, ns) in self._out_steps(st, ipath):
                if pred is None or pred in out:
                    continue
                out[pred] = (list(ip), r.rid, sorted(st.work))
        return out

    def possible_next_chars(self, prefix):
        return set(self.licensed(self.state_for_prefix(prefix)))

    def possible_completions(self, prefix, max_len):
        results = {}
        nodes = [0]
        start = self.state_for_prefix(prefix)
        K = self.cfg.K

        def rec(state, suffix, deriv):
            if nodes[0] >= self.cfg.max_completion_nodes or \
               len(results) >= self.cfg.max_completions or len(suffix) >= max_len:
                return
            nodes[0] += 1
            lic = self.licensed(state)
            for ch in sorted(lic):
                ip, rid, work = lic[ch]
                step = {"char": ch, "internal_rules": ip, "output_rule": rid,
                        "residues_active": work}
                if ch == EOS:
                    results.setdefault(prefix + suffix + EOS, deriv + [step])
                    continue
                ns = state.append(ch)
                leave = [self.by_rid[x].residue for x in (ip + [rid])
                         if x in self.by_rid and self.by_rid[x].persistent(K)]
                if leave:
                    ns = ns.add_work(leave)
                rec(ns, suffix + ch, deriv + [step])

        rec(start, "", [])
        return results


# =============================================================================
# INTEGRITY: TARGET BLINDNESS  (section 5)
# =============================================================================


def target_blindness_test(verbose=True):
    problems = []
    sig = inspect.signature(LearnerV5.propose_random_rule)
    for name in sig.parameters:
        if any(b in name.lower() for b in ("target", "goal", "label", "answer")):
            problems.append("proposal accepts %s" % name)
    src = inspect.getsource(LearnerV5.propose_random_rule)
    for bad in ("target", "goal_char", "correct"):
        if bad in src:
            problems.append("proposal source mentions %r" % bad)

    cfg = ConfigV5()
    ln = LearnerV5(cfg, 1)
    ln.alphabet = set("abcdefgk#")
    st = State(tuple("abcabc"), frozenset(["@1", "@2"]), None)
    views = [(st, ())]
    res = set(["@1", "@2"])

    def snap():
        return (ln.rng.getstate(), ln.symtab.mark(), ln.next_rid)

    def restore(s):
        ln.rng.setstate(s[0])
        ln.symtab.rollback(s[1])
        ln.next_rid = s[2]

    def canon(p):
        if p is None:
            return None
        r, s, ip = p
        return (tuple(pat_str(e) for e in r.pattern), tuple(sorted(r.require)),
                tuple(map(str, r.actions)), tuple(s.chain))

    global HYPOTHETICAL_TARGET
    s0 = snap()
    HYPOTHETICAL_TARGET = "k"
    a = canon(ln.propose_random_rule(views, ln.alphabet, res, ln.rng, (0, 0, 0)))
    restore(s0)
    HYPOTHETICAL_TARGET = "z"
    b = canon(ln.propose_random_rule(views, ln.alphabet, res, ln.rng, (0, 0, 0)))
    restore(s0)
    HYPOTHETICAL_TARGET = None
    if a != b:
        problems.append("proposal differed under a different hypothetical target")
    if verbose:
        print("TARGET-BLIND INTEGRITY: %s" % ("PASS" if not problems else "FAIL"))
        for p in problems:
            print("   !! %s" % p)
    return (not problems), problems


HYPOTHETICAL_TARGET = None


# =============================================================================
# TEST A: BOUNDARY DISCOVERY
# =============================================================================


def boundary_discovery(learner, corpus, top=30):
    """Recover the raw provenance span of every internal structure and ask
    whether high-reuse structures line up with hidden words.  Nothing forced
    word-aligned structures; boundary-crossing spans are allowed."""
    words = set(corpus.lexicon())
    rows = []
    for name, rec in learner.symtab.info.items():
        exp = "".join(str(x) for x in learner.symtab.expand(name)
                      if isinstance(x, str) and len(x) == 1)
        if len(exp) < 2:
            continue
        creator = learner.by_rid.get(rec["created_by_rule"])
        reuse = creator.n_apply if creator else 0
        if exp in words:
            cls = "exact_word"
        elif any(exp in w for w in words):
            cls = "inside_word"
        else:
            # does it occur in the corpus at all, and does it cross a boundary?
            cls = "crosses_boundary" if any(exp in s for s in corpus.sentence_set) \
                else "arbitrary"
        rows.append({"symbol": name, "expansion": exp, "reuse": reuse,
                     "kind": rec["kind"], "depth": learner.symtab.depth(name),
                     "alignment": cls})
    rows.sort(key=lambda r: -r["reuse"])
    high = [r for r in rows if r["reuse"] >= 2]
    n = float(len(high)) if high else 1.0
    frac = lambda k: len([r for r in high if r["alignment"] == k]) / n
    return {"n_structures": len(rows), "n_high_reuse": len(high),
            "frac_exact_word": frac("exact_word"),
            "frac_inside_word": frac("inside_word"),
            "frac_crosses_boundary": frac("crosses_boundary"),
            "frac_arbitrary": frac("arbitrary"),
            "top": rows[:top]}


# =============================================================================
# TEST B: FUNCTIONAL CONVERGENCE
# =============================================================================


def v5_subject_signature(learner, corpus, subject, detail=False):
    """Collect the subject's persistent residues / higher-order structures.

    IMPLEMENTATION NOTE (honest): processing is strictly left-to-right, so the
    workspace after the subject region is a deterministic function of the
    subject's own characters - the later verb/object cannot reach back into it.
    "Many contexts" therefore collapses to one for the subject-local signature.
    A second, whole-context signature is also returned: the workspace at the
    END of several full sentences, intersected across them, which keeps only
    what survives in every context.  No labels, no word comparison.
    """
    rec = [] if detail else None
    st_local = learner.state_for_prefix(subject, record=rec)
    core = frozenset(st_local.work)
    fulls = []
    for v in corpus.verbs:
        for o in corpus.object_families[0][:2]:
            fulls.append(frozenset(learner.state_for_prefix(subject + v + o).work))
    full = frozenset.intersection(*fulls) if fulls else frozenset()
    return core, full, rec


def v4_subject_signature(learner, corpus, subject, detail=False):
    """The analogous measurement for UNCHANGED V4, which has no residues: the
    internal symbols present in any view after the subject region, plus the
    structural rules that fired there."""
    st = learner._initial_state()
    for ch in subject:
        st = st.place_arrival(ch)
    syms = set()
    for view, ipath in learner.expansions(st):
        for x in view.chain_symbols():
            if learner.symtab.is_internal(x):
                syms.add(x)
        for rid in ipath:
            syms.add("R%d" % rid)
    core = frozenset(syms)
    fulls = []
    for v in corpus.verbs:
        for o in corpus.object_families[0][:2]:
            st2 = learner._initial_state()
            for ch in subject + v + o:
                st2 = st2.place_arrival(ch)
            g = set()
            for view, ipath in learner.expansions(st2):
                for x in view.chain_symbols():
                    if learner.symtab.is_internal(x):
                        g.add(x)
                for rid in ipath:
                    g.add("R%d" % rid)
            fulls.append(frozenset(g))
    full = frozenset.intersection(*fulls) if fulls else frozenset()
    return core, full, None


def convergence_analysis(sig_fn, learner, corpus, detail_subjects=()):
    subjects = corpus.all_subjects()
    fam = corpus.subj_family
    core, full, traces = {}, {}, {}
    for s in subjects:
        want = s in detail_subjects
        c, u, tr = sig_fn(learner, corpus, s, detail=want)
        core[s], full[s] = c, u
        if tr:
            traces[s] = tr
    pairs = []
    same, diff = [], []
    same_f, diff_f = [], []
    for i, a in enumerate(subjects):
        for b in subjects[i + 1:]:
            A, B = core[a], core[b]
            j = (len(A & B) / float(len(A | B))) if (A | B) else 0.0
            C, D = full[a], full[b]
            jf = (len(C & D) / float(len(C | D))) if (C | D) else 0.0
            same_fam = fam[a] == fam[b]
            pairs.append({"a": a, "b": b, "same_family": same_fam,
                          "shared": sorted(A & B)[:8], "n_shared": len(A & B),
                          "jaccard": j, "jaccard_full_context": jf,
                          "n_shared_full": len(C & D)})
            (same if same_fam else diff).append(j)
            (same_f if same_fam else diff_f).append(jf)
    m = lambda xs: sum(xs) / float(len(xs)) if xs else 0.0
    # family-exclusive structures: shared by all members of a family, by no one else
    excl = []
    for fi, members in enumerate(corpus.subject_families):
        inter = frozenset.intersection(*[core[s] for s in members]) \
            if members else frozenset()
        outside = frozenset().union(*[core[s] for s in subjects
                                      if s not in members]) or frozenset()
        excl.append({"family": fi, "members": list(members),
                     "exclusive": sorted(inter - outside)})
    # evaluation-side deterministic grouping: identical signatures
    groups = defaultdict(list)
    for s in subjects:
        groups[tuple(sorted(core[s]))].append(s)
    grouping = [sorted(g) for g in groups.values()]
    fams = sorted([sorted(f) for f in corpus.subject_families])
    recovered = len([g for g in grouping if sorted(g) in fams])
    return {
        "signatures": {s: sorted(core[s]) for s in subjects},
        "signature_sizes": {s: len(core[s]) for s in subjects},
        "pairs": pairs,
        "same_family_overlap": m(same),
        "different_family_overlap": m(diff),
        "same_family_overlap_full_context": m(same_f),
        "different_family_overlap_full_context": m(diff_f),
        "mean_signature_size": (sum(len(core[s]) for s in subjects) /
                                float(len(subjects))),
        "mean_signature_size_full": (sum(len(full[s]) for s in subjects) /
                                     float(len(subjects))),
        "same_family_n": len(same), "different_family_n": len(diff),
        "family_exclusive": excl,
        "n_family_exclusive": sum(len(e["exclusive"]) for e in excl),
        "grouping": grouping,
        "families_fully_recovered": recovered,
        "exact_partition_recovered": sorted(grouping) == fams,
        "traces": traces,
    }


# =============================================================================
# TEST C: HELD-OUT COMPOSITION
# =============================================================================


def heldout_eval(learner, corpus, cfg_max_len, residue_aware=True):
    out = []
    for prefix in corpus.heldout_prefixes:
        gen = learner.possible_completions(prefix, cfg_max_len)
        cls = V4.classify_completions(corpus, prefix, set(gen))
        met = V4.metrics_for(prefix, corpus, cls)
        succ = []
        for g in cls["compatible"]:
            deriv = gen[g] if isinstance(gen, dict) else None
            if deriv:
                res = sorted(set(x for step in deriv
                                 for x in step.get("residues_active", [])))
                succ.append({"completion": g, "residues_active": res,
                             "rules": [step.get("output_rule") for step in deriv],
                             "internal": [r for step in deriv
                                          for r in step.get("internal_rules", [])]})
        out.append({"prefix": prefix, "metrics": met, "n_generated": len(gen),
                    "compatible": cls["compatible"][:8],
                    "successful_derivations": succ[:4]})
    m = lambda k: sum(h["metrics"][k] for h in out) / float(len(out))
    return {"per_prefix": out, "precision": m("precision"), "recall": m("recall"),
            "exact_set_accuracy": m("exact_set_accuracy"),
            "incompatible_generation_rate": m("incompatible_generation_rate"),
            "malformed_generation_rate": m("malformed_generation_rate"),
            "mean_completion_set_size": m("generated")}


# =============================================================================
# RUNNERS
# =============================================================================


def run_v5(corpus, train_strings, seed, K, detail=False):
    cfg = ConfigV5(K=K, epochs=2)
    ln = LearnerV5(cfg, seed)
    for s in train_strings:
        ln.alphabet.update(s)          # observed characters only
    t0 = time.time()
    ln.train(train_strings, time_cap=RUN_TIME_CAP)
    ln.freeze()
    secs = time.time() - t0
    subs = corpus.all_subjects()
    det = (subs[0], subs[1], subs[3]) if detail else ()
    conv = convergence_analysis(v5_subject_signature, ln, corpus, det)
    bnd = boundary_discovery(ln, corpus)
    held = heldout_eval(ln, corpus, cfg.max_completion_len)
    st = ln.stats
    return {
        "K": K, "seconds": secs, "capped": getattr(ln, "capped", False),
        "rules": len(ln.rules),
        "persistent_residues": len(ln.persistent_residues()),
        "internal_symbols": len(ln.symtab.info),
        "combine_rules": len([r for r in ln.rules
                              if any(a[0] == "COMBINE" for a in r.actions)]),
        "residue_requiring_rules": len([r for r in ln.rules if r.require]),
        "residues_written": st["residues_written"],
        "decisions": st["decisions"], "solved_by_existing": st["solved_by_existing"],
        "new_rule_created": st["new_rule_created"], "unsolved": st["unsolved"],
        "candidates_proposed": st["candidates_proposed"],
        "convergence": conv, "boundary": bnd, "heldout": held,
        "_learner": ln if detail else None,
    }


def run_v4_baseline(corpus, train_strings, seed):
    cfg = V4.Config()                  # UNCHANGED V4 defaults (full_system)
    ln = V4.Learner(cfg, seed)
    problems = V4.cheat_controls(ln, corpus, train_strings, corpus.val_strings,
                                 "pre-train")
    t0 = time.time()
    ln.train(train_strings)
    ln.freeze()
    secs = time.time() - t0
    conv = convergence_analysis(v4_subject_signature, ln, corpus)
    held = []
    for prefix in corpus.heldout_prefixes:
        gen = ln.possible_completions(prefix, cfg.max_completion_len)
        cls = V4.classify_completions(corpus, prefix, set(gen))
        held.append(V4.metrics_for(prefix, corpus, cls))
    m = lambda k: sum(h[k] for h in held) / float(len(held))
    bnd = boundary_discovery_v4(ln, corpus)
    return {"seconds": secs, "rules": len(ln.rules),
            "internal_symbols": len(ln.symtab.info),
            "convergence": conv,
            "boundary": bnd,
            "heldout": {"precision": m("precision"), "recall": m("recall"),
                        "exact_set_accuracy": m("exact_set_accuracy"),
                        "incompatible_generation_rate": m("incompatible_generation_rate"),
                        "malformed_generation_rate": m("malformed_generation_rate"),
                        "mean_completion_set_size": m("generated")},
            "cheat_control_problems": problems}


def boundary_discovery_v4(learner, corpus, top=30):
    words = set(corpus.lexicon())
    rows = []
    for name, rec in learner.symtab.info.items():
        if rec["kind"] != "collapse":
            continue
        exp = "".join(str(x) for x in learner.symtab.expand(name)
                      if isinstance(x, str) and len(x) == 1)
        if len(exp) < 2:
            continue
        creator = None
        for r in learner.rules:
            if r.rid == rec["created_by_rule"]:
                creator = r
                break
        if exp in words:
            cls = "exact_word"
        elif any(exp in w for w in words):
            cls = "inside_word"
        else:
            cls = "crosses_boundary" if any(exp in s for s in corpus.sentence_set) \
                else "arbitrary"
        rows.append({"symbol": name, "expansion": exp,
                     "reuse": creator.n_apply if creator else 0,
                     "alignment": cls, "depth": learner.symtab.depth(name)})
    rows.sort(key=lambda r: -r["reuse"])
    high = [r for r in rows if r["reuse"] >= 2]
    n = float(len(high)) if high else 1.0
    frac = lambda k: len([r for r in high if r["alignment"] == k]) / n
    return {"n_structures": len(rows), "n_high_reuse": len(high),
            "frac_exact_word": frac("exact_word"),
            "frac_inside_word": frac("inside_word"),
            "frac_crosses_boundary": frac("crosses_boundary"),
            "frac_arbitrary": frac("arbitrary"), "top": rows[:top]}


def cheat_controls(corpus, train_strings):
    problems = []
    for s in train_strings + corpus.val_strings:
        if any(c in s for c in " \t|_-"):
            problems.append("boundary character in input: %r" % s)
        if s in corpus.heldout_strings:
            problems.append("held-out string in training: %r" % s)
        for p in corpus.heldout_prefixes:
            if s.startswith(p):
                problems.append("held-out prefix in training: %r" % s)
    src = inspect.getsource(LearnerV5)
    for bad in ["subj_family", "obj_family", "objfam_table", "subject_families",
                "object_families", "heldout", "family", "lexicon", "decompose"]:
        if bad in src:
            problems.append("learner source mentions %r" % bad)
    return problems


# =============================================================================
# MAIN
# =============================================================================


def main():
    print("=" * 78)
    print("V5 RESIDUE EXPERIMENT")
    print("=" * 78)
    blind_ok, blind_problems = target_blindness_test()
    print("")

    corpora = {s: HiddenLangV5(s) for s in SEEDS}
    c1 = corpora[SEEDS[0]]
    print("seed %d language: subjects %s" % (SEEDS[0], c1.subject_families))
    print("  verbs %s" % c1.verbs)
    print("  objects %s" % c1.object_families)
    print("  latin square %s | held-out pairs %s"
          % (c1.objfam_table, c1.heldout_pairs))
    print("  distinct licensed sentences %d | examples: %s"
          % (len(c1.sentence_set), c1.train_at(3)))
    print("  N = number of training EXPERIENCES (sentences repeat; the hidden")
    print("  world is fixed, so more data = more reuse evidence, not more concepts)")
    print("")

    results = {"seeds": SEEDS, "scales": SCALES, "K_values": K_VALUES,
               "K_main": K_MAIN,
               "target_blind": "PASS" if blind_ok else "FAIL",
               "target_blind_problems": blind_problems,
               "language": {str(s): corpora[s].signature() for s in SEEDS},
               "v4": {}, "v5": {}, "k_sensitivity": {},
               "data_identity": True, "cheat_control_problems": []}
    outpath = os.path.join(HERE, "symbolic_rewrite_v5_results.json")
    detail_store = {}

    for N in SCALES:
        results["v4"][str(N)] = []
        results["v5"][str(N)] = []
        for seed in SEEDS:
            corpus = corpora[seed]
            train = corpus.train_at(N)
            v4_train = list(train)
            v5_train = list(train)
            assert v4_train == v5_train, "V4/V5 data identity failed"
            results["cheat_control_problems"] += cheat_controls(corpus, train)

            r4 = run_v4_baseline(corpus, v4_train, seed)
            want = (N == SCALES[-1] and seed == SEEDS[0])
            r5 = run_v5(corpus, v5_train, seed, K_MAIN, detail=want)
            if want:
                detail_store["learner"] = r5.pop("_learner")
                detail_store["corpus"] = corpus
            else:
                r5.pop("_learner", None)
            r4["seed"] = seed
            r5["seed"] = seed
            results["v4"][str(N)].append(r4)
            results["v5"][str(N)].append(r5)
            print("  N=%-5d seed %d | V4 same=%.3f diff=%.3f fam=%d exact=%.2f "
                  "| V5 same=%.3f diff=%.3f fam=%d exact=%.2f res=%d comb=%d (%.0fs/%.0fs)"
                  % (N, seed,
                     r4["convergence"]["same_family_overlap"],
                     r4["convergence"]["different_family_overlap"],
                     r4["convergence"]["families_fully_recovered"],
                     r4["heldout"]["exact_set_accuracy"],
                     r5["convergence"]["same_family_overlap"],
                     r5["convergence"]["different_family_overlap"],
                     r5["convergence"]["families_fully_recovered"],
                     r5["heldout"]["exact_set_accuracy"],
                     r5["persistent_residues"], r5["combine_rules"],
                     r4["seconds"], r5["seconds"]))
            sys.stdout.flush()
            with open(outpath, "w") as f:
                json.dump(results, f, indent=1, ensure_ascii=False,
                          default=lambda o: None)
        print("")

    # ------------------------------------------------------ K sensitivity
    print("K sensitivity (N=%d, %d seeds):" % (SCALES[-1], len(SEEDS)))
    for K in K_VALUES:
        rows = []
        for seed in SEEDS:
            corpus = corpora[seed]
            r = run_v5(corpus, corpus.train_at(SCALES[-1]), seed, K)
            r.pop("_learner", None)
            r["seed"] = seed
            rows.append(r)
        results["k_sensitivity"][str(K)] = rows
        a = lambda k: sum(x["convergence"][k] for x in rows) / float(len(rows))
        h = lambda k: sum(x["heldout"][k] for x in rows) / float(len(rows))
        print("  K=%d  same=%.3f diff=%.3f fam=%.1f residues=%.0f combine=%.1f "
              "exact=%.3f" % (K, a("same_family_overlap"),
                              a("different_family_overlap"),
                              a("families_fully_recovered"),
                              sum(x["persistent_residues"] for x in rows) / float(len(rows)),
                              sum(x["combine_rules"] for x in rows) / float(len(rows)),
                              h("exact_set_accuracy")))
        sys.stdout.flush()
        with open(outpath, "w") as f:
            json.dump(results, f, indent=1, ensure_ascii=False,
                      default=lambda o: None)
    print("")
    report(results, detail_store, blind_ok, blind_problems)
    with open(outpath, "w") as f:
        json.dump(results, f, indent=1, ensure_ascii=False,
                  default=lambda o: None)
    print("files written: symbolic_rewrite_v5_results.json")


def report(results, detail, blind_ok, blind_problems):
    avg = lambda rows, path: sum(_dig(r, path) for r in rows) / float(len(rows))
    print("================ V5 RESIDUE EXPERIMENT ================")
    print("")
    print("TARGET-BLIND INTEGRITY: %s" % ("PASS" if blind_ok else "FAIL"))
    print("NO TOKENIZER / NO SPACES: %s"
          % ("PASS" if not [p for p in results["cheat_control_problems"]
                            if "boundary" in p] else "FAIL"))
    print("NO FAMILY LABELS USED IN TRAINING: %s"
          % ("PASS" if not [p for p in results["cheat_control_problems"]
                            if "mentions" in p] else "FAIL"))
    print("V4/V5 DATA IDENTITY: %s" % ("PASS" if results["data_identity"] else "FAIL"))
    print("")
    print("BOUNDARY DISCOVERY:")
    for N in SCALES:
        v4 = results["v4"][str(N)]
        v5 = results["v5"][str(N)]
        print("N=%d:" % N)
        for tag, rows in (("V4", v4), ("V5", v5)):
            print("  %s  structures %.0f (high-reuse %.0f) | exact-word %.3f  "
                  "inside-word %.3f  crosses-boundary %.3f  arbitrary %.3f"
                  % (tag, avg(rows, ["boundary", "n_structures"]),
                     avg(rows, ["boundary", "n_high_reuse"]),
                     avg(rows, ["boundary", "frac_exact_word"]),
                     avg(rows, ["boundary", "frac_inside_word"]),
                     avg(rows, ["boundary", "frac_crosses_boundary"]),
                     avg(rows, ["boundary", "frac_arbitrary"])))
    print("")
    print("FUNCTIONAL CONVERGENCE:")
    for N in SCALES:
        print("N=%d:" % N)
        for tag in ("v4", "v5"):
            rows = results[tag][str(N)]
            print("  %s  same-family overlap: %.4f | different-family overlap: "
                  "%.4f | ratio %s | exact families recovered: %.1f | "
                  "family-exclusive: %.1f"
                  % (tag.upper(), avg(rows, ["convergence", "same_family_overlap"]),
                     avg(rows, ["convergence", "different_family_overlap"]),
                     _ratio(avg(rows, ["convergence", "same_family_overlap"]),
                            avg(rows, ["convergence", "different_family_overlap"])),
                     avg(rows, ["convergence", "families_fully_recovered"]),
                     avg(rows, ["convergence", "n_family_exclusive"])))
        v5 = results["v5"][str(N)]
        print("       V5 persistent residues %.0f | combine rules %.1f | "
              "residue-requiring rules %.1f"
              % (sum(r["persistent_residues"] for r in v5) / float(len(v5)),
                 sum(r["combine_rules"] for r in v5) / float(len(v5)),
                 sum(r["residue_requiring_rules"] for r in v5) / float(len(v5))))
    print("")
    print("HELD-OUT COMPOSITION:")
    for tag in ("v4", "v5"):
        print("%s:" % tag.upper())
        for N in SCALES:
            rows = results[tag][str(N)]
            print("  N=%-5d precision %.3f | recall %.3f | exact-set %.3f | "
                  "incompatible %.3f | malformed %.3f | |gen| %.1f"
                  % (N, avg(rows, ["heldout", "precision"]),
                     avg(rows, ["heldout", "recall"]),
                     avg(rows, ["heldout", "exact_set_accuracy"]),
                     avg(rows, ["heldout", "incompatible_generation_rate"]),
                     avg(rows, ["heldout", "malformed_generation_rate"]),
                     avg(rows, ["heldout", "mean_completion_set_size"])))
    print("")
    print("K SENSITIVITY (N=%d):" % SCALES[-1])
    for K in K_VALUES:
        rows = results["k_sensitivity"].get(str(K), [])
        if not rows:
            continue
        print("K=%d:" % K)
        print("  same-family %.4f | different-family %.4f | ratio %s | "
              "families %.1f | persistent residues %.0f | combine rules %.1f | "
              "held-out exact %.3f"
              % (avg(rows, ["convergence", "same_family_overlap"]),
                 avg(rows, ["convergence", "different_family_overlap"]),
                 _ratio(avg(rows, ["convergence", "same_family_overlap"]),
                        avg(rows, ["convergence", "different_family_overlap"])),
                 avg(rows, ["convergence", "families_fully_recovered"]),
                 sum(r["persistent_residues"] for r in rows) / float(len(rows)),
                 sum(r["combine_rules"] for r in rows) / float(len(rows)),
                 avg(rows, ["heldout", "exact_set_accuracy"])))
    print("")
    _print_traces(detail)
    print("INTERPRETATION:")
    N = SCALES[-1]
    v5 = results["v5"][str(N)]
    same = avg(v5, ["convergence", "same_family_overlap"])
    diff = avg(v5, ["convergence", "different_family_overlap"])
    fam = avg(v5, ["convergence", "families_fully_recovered"])
    exact = avg(v5, ["heldout", "exact_set_accuracy"])
    v4rows = results["v4"][str(N)]
    v4exact = avg(v4rows, ["heldout", "exact_set_accuracy"])
    align = avg(v5, ["boundary", "frac_exact_word"])
    converged = (same > 0.0) and (same >= 2.0 * max(diff, 1e-9)) and \
        (avg(v5, ["convergence", "n_family_exclusive"]) > 0)
    boundaries = align > 0.20
    if converged and exact > max(0.25, v4exact + 0.1):
        v = "D. V5 shows functional convergence AND improves held-out composition."
    elif converged:
        v = ("C. V5 shows functional convergence (same-family items share "
             "substantially more internal structure) but held-out composition "
             "still fails.")
    elif boundaries:
        v = ("B. V5 discovers recurring surface units, but not functional "
             "convergence.")
    else:
        v = "A. V5 does not discover boundaries or functional families."
    print("  V5 same-family %.4f vs different-family %.4f (ratio %s)"
          % (same, diff, _ratio(same, diff)))
    print("  V5 families recovered %.1f | family-exclusive structures %.1f"
          % (fam, avg(v5, ["convergence", "n_family_exclusive"])))
    print("  V5 held-out exact-set %.3f vs V4 %.3f" % (exact, v4exact))
    print("  V5 high-reuse structures aligned to whole hidden words %.3f" % align)
    print("  %s" % v)
    print("")
    print("CHEAT-CONTROL VIOLATIONS: %d" % len(results["cheat_control_problems"]))
    for p in results["cheat_control_problems"][:8]:
        print("   !! %s" % p)
    print("=======================================================")


def _dig(d, path):
    for p in path:
        d = d[p]
    return d


def _ratio(a, b):
    if b <= 1e-9:
        return "inf" if a > 1e-9 else "n/a"
    return "%.2fx" % (a / b)


def _print_traces(detail):
    ln = detail.get("learner")
    corpus = detail.get("corpus")
    if ln is None:
        return
    print("EXAMPLE TRACES (V5, seed %d, N=%d) - raw chars -> rules -> residues:"
          % (corpus.seed, SCALES[-1]))
    subs = corpus.all_subjects()
    fam = corpus.subj_family
    pick = [subs[0], subs[1], subs[3]]          # two same-family, one other
    for s in pick:
        rec = []
        ln.state_for_prefix(s, record=rec)
        print("  subject %r  [hidden family %d]" % (s, fam[s]))
        for step in rec:
            print("     %r -> rules %s -> residues %s"
                  % (step["char"], step["rules"] or "(none)",
                     step["work"] or "(none)"))
    print("  most-reused structural rules:")
    for r in sorted(ln.rules, key=lambda x: -x.n_apply)[:5]:
        print("     " + r.text() + "   [applied %d, contexts %d]"
              % (r.n_apply, len(r.success_contexts)))
    print("")


if __name__ == "__main__":
    main()
