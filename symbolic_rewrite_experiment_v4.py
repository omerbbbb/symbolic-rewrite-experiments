#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
symbolic_rewrite_experiment_v4.py

V4 of the purely discrete, symbolic autoregressive learner over raw character
strings.  Patched from symbolic_rewrite_experiment.py (V3); the architecture,
the raw-character input, the random 2D grid, the arrival links, the transparent
internal symbols, the rewrite rules, the provenance/explain_symbol machinery,
the frozen inference, the evaluation framework, the multi-seed runner, the
ablations and the JSON outputs are all preserved.

NO numerical learning of any kind is used inside the learner:
  no tokenizer, no embeddings, no neural nets, no gradients, no probabilities,
  no similarity scores, no vector distances, no softmax, no learned floats.
The learner's entire state is:
  * a 2D grid of discrete symbol cells (including one reserved OUT register),
  * symbolic previous/next links between arrivals,
  * a set of discrete rewrite rules,
  * a provenance table for internally created symbols.

Run:   python3 symbolic_rewrite_experiment_v4.py
Saves: results_v4.json, rules_v4.json, traces_v4.json

Standard library only.
"""

import inspect
import json
import os
import random
import sys
from collections import defaultdict, deque

# =============================================================================
# SECTION 0.  CONFIGURATION
# =============================================================================


class Config(object):
    """All computational cutoffs live here.

    IMPORTANT: every limit below is a *computational* cutoff whose only purpose
    is to keep the program finite.  None of them is a model score, a
    hyper-parameter of a learned function, or a ranking criterion.
    """

    def __init__(self, **kw):
        # ---- grid ----------------------------------------------------------
        self.grid_size = 16                # 16 x 16 symbolic grid (configurable)

        # ---- rule capacity (configurable; can be raised to whole grid) ------
        self.max_read_cells = 8            # max locations a rule may inspect
        self.max_write_cells = 8           # max locations a rule may modify
        self.max_pattern_len = 4           # linked-tail positions inspected
        self.max_collapse_span = 5         # max linked positions rewritten at once

        # ---- symbolic search cutoffs ---------------------------------------
        self.max_internal_depth = 2        # depth of epsilon-rewrite exploration
        self.max_expansion_states = 8      # branching cap during exploration
        self.max_out_path = 2              # max rewrite steps that touch OUT

        # ---- new-rule random search (V4: target-blind) ----------------------
        self.new_rule_budget = 60          # safety budget of random candidates
        self.p_propose_collapse = 0.7      # chance a candidate proposes structure
        self.p_propose_out = 0.85          # chance a candidate touches OUT
        self.p_propose_out_pair = 0.15     # chance it writes OUT twice / erases
        self.p_propose_marker = 0.12       # chance a candidate writes a marker
        self.p_propose_require = 0.10      # chance a candidate adds a presence test

        # ---- generalization -------------------------------------------------
        self.allow_generalization = True
        self.allow_reuse = True            # reuse in *new* contexts
        self.allow_composition = True      # internal symbols / hierarchy
        self.counterexample_check = True   # V4: negative-evidence veto
        self.max_relax_positions = 1       # positions wildcarded per event
        self.max_contexts_per_rule = 16

        # ---- training -------------------------------------------------------
        self.epochs = 2
        self.shuffle_input = False         # shuffled-character control

        # ---- inference ------------------------------------------------------
        self.max_completion_len = 8
        self.max_completion_nodes = 3000
        self.max_completions = 200

        for k, v in kw.items():
            if not hasattr(self, k):
                raise KeyError("unknown config key: %s" % k)
            setattr(self, k, v)

    def as_dict(self):
        return dict(self.__dict__)


EOS = "#"       # end-of-sequence character (allowed by the specification)
BOS = "^"       # start-of-sequence marker placed in the grid at init.
#               NOTE: BOS is *not* a hidden separator; it marks only the start
#               of the whole sequence, never a word boundary.  There is exactly
#               one BOS per string, placed before any character arrives.

WILD = ("A",)                     # wildcard pattern element


def lit(sym):
    return ("L", sym)


def var(name):
    return ("V", name)            # supported by the matcher (variable binding)


def pat_str(elem):
    if elem[0] == "L":
        return repr(elem[1])
    if elem[0] == "V":
        return "?%s" % elem[1]
    return "*"


def pattern_matches(pattern, tail):
    """Discrete match of a rule pattern against a concrete symbol tuple."""
    if len(pattern) != len(tail):
        return False
    bind = {}
    for pe, s in zip(pattern, tail):
        t = pe[0]
        if t == "L":
            if pe[1] != s:
                return False
        elif t == "V":
            if pe[1] in bind:
                if bind[pe[1]] != s:
                    return False
            else:
                bind[pe[1]] = s
    return True


# =============================================================================
# SECTION 1.  HIDDEN SYNTHETIC LANGUAGE (GENERATOR METADATA)
#
# This object holds the hidden classes.  It is NEVER passed to the learner and
# never referenced from learner code.  A cheat-control checks both facts.
#
# V4 (CHANGE 6 + CHANGE 8): the language is genuinely compositional.  The verbs
# are SHARED by all subject families, and the licensed object family is a
# function of the PAIR (subject family, verb) laid out as a Latin square, so the
# verb alone determines nothing.  A variable neutral filler word is inserted
# between the subject and the verb (CHANGE 8, option A), so the subject is never
# locally adjacent to the position where the object must be decided.
# =============================================================================

CONSONANTS = "bdfgklmnprstvz"
VOWELS = "aeiou"


def _make_word(rng, nsyl, coda):
    w = ""
    for _ in range(nsyl):
        w += rng.choice(CONSONANTS) + rng.choice(VOWELS)
    if coda:
        w += rng.choice("nrslm")
    return w


class HiddenCorpus(object):
    """Generates the synthetic language.  Knows the hidden classes.

    Hidden structure (learner never sees any of it):
        * 3 subject families, 3 subjects each
        * 3 verbs, SHARED by every subject family
        * 3 object families, 4 objects each
        * 3 neutral filler words, shared by everything, carrying no information
        * compatibility (Latin square):
              objfam( subject_family i , verb j ) = (i + j) % 3
          so neither the subject alone nor the verb alone determines the
          licensed object family - only their combination does.
        * surface form:   subject + filler + verb + object + '#'
    """

    def __init__(self, seed):
        rng = random.Random(seed * 7919 + 13)
        self.seed = seed
        self._access_count = 0

        words = set()

        def fresh(family_words):
            # variable-length nonsense words (4, 5 or 6 characters) so that no
            # boundary can be inferred from a fixed length.
            for _ in range(8000):
                nsyl = rng.choice([2, 2, 3])
                coda = rng.random() < 0.4 and nsyl == 2
                w = _make_word(rng, nsyl, coda)
                if w in words:
                    continue
                # no word may be a prefix of another word (keeps the synthetic
                # language uniquely segmentable in principle; documented choice)
                bad = False
                for u in words:
                    if u.startswith(w) or w.startswith(u):
                        bad = True
                        break
                if bad:
                    continue
                # CHANGE 6: family membership must NOT be encoded by a shared
                # affix, so no two words of the same family may share a 2-char
                # prefix or a 2-char suffix.
                for u in family_words:
                    if u[:2] == w[:2] or u[-2:] == w[-2:]:
                        bad = True
                        break
                if bad:
                    continue
                words.add(w)
                return w
            raise RuntimeError("lexicon generation failed")

        def fresh_family(n):
            fam = []
            for _ in range(n):
                fam.append(fresh(fam))
            return fam

        self.subject_families = [fresh_family(3) for _ in range(3)]
        self.verbs = fresh_family(3)                 # SHARED across families
        self.object_families = [fresh_family(4) for _ in range(3)]
        self.fillers = fresh_family(3)               # neutral, unlabelled

        # Latin square: the (subject family, verb) PAIR selects the object family
        self.objfam_table = [[(fi + vi) % 3 for vi in range(3)] for fi in range(3)]

        self.subj_family = {}
        for fi, fam in enumerate(self.subject_families):
            for s in fam:
                self.subj_family[s] = fi
        self.obj_family = {}
        for fi, fam in enumerate(self.object_families):
            for o in fam:
                self.obj_family[o] = fi
        self.verb_index = {v: i for i, v in enumerate(self.verbs)}

        # ---- full grammatical language -------------------------------------
        self.all_quads = []                 # (subject, verb, object)
        for fi in range(3):
            for s in self.subject_families[fi]:
                for vi, v in enumerate(self.verbs):
                    for o in self.object_families[self.objfam_table[fi][vi]]:
                        self.all_quads.append((s, v, o))

        # ---- held-out COMBINATIONS (entire (subject, verb) pairs) -----------
        # CHANGE 7: one held-out pair per subject family, each with a different
        # verb, so both the subject and the verb remain observable in other
        # combinations and only the COMBINATION is unseen.
        # The three held-out pairs are additionally required to license three
        # DIFFERENT object families, so that no constant answer can score well.
        perms = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]]
        good = [p for p in perms
                if len(set(self.objfam_table[fi][p[fi]] for fi in range(3))) == 3]
        vperm = good[rng.randrange(len(good))]
        self.heldout_pairs = []
        for fi in range(3):
            s = self.subject_families[fi][rng.randrange(3)]
            v = self.verbs[vperm[fi]]
            self.heldout_pairs.append((s, v))
        self.heldout_pair_set = set(self.heldout_pairs)

        # canonical filler used to build the held-out probe prefixes
        self.probe_filler = self.fillers[0]

        self.heldout_compatible = {}     # prefix -> set of compatible strings
        self.heldout_incompatible = {}   # prefix -> set of incompatible strings
        for (s, v) in self.heldout_pairs:
            fi = self.subj_family[s]
            vi = self.verb_index[v]
            good = self.objfam_table[fi][vi]
            prefix = s + self.probe_filler + v
            self.heldout_compatible[prefix] = set(
                prefix + o + EOS for o in self.object_families[good])
            bad = set()
            for gi in range(3):
                if gi == good:
                    continue
                for o in self.object_families[gi]:
                    bad.add(prefix + o + EOS)
            self.heldout_incompatible[prefix] = bad

        self.heldout_prefixes = sorted(self.heldout_compatible)

        # every surface string of a held-out (subject, verb) pair is excluded
        # from training AND validation, for every filler and every object.
        self.heldout_strings = set()
        for (s, v) in self.heldout_pairs:
            fi = self.subj_family[s]
            vi = self.verb_index[v]
            for f in self.fillers:
                for o in self.object_families[self.objfam_table[fi][vi]]:
                    self.heldout_strings.add(s + f + v + o + EOS)

        # ---- corpus ---------------------------------------------------------
        remaining = []
        for (s, v, o) in self.all_quads:
            if (s, v) in self.heldout_pair_set:
                continue
            f = self.fillers[rng.randrange(len(self.fillers))]
            remaining.append(s + f + v + o + EOS)
        remaining = sorted(set(remaining))
        rng.shuffle(remaining)
        cut = int(len(remaining) * 0.8)
        self.train_strings = sorted(remaining[:cut])
        self.val_strings = sorted(remaining[cut:])

        self._sanity()

    # ---- guarded accessors (used ONLY by the evaluation harness) ------------
    def touch(self):
        self._access_count += 1

    @property
    def access_count(self):
        return self._access_count

    def _sanity(self):
        # CHANGE 7: enough indirect evidence must remain in TRAINING
        for (s, v) in self.heldout_pairs:
            fi = self.subj_family[s]
            # the held-out subject appears with the other verbs
            assert any(t.startswith(s) for t in self.train_strings), \
                "held-out subject never appears in training"
            # the held-out verb appears with the sibling subjects of the family
            sibs = [x for x in self.subject_families[fi] if x != s]
            assert any(t.startswith(x) and v in t
                       for t in self.train_strings for x in sibs), \
                "held-out verb never appears with a sibling subject in training"
        for t in self.train_strings:
            assert t not in self.heldout_strings
        for t in self.val_strings:
            assert t not in self.heldout_strings
        for t in self.train_strings + self.val_strings:
            assert " " not in t and t.endswith(EOS) and t.count(EOS) == 1
        for p in self.heldout_prefixes:
            for t in self.train_strings + self.val_strings:
                assert not t.startswith(p), "held-out prefix leaked into corpus"

    def decompose(self, s):
        """EVALUATION ONLY.  Returns (subject, filler, verb, object) or None."""
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
                    if not r2.startswith(v):
                        continue
                    o = r2[len(v):]
                    if o in self.obj_family:
                        return (subj, f, v, o)
        return None

    def is_grammatical(self, s):
        """EVALUATION ONLY: well-formed AND obeying the Latin-square constraint."""
        d = self.decompose(s)
        if d is None:
            return False
        subj, _f, v, o = d
        fi = self.subj_family[subj]
        vi = self.verb_index[v]
        return self.obj_family[o] == self.objfam_table[fi][vi]

    def lexicon(self):
        self.touch()
        out = []
        for fam in self.subject_families:
            out.extend(fam)
        out.extend(self.verbs)
        out.extend(self.fillers)
        for fam in self.object_families:
            out.extend(fam)
        return out


# =============================================================================
# SECTION 2.  TRANSPARENT INTERNAL SYMBOLS (PROVENANCE)   [UNCHANGED FROM V3]
# =============================================================================


class SymbolTable(object):
    """Every internally created symbol is fully transparent.

    For each created symbol we store:
      * the rule that created it,
      * the source symbols/cells that produced it,
      * (recursively) how to expand it back to raw characters.
    """

    def __init__(self):
        self.n = 0
        self.info = {}          # name -> dict
        self.order = []         # creation order (for rollback of discards)

    def new_symbol(self, rule_id, sources, kind="collapse"):
        self.n += 1
        name = u"§%d" % self.n          # e.g. §7
        self.info[name] = {
            "name": name,
            "created_by_rule": rule_id,
            "kind": kind,
            "sources": tuple(sources),
            "alt_sources": [],
        }
        self.order.append(name)
        return name

    def mark(self):
        return self.n

    def rollback(self, n):
        """Discard symbols created after mark() - used when a random candidate
        is REJECTED, so that a discarded candidate leaves no trace."""
        while self.order and self.n > n:
            name = self.order.pop()
            self.info.pop(name, None)
            self.n -= 1

    def is_internal(self, sym):
        return sym in self.info

    def record_alternate(self, name, sources):
        rec = self.info.get(name)
        if rec is None:
            return
        t = tuple(sources)
        if t != rec["sources"] and t not in rec["alt_sources"] and len(rec["alt_sources"]) < 8:
            rec["alt_sources"].append(t)

    def expand(self, sym, _guard=0):
        """Recursively expand a symbol back to a tuple of raw characters."""
        if _guard > 64:
            return (sym,)
        rec = self.info.get(sym)
        if rec is None:
            return (sym,)
        out = []
        for s in rec["sources"]:
            out.extend(self.expand(s, _guard + 1))
        return tuple(out)

    def depth(self, sym, _guard=0):
        rec = self.info.get(sym)
        if rec is None or _guard > 64:
            return 0
        d = 0
        for s in rec["sources"]:
            d = max(d, self.depth(s, _guard + 1))
        return d + 1

    def explain(self, sym, indent=0, lines=None, _guard=0):
        """Print / return the derivation tree of a symbol."""
        if lines is None:
            lines = []
        pad = "  " * indent
        rec = self.info.get(sym)
        if rec is None:
            lines.append("%s%s   [raw character]" % (pad, repr(sym)))
            return lines
        lines.append("%s%s   <- rule R%s (%s)   expands to %r"
                     % (pad, sym, rec["created_by_rule"], rec["kind"],
                        "".join(self.expand(sym))))
        if _guard > 32:
            return lines
        for s in rec["sources"]:
            self.explain(s, indent + 1, lines, _guard + 1)
        return lines


# =============================================================================
# SECTION 3.  THE 2D SYMBOLIC GRID + ARRIVAL LINKS + THE **OUT** REGISTER
#
# CHANGE 2: one reserved symbolic cell called OUT.  It starts EMPTY.  Rewrite
# rules may write a raw character into it, erase it, or change it.  The
# predicted character IS whatever OUT holds after the rewrite path has run.
# =============================================================================


class State(object):
    """Discrete state: a grid of symbol cells plus prev/next arrival links,
    plus the reserved OUT register.

    * Cells contain raw characters, internal symbols, or nothing (absent key).
    * Grid coordinates are RANDOM and carry no order information.
    * Input order is retained ONLY by the symbolic prev/next links.
    """

    __slots__ = ("grid", "head", "prv", "nxt", "free", "free_idx", "out",
                 "_chain", "_sig", "_offchain")

    def __init__(self, grid, head, prv, nxt, free, free_idx, out=None):
        self.grid = grid
        self.head = head
        self.prv = prv
        self.nxt = nxt
        self.free = free              # shared, pre-shuffled coordinate tuple
        self.free_idx = free_idx
        self.out = out                # the OUT register (None == EMPTY)
        self._chain = None
        self._sig = None
        self._offchain = None

    # ---- construction ------------------------------------------------------
    @staticmethod
    def initial(cfg, seed):
        """Initialize the grid from a reproducible random seed."""
        rng = random.Random(seed)
        coords = [(r, c) for r in range(cfg.grid_size) for c in range(cfg.grid_size)]
        rng.shuffle(coords)
        coords = tuple(coords)
        c0 = coords[0]
        return State({c0: BOS}, c0, {}, {}, coords, 1, None)

    def copy(self):
        return State(dict(self.grid), self.head, dict(self.prv), dict(self.nxt),
                     self.free, self.free_idx, self.out)

    # ---- link walking (order lives in the links, not in coordinates) -------
    def chain_coords(self):
        out = []
        c = self.head
        while c is not None:
            out.append(c)
            c = self.prv.get(c)
        out.reverse()
        return out

    def chain_symbols(self):
        if self._chain is None:
            out = []
            c = self.head
            while c is not None:
                out.append(self.grid[c])
                c = self.prv.get(c)
            out.reverse()
            self._chain = tuple(out)
        return self._chain

    def offchain_symbols(self):
        if self._offchain is None:
            on = set(self.chain_coords())
            self._offchain = tuple(sorted(v for k, v in self.grid.items() if k not in on))
        return self._offchain

    def sig(self):
        if self._sig is None:
            self._sig = (self.chain_symbols(), self.offchain_symbols(), self.out)
        return self._sig

    # ---- primitive discrete operations ------------------------------------
    def _next_free(self):
        i = self.free_idx
        while i < len(self.free) and self.free[i] in self.grid:
            i += 1
        if i >= len(self.free):
            raise RuntimeError("grid exhausted")
        return self.free[i], i + 1

    def place_arrival(self, symbol):
        """Place a symbol in a RANDOMLY selected available location and link it
        after the current head.  (The location is random; only the link carries
        order information.)  Arrival clears the OUT register."""
        ns = self.copy()
        coord, nidx = ns._next_free()
        ns.free_idx = nidx
        ns.grid[coord] = symbol
        if ns.head is not None:
            ns.nxt[ns.head] = coord
            ns.prv[coord] = ns.head
        ns.head = coord
        ns.out = None
        ns._chain = None
        ns._sig = None
        ns._offchain = None
        return ns

    # ---- the OUT register --------------------------------------------------
    def set_out(self, value):
        """Return a state identical to this one except for the OUT register.

        The grid/link dictionaries are SHARED, never mutated in place: every
        other primitive copies before writing, so sharing is safe and keeps the
        OUT operations cheap."""
        ns = State(self.grid, self.head, self.prv, self.nxt, self.free,
                   self.free_idx, value)
        ns._chain = self._chain
        ns._offchain = self._offchain
        return ns

    def write_marker(self, symbol):
        ns = self.copy()
        coord, nidx = ns._next_free()
        ns.free_idx = nidx
        ns.grid[coord] = symbol
        ns._sig = None
        ns._offchain = None
        return ns

    def erase_marker(self, symbol):
        ns = self.copy()
        on = set(ns.chain_coords())
        for k, v in list(ns.grid.items()):
            if v == symbol and k not in on:
                del ns.grid[k]
                break
        ns._sig = None
        ns._offchain = None
        return ns

    def relabel(self, idx_from_end, symbol):
        cc = self.chain_coords()
        if idx_from_end >= len(cc):
            return None
        ns = self.copy()
        ns.grid[cc[-1 - idx_from_end]] = symbol
        ns._chain = None
        ns._sig = None
        ns._offchain = None
        return ns

    def collapse_segment(self, start, k, symbol):
        """Erase k linked cells starting at chain position `start` and write ONE
        new cell holding `symbol`, re-linking it in their place.  This modifies
        k + 1 grid cells in a single rule application.  The segment may sit
        anywhere in the chain, not only at its end."""
        cc = self.chain_coords()
        if start < 0 or start + k > len(cc) or k < 1:
            return None
        ns = self.copy()
        victims = cc[start:start + k]
        anchor = cc[start - 1] if start > 0 else None
        after = cc[start + k] if start + k < len(cc) else None
        for v in victims:
            del ns.grid[v]
            ns.prv.pop(v, None)
            ns.nxt.pop(v, None)
        coord, nidx = ns._next_free()
        ns.free_idx = nidx
        ns.grid[coord] = symbol
        if anchor is not None:
            ns.nxt[anchor] = coord
            ns.prv[coord] = anchor
        if after is not None:
            ns.nxt[coord] = after
            ns.prv[after] = coord
        else:
            ns.head = coord
        ns._chain = None
        ns._sig = None
        ns._offchain = None
        return ns


def read_output(state):
    """CHANGE 2: the prediction is simply whatever the OUT register holds."""
    return state.out


# =============================================================================
# SECTION 4.  DISCRETE REWRITE RULES
#
# CHANGE 2: a rule no longer carries `incoming = <the character it licenses>`.
# A rule that produces output does so by an OUT-writing ACTION whose character
# was drawn at random from the learner's observed alphabet when the rule was
# proposed.  Meaning is read out of the symbolic state, not stored in the rule.
# =============================================================================


class Rule(object):
    """A discrete symbolic transformation.

    Conditions (all discrete, no scores):
      * `pattern`   : a tuple matched against the last N linked positions.
                      Elements are literals, wildcards, or variables.
      * `require`   : symbols that must be present somewhere in the grid.

    Effects (`actions`), applied in order:
      ('COLLAPSE', k, sym)  replace k linked symbols from the match with one
      ('MARK', sym)         write a reusable internal marker into a free cell
      ('ERASE', sym)        erase a marker
      ('RELABEL', i, sym)   replace one symbol i positions from the head
      ('OUT', ch)           write raw character ch into the OUT register
      ('OUTCLEAR',)         erase the OUT register

    `kind` is derived: a rule that touches OUT is an OUTPUT rule, otherwise it
    is an INTERNAL (epsilon) rule.
    """

    __slots__ = ("rid", "kind", "pattern", "protected", "require",
                 "actions", "created_at", "origin", "contexts", "active",
                 "n_apply", "n_reuse", "n_generalized", "history",
                 "merged_from", "subsumed_by", "state_sigs", "emit_char",
                 "generalized", "deactivated_reason")

    def __init__(self, rid, pattern, require, actions, created_at, origin,
                 protected=None):
        self.rid = rid
        self.pattern = tuple(pattern)
        self.protected = frozenset(protected or ())
        self.require = tuple(require)
        self.actions = tuple(actions)
        touches_out = any(a[0] in ("OUT", "OUTCLEAR") for a in self.actions)
        self.kind = "OUTPUT" if touches_out else "INTERNAL"
        # static consequence of the action sequence on an EMPTY OUT register
        cur = None
        for a in self.actions:
            if a[0] == "OUT":
                cur = a[1]
            elif a[0] == "OUTCLEAR":
                cur = None
        self.emit_char = cur
        self.created_at = created_at      # (epoch, step, string index)
        self.origin = origin              # 'random-search'
        self.contexts = []                # successful matched tails
        self.state_sigs = []              # full situations it succeeded in
        self.active = True
        self.n_apply = 0
        self.n_reuse = 0                  # successes after the creating one
        self.n_generalized = 0
        self.generalized = False
        self.deactivated_reason = None
        self.history = []                 # list of previous patterns
        self.merged_from = []
        self.subsumed_by = None

    # ---- matching ----------------------------------------------------------
    def match(self, state):
        chain = state.chain_symbols()
        L = len(self.pattern)
        if L > len(chain) or L == 0:
            return None
        tail = chain[len(chain) - L:]
        if not pattern_matches(self.pattern, tail):
            return None
        if self.require:
            present = state.offchain_symbols()
            for sym in self.require:
                if sym not in present:
                    return None
        return (tail, {})

    def index_key(self):
        """Index by the last pattern element (fast candidate lookup)."""
        if not self.pattern:
            return "*"
        e = self.pattern[-1]
        return e[1] if e[0] == "L" else "*"

    def first_key(self):
        if not self.pattern:
            return "*"
        e = self.pattern[0]
        return e[1] if e[0] == "L" else "*"

    # ---- application -------------------------------------------------------
    def occurrences(self, state):
        """INTERNAL rules match a contiguous linked SEGMENT anywhere in the
        chain (not only its end).  Returns the list of start positions."""
        chain = state.chain_symbols()
        L = len(self.pattern)
        if L == 0 or L > len(chain):
            return []
        if self.require:
            present = state.offchain_symbols()
            for sym in self.require:
                if sym not in present:
                    return []
        out = []
        for i in range(len(chain) - L + 1):
            if pattern_matches(self.pattern, chain[i:i + L]):
                out.append(i)
        return out

    def apply_at(self, state, start, symtab, record=True):
        """Apply the rule's effects at a given segment start position."""
        st = state
        for act in self.actions:
            if act[0] == "COLLAPSE":
                k, sym = act[1], act[2]
                chain = st.chain_symbols()
                if start + k > len(chain):
                    return None
                if record:
                    symtab.record_alternate(sym, chain[start:start + k])
                st = st.collapse_segment(start, k, sym)
            elif act[0] == "MARK":
                st = st.write_marker(act[1])
            elif act[0] == "ERASE":
                st = st.erase_marker(act[1])
            elif act[0] == "RELABEL":
                st = st.relabel(act[1], act[2])
            elif act[0] == "OUT":
                st = st.set_out(act[1])
            elif act[0] == "OUTCLEAR":
                st = st.set_out(None)
            if st is None:
                return None
        return st

    def apply_at_tail(self, state, symtab, record=True):
        chain = state.chain_symbols()
        return self.apply_at(state, len(chain) - len(self.pattern), symtab, record)

    def modifies_chain(self):
        return any(a[0] in ("COLLAPSE", "RELABEL") for a in self.actions)

    # ---- capacity accounting ----------------------------------------------
    def reads(self):
        return len(self.pattern) + len(self.require)

    def writes(self):
        w = 0
        for a in self.actions:
            if a[0] == "COLLAPSE":
                w += a[1] + 1
            elif a[0] in ("MARK", "ERASE", "RELABEL", "OUT", "OUTCLEAR"):
                w += 1
        return w

    # ---- human-readable form ----------------------------------------------
    def text(self):
        cond = ["linked-tail=(%s)" % ",".join(pat_str(e) for e in self.pattern)]
        if self.require:
            cond.append("grid-contains{%s}" % ",".join(self.require))
        eff = []
        for a in self.actions:
            if a[0] == "COLLAPSE":
                eff.append("replace the matched %d linked symbols with %s"
                           % (a[1], a[2]))
            elif a[0] == "MARK":
                eff.append("write marker %s" % a[1])
            elif a[0] == "ERASE":
                eff.append("erase marker %s" % a[1])
            elif a[0] == "RELABEL":
                eff.append("relabel position -%d as %s" % (a[1], a[2]))
            elif a[0] == "OUT":
                eff.append("OUT := %r" % a[1])
            elif a[0] == "OUTCLEAR":
                eff.append("OUT := EMPTY")
        return "R%-4d %-8s IF %s THEN %s" % (
            self.rid, self.kind, " AND ".join(cond), "; ".join(eff))

    def to_json(self, symtab):
        return {
            "rid": self.rid,
            "kind": self.kind,
            "emit_char": self.emit_char,
            "pattern": [pat_str(e) for e in self.pattern],
            "pattern_uses_internal_symbols": any(
                e[0] == "L" and symtab.is_internal(e[1]) for e in self.pattern),
            "require": list(self.require),
            "actions": [list(map(str, a)) for a in self.actions],
            "origin": self.origin,
            "created_at": list(self.created_at),
            "active": self.active,
            "generalized": self.generalized,
            "deactivated_reason": self.deactivated_reason,
            "n_apply": self.n_apply,
            "n_reuse": self.n_reuse,
            "n_generalized": self.n_generalized,
            "n_contexts": len(self.contexts),
            "n_situations": len(self.state_sigs),
            "history": [[pat_str(e) for e in p] for p in self.history],
            "merged_from": list(self.merged_from),
            "subsumed_by": self.subsumed_by,
            "reads": self.reads(),
            "writes": self.writes(),
            "text": self.text(),
        }


# =============================================================================
# SECTION 4b.  THE TRAINING ARCHIVE  (CHANGE 4)
#
# An archive of ACTUAL training decisions: for each observed symbolic tail, the
# raw characters that were actually correct next characters in training.
#
# It may ONLY be used to test a proposed generalization for contradiction.
# It never proposes rules, never selects rules, never generates or ranks
# output, and it is LOCKED at freeze time so that it is provably unavailable
# during inference.
# =============================================================================


class TrainingArchive(object):

    def __init__(self):
        self.by_len = defaultdict(dict)   # L -> {tail tuple: set of correct chars}
        self.version = 0
        self.n_decisions = 0
        self.locked = False
        self.access_while_locked = 0
        self._cache = {}

    def lock(self):
        self.locked = True

    def _guard(self):
        if self.locked:
            self.access_while_locked += 1
            raise RuntimeError("training archive consulted after freeze")

    def record(self, tails_by_len, correct_char):
        """Store one actual training decision.  Returns the list of
        (L, tail) entries whose correct-character set actually changed."""
        self._guard()
        self.n_decisions += 1
        changed = []
        for L, tails in tails_by_len.items():
            d = self.by_len[L]
            for t in tails:
                s = d.get(t)
                if s is None:
                    d[t] = set([correct_char])
                    changed.append((L, t))
                    self.version += 1
                elif correct_char not in s:
                    s.add(correct_char)
                    changed.append((L, t))
                    self.version += 1
        if changed:
            self._cache.clear()
        return changed

    def correct_chars(self, L, tail):
        self._guard()
        return self.by_len.get(L, {}).get(tail)

    def contradicted(self, pattern, emit_char):
        """Boolean: does this (pattern -> emit_char) rule apply to some archived
        training decision for which `emit_char` was NOT a correct next
        character?  consistent / contradicted, no error score."""
        self._guard()
        if emit_char is None:
            return None
        key = (pattern, emit_char, self.version)
        hit = self._cache.get(key)
        if hit is not None:
            return hit[0] if hit[0] is None else hit[1]
        L = len(pattern)
        d = self.by_len.get(L)
        found = None
        if d:
            for tail, chars in d.items():
                if emit_char in chars:
                    continue
                if pattern_matches(pattern, tail):
                    found = tail
                    break
        self._cache[key] = (found, found)
        return found

    def counterexample_for_tail(self, pattern, emit_char, L, tail):
        """Cheap single-entry check used for later-contradiction rollback."""
        self._guard()
        if emit_char is None or len(pattern) != L:
            return False
        chars = self.by_len.get(L, {}).get(tail)
        if chars is None or emit_char in chars:
            return False
        return pattern_matches(pattern, tail)


# =============================================================================
# SECTION 5.  THE LEARNER
#
# The learner receives ONLY raw strings.  It never receives, and has no way to
# obtain, word boundaries, class labels, grammar rules or test labels.
# =============================================================================


class Learner(object):

    def __init__(self, cfg, seed):
        self.cfg = cfg
        self.seed = seed
        self.rng = random.Random(seed * 104729 + 7)
        self.symtab = SymbolTable()
        self.rules = []
        self.next_rid = 1
        self.frozen = False
        self.rid_at_freeze = None
        self.alphabet = set()            # grown from the raw characters observed
        self.archive = TrainingArchive()

        # indices: pattern element -> rules
        self._out_idx = defaultdict(list)
        self._int_idx = defaultdict(list)
        self._exact = {}          # exact rule signature -> rule (dedup)
        self._int_pat = {}        # internal collapse pattern -> rule (dedup)
        # rules that produce the same output by the same effects over the same
        # number of inspected positions (candidate partners for relaxation)
        self._merge_buckets = defaultdict(list)
        self._by_rid = {}

        # diagnostics (counts only; never used to choose rules)
        self.stats = {
            "decisions": 0,
            "solved_by_existing": 0,
            "required_new_rule_search": 0,
            "new_rule_created": 0,
            "unsolved": 0,
            "candidates_proposed": 0,
            "candidates_rejected": 0,
            "candidates_accidentally_correct": 0,
            "candidate_no_output": 0,
            "generalization_events": 0,
            "generalization_rejected_positive": 0,
            "generalization_rejected_counterexample": 0,
            "rollback_events": 0,
            "deactivation_events": 0,
            "reuse_events": 0,
        }
        self.dynamics = []               # per-decision: 0 existing, 1 new, 2 unsolved
        self.traces = []                 # successful derivation traces
        self.rollback_log = []

    # ------------------------------------------------------------------ util
    def _register(self, rule):
        self.rules.append(rule)
        self._by_rid[rule.rid] = rule
        self._exact[(rule.pattern, rule.require, rule.actions)] = rule
        if rule.kind == "INTERNAL":
            self._int_pat[rule.pattern] = rule
            self._int_idx[rule.first_key()].append(rule)
        else:
            self._out_idx[rule.index_key()].append(rule)
            self._merge_buckets[(rule.emit_char, len(rule.pattern),
                                 rule.actions, rule.require)].append(rule)

    def _reindex(self, rule, old_key):
        idx = self._out_idx if rule.kind == "OUTPUT" else self._int_idx
        try:
            idx[old_key].remove(rule)
        except ValueError:
            pass
        idx[rule.index_key()].append(rule)

    def _internal_candidates(self, state):
        """Internal rules whose first condition can start somewhere in the
        chain (index lookup keeps this cheap)."""
        out = []
        seen = set()
        syms = set(state.chain_symbols())
        for key in list(syms) + ["*"]:
            for r in self._int_idx.get(key, ()):
                if r.rid not in seen:
                    seen.add(r.rid)
                    out.append(r)
        out.sort(key=lambda r: r.rid)
        return out

    def _output_candidates(self, state):
        chain = state.chain_symbols()
        last = chain[-1] if chain else None
        out = self._out_idx.get("*", [])
        if last is not None and last in self._out_idx:
            out = out + self._out_idx[last]
        out.sort(key=lambda r: r.rid)
        return out

    def _usable(self, rule, state):
        """Reuse control.

        NO-REUSE ABLATION: a rule may fire only in a situation identical to one
        in which it has already succeeded (identical full linked-chain
        signature), so nothing is ever carried over to a new situation.
        """
        if rule.subsumed_by is not None or not rule.active:
            return False
        if self.cfg.allow_reuse:
            return True
        return state.sig()[0] in rule.state_sigs

    # ------------------------------------------------- symbolic exploration
    def expansions(self, state):
        """Explore the VIEWS of the current state reachable by epsilon
        (INTERNAL) rewrites.

        The persistent state is always the raw arrival chain.  Internal rules
        rewrite contiguous linked segments into reusable internal symbols, so a
        view is a re-derivable hierarchical reading of the same raw input.
        Because views are recomputed from the raw chain at every step, no
        irreversible commitment to one segmentation is ever made.

        Search branches.  The cutoffs are purely computational; nothing is
        scored.  The final ordering (fewest rewrites first, i.e. the raw view
        before its restructurings) is a search ORDER policy: it decides which
        applicable rewrite is examined first, never whether a rule applies.
        """
        cfg = self.cfg
        results = [(state, ())]
        if not cfg.allow_composition:
            return results
        seen = {state.sig()}
        frontier = deque([(state, ())])
        while frontier and len(results) < cfg.max_expansion_states:
            st, path = frontier.popleft()
            if len(path) >= cfg.max_internal_depth:
                continue
            for r in self._internal_candidates(st):
                if not self._usable(r, st):
                    continue
                for pos in r.occurrences(st):
                    ns = r.apply_at(st, pos, self.symtab, record=not self.frozen)
                    if ns is None:
                        continue
                    if not self.frozen:
                        # bookkeeping only: how often a structural rule is
                        # actually used.  Never consulted when choosing rules.
                        r.n_apply += 1
                        if r.n_apply > 1:
                            r.n_reuse += 1
                    sg = ns.sig()
                    if sg in seen:
                        continue
                    seen.add(sg)
                    results.append((ns, path + (r.rid,)))
                    frontier.append((ns, path + (r.rid,)))
                    if len(results) >= cfg.max_expansion_states:
                        break
                if len(results) >= cfg.max_expansion_states:
                    break
        results.sort(key=lambda x: (len(x[1]), len(x[0].chain_symbols())))
        return results

    def _out_steps(self, st, ipath, record):
        """Apply every applicable OUTPUT rule to a view and read the resulting
        OUT register.  Yields (out_char, ipath, rule, tail, new_state).

        Paths may branch: a rule that also modifies the chain leaves a state on
        which a further OUT-touching rule may fire (write / erase / change OUT),
        up to cfg.max_out_path steps.  Nothing is scored.
        """
        budget = self.cfg.max_out_path
        frontier = [(st, ipath, (), 0)]
        while frontier:
            cur, ip, opath, depth = frontier.pop(0)
            for r in self._output_candidates(cur):
                if not self._usable(r, cur):
                    continue
                m = r.match(cur)
                if m is None:
                    continue
                ns = r.apply_at_tail(cur, self.symtab, record=record)
                if ns is None:
                    continue
                yield (read_output(ns), ip, r, m[0], ns, opath + (r.rid,))
                if depth + 1 < budget and r.modifies_chain():
                    frontier.append((ns, ip, opath + (r.rid,), depth + 1))

    def find_path(self, state, target):
        """CHANGE 3: try to reach the target using ONLY rules that already
        exist.  A rule either applies or does not.  A path either produces
        OUT == target or does not.  Nothing is scored, nothing is ranked; the
        known target enters only as the final boolean test.
        """
        for st, ipath in self.expansions(state):
            for (predicted, ip, r, tail, ns, opath) in self._out_steps(
                    st, ipath, record=not self.frozen):
                if predicted == target:            # boolean test, after execution
                    return (st, ip, r, tail, opath)
        return None

    # --------------------------------------------------- new rule creation
    def _new_rid(self):
        r = self.next_rid
        self.next_rid += 1
        return r

    def _pattern_from_state(self, st, length):
        chain = st.chain_symbols()
        length = min(length, len(chain), self.cfg.max_pattern_len)
        return tuple(lit(s) for s in chain[len(chain) - length:])

    def _rule_exists(self, pattern, require, actions):
        if actions is None:
            return self._int_pat.get(pattern)
        return self._exact.get((pattern, tuple(require), tuple(actions)))

    def _propose_structural(self, st, where):
        """Randomly propose a new structural (epsilon) rewrite rule: replace a
        randomly chosen contiguous linked segment of the current view with one
        reusable internal symbol.  The learner is told nothing about where
        boundaries might be; the segment is drawn at random."""
        cfg = self.cfg
        chain = st.chain_symbols()
        kmax = min(cfg.max_collapse_span, len(chain), cfg.max_write_cells - 2,
                   cfg.max_pattern_len)
        if kmax < 2:
            return None, st
        k = self.rng.randint(2, kmax)
        start = self.rng.randrange(0, len(chain) - k + 1)
        seg = chain[start:start + k]
        pat = tuple(lit(x) for x in seg)
        if self._rule_exists(pat, (), None) is not None:
            return None, st
        rid = self._new_rid()
        sym = self.symtab.new_symbol(rid, seg)
        acts = [("COLLAPSE", k, sym)]
        if self.rng.random() < cfg.p_propose_marker:
            acts.append(("MARK", self.symtab.new_symbol(rid, seg, kind="marker")))
        cand = Rule(rid, pat, (), tuple(acts), where, "random-search",
                    protected=range(len(pat)))
        if cand.reads() > cfg.max_read_cells or cand.writes() > cfg.max_write_cells:
            return None, st
        ns = cand.apply_at(st, start, self.symtab)
        if ns is None:
            return None, st
        return cand, ns

    # ---------------------------------------------------------------------
    # CHANGE 1: TARGET-BLIND CANDIDATE GENERATION.
    #
    # This function does not take, see, read, or in any way depend on the
    # character that the training step happens to require.  It draws
    #   * which restructured VIEW of the state is inspected,
    #   * whether a new structural rewrite is proposed and over which segment,
    #   * which reusable symbol is created,
    #   * whether a presence test is added and over which symbol,
    #   * whether the OUT register is written / erased / overwritten, and
    #   * WHICH raw character is written into OUT - drawn uniformly from the
    #     characters the learner has actually observed so far.
    #
    # DESIGN DECISION kept from V3 (see CHANGE 5: "new rules begin specific"):
    # within the randomly chosen view the inspected conditions are the
    # maximally specific ones available (the longest inspectable linked tail).
    # Generality is never invented at creation time; it can arise only later,
    # out of successful reuse.
    # ---------------------------------------------------------------------
    def propose_random_rule(self, views, alphabet, rng, where):
        cfg = self.cfg
        st, ipath = views[rng.randrange(len(views))]
        structural = None
        st2 = st
        if cfg.allow_composition and rng.random() < cfg.p_propose_collapse:
            structural, st2 = self._propose_structural(st, where)

        pat = self._pattern_from_state(st2, cfg.max_pattern_len)
        if not pat:
            return None
        require = ()
        if rng.random() < cfg.p_propose_require:
            off = st2.offchain_symbols()
            if off:
                require = (off[rng.randrange(len(off))],)

        acts = []
        letters = sorted(alphabet)
        if letters and rng.random() < cfg.p_propose_out:
            acts.append(("OUT", letters[rng.randrange(len(letters))]))
            if rng.random() < cfg.p_propose_out_pair:
                if rng.random() < 0.5:
                    acts.append(("OUTCLEAR",))
                else:
                    acts.append(("OUT", letters[rng.randrange(len(letters))]))
        if not acts:
            # a proposal that does not touch OUT at all: legal, but it will
            # produce no output and therefore cannot pass the success test.
            acts.append(("OUTCLEAR",))

        if self._rule_exists(pat, require, acts) is not None:
            return None
        rid = self._new_rid()
        cand = Rule(rid, pat, require, tuple(acts), where, "random-search")
        if cand.reads() > cfg.max_read_cells or cand.writes() > cfg.max_write_cells:
            return None
        return (cand, structural, st2, ipath)

    def search_new_rule(self, state, target, where):
        """Random search.  The proposal function above never sees `target`;
        the value is used ONLY here, after the candidate has been executed and
        the OUT register has been read.  There is no fallback that constructs a
        rule emitting the required character: if the budget is exhausted the
        decision is left UNSOLVED.
        """
        assert not self.frozen, "no rule creation at inference time"
        cfg = self.cfg
        views = self.expansions(state)
        for _attempt in range(cfg.new_rule_budget):
            sym_mark = self.symtab.mark()
            rid_mark = self.next_rid
            proposal = self.propose_random_rule(views, self.alphabet, self.rng, where)
            if proposal is None:
                self.symtab.rollback(sym_mark)
                self.next_rid = rid_mark
                continue
            cand, structural, st2, ipath = proposal
            self.stats["candidates_proposed"] += 1

            # ---- execute the candidate, then read the OUT register ----------
            ns = cand.apply_at_tail(st2, self.symtab)
            predicted = read_output(ns) if ns is not None else None
            if predicted is None:
                self.stats["candidate_no_output"] += 1

            # ---- ONLY NOW may the known target be consulted, as a boolean ---
            if predicted == target:
                self.stats["candidates_accidentally_correct"] += 1
                opath = (cand.rid,)
                if structural is not None:
                    self._register(structural)
                    ipath = ipath + (structural.rid,)
                self._register(cand)
                return (st2, ipath, cand, cand.match(st2)[0], opath)

            # ---- discarded: the candidate leaves no trace -------------------
            self.stats["candidates_rejected"] += 1
            self.symtab.rollback(sym_mark)
            self.next_rid = rid_mark
        return None

    # ------------------------------------------------------ generalization
    def _antiunify(self, p, tail):
        """Position-wise: agreeing conditions are retained, differing ones are
        replaced by wildcards.  Purely structural; no similarity score."""
        out = []
        diffs = []
        for i, (pe, s) in enumerate(zip(p, tail)):
            if pe[0] == "L" and pe[1] == s:
                out.append(pe)
            elif pe[0] == "A":
                out.append(pe)
            else:
                out.append(WILD)
                diffs.append(i)
        return tuple(out), diffs

    def _reproduces(self, pattern, contexts):
        """A rule may never be generalized if the relaxed form fails to
        reproduce one of the successful cases it must still cover."""
        for tail in contexts:
            if len(tail) != len(pattern):
                return False
            if not pattern_matches(pattern, tail):
                return False
        return True

    def _relax(self, rule, new_pattern, reason, extra_contexts=()):
        # CHANGE 5.2: at least one meaningful symbolic condition must survive
        if not any(e[0] != "A" for e in new_pattern):
            self.stats["generalization_rejected_positive"] += 1
            return False
        # positive check (V3 behaviour): the relaxed rule must still cover the
        # successful cases it came from
        if not self._reproduces(new_pattern, list(rule.contexts) + list(extra_contexts)):
            self.stats["generalization_rejected_positive"] += 1
            return False
        # CHANGE 4: NEGATIVE check against the archive of real training decisions
        if self.cfg.counterexample_check:
            ce = self.archive.contradicted(new_pattern, rule.emit_char)
            if ce is not None:
                self.stats["generalization_rejected_counterexample"] += 1
                return False
        old_key = rule.index_key()
        rule.history.append(rule.pattern)
        rule.pattern = new_pattern
        rule.n_generalized += 1
        rule.generalized = True
        for t in extra_contexts:
            if t not in rule.contexts and len(rule.contexts) < self.cfg.max_contexts_per_rule:
                rule.contexts.append(t)
        self._reindex(rule, old_key)
        self.stats["generalization_events"] += 1
        rule.merged_from.append(reason)
        return True

    def try_generalize_by_reuse(self, rule):
        """CHANGE 5: relaxation may be proposed ONLY for a rule that has already
        proved REUSABLE (it succeeded again after the success that created it).

        A rule whose conditions are all literals can, by construction, only ever
        match one tail, so a reused rule is compared with ANOTHER rule that has
        itself already succeeded, produces the same output by the same effects,
        and inspects the same number of positions.  The conditions the two
        share stay fixed; the positions on which they differ become wildcards.
        Purely structural anti-unification - no similarity measure, no score.
        The result must still pass the positive coverage check and the global
        counterexample check before it is accepted.
        """
        if not self.cfg.allow_generalization or rule.kind != "OUTPUT":
            return False
        if rule.n_reuse < 1:
            return False                     # never generalize at creation time
        bucket = self._merge_buckets.get(
            (rule.emit_char, len(rule.pattern), rule.actions, rule.require), ())
        for other in bucket:
            if other is rule or not other.active or other.subsumed_by is not None:
                continue
            if other.n_apply < 1:
                continue
            merged = []
            diffs = 0
            ok = True
            for i, (a, b) in enumerate(zip(rule.pattern, other.pattern)):
                if a == b:
                    merged.append(a)
                    continue
                if i in rule.protected or i in other.protected:
                    ok = False
                    break
                merged.append(WILD)
                diffs += 1
            if not ok or diffs == 0 or diffs > self.cfg.max_relax_positions:
                continue
            if self._relax(rule, tuple(merged), "reuse-merge-with-R%d" % other.rid,
                           tuple(other.contexts)):
                other.active = False
                other.subsumed_by = rule.rid
                return True
        return False

    def try_self_generalize(self, rule, tail):
        """CHANGE 5: generalization is triggered ONLY when a rule that already
        exists succeeds again in a genuinely DIFFERENT context.  Shared
        conditions stay fixed; differing ones may become wildcards."""
        if not self.cfg.allow_generalization or rule.kind != "OUTPUT":
            return False
        newpat, diffs = self._antiunify(rule.pattern, tail)
        diffs = [i for i in diffs if i not in rule.protected]
        if not diffs:
            return False
        if len(diffs) > self.cfg.max_relax_positions:
            # relax gradually: only the leftmost differing position this time
            keep = diffs[: self.cfg.max_relax_positions]
            newpat = list(rule.pattern)
            for i in keep:
                newpat[i] = WILD
            newpat = tuple(newpat)
        return self._relax(rule, newpat, "reuse-in-new-context", (tail,))

    def recheck_generalizations(self, changed):
        """CHANGE 4: a rule that was generalized earlier may become contradicted
        when new training data arrives.  Roll it back to its last safe version,
        or deactivate it if no safe version exists."""
        if not self.cfg.counterexample_check or not changed:
            return
        for rule in self.rules:
            if not rule.active or not rule.generalized or rule.kind != "OUTPUT":
                continue
            for (L, tail) in changed:
                if not self.archive.counterexample_for_tail(
                        rule.pattern, rule.emit_char, L, tail):
                    continue
                # contradiction found -> roll back
                rolled = False
                while rule.history:
                    old_key = rule.index_key()
                    rule.pattern = rule.history.pop()
                    self._reindex(rule, old_key)
                    if self.archive.contradicted(rule.pattern, rule.emit_char) is None:
                        rolled = True
                        break
                if rolled:
                    self.stats["rollback_events"] += 1
                    rule.generalized = bool(rule.history)
                    self.rollback_log.append({
                        "rule": rule.rid, "event": "rollback",
                        "restored_pattern": [pat_str(e) for e in rule.pattern],
                        "counterexample_tail": [str(x) for x in tail],
                        "emit": rule.emit_char,
                    })
                else:
                    rule.active = False
                    rule.deactivated_reason = "contradicted-after-generalization"
                    self.stats["deactivation_events"] += 1
                    self.rollback_log.append({
                        "rule": rule.rid, "event": "deactivated",
                        "counterexample_tail": [str(x) for x in tail],
                        "emit": rule.emit_char,
                    })
                break

    # ------------------------------------------------------------ training
    def train(self, strings):
        assert not self.frozen
        cfg = self.cfg
        for s in strings:
            assert isinstance(s, str) and " " not in s, "learner sees raw strings only"
        for ep in range(cfg.epochs):
            for si, s in enumerate(strings):
                self._train_string(s, ep, si)

    def _record_success(self, rule, tail, st):
        rule.n_apply += 1
        if tail not in rule.contexts:
            if len(rule.contexts) < self.cfg.max_contexts_per_rule:
                rule.contexts.append(tail)
        sg = st.sig()[0]
        if sg not in rule.state_sigs and len(rule.state_sigs) < 64:
            rule.state_sigs.append(sg)

    def _tails_of(self, views):
        out = defaultdict(set)
        for st, _ip in views:
            chain = st.chain_symbols()
            for L in range(1, self.cfg.max_pattern_len + 1):
                if L <= len(chain):
                    out[L].add(chain[len(chain) - L:])
        return out

    def _train_string(self, s, epoch, si):
        # reproducible per-sequence grid initialization
        state = State.initial(self.cfg, self.seed * 1000003 + epoch * 9176 + si)
        trace = {"string_index": si, "epoch": epoch, "steps": []}
        for t in range(len(s)):
            target = s[t]
            self.stats["decisions"] += 1
            views = self.expansions(state)

            outcome = 0
            rule = None
            ipath = ()
            opath = ()
            tail = ()

            # (CHANGE 3) FIRST try to reach the target with existing rules only
            found = self.find_path(state, target)
            if found is not None:
                st, ipath, rule, tail, opath = found
                first_time = (len(rule.contexts) == 0)
                if not first_time:
                    rule.n_reuse += 1
                    self.stats["reuse_events"] += 1
                    # (CHANGE 5) the rule has proved reusable: only now may a
                    # relaxed version of it be proposed.
                    if tail not in rule.contexts:
                        # succeeded again on a genuinely different tail
                        self.try_self_generalize(rule, tail)
                    else:
                        self.try_generalize_by_reuse(rule)
                self._record_success(rule, tail, st)
                self.stats["solved_by_existing"] += 1
            else:
                # (CHANGE 1) only now may random new-rule search begin
                self.stats["required_new_rule_search"] += 1
                res = self.search_new_rule(state, target, (epoch, t, si))
                if res is None:
                    outcome = 2                       # UNSOLVED, not repaired
                    self.stats["unsolved"] += 1
                else:
                    st, ipath, rule, tail, opath = res
                    self._record_success(rule, tail, st)
                    outcome = 1
                    self.stats["new_rule_created"] += 1

            # archive the ACTUAL training decision (CHANGE 4) and re-check the
            # generalized rules against the newly observed evidence
            changed = self.archive.record(self._tails_of(views), target)
            self.recheck_generalizations(changed)

            # the persistent state is always the raw arrival chain: the incoming
            # character simply enters the grid at a random free location, linked
            # after the previous arrival.  Only now does the learner know this
            # character exists, so only now does it enter the alphabet.
            self.alphabet.add(target)
            state = state.place_arrival(target)
            self.dynamics.append(outcome)
            trace["steps"].append({
                "t": t, "target": target,
                "outcome": ["existing", "new_rule", "unsolved"][outcome],
                "internal_rules": list(ipath),
                "output_rules": list(opath),
                "matched_tail": [str(x) for x in tail],
            })
        if len(self.traces) < 40:
            self.traces.append(trace)

    # ----------------------------------------------------------- inference
    def freeze(self):
        """CHANGE 10: no new rules, no target, no archive, nothing else."""
        self.frozen = True
        self.rid_at_freeze = self.next_rid
        self.archive.lock()

    def _initial_state(self):
        # Reproducible initialization for inference.  The physical cell
        # locations differ from those used in training, which is harmless
        # precisely because no rule ever refers to a coordinate.
        return State.initial(self.cfg, self.seed * 31 + 5)

    def _state_for_prefix(self, prefix):
        """Feed a raw prefix into the grid one character at a time.

        Arrival is the machine's base operation, not a prediction: an observed
        character always enters the grid.  What the learned rule system governs
        is which characters may FOLLOW.  `licensed_steps` records, for
        diagnostics only, how many of the prefix characters an existing rule
        would itself have licensed.
        """
        assert self.frozen, "inference must run on a frozen rule system"
        st = self._initial_state()
        licensed = 0
        for ch in prefix:
            if ch in self._licensed(st):
                licensed += 1
            st = st.place_arrival(ch)
        return st, licensed

    def _licensed_with_derivation(self, state):
        """Execute every reachable rewrite path and collect the OUT registers.
        Returns {char: (internal_rule_ids, output_rule_ids, composed_flag)}.
        Unranked; the first derivation found for a character is kept for
        reporting only."""
        out = {}
        for st, ipath in self.expansions(state):
            for (predicted, ip, r, _tail, _ns, opath) in self._out_steps(
                    st, ipath, record=not self.frozen):
                if predicted is None or predicted in out:
                    continue
                syms = set()
                for rid in list(ip) + list(opath):
                    rr = self._by_rid.get(rid)
                    if rr is None:
                        continue
                    for e in rr.pattern:
                        if e[0] == "L" and self.symtab.is_internal(e[1]):
                            syms.add(e[1])
                    for a in rr.actions:
                        if a[0] == "COLLAPSE" and self.symtab.is_internal(a[2]):
                            syms.add(a[2])
                composed = bool(ip) or any(
                    e[0] == "L" and self.symtab.is_internal(e[1]) for e in r.pattern)
                out[predicted] = (list(ip), list(opath), composed, sorted(syms),
                                  [pat_str(e) for e in r.pattern])
        return out

    def _licensed(self, state):
        return set(self._licensed_with_derivation(state))

    def prefix_licensing_coverage(self, prefix):
        """Diagnostic: how many characters of a raw prefix an existing rule
        would itself have licensed (arrival happens regardless)."""
        _st, lic = self._state_for_prefix(prefix)
        return lic, len(prefix)

    def possible_next_chars(self, prefix):
        """Return the SET of characters for which an existing symbolic
        derivation survives.  No ranking, no probabilities, no scores."""
        st, _ = self._state_for_prefix(prefix)
        return self._licensed(st)

    def possible_completions(self, prefix, max_length):
        """Recursively continue only with next characters allowed by existing
        symbolic rules.  Returns {complete string: derivation}."""
        results = {}
        nodes = [0]
        start, _ = self._state_for_prefix(prefix)

        def rec(state, suffix, deriv):
            if nodes[0] >= self.cfg.max_completion_nodes:
                return
            if len(results) >= self.cfg.max_completions:
                return
            if len(suffix) >= max_length:
                return
            nodes[0] += 1
            lic = self._licensed_with_derivation(state)
            for ch in sorted(lic):
                ip, op, composed, syms, pat = lic[ch]
                step = {"char": ch, "internal_rules": ip, "output_rules": op,
                        "composed": composed, "internal_symbols": syms,
                        "output_rule_conditions": pat,
                        "symbol_expansions": {
                            s: "".join(self.symtab.expand(s)) for s in syms}}
                if ch == EOS:
                    results.setdefault(prefix + suffix + EOS, deriv + [step])
                    continue
                rec(state.place_arrival(ch), suffix + ch, deriv + [step])

        rec(start, "", [])
        return results

    # -------------------------------------------------------- transparency
    def explain_symbol(self, symbol, out=None):
        lines = self.symtab.explain(symbol)
        if out is None:
            out = sys.stdout
        for ln in lines:
            out.write(ln + "\n")
        return lines


# =============================================================================
# SECTION 6.  CONTROLS AGAINST CHEATING  (CHANGE 15)
# =============================================================================

FORBIDDEN_IN_LEARNER_SOURCE = [
    "HiddenCorpus", "subject_famil", "object_famil", "objfam", "heldout",
    "lexicon", "decompose", "is_grammatical", "verb",
]


def target_blindness_test(verbose=True):
    """CHANGE 1 / CHANGE 15: prove that candidate generation cannot see the
    target.

      1. the proposal function does not even accept a target parameter;
      2. its source code never mentions one;
      3. with the RNG state saved and restored, and a DIFFERENT hypothetical
         target in force, it proposes the byte-identical rule.
    """
    problems = []
    sig = inspect.signature(Learner.propose_random_rule)
    for name in sig.parameters:
        if "target" in name.lower() or "goal" in name.lower() or "label" in name.lower():
            problems.append("propose_random_rule accepts a target-like parameter: %s" % name)
    src = inspect.getsource(Learner.propose_random_rule) + \
        inspect.getsource(Learner._propose_structural)
    for bad in ("target", "goal_char", "self.target"):
        if bad in src:
            problems.append("proposal source mentions %r" % bad)

    # ---- behavioural test ---------------------------------------------------
    cfg = Config()
    learner = Learner(cfg, 1)
    learner.alphabet = set("abcdefgk#")
    st = State.initial(cfg, 12345)
    for ch in "abcabc":
        st = st.place_arrival(ch)
    views = learner.expansions(st)

    def snapshot():
        return (learner.rng.getstate(), learner.symtab.mark(), learner.next_rid)

    def restore(s):
        learner.rng.setstate(s[0])
        learner.symtab.rollback(s[1])
        learner.next_rid = s[2]

    def canonical(p):
        if p is None:
            return None
        cand, structural, st2, ipath = p
        return (tuple(pat_str(e) for e in cand.pattern), cand.require,
                tuple(map(str, cand.actions)),
                None if structural is None else
                (tuple(pat_str(e) for e in structural.pattern),
                 tuple(map(str, structural.actions))),
                tuple(st2.chain_symbols()), ipath)

    global HYPOTHETICAL_TARGET
    s0 = snapshot()
    HYPOTHETICAL_TARGET = "k"
    a = canonical(learner.propose_random_rule(views, learner.alphabet,
                                              learner.rng, (0, 0, 0)))
    restore(s0)
    HYPOTHETICAL_TARGET = "z"
    b = canonical(learner.propose_random_rule(views, learner.alphabet,
                                              learner.rng, (0, 0, 0)))
    restore(s0)
    HYPOTHETICAL_TARGET = None
    if a != b:
        problems.append("proposal differed under a different hypothetical target")

    ok = not problems
    if verbose:
        if ok:
            print("TARGET-BLIND CANDIDATE GENERATION: PASS")
        else:
            print("TARGET-BLIND CANDIDATE GENERATION: FAIL")
            for p in problems:
                print("   !! %s" % p)
    return ok, problems


HYPOTHETICAL_TARGET = None


def cheat_controls(learner, corpus, train_strings, val_strings, phase):
    """Explicit assertions that the learner never accesses hidden information."""
    problems = []

    # 1. the learner holds no reference to the corpus generator
    for k, v in vars(learner).items():
        if v is corpus:
            problems.append("learner attribute %s references the generator" % k)
        if isinstance(v, (list, tuple, set)):
            for item in v:
                if item is corpus:
                    problems.append("learner container %s holds the generator" % k)

    # 2. the learner only ever received raw single-token strings; NO BOUNDARIES
    for s in train_strings + val_strings:
        if " " in s or "\t" in s or "|" in s or "_" in s:
            problems.append("boundary/separator character found in learner input")
        if s.count(EOS) != 1 or not s.endswith(EOS):
            problems.append("malformed learner input: %r" % s)

    # 3. no hidden class label or hidden word list ever entered the alphabet
    for a in learner.alphabet:
        if len(a) != 1:
            problems.append("non-character symbol in learner alphabet: %r" % a)
        if a not in "".join(train_strings):
            problems.append("alphabet character never observed in input: %r" % a)

    # 4. held-out strings never appear in training material
    for s in train_strings + val_strings:
        if s in corpus.heldout_strings:
            problems.append("held-out string leaked into training: %r" % s)
        for p in corpus.heldout_prefixes:
            if s.startswith(p):
                problems.append("held-out prefix leaked into training: %r" % s)

    # 5. no learner method may mention generator metadata
    if phase == "pre-train":
        src = inspect.getsource(Learner)
        for bad in FORBIDDEN_IN_LEARNER_SOURCE:
            if bad in src:
                problems.append("learner source mentions generator metadata: %r" % bad)

    # 6. at inference no rule may be created and the archive must be locked
    if phase == "post-inference":
        if learner.next_rid != learner.rid_at_freeze:
            problems.append("rules were created after freezing")
        if not learner.frozen:
            problems.append("learner was not frozen at inference")
        if not learner.archive.locked:
            problems.append("training archive was not locked at inference")
        if learner.archive.access_while_locked:
            problems.append("training archive was consulted %d times at inference"
                            % learner.archive.access_while_locked)

    # 7. no rule may store the target: a rule's output character must be one
    #    that the proposal drew from the observed alphabet
    for r in learner.rules:
        if r.emit_char is not None and r.emit_char not in learner.alphabet:
            problems.append("rule R%d emits a character outside the observed "
                            "alphabet" % r.rid)
    return problems


def final_contradiction_audit(learner):
    """CHANGE 4: active generalized rules with known archived contradictions."""
    was_locked = learner.archive.locked
    learner.archive.locked = False          # audit runs OUTSIDE the learner
    bad = []
    for r in learner.rules:
        if not r.active or not r.generalized or r.kind != "OUTPUT":
            continue
        ce = learner.archive.contradicted(r.pattern, r.emit_char)
        if ce is not None:
            bad.append({"rule": r.rid, "emit": r.emit_char,
                        "counterexample": [str(x) for x in ce]})
    learner.archive.locked = was_locked
    return bad


# =============================================================================
# SECTION 7.  EVALUATION (OUTSIDE THE LEARNER)
# =============================================================================


def classify_completions(corpus, prefix, generated):
    compat = corpus.heldout_compatible[prefix]
    incompat = corpus.heldout_incompatible[prefix]
    out = {"compatible": [], "incompatible": [], "malformed": []}
    for g in sorted(generated):
        if g in compat:
            out["compatible"].append(g)
        elif g in incompat:
            out["incompatible"].append(g)
        elif corpus.decompose(g) is not None:
            # a well-formed word sequence that is not a licensed continuation
            out["incompatible"].append(g)
        else:
            out["malformed"].append(g)
    return out


def metrics_for(prefix, corpus, cls):
    target = corpus.heldout_compatible[prefix]
    n = len(cls["compatible"]) + len(cls["incompatible"]) + len(cls["malformed"])
    tp = len(cls["compatible"])
    precision = (tp / float(n)) if n else 0.0
    recall = tp / float(len(target))
    exact = 1.0 if (set(cls["compatible"]) == target and
                    not cls["incompatible"] and not cls["malformed"]) else 0.0
    return {"generated": n, "true_positive": tp, "precision": precision,
            "recall": recall, "exact_set_accuracy": exact,
            "incompatible_generation_rate":
                (len(cls["incompatible"]) / float(n)) if n else 0.0,
            "malformed_generation_rate":
                (len(cls["malformed"]) / float(n)) if n else 0.0}


def rule_diagnostics(learner):
    rules = learner.rules
    active = [r for r in rules if r.active]
    output = [r for r in rules if r.kind == "OUTPUT"]
    internal = [r for r in rules if r.kind == "INTERNAL"]
    reused = [r for r in rules if r.n_reuse > 0]
    generalized = [r for r in rules if r.n_generalized > 0]
    never = [r for r in rules if r.n_apply <= 1 and r.n_reuse == 0]

    def pdepth(r):
        d = 0
        for e in r.pattern:
            if e[0] == "L":
                d = max(d, learner.symtab.depth(e[1]))
        for a in r.actions:
            if a[0] == "COLLAPSE":
                d = max(d, learner.symtab.depth(a[2]))
        return d
    comp = [pdepth(r) for r in rules] or [0]
    der = [learner.symtab.depth(s) for s in learner.symtab.info] or [0]
    return {
        "total_rules": len(rules),
        "active_rules": len(active),
        "output_rules": len(output),
        "internal_rules": len(internal),
        "reused_rules": len(reused),
        "generalized_rules": len(generalized),
        "active_generalized_rules": len([r for r in generalized if r.active]),
        "never_reused_rules": len(never),
        "max_rule_composition_depth": max(comp),
        "mean_rule_composition_depth": sum(comp) / float(len(comp)),
        "max_derivation_depth": max(der),
        "internal_symbols": len(learner.symtab.info),
        "rules_referencing_internal_symbols": len(
            [r for r in rules if any(e[0] == "L" and learner.symtab.is_internal(e[1])
                                     for e in r.pattern)]),
    }


def learning_dynamics(learner, buckets=10):
    d = learner.dynamics
    if not d:
        return []
    n = len(d)
    out = []
    for i in range(buckets):
        a = (n * i) // buckets
        b = (n * (i + 1)) // buckets
        seg = d[a:b]
        if not seg:
            continue
        newsearch = len([x for x in seg if x != 0])
        out.append({
            "portion": "%d/%d" % (i + 1, buckets),
            "decisions": len(seg),
            "solved_by_existing_rules": len([x for x in seg if x == 0]),
            "new_rule_search_required": newsearch,
            "new_rule_created": len([x for x in seg if x == 1]),
            "unsolved": len([x for x in seg if x == 2]),
            "new_rule_rate": newsearch / float(len(seg)),
        })
    return out


def emergent_structures(learner, corpus, train_strings, top=12):
    """Report internal structures whose provenance expands into recurring raw
    character patterns.  The learner never labels them; the comparison with the
    hidden vocabulary happens here, for evaluation only."""
    lex = set(corpus.lexicon())
    rows = []
    for name, rec in learner.symtab.info.items():
        if rec["kind"] != "collapse":
            continue
        exp = "".join(learner.symtab.expand(name))
        if len(exp) < 2:
            continue
        occ = sum(s.count(exp) for s in train_strings)
        creator = None
        for r in learner.rules:
            if r.rid == rec["created_by_rule"]:
                creator = r
                break
        rows.append({
            "symbol": name,
            "expansion": exp,
            "corpus_occurrences": occ,
            "rule_applications": creator.n_apply if creator else 0,
            "depth": learner.symtab.depth(name),
            "matches_hidden_word": exp in lex,          # EVALUATION ONLY
        })
    rows.sort(key=lambda r: (-r["rule_applications"], -r["corpus_occurrences"]))
    exact = [r for r in rows if r["matches_hidden_word"]]
    return {
        "examples": rows[:top],
        "examples_matching_hidden_words": exact[:top],
        "n_internal_structures": len(rows),
        "n_exact_hidden_word_matches": len(exact),
        "exact_hidden_word_matches": sorted(set(r["expansion"] for r in exact)),
        "derivation_depths": sorted(set(r["depth"] for r in rows)),
    }


def validation_coverage(learner, val_strings):
    """How much of the unseen validation material the frozen rule system can
    still derive (evaluation only)."""
    ok = 0
    total = 0
    for s in val_strings:
        total += 1
        derivable = True
        for i in range(len(s)):
            nxt = learner.possible_next_chars(s[:i])
            if s[i] not in nxt:
                derivable = False
                break
        if derivable:
            ok += 1
    return {"validation_strings": total,
            "fully_derivable": ok,
            "rate": (ok / float(total)) if total else 0.0}


# =============================================================================
# SECTION 7b.  POST-HOC SUBJECT-FAMILY PROBE   (EVALUATION ONLY)
#
# Everything in this section runs on the ALREADY FROZEN learner, after
# inference.  It creates no rules, changes no rule, and feeds nothing back into
# training (asserted below).  The hidden subject-family labels are used ONLY at
# the very end, to compare the groups the probe discovered with the truth.
#
# No similarity score is used anywhere: the groupings are exact set identities.
# =============================================================================


def _probe_prefix(corpus, subject, verb):
    """The same filler condition the held-out evaluation uses."""
    return subject + corpus.probe_filler + verb


def behavioral_signature(learner, corpus, subjects, verbs):
    """DIAGNOSTIC 1.  For every (subject, verb) query the frozen learner and
    record the complete SET of object completions it licenses."""
    sig = {}
    completions = {}
    for s in subjects:
        per_verb = {}
        for v in verbs:
            pfx = _probe_prefix(corpus, s, v)
            gen = learner.possible_completions(pfx, learner.cfg.max_completion_len)
            objs = set()
            for g in gen:
                if g.startswith(pfx) and g.endswith(EOS):
                    objs.add(g[len(pfx):-1])
            per_verb[v] = frozenset(objs)
            completions[(s, v)] = gen
        sig[s] = per_verb
    return sig, completions


def group_by_identical(sig, subjects, verbs):
    """Exact identity of behavioural signatures - no similarity measure."""
    buckets = defaultdict(list)
    for s in subjects:
        key = tuple(sorted((v, tuple(sorted(sig[s][v]))) for v in verbs))
        buckets[key].append(s)
    return [sorted(g) for g in buckets.values()]


def internal_footprint(learner, corpus, subjects, verbs):
    """DIAGNOSTIC 2.  Replay each probe prefix through the FROZEN learner and
    collect the rules and internal structures that its processing touches."""
    out = {}
    for s in subjects:
        rec = {
            "rules_applied": set(),
            "generalized_rules_applied": set(),
            "internal_rules_applied": set(),
            "internal_symbols_in_views": set(),
            "internal_symbols_active_at_verb": set(),
        }
        for v in verbs:
            pfx = _probe_prefix(corpus, s, v)
            verb_start = len(s) + len(corpus.probe_filler)   # evaluation only
            st = learner._initial_state()
            for i, ch in enumerate(pfx):
                in_verb = (i >= verb_start)
                for view, ipath in learner.expansions(st):
                    for rid in ipath:
                        rec["internal_rules_applied"].add(rid)
                        rec["rules_applied"].add(rid)
                    for sym in view.chain_symbols():
                        if learner.symtab.is_internal(sym):
                            rec["internal_symbols_in_views"].add(sym)
                            if in_verb:
                                rec["internal_symbols_active_at_verb"].add(sym)
                    for (predicted, _ip, r, _tail, _ns, opath) in learner._out_steps(
                            view, ipath, record=False):
                        if predicted != ch:
                            continue          # only rules that license this char
                        for rid in opath:
                            rec["rules_applied"].add(rid)
                            rr = learner._by_rid.get(rid)
                            if rr is not None and rr.generalized:
                                rec["generalized_rules_applied"].add(rid)
                st = st.place_arrival(ch)
        out[s] = {k: sorted(vv) for k, vv in rec.items()}
    return out


def _replay_out_sequence(learner, state, rids, char):
    """Can this exact sequence of stored output rules be applied to `state`
    (in some view reachable by existing internal rules) and yield `char`?
    Returns the internal path used, or None.  Creates nothing."""
    for view, ipath in learner.expansions(state):
        cur = view
        ok = True
        for rid in rids:
            r = learner._by_rid.get(rid)
            if r is None or not r.active or r.subsumed_by is not None:
                return None
            if r.match(cur) is None:
                ok = False
                break
            nxt = r.apply_at_tail(cur, learner.symtab, record=False)
            if nxt is None:
                ok = False
                break
            cur = nxt
        if ok and read_output(cur) == char:
            return ipath
    return None


def _path_replays(learner, prefix, steps):
    """Replay a whole recorded derivation (one step per generated character)
    starting from a DIFFERENT raw prefix, using only existing rules."""
    st, _ = learner._state_for_prefix(prefix)
    same_internal = True
    for step in steps:
        ip = _replay_out_sequence(learner, st, step["output_rules"], step["char"])
        if ip is None:
            return False, False
        if tuple(ip) != tuple(step["internal_rules"]):
            same_internal = False
        st = st.place_arrival(step["char"])
    return True, same_internal


def substitution_matrix(learner, corpus, subjects, verbs, completions):
    """DIAGNOSTIC 3.  s1 ~ s2 iff, for every shared verb, every successful
    frozen derivation path recorded for s1 can be replayed with s2 substituted
    for s1, and vice versa - without creating any rule."""
    transfer = {}          # (s1, s2) -> bool  (s1's paths replay under s2)
    detail = {}
    for s1 in subjects:
        for s2 in subjects:
            if s1 == s2:
                continue
            ok_all = True
            per_verb = {}
            for v in verbs:
                paths = completions[(s1, v)]
                pfx2 = _probe_prefix(corpus, s2, v)
                n_ok = 0
                n_same_internal = 0
                for _g, steps in sorted(paths.items()):
                    r_ok, r_same = _path_replays(learner, pfx2, steps)
                    if r_ok:
                        n_ok += 1
                        if r_same:
                            n_same_internal += 1
                per_verb[v] = {"paths": len(paths), "replayed": n_ok,
                               "replayed_with_identical_internal_path": n_same_internal}
                if len(paths) == 0 or n_ok < len(paths):
                    ok_all = False
            transfer[(s1, s2)] = ok_all
            detail[(s1, s2)] = per_verb
    equiv = {}
    for s1 in subjects:
        for s2 in subjects:
            if s1 == s2:
                continue
            equiv[(s1, s2)] = transfer[(s1, s2)] and transfer[(s2, s1)]
    return equiv, transfer, detail


def _groups_from_equivalence(subjects, equiv):
    """Connected components of the symmetric relation (reported as-is; the
    relation is not assumed to be transitive, so components are the honest
    reading and non-transitivity is reported separately)."""
    groups = []
    seen = set()
    for s in subjects:
        if s in seen:
            continue
        comp = [s]
        seen.add(s)
        frontier = [s]
        while frontier:
            a = frontier.pop()
            for b in subjects:
                if b in seen:
                    continue
                if equiv.get((a, b)):
                    seen.add(b)
                    comp.append(b)
                    frontier.append(b)
        groups.append(sorted(comp))
    return groups


def compare_with_hidden_families(corpus, subjects, groups):
    """EVALUATION ONLY: compare discovered groups with the hidden families."""
    fam_of = {}
    for fi, fam in enumerate(corpus.subject_families):
        for s in fam:
            fam_of[s] = fi
    grp_of = {}
    for gi, g in enumerate(groups):
        for s in g:
            grp_of[s] = gi
    same_ok, same_total, diff_bad, diff_total = 0, 0, 0, 0
    same_pairs_ok, diff_pairs_bad = [], []
    for i, s1 in enumerate(subjects):
        for s2 in subjects[i + 1:]:
            together = (grp_of[s1] == grp_of[s2])
            if fam_of[s1] == fam_of[s2]:
                same_total += 1
                if together:
                    same_ok += 1
                    same_pairs_ok.append((s1, s2))
            else:
                diff_total += 1
                if together:
                    diff_bad += 1
                    diff_pairs_bad.append((s1, s2))
    exact = sorted([sorted(f) for f in corpus.subject_families]) == \
        sorted([sorted(g) for g in groups])
    return {
        "same_family_pairs_grouped": same_ok,
        "same_family_pairs_total": same_total,
        "different_family_pairs_grouped": diff_bad,
        "different_family_pairs_total": diff_total,
        "same_family_pairs_correctly_grouped": [list(p) for p in same_pairs_ok],
        "different_family_pairs_incorrectly_grouped": [list(p) for p in diff_pairs_bad],
        "groups_equal_hidden_families": exact,
    }


def family_common_structures(corpus, footprint, subjects):
    """EVALUATION ONLY: which rules / internal symbols are shared by ALL members
    of a hidden family and by NO member of any other family.  Exact set
    intersections; nothing is scored."""
    out = []
    keys = ["rules_applied", "generalized_rules_applied",
            "internal_rules_applied", "internal_symbols_in_views",
            "internal_symbols_active_at_verb"]
    for fi, fam in enumerate(corpus.subject_families):
        others = [s for s in subjects if s not in fam]
        row = {"family": fi, "members": list(fam)}
        for k in keys:
            inter = None
            for s in fam:
                cur = set(footprint[s][k])
                inter = cur if inter is None else (inter & cur)
            outside = set()
            for s in others:
                outside |= set(footprint[s][k])
            row[k + "_shared_by_family"] = len(inter or ())
            row[k + "_exclusive_to_family"] = sorted((inter or set()) - outside)
        out.append(row)
    return out


def subject_family_probe(learner, corpus, do_substitution=True, verbose=False):
    """Run all three post-hoc diagnostics on the frozen learner."""
    assert learner.frozen, "the probe runs only on a frozen learner"
    rid_before = learner.next_rid
    n_rules_before = len(learner.rules)

    subjects = sorted(learner_visible_subjects(corpus))
    verbs = list(corpus.verbs)

    sig, completions = behavioral_signature(learner, corpus, subjects, verbs)
    beh_groups = group_by_identical(sig, subjects, verbs)
    beh_cmp = compare_with_hidden_families(corpus, subjects, beh_groups)

    footprint = internal_footprint(learner, corpus, subjects, verbs)
    fam_common = family_common_structures(corpus, footprint, subjects)

    sub = None
    if do_substitution:
        equiv, transfer, detail = substitution_matrix(
            learner, corpus, subjects, verbs, completions)
        sub_groups = _groups_from_equivalence(subjects, equiv)
        sub_cmp = compare_with_hidden_families(corpus, subjects, sub_groups)
        # honesty check: is the discovered relation actually an equivalence?
        nontransitive = []
        for a in subjects:
            for b in subjects:
                for c in subjects:
                    if a == b or b == c or a == c:
                        continue
                    if equiv.get((a, b)) and equiv.get((b, c)) and not equiv.get((a, c)):
                        nontransitive.append([a, b, c])
        sub = {
            "matrix": {"%s|%s" % k: v for k, v in equiv.items()},
            "one_way_transfer": {"%s|%s" % k: v for k, v in transfer.items()},
            "detail": {"%s|%s" % k: v for k, v in detail.items()},
            "groups": sub_groups,
            "comparison": sub_cmp,
            "nontransitive_triples": nontransitive[:20],
            "n_nontransitive_triples": len(nontransitive),
        }

    # nothing may have been created or changed by the probe
    assert learner.next_rid == rid_before, "the probe created a rule"
    assert len(learner.rules) == n_rules_before, "the probe added a rule"
    assert learner.next_rid == learner.rid_at_freeze, "rules exist after freeze"

    return {
        "subjects": subjects,
        "verbs": verbs,
        "probe_filler": corpus.probe_filler,
        "true_hidden_families": [list(f) for f in corpus.subject_families],
        "signatures": {s: {v: sorted(sig[s][v]) for v in verbs} for s in subjects},
        "behavioral_groups": beh_groups,
        "behavioral_comparison": beh_cmp,
        "footprint": footprint,
        "family_common_structures": fam_common,
        "substitution": sub,
    }


def learner_visible_subjects(corpus):
    """EVALUATION ONLY helper: the list of subject words to probe.  The learner
    is not told that these strings are subjects, nor which family they belong
    to; the probe simply feeds raw prefixes built from them."""
    out = []
    for fam in corpus.subject_families:
        out.extend(fam)
    return out


def print_subject_family_probe(probe, corpus):
    subjects = probe["subjects"]
    verbs = probe["verbs"]
    fam_of = {}
    for fi, fam in enumerate(corpus.subject_families):
        for s in fam:
            fam_of[s] = fi

    print("-" * 78)
    print("DIAGNOSTIC 1: BEHAVIOURAL SIGNATURES  (frozen learner, filler=%r)"
          % probe["probe_filler"])
    print("-" * 78)
    for s in subjects:
        print("  subject %-8r  [hidden family %d]" % (s, fam_of[s]))
        for v in verbs:
            objs = probe["signatures"][s][v]
            print("      + %-8r -> {%s}" % (v, ", ".join(objs) if objs else ""))
    print("")
    print("  groups with EXACTLY identical signatures:")
    for g in probe["behavioral_groups"]:
        print("      %s   (hidden families %s)"
              % (g, sorted(set(fam_of[x] for x in g))))
    bc = probe["behavioral_comparison"]
    print("  same-family pairs with identical signatures:      %d / %d"
          % (bc["same_family_pairs_grouped"], bc["same_family_pairs_total"]))
    print("  different-family pairs with identical signatures: %d / %d"
          % (bc["different_family_pairs_grouped"], bc["different_family_pairs_total"]))
    print("")

    print("-" * 78)
    print("DIAGNOSTIC 2: INTERNAL RULE FOOTPRINT")
    print("-" * 78)
    for s in subjects:
        f = probe["footprint"][s]
        print("  footprint[%r]  (hidden family %d)" % (s, fam_of[s]))
        print("      rules applied ................... %d %s"
              % (len(f["rules_applied"]), f["rules_applied"][:10]))
        print("      generalized rules applied ....... %d %s"
              % (len(f["generalized_rules_applied"]),
                 f["generalized_rules_applied"][:10]))
        print("      internal rules applied .......... %d" % len(f["internal_rules_applied"]))
        print("      internal symbols in views ....... %d %s"
              % (len(f["internal_symbols_in_views"]), f["internal_symbols_in_views"][:8]))
        print("      internal symbols alive at verb .. %d %s"
              % (len(f["internal_symbols_active_at_verb"]),
                 f["internal_symbols_active_at_verb"][:8]))
    print("")
    print("  structures shared by ALL members of a hidden family and by NO")
    print("  member of any other family (exact set intersection):")
    for row in probe["family_common_structures"]:
        print("    family %d %s" % (row["family"], row["members"]))
        for k in ["rules_applied", "generalized_rules_applied",
                  "internal_rules_applied", "internal_symbols_in_views",
                  "internal_symbols_active_at_verb"]:
            print("      %-34s shared=%-4d exclusive=%s"
                  % (k, row[k + "_shared_by_family"], row[k + "_exclusive_to_family"][:8]))
    print("")

    if probe["substitution"] is not None:
        print("-" * 78)
        print("DIAGNOSTIC 3: RULE-SUBSTITUTION TEST")
        print("-" * 78)
        sub = probe["substitution"]
        w = max(8, max(len(s) for s in subjects) + 1)
        print("      " + "".join(("%-" + str(w) + "s") % s[:w - 1] for s in subjects))
        for s1 in subjects:
            cells = []
            for s2 in subjects:
                if s1 == s2:
                    cells.append("-")
                else:
                    cells.append("Y" if sub["matrix"]["%s|%s" % (s1, s2)] else "N")
            print(("%-" + str(w) + "s") % s1[:w - 1] +
                  "".join(("%-" + str(w) + "s") % c for c in cells))
        print("")
        print("  equivalence groups induced by substitution:")
        for g in sub["groups"]:
            print("      %s   (hidden families %s)"
                  % (g, sorted(set(fam_of[x] for x in g))))
        print("  non-transitive triples in the discovered relation: %d"
              % sub["n_nontransitive_triples"])
        print("")


# =============================================================================
# SECTION 8.  EXPERIMENT RUNNER
# =============================================================================

ABLATIONS = [
    ("full_system", {}),
    ("no_reuse", {"allow_reuse": False}),
    ("no_generalization", {"allow_generalization": False}),
    ("no_composition", {"allow_composition": False}),
    ("shuffled_input_order", {"shuffle_input": True}),
    ("no_counterexample_check", {"counterexample_check": False}),
]


def shuffle_strings(strings, seed):
    """Shuffled-character control: destroy character order inside every training
    example.  The terminator is kept in final position so that the control
    learner still has a well defined end-of-sequence event."""
    rng = random.Random(seed * 6151 + 3)
    out = []
    for s in strings:
        body = list(s[:-1])
        rng.shuffle(body)
        out.append("".join(body) + EOS)
    return out


def run_one(ablation, overrides, seed, collect_detail=False, probe=False,
            probe_substitution=False):
    cfg = Config(**overrides)
    corpus = HiddenCorpus(seed)
    train = list(corpus.train_strings)
    val = list(corpus.val_strings)

    learner_input = shuffle_strings(train, seed) if cfg.shuffle_input else train

    learner = Learner(cfg, seed)
    pre_access = corpus.access_count
    problems = cheat_controls(learner, corpus, learner_input, val, "pre-train")
    learner.train(learner_input)
    assert corpus.access_count == pre_access, \
        "the corpus generator was queried during training"

    learner.freeze()

    # ---------------- inference on the held-out combinations ----------------
    heldout_results = []
    composed_success = 0
    flat_success = 0
    success_derivations = []
    for prefix in corpus.heldout_prefixes:          # constructed OUTSIDE learner
        pre = corpus.access_count
        lic, tot = learner.prefix_licensing_coverage(prefix)
        nxt = sorted(learner.possible_next_chars(prefix))
        gen = learner.possible_completions(prefix, cfg.max_completion_len)
        assert corpus.access_count == pre, \
            "the corpus generator was queried during inference"
        cls = classify_completions(corpus, prefix, set(gen))
        met = metrics_for(prefix, corpus, cls)
        for g in cls["compatible"]:
            deriv = gen[g]
            composed = any(step["composed"] for step in deriv)
            if composed:
                composed_success += 1
            else:
                flat_success += 1
            if len(success_derivations) < 12:
                success_derivations.append({
                    "seed": seed, "prefix": prefix, "completion": g,
                    "classification": "COMPOSED SUCCESS" if composed else "FLAT SUCCESS",
                    "steps": deriv,
                })
        heldout_results.append({
            "prefix": prefix,
            "prefix_chars_licensed_by_existing_rules": lic,
            "prefix_length": tot,
            "possible_next_chars": nxt,
            "n_generated": len(gen),
            "generated": sorted(gen)[:60],
            "compatible": cls["compatible"],
            "incompatible": cls["incompatible"][:30],
            "n_incompatible": len(cls["incompatible"]),
            "malformed": cls["malformed"][:30],
            "n_malformed": len(cls["malformed"]),
            "target_compatible_set": sorted(corpus.heldout_compatible[prefix]),
            "metrics": met,
        })

    problems += cheat_controls(learner, corpus, learner_input, val, "post-inference")
    contradiction_audit = final_contradiction_audit(learner)

    # POST-HOC diagnostic on the frozen learner (evaluation only; creates
    # nothing, feeds nothing back).  The hidden family labels are touched only
    # inside the comparison helpers, after every query has been made.
    # (the probe MAY read the hidden labels - but only to compare the groups it
    # discovered from the learner's behaviour with the truth, after the fact.)
    probe_result = None
    if probe:
        probe_result = subject_family_probe(
            learner, corpus, do_substitution=probe_substitution)
        assert learner.next_rid == learner.rid_at_freeze, \
            "a rule was created during the post-hoc probe"

    diag = rule_diagnostics(learner)
    dyn = learning_dynamics(learner)
    emerg = emergent_structures(learner, corpus, train)
    valcov = validation_coverage(learner, val)

    mean = lambda xs: (sum(xs) / float(len(xs))) if xs else 0.0
    agg = {
        "precision": mean([h["metrics"]["precision"] for h in heldout_results]),
        "recall": mean([h["metrics"]["recall"] for h in heldout_results]),
        "exact_set_accuracy": mean([h["metrics"]["exact_set_accuracy"] for h in heldout_results]),
        "incompatible_generation_rate": mean(
            [h["metrics"]["incompatible_generation_rate"] for h in heldout_results]),
        "malformed_generation_rate": mean(
            [h["metrics"]["malformed_generation_rate"] for h in heldout_results]),
        "mean_completion_set_size": mean([h["metrics"]["generated"] for h in heldout_results]),
        "prefix_licensed_rate": mean(
            [h["prefix_chars_licensed_by_existing_rules"] / float(h["prefix_length"])
             for h in heldout_results]),
        "composed_successes": composed_success,
        "flat_successes": flat_success,
    }

    result = {
        "ablation": ablation,
        "seed": seed,
        "config": cfg.as_dict(),
        "dataset": {
            "seed": seed,
            "n_train": len(train),
            "n_val": len(val),
            "train_strings": train,
            "val_strings": val,
            "heldout_prefixes": corpus.heldout_prefixes,
            "heldout_compatible_strings": {p: sorted(corpus.heldout_compatible[p])
                                           for p in corpus.heldout_compatible},
            "heldout_incompatible_strings": {p: sorted(corpus.heldout_incompatible[p])
                                             for p in corpus.heldout_incompatible},
            "hidden_lexicon_FOR_EVALUATION_ONLY": {
                "subject_families": corpus.subject_families,
                "verbs": corpus.verbs,
                "fillers": corpus.fillers,
                "object_family_table_subjfam_x_verb": corpus.objfam_table,
                "object_families": corpus.object_families,
                "heldout_pairs": [list(p) for p in corpus.heldout_pairs],
            },
        },
        "learner_stats": dict(learner.stats),
        "rule_diagnostics": diag,
        "learning_dynamics": dyn,
        "emergent_structures": emerg,
        "validation_coverage": valcov,
        "heldout": heldout_results,
        "successful_heldout_derivations": success_derivations,
        "rollback_log": learner.rollback_log[:40],
        "active_generalized_rules_with_contradictions": contradiction_audit,
        "aggregate": agg,
        "cheat_control_problems": problems,
        "subject_family_probe": probe_result,
    }
    if collect_detail:
        result["_learner"] = learner
        result["_corpus"] = corpus
    return result


def _grouping_verdict(cmp_rows):
    """Deterministic, score-free verdict over the per-seed comparisons."""
    n = len(cmp_rows)
    if n == 0:
        return "not discovered", 0, 0, 0
    exact = len([c for c in cmp_rows if c["groups_equal_hidden_families"]])
    pure = len([c for c in cmp_rows
                if c["different_family_pairs_grouped"] == 0
                and c["same_family_pairs_grouped"] > 0])
    complete = len([c for c in cmp_rows
                    if c["same_family_pairs_grouped"] == c["same_family_pairs_total"]
                    and c["different_family_pairs_grouped"] > 0])
    if exact * 2 >= n:
        return "clearly discovered", exact, pure, complete
    if (pure + complete) * 2 >= n:
        return "partially discovered", exact, pure, complete
    return "not discovered", exact, pure, complete


def aggregate_probe(rows):
    """Aggregate the post-hoc probe over seeds (evaluation only)."""
    probes = [r["subject_family_probe"] for r in rows
              if r.get("subject_family_probe")]
    if not probes:
        return None
    beh = [p["behavioral_comparison"] for p in probes]
    sub = [p["substitution"]["comparison"] for p in probes
           if p.get("substitution")]
    S = lambda rs, k: sum(c[k] for c in rs)
    bverdict, bex, bpure, bcomp = _grouping_verdict(beh)
    sverdict, sex, spure, scomp = _grouping_verdict(sub) if sub else \
        ("not run", 0, 0, 0)
    return {
        "seeds": len(probes),
        "behavioral": {
            "same_family_pairs_grouped": S(beh, "same_family_pairs_grouped"),
            "same_family_pairs_total": S(beh, "same_family_pairs_total"),
            "different_family_pairs_grouped": S(beh, "different_family_pairs_grouped"),
            "different_family_pairs_total": S(beh, "different_family_pairs_total"),
            "seeds_groups_equal_families": bex,
            "seeds_pure": bpure, "seeds_complete": bcomp,
            "verdict": bverdict,
        },
        "substitution": {
            "same_family_pairs_grouped": S(sub, "same_family_pairs_grouped"),
            "same_family_pairs_total": S(sub, "same_family_pairs_total"),
            "different_family_pairs_grouped": S(sub, "different_family_pairs_grouped"),
            "different_family_pairs_total": S(sub, "different_family_pairs_total"),
            "seeds_groups_equal_families": sex,
            "seeds_pure": spure, "seeds_complete": scomp,
            "verdict": sverdict,
        } if sub else None,
        "family_exclusive_structure_counts": [
            sum(len(row[k + "_exclusive_to_family"])
                for p in probes for row in p["family_common_structures"])
            for k in ["rules_applied", "generalized_rules_applied",
                      "internal_rules_applied", "internal_symbols_in_views",
                      "internal_symbols_active_at_verb"]],
    }


def aggregate(rows):
    mean = lambda xs: (sum(xs) / float(len(xs))) if xs else 0.0

    def std(xs):
        if len(xs) < 2:
            return 0.0
        m = mean(xs)
        return (sum((x - m) ** 2 for x in xs) / float(len(xs) - 1)) ** 0.5
    keys = ["precision", "recall", "exact_set_accuracy",
            "incompatible_generation_rate", "malformed_generation_rate",
            "mean_completion_set_size", "prefix_licensed_rate"]
    out = {}
    for k in keys:
        xs = [r["aggregate"][k] for r in rows]
        out[k + "_mean"] = mean(xs)
        out[k + "_std"] = std(xs)
    S = lambda key: sum(r["learner_stats"][key] for r in rows)
    out["rules_mean"] = mean([r["rule_diagnostics"]["total_rules"] for r in rows])
    out["generalized_rules_mean"] = mean([r["rule_diagnostics"]["generalized_rules"] for r in rows])
    out["reused_rules_mean"] = mean([r["rule_diagnostics"]["reused_rules"] for r in rows])
    out["internal_symbols_mean"] = mean([r["rule_diagnostics"]["internal_symbols"] for r in rows])
    out["exact_hidden_word_matches_mean"] = mean(
        [r["emergent_structures"]["n_exact_hidden_word_matches"] for r in rows])
    out["new_rule_rate_first_decile"] = mean(
        [r["learning_dynamics"][0]["new_rule_rate"] for r in rows if r["learning_dynamics"]])
    out["new_rule_rate_last_decile"] = mean(
        [r["learning_dynamics"][-1]["new_rule_rate"] for r in rows if r["learning_dynamics"]])
    out["validation_derivable_rate"] = mean([r["validation_coverage"]["rate"] for r in rows])
    out["total_candidates_proposed"] = S("candidates_proposed")
    out["total_candidates_rejected"] = S("candidates_rejected")
    out["total_candidates_accidentally_correct"] = S("candidates_accidentally_correct")
    out["total_decisions"] = S("decisions")
    out["total_solved_by_existing"] = S("solved_by_existing")
    out["total_required_new_rule_search"] = S("required_new_rule_search")
    out["total_unsolved"] = S("unsolved")
    out["total_reuse_events"] = S("reuse_events")
    out["total_generalization_events"] = S("generalization_events")
    out["total_generalization_rejected_counterexample"] = \
        S("generalization_rejected_counterexample")
    out["total_generalization_rejected_positive"] = S("generalization_rejected_positive")
    out["total_rollback_events"] = S("rollback_events") + S("deactivation_events")
    out["active_generalized_with_contradictions"] = sum(
        len(r["active_generalized_rules_with_contradictions"]) for r in rows)
    out["composed_successes"] = sum(r["aggregate"]["composed_successes"] for r in rows)
    out["flat_successes"] = sum(r["aggregate"]["flat_successes"] for r in rows)
    out["seeds"] = len(rows)
    return out


# =============================================================================
# SECTION 9.  MAIN
# =============================================================================

SEEDS = list(range(1, 11))       # 10 independent random seeds


CHANGELOG = """
FUNCTIONS CHANGED FROM V3
-------------------------
HiddenCorpus.__init__/_sanity/decompose (+ new is_grammatical)
    rebuilt: 3 SHARED verbs used by all 3 subject families, object family
    selected by the (subject-family, verb) PAIR as a Latin square, and a
    variable neutral filler word inserted between subject and verb.  The old
    version let the verb alone determine the object family and put the subject
    directly next to the verb, which made a local raw-tail rule sufficient.
    Held-out (subject,verb) pairs: one per family, each with a different verb.

State (+ set_out, read_output)
    added the reserved OUT register.  A rule's meaning is no longer stored in
    the rule; it is written into the symbolic state and read back out.

Rule.__init__/match/apply_at/text/to_json  (removed: `incoming`)
    rules no longer carry `incoming = <the character they license>`.  Output is
    an ACTION - ('OUT', ch) / ('OUTCLEAR',) - whose character was drawn at
    random from the observed alphabet.  `kind` is derived from the actions.

Learner.create_rule_for_target  ->  split into
    Learner.propose_random_rule(views, alphabet, rng, where)   [TARGET-BLIND]
    Learner.search_new_rule(state, target, where)              [propose/execute/test]
    the old function received `target` and built a rule around it, and had a
    `fallback` branch that directly constructed a rule licensing the target.
    Both are gone.  Rejected candidates are rolled back and leave no trace;
    an exhausted budget leaves the decision UNSOLVED.

Learner.find_path / _licensing_rules -> Learner.find_path / _out_steps
    the old version pre-filtered candidate rules by `r.incoming == target`.
    Now every applicable path is EXECUTED and the OUT register is read; the
    target enters only as the final boolean test.  Paths may branch over up to
    cfg.max_out_path OUT-touching steps.

Learner._licensed -> _licensed_with_derivation
    returns, per licensed character, the internal rules and output rules used,
    plus a COMPOSED/FLAT flag (CHANGE 12).  possible_completions now returns
    the derivation for every generated string.

Learner.try_merge   REMOVED
    it generalized two rules at creation time, which CHANGE 5 forbids.
    Generalization now happens only through try_self_generalize, i.e. only
    after a rule has succeeded again in a genuinely different context.

Learner._relax
    now also runs the NEGATIVE archive check (CHANGE 4) before accepting a
    relaxation, and counts positive- and counterexample-rejections separately.

NEW: TrainingArchive, Learner.recheck_generalizations, final_contradiction_audit
    archive of the actual training decisions, used ONLY to veto or roll back
    generalizations; locked at freeze so it is provably unavailable at
    inference.

NEW: target_blindness_test, expanded cheat_controls
    signature/source/RNG-replay proof of target blindness, archive lock check,
    boundary check, alphabet check, learner-source metadata scan.

SymbolTable.mark/rollback  NEW
    so a rejected random candidate leaves no internal symbols behind.

ABLATIONS
    replaced by the exact six required: full_system, no_reuse,
    no_generalization, no_composition, shuffled_input_order,
    no_counterexample_check.

NOTE ON CHANGE 8: option A was implemented (a variable neutral filler word
between subject and verb, never labelled, no boundary revealed, raw order
preserved).  With max_pattern_len = 4 a raw-tail rule reaches only the end of
the verb, which under the Latin square is not sufficient; the subject-family
information must be carried by reusable internal/composed structure.
"""


def main():
    outdir = os.path.dirname(os.path.abspath(__file__))
    print(CHANGELOG)

    print("=" * 78)
    print("SYMBOLIC REWRITE EXPERIMENT V4")
    print("=" * 78)
    blind_ok, blind_problems = target_blindness_test()
    print("")

    all_rows = []
    by_ablation = {}
    detail = None

    base = Config()
    print("grid %dx%d | max_read_cells=%d | max_write_cells=%d | pattern<=%d "
          "| new_rule_budget=%d"
          % (base.grid_size, base.grid_size, base.max_read_cells,
             base.max_write_cells, base.max_pattern_len, base.new_rule_budget))
    print("seeds: %s" % SEEDS)
    print("ablations: %s" % ", ".join(a for a, _ in ABLATIONS))
    print("")

    for ablation, ov in ABLATIONS:
        rows = []
        for seed in SEEDS:
            want_detail = (ablation == "full_system" and seed == SEEDS[0])
            want_probe = (ablation == "full_system")
            r = run_one(ablation, ov, seed, collect_detail=want_detail,
                        probe=want_probe, probe_substitution=want_probe)
            if want_detail:
                detail = (r.pop("_learner"), r.pop("_corpus"), r)
            rows.append(r)
            all_rows.append(r)
            st = r["learner_stats"]
            print("  [%-24s seed %2d]  rules=%-5d gen=%-3d unsolved=%-4d "
                  "P=%.2f R=%.2f exact=%.0f incompat=%.2f malformed=%.2f |gen|=%.1f"
                  % (ablation, seed, r["rule_diagnostics"]["total_rules"],
                     r["rule_diagnostics"]["generalized_rules"], st["unsolved"],
                     r["aggregate"]["precision"], r["aggregate"]["recall"],
                     r["aggregate"]["exact_set_accuracy"],
                     r["aggregate"]["incompatible_generation_rate"],
                     r["aggregate"]["malformed_generation_rate"],
                     r["aggregate"]["mean_completion_set_size"]))
            sys.stdout.flush()
        by_ablation[ablation] = {"per_seed": rows, "aggregate": aggregate(rows),
                                 "probe_aggregate": aggregate_probe(rows)}
        print("")

    # ------------------------------------------------------------------ files
    results = {
        "version": "v4",
        "target_blind_candidate_generation": "PASS" if blind_ok else "FAIL",
        "target_blindness_problems": blind_problems,
        "config_defaults": Config().as_dict(),
        "seeds": SEEDS,
        "ablations": {k: {"aggregate": v["aggregate"],
                          "per_seed": [{
                              "seed": r["seed"],
                              "aggregate": r["aggregate"],
                              "learner_stats": r["learner_stats"],
                              "rule_diagnostics": r["rule_diagnostics"],
                              "learning_dynamics": r["learning_dynamics"],
                              "validation_coverage": r["validation_coverage"],
                              "emergent_structures": r["emergent_structures"],
                              "heldout": r["heldout"],
                              "successful_heldout_derivations":
                                  r["successful_heldout_derivations"],
                              "rollback_log": r["rollback_log"],
                              "active_generalized_rules_with_contradictions":
                                  r["active_generalized_rules_with_contradictions"],
                              "cheat_control_problems": r["cheat_control_problems"],
                          } for r in v["per_seed"]]}
                      for k, v in by_ablation.items()},
        "dataset_examples": {
            "seed_%d" % SEEDS[0]: by_ablation["full_system"]["per_seed"][0]["dataset"]
        },
    }
    with open(os.path.join(outdir, "results_v4.json"), "w") as f:
        json.dump(results, f, indent=1, ensure_ascii=False)

    learner, corpus, dr = detail
    rules_json = {
        "note": "complete rule system of ablation=full_system, seed=%d" % SEEDS[0],
        "n_rules": len(learner.rules),
        "rules": [r.to_json(learner.symtab) for r in learner.rules],
        "internal_symbols": {
            name: {
                "created_by_rule": rec["created_by_rule"],
                "kind": rec["kind"],
                "sources": list(rec["sources"]),
                "alt_sources": [list(a) for a in rec["alt_sources"]],
                "expansion": "".join(learner.symtab.expand(name)),
                "derivation_depth": learner.symtab.depth(name),
            } for name, rec in learner.symtab.info.items()},
    }
    with open(os.path.join(outdir, "rules_v4.json"), "w") as f:
        json.dump(rules_json, f, indent=1, ensure_ascii=False)

    traces_json = {
        "note": "training traces + successful held-out derivations, "
                "ablation=full_system, seed=%d" % SEEDS[0],
        "training_traces": learner.traces,
        "heldout_success_derivations": dr["successful_heldout_derivations"],
        "rollback_log": learner.rollback_log,
    }
    with open(os.path.join(outdir, "traces_v4.json"), "w") as f:
        json.dump(traces_json, f, indent=1, ensure_ascii=False)

    probe_json = {
        "note": "post-hoc subject-family probe on the FROZEN full_system "
                "learners; evaluation only, nothing fed back into training",
        "aggregate": by_ablation["full_system"]["probe_aggregate"],
        "per_seed": [{"seed": r["seed"], "probe": r["subject_family_probe"]}
                     for r in by_ablation["full_system"]["per_seed"]],
    }
    with open(os.path.join(outdir, "probe_v4.json"), "w") as f:
        json.dump(probe_json, f, indent=1, ensure_ascii=False)

    # ------------------------------------------------------- readable summary
    print("-" * 78)
    print("EXAMPLE RULE SYSTEM (full_system, seed %d)" % SEEDS[0])
    print("-" * 78)
    ints = [r for r in learner.rules if r.kind == "INTERNAL" and r.n_apply > 1]
    ints.sort(key=lambda r: -r.n_apply)
    print(" most-used structural (epsilon) rules:")
    for r in ints[:5]:
        print("  " + r.text() + "    [applied %d, reused %d]" % (r.n_apply, r.n_reuse))
    cons = [r for r in learner.rules if r.kind == "OUTPUT" and r.n_apply > 1]
    cons.sort(key=lambda r: -r.n_apply)
    print(" most-used output rules:")
    for r in cons[:6]:
        print("  " + r.text() + "    [applied %d, reused %d, generalized %d]"
              % (r.n_apply, r.n_reuse, r.n_generalized))
    gen = [r for r in learner.rules if r.n_generalized > 0]
    gen.sort(key=lambda r: -r.n_generalized)
    print(" generalized rules (conditions relaxed by successful reuse):")
    for r in gen[:6]:
        print("  " + r.text() + "    [applied %d, generalizations %d, active=%s]"
              % (r.n_apply, r.n_generalized, r.active))
        for h in r.history:
            print("        was: (%s)" % ",".join(pat_str(e) for e in h))
    print("")

    # ------------------------------------------------------- FINAL RESULTS
    print("================ FINAL RESULTS V4 ================")
    print("")
    print("TARGET-BLIND CANDIDATE GENERATION: %s" % ("PASS" if blind_ok else "FAIL"))
    for p in blind_problems:
        print("   !! %s" % p)
    print("")
    for ablation, _ in ABLATIONS:
        a = by_ablation[ablation]["aggregate"]
        print("%s:" % ablation)
        print("  precision:                     %.4f  (sd %.4f)"
              % (a["precision_mean"], a["precision_std"]))
        print("  recall:                        %.4f  (sd %.4f)"
              % (a["recall_mean"], a["recall_std"]))
        print("  exact_set_accuracy:            %.4f" % a["exact_set_accuracy_mean"])
        print("  incompatible_generation_rate:  %.4f"
              % a["incompatible_generation_rate_mean"])
        print("  malformed_generation_rate:     %.4f"
              % a["malformed_generation_rate_mean"])
        print("  mean_completion_set_size:      %.2f"
              % a["mean_completion_set_size_mean"])
        print("")

    fa = by_ablation["full_system"]["aggregate"]
    print("random candidate proposals:        %d" % fa["total_candidates_proposed"])
    print("candidates rejected:               %d" % fa["total_candidates_rejected"])
    print("successful accidental target hits: %d"
          % fa["total_candidates_accidentally_correct"])
    print("training decisions:                %d" % fa["total_decisions"])
    print("solved using existing rules:       %d" % fa["total_solved_by_existing"])
    print("required new-rule search:          %d" % fa["total_required_new_rule_search"])
    print("unsolved training decisions:       %d" % fa["total_unsolved"])
    print("")
    print("new-rule requirement:")
    print("  first decile:                    %.4f" % fa["new_rule_rate_first_decile"])
    print("  last decile:                     %.4f" % fa["new_rule_rate_last_decile"])
    print("  (full_system, seed %d, by portion of training:)" % SEEDS[0])
    for row in by_ablation["full_system"]["per_seed"][0]["learning_dynamics"]:
        print("    %-6s decisions=%-4d existing=%-4d new_rule=%-4d unsolved=%-4d rate=%.3f"
              % (row["portion"], row["decisions"], row["solved_by_existing_rules"],
                 row["new_rule_created"], row["unsolved"], row["new_rule_rate"]))
    print("")
    print("rules reused:                      %.1f per seed (%d reuse events)"
          % (fa["reused_rules_mean"], fa["total_reuse_events"]))
    print("rules generalized:                 %.1f per seed (%d events)"
          % (fa["generalized_rules_mean"], fa["total_generalization_events"]))
    print("")
    print("generalizations rejected by counterexamples: %d"
          % fa["total_generalization_rejected_counterexample"])
    print("  (also rejected by the positive/coverage check: %d)"
          % fa["total_generalization_rejected_positive"])
    print("rollback/deactivation events:                %d" % fa["total_rollback_events"])
    print("active generalized rules with known contradictions: %d"
          % fa["active_generalized_with_contradictions"])
    print("  same count for no_counterexample_check ablation:  %d"
          % by_ablation["no_counterexample_check"]["aggregate"]
            ["active_generalized_with_contradictions"])
    print("")
    print("held-out successes using composition:        %d" % fa["composed_successes"])
    print("held-out successes using flat/raw rules only:%d" % fa["flat_successes"])
    print("")
    em = dr["emergent_structures"]
    print("emergent transparent structures:")
    print("  internal structures:            %d" % em["n_internal_structures"])
    print("  exact hidden-word matches:      %d (seed %d) / %.1f mean per seed"
          % (em["n_exact_hidden_word_matches"], SEEDS[0],
             fa["exact_hidden_word_matches_mean"]))
    print("  examples: %s" % ", ".join(
        "%s=%r(applied %d)" % (r["symbol"], r["expansion"], r["rule_applications"])
        for r in em["examples"][:6]))
    print("  matching hidden words: %s" % ", ".join(
        "%s=%r" % (r["symbol"], r["expansion"])
        for r in em["examples_matching_hidden_words"][:6]))
    print("  derivation depths: %s" % em["derivation_depths"])
    deepest = None
    for name in learner.symtab.info:
        if learner.symtab.info[name]["kind"] != "collapse":
            continue
        if deepest is None or learner.symtab.depth(name) > learner.symtab.depth(deepest):
            deepest = name
    if deepest is not None:
        print("  provenance of the deepest internal symbol (explain_symbol):")
        for ln in learner.symtab.explain(deepest)[:10]:
            print("    " + ln)
    print("")
    print("representative successful held-out derivations:")
    pool = []
    for ablrow in by_ablation["full_system"]["per_seed"]:
        pool.extend(ablrow["successful_heldout_derivations"])
    composed_ex = [d for d in pool if d["classification"] == "COMPOSED SUCCESS"]
    flat_ex = [d for d in pool if d["classification"] == "FLAT SUCCESS"]
    picked = composed_ex[:3] + flat_ex[:1]
    for d in picked:
        print("  [%s]  raw held-out prefix %r  ->  %r"
              % (d["classification"], d["prefix"], d["completion"]))
        for step in d["steps"]:
            print("      structural rewrite(s) %-14s -> internal symbol(s) %-16s"
                  " -> output rule %s IF (%s) -> OUT=%r"
                  % (step["internal_rules"] or "(none)",
                     ",".join("%s=%r" % (s, step["symbol_expansions"][s])
                              for s in step["internal_symbols"]) or "(none)",
                     step["output_rules"],
                     ",".join(step["output_rule_conditions"]), step["char"]))
        seen_syms = []
        for step in d["steps"]:
            for s in step["internal_symbols"]:
                if s not in seen_syms:
                    seen_syms.append(s)
        if d.get("seed") == SEEDS[0]:      # symbol names are per-seed
            for s in seen_syms[:2]:
                print("      recursive expansion of %s:" % s)
                for ln in learner.symtab.explain(s)[:8]:
                    print("        " + ln)
        print("      final completion: %r" % d["completion"])
    if not picked:
        print("  (none - no held-out completion was compatible)")
    print("")
    print("held-out generation, full_system, seed %d:" % SEEDS[0])
    for h in by_ablation["full_system"]["per_seed"][0]["heldout"]:
        print("  prefix %r" % h["prefix"])
        print("    possible_next_chars: %s" % h["possible_next_chars"])
        print("    generated (%d): %s" % (h["n_generated"], h["generated"][:8]))
        print("    compatible  : %s" % h["compatible"])
        print("    incompatible (%d): %s" % (h["n_incompatible"], h["incompatible"][:5]))
        print("    malformed    (%d): %s" % (h["n_malformed"], h["malformed"][:5]))
        print("    P=%.3f R=%.3f exact=%.0f"
              % (h["metrics"]["precision"], h["metrics"]["recall"],
                 h["metrics"]["exact_set_accuracy"]))
    print("")
    print("validation strings fully derivable by the frozen system: %.3f (full_system)"
          % fa["validation_derivable_rate"])
    print("")
    nprob = sum(len(r["cheat_control_problems"]) for r in all_rows)
    print("cheat-control violations:")
    print("  target-blind candidate generation .......... %s"
          % ("ok" if blind_ok else "VIOLATION"))
    print("  training archive unavailable at inference ... %s"
          % ("ok" if not any("archive" in p for r in all_rows
                             for p in r["cheat_control_problems"]) else "VIOLATION"))
    print("  no generator metadata inside the learner .... %s"
          % ("ok" if not any("metadata" in p or "generator" in p
                             for r in all_rows for p in r["cheat_control_problems"])
             else "VIOLATION"))
    print("  no word boundaries supplied ................. %s"
          % ("ok" if not any("boundary" in p for r in all_rows
                             for p in r["cheat_control_problems"]) else "VIOLATION"))
    print("  held-out strings absent from train/val ...... %s"
          % ("ok" if not any("held-out" in p for r in all_rows
                             for p in r["cheat_control_problems"]) else "VIOLATION"))
    print("  no rules created after freeze ............... %s"
          % ("ok" if not any("after freezing" in p for r in all_rows
                             for p in r["cheat_control_problems"]) else "VIOLATION"))
    print("  no target value stored in a candidate ....... %s"
          % ("ok" if not any("outside the observed" in p for r in all_rows
                             for p in r["cheat_control_problems"]) else "VIOLATION"))
    print("  total violation messages across %d runs: %d" % (len(all_rows), nprob))
    for r in all_rows:
        for p in r["cheat_control_problems"][:3]:
            print("    !! [%s seed %d] %s" % (r["ablation"], r["seed"], p))
    print("")
    print("files written: results_v4.json rules_v4.json traces_v4.json "
          "probe_v4.json")
    print("===================================================")

    # ---------------------------------------------------------------------
    # POST-HOC SUBJECT-FAMILY PROBE (frozen learner, evaluation only)
    # ---------------------------------------------------------------------
    print("")
    probe0 = by_ablation["full_system"]["per_seed"][0]["subject_family_probe"]
    print_subject_family_probe(probe0, corpus)

    pa = by_ablation["full_system"]["probe_aggregate"]
    fam_of = {}
    for fi, fam in enumerate(corpus.subject_families):
        for s in fam:
            fam_of[s] = fi

    print("================ SUBJECT FAMILY PROBE ================")
    print("(post-hoc, frozen learner, full_system; detail = seed %d, "
          "aggregates over %d seeds)" % (SEEDS[0], pa["seeds"]))
    print("")
    print("TRUE HIDDEN FAMILIES:")
    for fi, fam in enumerate(probe0["true_hidden_families"]):
        print("  family %d: %s" % (fi, sorted(fam)))
    print("")
    print("BEHAVIORAL GROUPS DISCOVERED:")
    for g in probe0["behavioral_groups"]:
        print("  %s   (hidden families %s)"
              % (g, sorted(set(fam_of[x] for x in g))))
    print("  seeds whose groups equal the hidden families: %d / %d"
          % (pa["behavioral"]["seeds_groups_equal_families"], pa["seeds"]))
    print("")
    print("RULE-SUBSTITUTION GROUPS DISCOVERED:")
    if probe0["substitution"] is not None:
        for g in probe0["substitution"]["groups"]:
            print("  %s   (hidden families %s)"
                  % (g, sorted(set(fam_of[x] for x in g))))
        print("  non-transitive triples (seed %d): %d"
              % (SEEDS[0], probe0["substitution"]["n_nontransitive_triples"]))
        print("  seeds whose groups equal the hidden families: %d / %d"
              % (pa["substitution"]["seeds_groups_equal_families"], pa["seeds"]))
    print("")
    print("SAME-FAMILY PAIRS CORRECTLY GROUPED:")
    b = probe0["behavioral_comparison"]
    print("  behavioural   seed %d: %d / %d   %s"
          % (SEEDS[0], b["same_family_pairs_grouped"], b["same_family_pairs_total"],
             b["same_family_pairs_correctly_grouped"]))
    print("  behavioural   all seeds: %d / %d"
          % (pa["behavioral"]["same_family_pairs_grouped"],
             pa["behavioral"]["same_family_pairs_total"]))
    if pa["substitution"]:
        s = probe0["substitution"]["comparison"]
        print("  substitution  seed %d: %d / %d   %s"
              % (SEEDS[0], s["same_family_pairs_grouped"],
                 s["same_family_pairs_total"], s["same_family_pairs_correctly_grouped"]))
        print("  substitution  all seeds: %d / %d"
              % (pa["substitution"]["same_family_pairs_grouped"],
                 pa["substitution"]["same_family_pairs_total"]))
    print("")
    print("DIFFERENT-FAMILY PAIRS INCORRECTLY GROUPED:")
    print("  behavioural   seed %d: %d / %d   %s"
          % (SEEDS[0], b["different_family_pairs_grouped"],
             b["different_family_pairs_total"],
             b["different_family_pairs_incorrectly_grouped"][:6]))
    print("  behavioural   all seeds: %d / %d"
          % (pa["behavioral"]["different_family_pairs_grouped"],
             pa["behavioral"]["different_family_pairs_total"]))
    if pa["substitution"]:
        s = probe0["substitution"]["comparison"]
        print("  substitution  seed %d: %d / %d   %s"
              % (SEEDS[0], s["different_family_pairs_grouped"],
                 s["different_family_pairs_total"],
                 s["different_family_pairs_incorrectly_grouped"][:6]))
        print("  substitution  all seeds: %d / %d"
              % (pa["substitution"]["different_family_pairs_grouped"],
                 pa["substitution"]["different_family_pairs_total"]))
    print("")
    print("COMMON INTERNAL RULES/SYMBOLS PER FAMILY:")
    for row in probe0["family_common_structures"]:
        print("  family %d %s" % (row["family"], row["members"]))
        print("      shared by all members: rules=%d  generalized=%d  "
              "internal-rules=%d  symbols=%d  symbols-alive-at-verb=%d"
              % (row["rules_applied_shared_by_family"],
                 row["generalized_rules_applied_shared_by_family"],
                 row["internal_rules_applied_shared_by_family"],
                 row["internal_symbols_in_views_shared_by_family"],
                 row["internal_symbols_active_at_verb_shared_by_family"]))
        print("      EXCLUSIVE to this family (shared by all its members and by")
        print("      no outsider): rules=%s  generalized=%s  internal-rules=%s"
              % (row["rules_applied_exclusive_to_family"][:6] or "[]",
                 row["generalized_rules_applied_exclusive_to_family"][:6] or "[]",
                 row["internal_rules_applied_exclusive_to_family"][:6] or "[]"))
        print("                    symbols=%s  symbols-alive-at-verb=%s"
              % (row["internal_symbols_in_views_exclusive_to_family"][:6] or "[]",
                 row["internal_symbols_active_at_verb_exclusive_to_family"][:6] or "[]"))
    ex = pa["family_exclusive_structure_counts"]
    print("  total family-exclusive structures over %d seeds x 3 families:"
          % pa["seeds"])
    print("      rules=%d  generalized=%d  internal-rules=%d  symbols=%d  "
          "symbols-alive-at-verb=%d" % tuple(ex))
    print("")
    print("CONCLUSION:")
    print("  behavioural signatures ....... %s" % pa["behavioral"]["verdict"])
    if pa["substitution"]:
        print("  rule substitution ............ %s" % pa["substitution"]["verdict"])
    print("  family-exclusive internal structure: %s"
          % ("present" if sum(ex) > 0 else "none found"))
    verdicts = [pa["behavioral"]["verdict"]]
    if pa["substitution"]:
        verdicts.append(pa["substitution"]["verdict"])
    if "clearly discovered" in verdicts:
        overall = "subject families clearly discovered"
    elif "partially discovered" in verdicts:
        overall = "subject families partially discovered"
    else:
        overall = "subject families NOT discovered"
    print("  overall: %s" % overall)
    print("======================================================")


if __name__ == "__main__":
    main()
