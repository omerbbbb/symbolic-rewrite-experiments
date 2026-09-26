#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
symbolic_rewrite_experiment.py

A purely discrete, symbolic autoregressive learner over raw character strings.

NO numerical learning of any kind is used inside the learner:
  no tokenizer, no embeddings, no neural nets, no gradients, no probabilities,
  no similarity scores, no vector distances, no softmax, no learned floats.
The learner's entire state is:
  * a 2D grid of discrete symbol cells,
  * symbolic previous/next links between arrivals,
  * a set of discrete rewrite rules,
  * a provenance table for internally created symbols.

Randomness is used only where the specification demands it (random cell
placement, random candidate rule generation, seeding).  Random draws are not
scores and are never used to rank rules.

Run:   python3 symbolic_rewrite_experiment.py
Saves: results.json, rules.json, traces.json
Prints a concise human-readable summary and a final delimited block.

Standard library only.
"""

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

        # ---- new-rule random search ----------------------------------------
        self.new_rule_budget = 40          # safety budget of random candidates
        self.p_propose_collapse = 0.7      # chance a candidate proposes structure
        self.p_propose_marker = 0.12       # chance a candidate writes a marker
        self.p_propose_require = 0.10      # chance a candidate adds a presence test

        # ---- generalization -------------------------------------------------
        self.allow_generalization = True
        self.allow_reuse = True            # reuse in *new* contexts
        self.allow_composition = True      # internal symbols / hierarchy
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


# =============================================================================
# SECTION 1.  HIDDEN SYNTHETIC LANGUAGE (GENERATOR METADATA)
#
# This object holds the hidden classes.  It is NEVER passed to the learner and
# never referenced from learner code.  A cheat-control checks both facts.
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
        * 6 relation/verb words, 2 per subject family (verb sets are disjoint)
        * 3 object families, 4 objects each
        * compatibility:  subjects(family F) + verb v  ->  objects(fam_of[v])
          i.e. the licensed object family is a function of the (subject family,
          verb) pair; since verb sets are disjoint per subject family this is
          equivalent to a function of the verb.
    """

    def __init__(self, seed):
        rng = random.Random(seed * 7919 + 13)
        self.seed = seed
        self._access_count = 0

        words = set()

        def fresh():
            # variable-length nonsense words (4, 5 or 6 characters) so that no
            # boundary can be inferred from a fixed length.
            for _ in range(4000):
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
                words.add(w)
                return w
            raise RuntimeError("lexicon generation failed")

        self.subject_families = [[fresh() for _ in range(3)] for _ in range(3)]
        self.verbs = [fresh() for _ in range(6)]
        self.object_families = [[fresh() for _ in range(4)] for _ in range(3)]

        # verb sets are disjoint per subject family: family i uses verbs 2i,2i+1
        self.family_verbs = {i: [self.verbs[2 * i], self.verbs[2 * i + 1]]
                             for i in range(3)}
        # each verb licenses one object family
        objfam_assignment = [0, 1, 0, 2, 1, 2]
        self.verb_objfam = {self.verbs[i]: objfam_assignment[i] for i in range(6)}

        self.subj_family = {}
        for fi, fam in enumerate(self.subject_families):
            for s in fam:
                self.subj_family[s] = fi
        self.obj_family = {}
        for fi, fam in enumerate(self.object_families):
            for o in fam:
                self.obj_family[o] = fi

        # ---- full grammatical language -------------------------------------
        self.all_triples = []
        for fi in range(3):
            for s in self.subject_families[fi]:
                for v in self.family_verbs[fi]:
                    for o in self.object_families[self.verb_objfam[v]]:
                        self.all_triples.append((s, v, o))
        self.triple_of = {s + v + o + EOS: (s, v, o) for (s, v, o) in self.all_triples}

        # ---- held-out COMBINATIONS (entire (subject, verb) pairs) -----------
        pairs = []
        for fi in range(3):
            for s in self.subject_families[fi]:
                for v in self.family_verbs[fi]:
                    pairs.append((s, v))
        rng.shuffle(pairs)
        # two held-out pairs, from two different subject families
        chosen = []
        used_fams = set()
        for (s, v) in pairs:
            fi = self.subj_family[s]
            if fi in used_fams:
                continue
            chosen.append((s, v))
            used_fams.add(fi)
            if len(chosen) == 2:
                break
        self.heldout_pairs = chosen

        self.heldout_compatible = {}     # prefix -> set of compatible strings
        self.heldout_incompatible = {}   # prefix -> set of incompatible strings
        for (s, v) in self.heldout_pairs:
            good_fam = self.verb_objfam[v]
            self.heldout_compatible[s + v] = set(
                s + v + o + EOS for o in self.object_families[good_fam])
            bad = set()
            for fi in range(3):
                if fi == good_fam:
                    continue
                for o in self.object_families[fi]:
                    bad.add(s + v + o + EOS)
            self.heldout_incompatible[s + v] = bad

        heldout_strings = set()
        for pfx in self.heldout_compatible:
            heldout_strings |= self.heldout_compatible[pfx]
        self.heldout_strings = heldout_strings

        remaining = [s + v + o + EOS for (s, v, o) in self.all_triples
                     if (s + v + o + EOS) not in heldout_strings]
        rng.shuffle(remaining)
        cut = int(len(remaining) * 0.8)
        train = remaining[:cut]
        val = remaining[cut:]
        # repair the split so that every held-out subject and every held-out verb
        # is still observable in training *in other combinations* (otherwise the
        # hidden structure would not be recoverable even in principle)
        for (s, v) in self.heldout_pairs:
            for piece in (s, v):
                if not any(piece in t for t in train):
                    for i, t in enumerate(val):
                        if piece in t:
                            train.append(val.pop(i))
                            break
        self.train_strings = sorted(train)
        self.val_strings = sorted(val)

        self._sanity()

    # ---- guarded accessors (used ONLY by the evaluation harness) ------------
    def touch(self):
        self._access_count += 1

    @property
    def access_count(self):
        return self._access_count

    def _sanity(self):
        # every held-out subject and verb must still be observable separately
        for (s, v) in self.heldout_pairs:
            assert any(t.startswith(s) for t in self.train_strings), \
                "held-out subject never appears in training"
            assert any(v in t for t in self.train_strings), \
                "held-out verb never appears in training"
        for t in self.train_strings:
            assert t not in self.heldout_strings
        for t in self.val_strings:
            assert t not in self.heldout_strings
        for t in self.train_strings + self.val_strings:
            assert " " not in t and t.endswith(EOS) and t.count(EOS) == 1

    def decompose(self, s):
        """EVALUATION ONLY.  Returns (subject, verb, object) or None."""
        self.touch()
        if not s.endswith(EOS):
            return None
        body = s[:-1]
        for subj in self.subj_family:
            if not body.startswith(subj):
                continue
            rest = body[len(subj):]
            for v in self.verbs:
                if not rest.startswith(v):
                    continue
                o = rest[len(v):]
                if o in self.obj_family:
                    return (subj, v, o)
        return None

    def lexicon(self):
        self.touch()
        out = []
        for fam in self.subject_families:
            out.extend(fam)
        out.extend(self.verbs)
        for fam in self.object_families:
            out.extend(fam)
        return out


# =============================================================================
# SECTION 2.  TRANSPARENT INTERNAL SYMBOLS (PROVENANCE)
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

    def new_symbol(self, rule_id, sources, kind="collapse"):
        self.n += 1
        name = u"\u00a7%d" % self.n          # e.g. §7
        self.info[name] = {
            "name": name,
            "created_by_rule": rule_id,
            "kind": kind,
            "sources": tuple(sources),
            "alt_sources": [],
        }
        return name

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
# SECTION 3.  THE 2D SYMBOLIC GRID + ARRIVAL LINKS
# =============================================================================


class State(object):
    """Discrete state: a grid of symbol cells plus prev/next arrival links.

    * Cells contain raw characters, internal symbols, or nothing (absent key).
    * Grid coordinates are RANDOM and carry no order information.
    * Input order is retained ONLY by the symbolic prev/next links.
    """

    __slots__ = ("grid", "head", "prv", "nxt", "free", "free_idx",
                 "_chain", "_sig", "_offchain")

    def __init__(self, grid, head, prv, nxt, free, free_idx):
        self.grid = grid
        self.head = head
        self.prv = prv
        self.nxt = nxt
        self.free = free              # shared, pre-shuffled coordinate tuple
        self.free_idx = free_idx
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
        return State({c0: BOS}, c0, {}, {}, coords, 1)

    def copy(self):
        return State(dict(self.grid), self.head, dict(self.prv), dict(self.nxt),
                     self.free, self.free_idx)

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
            self._sig = (self.chain_symbols(), self.offchain_symbols())
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
        order information.)"""
        ns = self.copy()
        coord, nidx = ns._next_free()
        ns.free_idx = nidx
        ns.grid[coord] = symbol
        if ns.head is not None:
            ns.nxt[ns.head] = coord
            ns.prv[coord] = ns.head
        ns.head = coord
        ns._chain = None
        ns._sig = None
        ns._offchain = None
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


# =============================================================================
# SECTION 4.  DISCRETE REWRITE RULES
# =============================================================================


class Rule(object):
    """A discrete symbolic transformation.

    Conditions (all discrete, no scores):
      * `pattern`   : a tuple matched against the last N linked positions.
                      Elements are literals, wildcards, or variables.
      * `require`   : symbols that must be present somewhere in the grid.
      * `incoming`  : for CONSUME rules, the raw character being licensed;
                      None for INTERNAL (epsilon) rules.

    Effects (`actions`), applied in order:
      ('COLLAPSE', k, sym)  replace the last k linked symbols with one symbol
      ('MARK', sym)         write a reusable internal marker into a free cell
      ('ERASE', sym)        erase a marker
      ('RELABEL', i, sym)   replace one symbol i positions from the head

    The number of inspected / modified locations is bounded by the config and
    can be raised up to the whole grid.
    """

    __slots__ = ("rid", "kind", "incoming", "pattern", "protected", "require",
                 "actions", "created_at", "origin", "contexts", "active",
                 "n_apply", "n_reuse", "n_generalized", "history",
                 "merged_from", "subsumed_by", "state_sigs")

    def __init__(self, rid, kind, incoming, pattern, require, actions,
                 created_at, origin, protected=None):
        self.rid = rid
        self.kind = kind                  # 'CONSUME' | 'INTERNAL'
        self.incoming = incoming
        self.pattern = tuple(pattern)
        self.protected = frozenset(protected or ())
        self.require = tuple(require)
        self.actions = tuple(actions)
        self.created_at = created_at      # (epoch, step, string index)
        self.origin = origin              # 'random-search' | 'fallback'
        self.contexts = []                # successful matched tails
        self.state_sigs = []              # full situations it succeeded in
        self.active = True
        self.n_apply = 0
        self.n_reuse = 0                  # successes after the creating one
        self.n_generalized = 0
        self.history = []                 # list of previous patterns
        self.merged_from = []
        self.subsumed_by = None

    # ---- matching ----------------------------------------------------------
    def match(self, state):
        chain = state.chain_symbols()
        L = len(self.pattern)
        if L > len(chain):
            return None
        tail = chain[len(chain) - L:]
        bind = {}
        for pe, s in zip(self.pattern, tail):
            t = pe[0]
            if t == "L":
                if pe[1] != s:
                    return None
            elif t == "V":
                if pe[1] in bind:
                    if bind[pe[1]] != s:
                        return None
                else:
                    bind[pe[1]] = s
            # wildcard: always matches
        if self.require:
            present = state.offchain_symbols()
            for sym in self.require:
                if sym not in present:
                    return None
        return (tail, bind)

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
            bind = {}
            ok = True
            for pe, sy in zip(self.pattern, chain[i:i + L]):
                if pe[0] == "L":
                    if pe[1] != sy:
                        ok = False
                        break
                elif pe[0] == "V":
                    if pe[1] in bind:
                        if bind[pe[1]] != sy:
                            ok = False
                            break
                    else:
                        bind[pe[1]] = sy
            if ok:
                out.append(i)
        return out

    def apply_at(self, state, start, symtab):
        """Apply an INTERNAL rule at a given segment start position."""
        st = state
        for act in self.actions:
            if act[0] == "COLLAPSE":
                k, sym = act[1], act[2]
                chain = st.chain_symbols()
                if start + k > len(chain):
                    return None
                symtab.record_alternate(sym, chain[start:start + k])
                st = st.collapse_segment(start, k, sym)
            elif act[0] == "MARK":
                st = st.write_marker(act[1])
            elif act[0] == "ERASE":
                st = st.erase_marker(act[1])
            elif act[0] == "RELABEL":
                st = st.relabel(act[1], act[2])
            if st is None:
                return None
        return st

    # ---- capacity accounting ----------------------------------------------
    def reads(self):
        return len(self.pattern) + len(self.require)

    def writes(self):
        w = 0
        for a in self.actions:
            if a[0] == "COLLAPSE":
                w += a[1] + 1
            elif a[0] in ("MARK", "ERASE", "RELABEL"):
                w += 1
        if self.kind == "CONSUME":
            w += 1
        return w

    # ---- human-readable form ----------------------------------------------
    def text(self):
        cond = []
        if self.kind == "CONSUME":
            cond.append("incoming=%r" % self.incoming)
        cond.append("linked-tail=(%s)" % ",".join(pat_str(e) for e in self.pattern))
        if self.require:
            cond.append("grid-contains{%s}" % ",".join(self.require))
        eff = []
        if self.kind == "CONSUME":
            eff.append("license+append incoming")
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
        return "R%-4d %-8s IF %s THEN %s" % (
            self.rid, self.kind, " AND ".join(cond), "; ".join(eff))

    def to_json(self, symtab):
        return {
            "rid": self.rid,
            "kind": self.kind,
            "incoming": self.incoming,
            "pattern": [pat_str(e) for e in self.pattern],
            "require": list(self.require),
            "actions": [list(map(str, a)) for a in self.actions],
            "origin": self.origin,
            "created_at": list(self.created_at),
            "active": self.active,
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
        self.alphabet = set()            # derived from the raw training strings

        # indices: last-pattern-element -> rules
        self._cons_idx = defaultdict(list)
        self._int_idx = defaultdict(list)
        self._exact = {}          # exact rule signature -> rule (dedup)
        self._int_pat = {}        # internal collapse pattern -> rule (dedup)

        # diagnostics (counts only; never used to choose rules)
        self.stats = {
            "decisions": 0,
            "solved_by_existing": 0,
            "new_rule_created": 0,
            "fallback_rules": 0,
            "generalization_events": 0,
            "generalization_rejected": 0,
            "reuse_events": 0,
        }
        self.dynamics = []               # per-decision: 1 if a new rule was needed
        self.traces = []                 # successful derivation traces

    # ------------------------------------------------------------------ util
    def _register(self, rule):
        self.rules.append(rule)
        self._exact[(rule.kind, rule.incoming, rule.pattern, rule.require,
                     rule.actions)] = rule
        if rule.kind == "INTERNAL":
            self._int_pat[rule.pattern] = rule
            self._int_idx[rule.first_key()].append(rule)
        else:
            self._cons_idx[rule.index_key()].append(rule)

    def _reindex(self, rule, old_key):
        idx = self._cons_idx if rule.kind == "CONSUME" else self._int_idx
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

    def _candidates(self, state, idx):
        chain = state.chain_symbols()
        last = chain[-1] if chain else None
        out = idx.get("*", [])
        if last is not None and last in idx:
            out = out + idx[last]
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
                    ns = r.apply_at(st, pos, self.symtab)
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

    def _licensing_rules(self, st, target=None):
        for r in self._candidates(st, self._cons_idx):
            if target is not None and r.incoming != target:
                continue
            if not self._usable(r, st):
                continue
            m = r.match(st)
            if m is None:
                continue
            yield r, m[0]

    def find_path(self, state, target):
        """Try to reach `target` using ONLY rules that already exist.

        A rule either applies or does not.  A path either reaches the target or
        does not.  Nothing is scored, nothing is ranked.
        """
        for st, ipath in self.expansions(state):
            for r, tail in self._licensing_rules(st, target):
                return (st, ipath, r, tail)
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

    def _rule_exists(self, kind, incoming, pattern, require, actions):
        """Exact duplicate lookup.  `actions=None` means: any effect."""
        if actions is None:
            if kind == "INTERNAL":
                return self._int_pat.get(pattern)
            for key, r in self._exact.items():
                if key[0] == kind and key[1] == incoming and key[2] == pattern:
                    return r
            return None
        return self._exact.get((kind, incoming, pattern, tuple(require),
                                tuple(actions)))

    def _propose_internal(self, st, where):
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
        if self._rule_exists("INTERNAL", None, pat, (), None) is not None:
            return None, st
        rid = self._new_rid()
        sym = self.symtab.new_symbol(rid, seg)
        acts = [("COLLAPSE", k, sym)]
        if self.rng.random() < cfg.p_propose_marker:
            acts.append(("MARK", self.symtab.new_symbol(rid, seg, kind="marker")))
        cand = Rule(rid, "INTERNAL", None, pat, (), tuple(acts), where,
                    "random-search", protected=range(len(pat)))
        if cand.reads() > cfg.max_read_cells or cand.writes() > cfg.max_write_cells:
            self.next_rid -= 1
            return None, st
        ns = cand.apply_at(st, start, self.symtab)
        if ns is None:
            self.next_rid -= 1
            return None, st
        return cand, ns

    def create_rule_for_target(self, state, target, where):
        """Random candidate generation (specification section 8).

        Candidates are generated randomly.  A candidate is kept only if it
        participates in a path that reaches the known target; otherwise further
        candidates are drawn until the safety budget is exhausted.
        """
        assert not self.frozen, "no rule creation at inference time"
        cfg = self.cfg
        exps = self.expansions(state)
        for _attempt in range(cfg.new_rule_budget):
            # IMPLEMENTATION DECISION: half the random draws inspect the plain
            # raw view and half a randomly chosen restructured view, so that
            # composition never starves the flat route of candidates.
            if self.rng.random() < 0.5:
                st, ipath = exps[0]
            else:
                st, ipath = exps[self.rng.randrange(len(exps))]
            created_internal = None
            st2 = st
            if cfg.allow_composition and self.rng.random() < cfg.p_propose_collapse:
                created_internal, st2 = self._propose_internal(st, where)

            # DESIGN DECISION: the conditions of a candidate consuming rule are
            # always the maximally specific ones available (the longest
            # inspectable linked tail of the chosen view).  Generality is never
            # invented at creation time; it can only arise later, out of
            # successful reuse (specification section 9).
            pat = self._pattern_from_state(st2, cfg.max_pattern_len)
            require = ()
            if self.rng.random() < cfg.p_propose_require:
                off = st2.offchain_symbols()
                if off:
                    require = (off[self.rng.randrange(len(off))],)

            if self._rule_exists("CONSUME", target, pat, require, ()) is not None:
                # this consuming condition already exists; the candidate adds
                # nothing new, so another candidate is drawn.  Any new
                # structural rule proposed along the way is kept, since it does
                # enlarge the rewrite system.
                if created_internal is not None:
                    self._register(created_internal)
                    exps = self.expansions(state)
                continue

            rid = self._new_rid()
            cand = Rule(rid, "CONSUME", target, pat, require, (), where,
                        "random-search")
            if cand.reads() > cfg.max_read_cells or cand.writes() > cfg.max_write_cells:
                self.next_rid -= 1
                continue
            m = cand.match(st2)
            if m is None:
                self.next_rid -= 1
                continue
            if created_internal is not None:
                self._register(created_internal)
                ipath = ipath + (created_internal.rid,)
            self._register(cand)
            return (st2, ipath, cand, m[0])

        # ---- safety budget exhausted: maximally specific fallback ----------
        st, ipath = exps[0]
        pat = self._pattern_from_state(st, self.cfg.max_pattern_len)
        existing = self._rule_exists("CONSUME", target, pat, (), ())
        if existing is not None:
            return (st, ipath, existing, existing.match(st)[0])
        rid = self._new_rid()
        cand = Rule(rid, "CONSUME", target, pat, (), (), where, "fallback")
        self._register(cand)
        self.stats["fallback_rules"] += 1
        return (st, ipath, cand, cand.match(st)[0])

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
            for pe, s in zip(pattern, tail):
                if pe[0] == "L" and pe[1] != s:
                    return False
                if pe[0] == "V":
                    return False
        return True

    def _relax(self, rule, new_pattern, reason, extra_contexts=()):
        # a rule may never dissolve into an unconditional licence: at least one
        # symbolic condition must survive every generalization
        if not any(e[0] != "A" for e in new_pattern):
            self.stats["generalization_rejected"] += 1
            return False
        if not self._reproduces(new_pattern, list(rule.contexts) + list(extra_contexts)):
            self.stats["generalization_rejected"] += 1
            return False
        old_key = rule.index_key()
        rule.history.append(rule.pattern)
        rule.pattern = new_pattern
        rule.n_generalized += 1
        for t in extra_contexts:
            if t not in rule.contexts and len(rule.contexts) < self.cfg.max_contexts_per_rule:
                rule.contexts.append(t)
        self._reindex(rule, old_key)
        self.stats["generalization_events"] += 1
        rule.merged_from.append(reason)
        return True

    def try_merge(self, new_rule):
        """Generalization by shared symbolic structure.

        Two rules that license the same character with the same effects and the
        same pattern length are merged when their conditions differ in at most
        `max_relax_positions` relaxable positions: the agreeing conditions are
        retained and the differing ones become wildcards.  No numbers involved.
        """
        if not self.cfg.allow_generalization or new_rule.kind != "CONSUME":
            return None
        L = len(new_rule.pattern)
        for r in self.rules:
            if r is new_rule or not r.active or r.kind != "CONSUME":
                continue
            if r.incoming != new_rule.incoming or len(r.pattern) != L:
                continue
            if r.actions != new_rule.actions or r.require != new_rule.require:
                continue
            merged = []
            diffs = 0
            ok = True
            for i, (a, b) in enumerate(zip(r.pattern, new_rule.pattern)):
                if a == b:
                    merged.append(a)
                    continue
                if i in r.protected or i in new_rule.protected:
                    ok = False
                    break
                merged.append(WILD)
                diffs += 1
            if not ok or diffs == 0 or diffs > self.cfg.max_relax_positions:
                continue
            extra = list(new_rule.contexts)
            if self._relax(r, tuple(merged), "merge-with-R%d" % new_rule.rid, extra):
                new_rule.active = False
                new_rule.subsumed_by = r.rid
                return r
        return None

    def try_self_generalize(self, rule, tail):
        """Triggered when a known rule succeeds again in a DIFFERENT context."""
        if not self.cfg.allow_generalization or rule.kind != "CONSUME":
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

    # ------------------------------------------------------------ training
    def train(self, strings):
        assert not self.frozen
        cfg = self.cfg
        for s in strings:
            assert isinstance(s, str) and " " not in s, "learner sees raw strings only"
            self.alphabet.update(s)
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

    def _train_string(self, s, epoch, si):
        # reproducible per-sequence grid initialization
        state = State.initial(self.cfg, self.seed * 1000003 + epoch * 9176 + si)
        trace = {"string_index": si, "epoch": epoch, "steps": []}
        for t in range(len(s)):
            target = s[t]
            self.stats["decisions"] += 1

            # (7) FIRST try to reach the target with rules that already exist
            found = self.find_path(state, target)
            if found is not None:
                st, ipath, rule, tail = found
                new_rule = False
                first_time = (len(rule.contexts) == 0)
                if not first_time:
                    rule.n_reuse += 1
                    self.stats["reuse_events"] += 1
                    if tail not in rule.contexts:
                        # (9) the rule succeeded again in a DIFFERENT context
                        self.try_self_generalize(rule, tail)
                self._record_success(rule, tail, st)
                self.stats["solved_by_existing"] += 1
            else:
                # (8) only now may a new rule be searched for
                st, ipath, rule, tail = self.create_rule_for_target(
                    state, target, (epoch, t, si))
                self._record_success(rule, tail, st)
                merged = self.try_merge(rule)
                if merged is not None:
                    rule = merged
                new_rule = True
                self.stats["new_rule_created"] += 1

            # the persistent state is always the raw arrival chain: the incoming
            # character simply enters the grid at a random free location, linked
            # after the previous arrival.
            state = state.place_arrival(target)
            self.dynamics.append(1 if new_rule else 0)
            trace["steps"].append({
                "t": t, "target": target, "new_rule": new_rule,
                "internal_rules": list(ipath), "consume_rule": rule.rid,
                "matched_tail": [str(x) for x in tail],
            })
        if len(self.traces) < 40:
            self.traces.append(trace)

    # ----------------------------------------------------------- inference
    def freeze(self):
        self.frozen = True
        self.rid_at_freeze = self.next_rid

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

    def _licensed(self, state):
        chars = set()
        for st, _ip in self.expansions(state):
            for r, _tail in self._licensing_rules(st):
                chars.add(r.incoming)
        return chars

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
        symbolic rules.  Returns the set of complete strings ending in EOS."""
        results = set()
        nodes = [0]
        start, _ = self._state_for_prefix(prefix)

        def rec(state, suffix):
            if nodes[0] >= self.cfg.max_completion_nodes:
                return
            if len(results) >= self.cfg.max_completions:
                return
            if len(suffix) >= max_length:
                return
            nodes[0] += 1
            for ch in sorted(self._licensed(state)):
                if ch == EOS:
                    results.add(prefix + suffix + EOS)
                    continue
                rec(state.place_arrival(ch), suffix + ch)

        rec(start, "")
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
# SECTION 6.  CONTROLS AGAINST CHEATING
# =============================================================================


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

    # 2. the learner only ever received raw single-token strings
    for s in train_strings + val_strings:
        if " " in s or "\t" in s:
            problems.append("whitespace separator found in learner input")
        if s.count(EOS) != 1 or not s.endswith(EOS):
            problems.append("malformed learner input: %r" % s)

    # 3. no hidden class label or hidden word list ever entered the alphabet
    for a in learner.alphabet:
        if len(a) != 1:
            problems.append("non-character symbol in learner alphabet: %r" % a)

    # 4. held-out strings never appear in training material
    for s in train_strings + val_strings:
        if s in corpus.heldout_strings:
            problems.append("held-out string leaked into training: %r" % s)

    # 5. at inference no rule may be created
    if phase == "post-inference":
        if learner.next_rid != learner.rid_at_freeze:
            problems.append("rules were created after freezing")
        if not learner.frozen:
            problems.append("learner was not frozen at inference")
    return problems


# =============================================================================
# SECTION 7.  EVALUATION (OUTSIDE THE LEARNER)
#
# Everything below may use the hidden metadata.  None of it is ever visible to
# the learner and none of it feeds back into training or rule choice.
# =============================================================================


def classify_completions(corpus, prefix, generated):
    compat = corpus.heldout_compatible[prefix]
    incompat = corpus.heldout_incompatible[prefix]
    out = {"compatible": [], "incompatible": [], "not_in_language": []}
    for g in sorted(generated):
        if g in compat:
            out["compatible"].append(g)
        elif g in incompat:
            out["incompatible"].append(g)
        elif corpus.decompose(g) is not None:
            # grammatical string of the language, but not one of this prefix's
            # held-out combinations (can only happen if the prefix was extended
            # in a way that re-segments) -> treat as incompatible
            out["incompatible"].append(g)
        else:
            out["not_in_language"].append(g)
    return out


def metrics_for(prefix, corpus, cls):
    target = corpus.heldout_compatible[prefix]
    gen = cls["compatible"] + cls["incompatible"] + cls["not_in_language"]
    n = len(gen)
    tp = len(cls["compatible"])
    precision = (tp / float(n)) if n else 0.0
    recall = tp / float(len(target))
    exact = 1.0 if (set(cls["compatible"]) == target and
                    not cls["incompatible"] and not cls["not_in_language"]) else 0.0
    invalid = (len(cls["not_in_language"]) / float(n)) if n else 0.0
    return {"generated": n, "true_positive": tp, "precision": precision,
            "recall": recall, "exact_set_accuracy": exact,
            "invalid_generation_rate": invalid}


def rule_diagnostics(learner):
    rules = learner.rules
    active = [r for r in rules if r.active]
    consume = [r for r in rules if r.kind == "CONSUME"]
    internal = [r for r in rules if r.kind == "INTERNAL"]
    reused = [r for r in rules if r.n_reuse > 0]
    generalized = [r for r in rules if r.n_generalized > 0]
    never = [r for r in rules if r.n_apply <= 1 and r.n_reuse == 0]
    # composition depth: how deep the symbols a rule refers to are nested
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
        "consume_rules": len(consume),
        "internal_rules": len(internal),
        "created_by_random_search": len([r for r in rules if r.origin == "random-search"]),
        "created_by_fallback": len([r for r in rules if r.origin == "fallback"]),
        "reused_rules": len(reused),
        "generalized_rules": len(generalized),
        "never_reused_rules": len(never),
        "max_rule_composition_depth": max(comp),
        "mean_rule_composition_depth": sum(comp) / float(len(comp)),
        "max_derivation_depth": max(der),
        "internal_symbols": len(learner.symtab.info),
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
        out.append({
            "portion": "%d/%d" % (i + 1, buckets),
            "decisions": len(seg),
            "solved_by_existing_rules": len(seg) - sum(seg),
            "new_rule_created": sum(seg),
            "new_rule_rate": sum(seg) / float(len(seg)),
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
# SECTION 8.  EXPERIMENT RUNNER
# =============================================================================

ABLATIONS = [
    ("full_system", {}),
    ("no_reuse_or_generalization", {"allow_reuse": False,
                                    "allow_generalization": False}),
    ("no_composition", {"allow_composition": False}),
    ("shuffled_input_order", {"shuffle_input": True}),
    ("no_generalization_only", {"allow_generalization": False}),
    ("unrestricted_generalization", {"max_relax_positions": 4}),
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


def run_one(ablation, overrides, seed, collect_detail=False):
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
    for (subj, verb) in corpus.heldout_pairs:
        prefix = subj + verb          # constructed OUTSIDE the learner
        pre = corpus.access_count
        lic, tot = learner.prefix_licensing_coverage(prefix)
        nxt = sorted(learner.possible_next_chars(prefix))
        gen = learner.possible_completions(prefix, cfg.max_completion_len)
        assert corpus.access_count == pre, \
            "the corpus generator was queried during inference"
        cls = classify_completions(corpus, prefix, gen)
        met = metrics_for(prefix, corpus, cls)
        heldout_results.append({
            "prefix": prefix,
            "prefix_chars_licensed_by_existing_rules": lic,
            "prefix_length": tot,
            "possible_next_chars": nxt,
            "n_generated": len(gen),
            "generated": sorted(gen)[:60],
            "compatible": cls["compatible"],
            "incompatible": cls["incompatible"],
            "not_in_language": cls["not_in_language"][:30],
            "n_not_in_language": len(cls["not_in_language"]),
            "target_compatible_set": sorted(corpus.heldout_compatible[prefix]),
            "metrics": met,
        })

    problems += cheat_controls(learner, corpus, learner_input, val, "post-inference")

    diag = rule_diagnostics(learner)
    dyn = learning_dynamics(learner)
    emerg = emergent_structures(learner, corpus, train)
    valcov = validation_coverage(learner, val)

    mean = lambda xs: (sum(xs) / float(len(xs))) if xs else 0.0
    agg = {
        "precision": mean([h["metrics"]["precision"] for h in heldout_results]),
        "recall": mean([h["metrics"]["recall"] for h in heldout_results]),
        "exact_set_accuracy": mean([h["metrics"]["exact_set_accuracy"] for h in heldout_results]),
        "invalid_generation_rate": mean([h["metrics"]["invalid_generation_rate"] for h in heldout_results]),
        "mean_generated": mean([h["metrics"]["generated"] for h in heldout_results]),
        "prefix_licensed_rate": mean(
            [h["prefix_chars_licensed_by_existing_rules"] / float(h["prefix_length"])
             for h in heldout_results]),
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
            "heldout_prefixes": [s + v for (s, v) in corpus.heldout_pairs],
            "heldout_compatible_strings": {p: sorted(corpus.heldout_compatible[p])
                                           for p in corpus.heldout_compatible},
            "heldout_incompatible_strings": {p: sorted(corpus.heldout_incompatible[p])
                                             for p in corpus.heldout_incompatible},
            "hidden_lexicon_FOR_EVALUATION_ONLY": {
                "subject_families": corpus.subject_families,
                "verbs": corpus.verbs,
                "verb_object_family": corpus.verb_objfam,
                "object_families": corpus.object_families,
            },
        },
        "learner_stats": dict(learner.stats),
        "rule_diagnostics": diag,
        "learning_dynamics": dyn,
        "emergent_structures": emerg,
        "validation_coverage": valcov,
        "heldout": heldout_results,
        "aggregate": agg,
        "cheat_control_problems": problems,
    }
    if collect_detail:
        result["_learner"] = learner
        result["_corpus"] = corpus
    return result


def aggregate(rows):
    mean = lambda xs: (sum(xs) / float(len(xs))) if xs else 0.0
    def std(xs):
        if len(xs) < 2:
            return 0.0
        m = mean(xs)
        return (sum((x - m) ** 2 for x in xs) / float(len(xs) - 1)) ** 0.5
    keys = ["precision", "recall", "exact_set_accuracy",
            "invalid_generation_rate", "mean_generated", "prefix_licensed_rate"]
    out = {}
    for k in keys:
        xs = [r["aggregate"][k] for r in rows]
        out[k + "_mean"] = mean(xs)
        out[k + "_std"] = std(xs)
    out["rules_mean"] = mean([r["rule_diagnostics"]["total_rules"] for r in rows])
    out["generalized_rules_mean"] = mean([r["rule_diagnostics"]["generalized_rules"] for r in rows])
    out["internal_symbols_mean"] = mean([r["rule_diagnostics"]["internal_symbols"] for r in rows])
    out["exact_hidden_word_matches_mean"] = mean(
        [r["emergent_structures"]["n_exact_hidden_word_matches"] for r in rows])
    out["new_rule_rate_first_decile"] = mean(
        [r["learning_dynamics"][0]["new_rule_rate"] for r in rows if r["learning_dynamics"]])
    out["new_rule_rate_last_decile"] = mean(
        [r["learning_dynamics"][-1]["new_rule_rate"] for r in rows if r["learning_dynamics"]])
    out["validation_derivable_rate"] = mean([r["validation_coverage"]["rate"] for r in rows])
    out["seeds"] = len(rows)
    return out


# =============================================================================
# SECTION 9.  MAIN
# =============================================================================

SEEDS = list(range(1, 11))       # 10 independent random seeds


def main():
    outdir = os.path.dirname(os.path.abspath(__file__))
    all_rows = []
    by_ablation = {}
    detail = None

    print("=" * 78)
    print("SYMBOLIC REWRITE EXPERIMENT")
    print("=" * 78)
    base = Config()
    print("grid %dx%d | max_read_cells=%d | max_write_cells=%d | pattern<=%d"
          % (base.grid_size, base.grid_size, base.max_read_cells,
             base.max_write_cells, base.max_pattern_len))
    print("seeds: %s" % SEEDS)
    print("ablations: %s" % ", ".join(a for a, _ in ABLATIONS))
    print("")

    for ablation, ov in ABLATIONS:
        rows = []
        for seed in SEEDS:
            want_detail = (ablation == "full_system" and seed == SEEDS[0])
            r = run_one(ablation, ov, seed, collect_detail=want_detail)
            if want_detail:
                detail = (r.pop("_learner"), r.pop("_corpus"), r)
            rows.append(r)
            all_rows.append(r)
            print("  [%-28s seed %2d]  rules=%-4d  gen=%-3d  P=%.2f R=%.2f "
                  "exact=%.0f invalid=%.2f  |completions|=%.1f"
                  % (ablation, seed, r["rule_diagnostics"]["total_rules"],
                     r["rule_diagnostics"]["generalized_rules"],
                     r["aggregate"]["precision"], r["aggregate"]["recall"],
                     r["aggregate"]["exact_set_accuracy"],
                     r["aggregate"]["invalid_generation_rate"],
                     r["aggregate"]["mean_generated"]))
        by_ablation[ablation] = {"per_seed": rows, "aggregate": aggregate(rows)}
        print("")

    # ------------------------------------------------------------------ files
    results = {
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
                              "cheat_control_problems": r["cheat_control_problems"],
                          } for r in v["per_seed"]]}
                      for k, v in by_ablation.items()},
        "dataset_examples": {
            "seed_%d" % SEEDS[0]: by_ablation["full_system"]["per_seed"][0]["dataset"]
        },
    }
    with open(os.path.join(outdir, "results.json"), "w") as f:
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
    with open(os.path.join(outdir, "rules.json"), "w") as f:
        json.dump(rules_json, f, indent=1, ensure_ascii=False)

    traces_json = {
        "note": "successful derivation traces, ablation=full_system, seed=%d" % SEEDS[0],
        "traces": learner.traces,
    }
    with open(os.path.join(outdir, "traces.json"), "w") as f:
        json.dump(traces_json, f, indent=1, ensure_ascii=False)

    # ------------------------------------------------------- readable summary
    print("-" * 78)
    print("EXAMPLE RULE SYSTEM (full_system, seed %d)" % SEEDS[0])
    print("-" * 78)
    ints = [r for r in learner.rules if r.kind == "INTERNAL" and r.n_apply > 1]
    ints.sort(key=lambda r: -r.n_apply)
    print(" most-used structural (epsilon) rules:")
    for r in ints[:5]:
        print("  " + r.text() + "    [applied %d, reused %d]" % (r.n_apply, r.n_reuse))
    cons = [r for r in learner.rules if r.kind == "CONSUME" and r.n_apply > 1]
    cons.sort(key=lambda r: -r.n_apply)
    print(" most-used licensing rules:")
    for r in cons[:6]:
        print("  " + r.text() + "    [applied %d, reused %d, generalized %d]"
              % (r.n_apply, r.n_reuse, r.n_generalized))
    gen = [r for r in learner.rules if r.n_generalized > 0]
    gen.sort(key=lambda r: -r.n_generalized)
    print(" generalized rules (conditions relaxed by successful reuse):")
    for r in gen[:6]:
        print("  " + r.text() + "    [applied %d, generalizations %d]"
              % (r.n_apply, r.n_generalized))
        for h in r.history:
            print("        was: (%s)" % ",".join(pat_str(e) for e in h))
        for mf in r.merged_from:
            print("        via: %s" % mf)
    print("")
    em = dr["emergent_structures"]
    print("EMERGENT INTERNAL STRUCTURES (learner-internal, unlabelled;")
    print("ranked by how often the structural rule that created them fired):")
    for row in em["examples"][:8]:
        print("   %s  expands to %-8r  applied %-4d  occurrences in corpus %-4d"
              % (row["symbol"], row["expansion"], row["rule_applications"],
                 row["corpus_occurrences"]))
    print("   structures whose expansion coincides exactly with a hidden lexical")
    print("   item (comparison made OUTSIDE the learner, for evaluation only): %d"
          % em["n_exact_hidden_word_matches"])
    for row in em["examples_matching_hidden_words"][:6]:
        print("      %s = %-8r  structural rule fired %d times"
              % (row["symbol"], row["expansion"], row["rule_applications"]))
    print("")
    deepest = None
    for name in learner.symtab.info:
        if learner.symtab.info[name]["kind"] != "collapse":
            continue
        if deepest is None or learner.symtab.depth(name) > learner.symtab.depth(deepest):
            deepest = name
    if deepest is not None:
        print("PROVENANCE OF THE DEEPEST INTERNAL SYMBOL  (explain_symbol):")
        lines = learner.symtab.explain(deepest)
        for ln in lines[:12]:
            print("   " + ln)
    print("")

    # ------------------------------------------------------- FINAL RESULTS
    print("================ FINAL RESULTS ================")
    print("experiment: discrete symbolic rewrite learner, raw characters only")
    print("date-independent, standard library only, seeds=%s" % SEEDS)
    cfgd = Config().as_dict()
    print("config: grid=%dx%d max_read=%d max_write=%d max_pattern=%d "
          "internal_depth=%d expansion_states=%d epochs=%d "
          "new_rule_budget=%d max_relax_positions=%d"
          % (cfgd["grid_size"], cfgd["grid_size"], cfgd["max_read_cells"],
             cfgd["max_write_cells"], cfgd["max_pattern_len"],
             cfgd["max_internal_depth"], cfgd["max_expansion_states"],
             cfgd["epochs"], cfgd["new_rule_budget"],
             cfgd["max_relax_positions"]))
    d0 = by_ablation["full_system"]["per_seed"][0]["dataset"]
    print("")
    print("dataset (seed %d): train=%d val=%d heldout_prefixes=%s"
          % (SEEDS[0], d0["n_train"], d0["n_val"], d0["heldout_prefixes"]))
    print("  train strings: %s" % " ".join(d0["train_strings"][:12]) + " ...")
    for p in d0["heldout_prefixes"]:
        print("  held-out compatible for %r: %s"
              % (p, " ".join(d0["heldout_compatible_strings"][p])))
        print("  held-out incompatible for %r: %s"
              % (p, " ".join(d0["heldout_incompatible_strings"][p][:6]) + " ..."))
    print("")
    print("%-30s %6s %6s %6s %7s %8s %7s %7s"
          % ("ablation", "P", "R", "exact", "invalid", "|gen|", "rules", "genlz"))
    for ablation, _ in ABLATIONS:
        a = by_ablation[ablation]["aggregate"]
        print("%-30s %6.3f %6.3f %6.3f %7.3f %8.1f %7.1f %7.1f"
              % (ablation, a["precision_mean"], a["recall_mean"],
                 a["exact_set_accuracy_mean"], a["invalid_generation_rate_mean"],
                 a["mean_generated_mean"], a["rules_mean"],
                 a["generalized_rules_mean"]))
    print("")
    print("per-seed recall (compatible held-out strings recovered):")
    for ablation, _ in ABLATIONS:
        rs = ["%.2f" % r["aggregate"]["recall"]
              for r in by_ablation[ablation]["per_seed"]]
        print("  %-30s %s" % (ablation, " ".join(rs)))
    print("")
    print("held-out prefix characters that an existing rule would itself have")
    print("licensed (the prefix is observed input, so arrival happens anyway):")
    for ablation, _ in ABLATIONS:
        a = by_ablation[ablation]["aggregate"]
        print("  %-30s %.3f" % (ablation, a["prefix_licensed_rate_mean"]))
    print("")
    print("per-seed precision:")
    for ablation, _ in ABLATIONS:
        rs = ["%.2f" % r["aggregate"]["precision"]
              for r in by_ablation[ablation]["per_seed"]]
        print("  %-30s %s" % (ablation, " ".join(rs)))
    print("")
    print("learning dynamics (new-rule creation rate, first vs last decile):")
    for ablation, _ in ABLATIONS:
        a = by_ablation[ablation]["aggregate"]
        print("  %-30s first=%.3f  last=%.3f" % (ablation,
              a["new_rule_rate_first_decile"], a["new_rule_rate_last_decile"]))
    print("")
    print("full_system, seed %d, learning dynamics by portion of training"
          % SEEDS[0])
    print("(portions 1-5 are the first pass over the training set, 6-10 the second):")
    for row in by_ablation["full_system"]["per_seed"][0]["learning_dynamics"]:
        print("  portion %-6s decisions=%-4d solved_by_existing=%-4d new_rules=%-4d rate=%.3f"
              % (row["portion"], row["decisions"], row["solved_by_existing_rules"],
                 row["new_rule_created"], row["new_rule_rate"]))
    print("")
    print("rule system (full_system aggregate over seeds): rules=%.1f  "
          "internal_symbols=%.1f  generalized=%.1f  exact_hidden_word_matches=%.1f"
          % (by_ablation["full_system"]["aggregate"]["rules_mean"],
             by_ablation["full_system"]["aggregate"]["internal_symbols_mean"],
             by_ablation["full_system"]["aggregate"]["generalized_rules_mean"],
             by_ablation["full_system"]["aggregate"]["exact_hidden_word_matches_mean"]))
    print("validation strings fully derivable by the frozen system: %.3f (full_system)"
          % by_ablation["full_system"]["aggregate"]["validation_derivable_rate"])
    print("")
    print("held-out generation, full_system, seed %d:" % SEEDS[0])
    for h in by_ablation["full_system"]["per_seed"][0]["heldout"]:
        print("  prefix %r" % h["prefix"])
        print("    possible_next_chars: %s" % h["possible_next_chars"])
        print("    generated (%d): %s" % (h["n_generated"], h["generated"][:12]))
        print("    compatible  : %s" % h["compatible"])
        print("    incompatible: %s" % h["incompatible"][:8])
        print("    not in language (%d): %s"
              % (h["n_not_in_language"], h["not_in_language"][:6]))
        print("    P=%.3f R=%.3f exact=%.0f invalid=%.3f"
              % (h["metrics"]["precision"], h["metrics"]["recall"],
                 h["metrics"]["exact_set_accuracy"],
                 h["metrics"]["invalid_generation_rate"]))
    print("")
    nprob = sum(len(r["cheat_control_problems"]) for r in all_rows)
    print("cheat-control violations across all %d runs: %d" % (len(all_rows), nprob))
    print("files written: results.json rules.json traces.json")
    print("=================================================")


if __name__ == "__main__":
    main()
