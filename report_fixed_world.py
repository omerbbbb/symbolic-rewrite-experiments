#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Print the FIXED-WORLD SCALING RESULTS block from the saved results file.
Reads only fixed_world_scaling_results.json - runs nothing, trains nothing.
Works on a partial file (reports the number of seeds actually completed)."""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(HERE, "fixed_world_scaling_results.json")))
SCALES = d["scales"]
integ = d["integrity"]
avg = lambda rows, k: sum(r[k] for r in rows) / float(len(rows))
ps = lambda rows, k: sum(r["probe"][k] for r in rows)
pav = lambda rows, k: sum(r["probe"][k] for r in rows) / float(len(rows))
have = [N for N in SCALES if d["symbolic"].get(str(N))]

print("================ FIXED-WORLD SCALING RESULTS ================")
print("")
print("FIXED LEXICON ACROSS SCALES: %s" % ("PASS" if integ["fixed_lexicon"] else "FAIL"))
print("FIXED GRAMMAR ACROSS SCALES: %s" % ("PASS" if integ["fixed_grammar"] else "FAIL"))
print("FIXED HELD-OUT PAIRS ACROSS SCALES: %s" % ("PASS" if integ["fixed_heldout"] else "FAIL"))
print("NESTED TRAINING SETS: %s" % ("PASS" if integ["nested"] else "FAIL"))
print("SYMBOLIC/TRANSFORMER DATA IDENTITY: %s" % ("PASS" if d.get("data_identity") else "FAIL"))
print("seeds completed per scale: %s"
      % {N: len(d["symbolic"][str(N)]) for N in have})
print("")
print("SYMBOLIC V4:")
for N in have:
    r = d["symbolic"][str(N)]
    print("N=%d:  (%d seeds)" % (N, len(r)))
    print("  precision %.4f | recall %.4f | exact_set %.4f | incompatible %.4f "
          "| malformed %.4f"
          % (avg(r, "precision"), avg(r, "recall"), avg(r, "exact_set_accuracy"),
             avg(r, "incompatible_generation_rate"),
             avg(r, "malformed_generation_rate")))
    print("  new-rule-search fraction %.4f (first decile %.3f, last %.3f) | "
          "reuse events %.0f"
          % (avg(r, "new_rule_search_fraction"),
             avg(r, "new_rule_rate_first_decile"),
             avg(r, "new_rule_rate_last_decile"), avg(r, "reuse_events")))
    print("  active rules %.0f | internal symbols %.0f | family-exclusive %s"
          % (avg(r, "active_rules"), avg(r, "internal_symbols"),
             {k: sum(x["probe"]["family_exclusive_structures"][k] for x in r)
              for k in r[0]["probe"]["family_exclusive_structures"]}))
print("")
print("TRANSFORMER SMALL:")
for N in have:
    r = d["transformer"][str(N)]
    print("N=%d:  (%d seeds)" % (N, len(r)))
    print("  fit/ceiling %.4f | val loss %.4f | top4 %.4f | R@4 %.4f | pairwise "
          "%.4f | mass %.4f | mean rank %.2f"
          % (avg(r, "fit_fraction_of_ceiling"), avg(r, "val_loss"),
             avg(r, "top4_exact_family_accuracy"), avg(r, "recall_at_4"),
             avg(r, "pairwise_family_accuracy"),
             avg(r, "probability_mass_correct_family"),
             avg(r, "mean_rank_correct")))
print("")
print("SUBJECT FAMILY PROBE BY SCALE   (same% = same-family pairs grouped,")
print("diff% = different-family pairs falsely grouped, full = families recovered)")
print("  %-7s | %-30s | %-30s" % ("N", "SYMBOLIC V4", "TRANSFORMER SMALL"))
print("  %-7s | %-30s | %-30s" % ("", "same%   diff%   full   exact",
                                  "same%   diff%   full   exact"))
for N in have:
    f = lambda r: "%.3f   %.3f   %.1f    %d/%d" % (
        ps(r, "same_family_pairs_grouped") / float(ps(r, "same_family_pairs_total")),
        ps(r, "different_family_pairs_grouped") / float(ps(r, "different_family_pairs_total")),
        pav(r, "families_fully_recovered"),
        sum(1 for x in r if x["probe"]["exact_partition_recovered"]), len(r))
    print("  %-7d | %-30s | %-30s" % (N, f(d["symbolic"][str(N)]),
                                      f(d["transformer"][str(N)])))
print("")
print("HELD-OUT GENERALIZATION BY SCALE")
print("  %-7s | %-24s | %-34s" % ("N", "SYMBOLIC V4", "TRANSFORMER SMALL"))
print("  %-7s | %-24s | %-34s" % ("", "prec    recall  exact",
                                  "top4    R@4     pairwise  mass"))
for N in have:
    s, t = d["symbolic"][str(N)], d["transformer"][str(N)]
    print("  %-7d | %.3f   %.3f   %.3f  | %.3f   %.3f   %.3f     %.3f"
          % (N, avg(s, "precision"), avg(s, "recall"), avg(s, "exact_set_accuracy"),
             avg(t, "top4_exact_family_accuracy"), avg(t, "recall_at_4"),
             avg(t, "pairwise_family_accuracy"),
             avg(t, "probability_mass_correct_family")))
print("")
print("TRANSFORMER seen vs held-out (subject,verb) combinations by scale:")
for N in have:
    t = d["transformer"][str(N)]
    c = lambda k: sum(x["probe"]["combination_control"][k] for x in t) / float(len(t))
    print("  N=%-6d seen: R@4 %.3f pair %.3f mass %.3f | held-out: R@4 %.3f "
          "pair %.3f mass %.3f"
          % (N, c("seen_recall_at_4"), c("seen_pairwise_family_accuracy"),
             c("seen_probability_mass_correct_family"), c("heldout_recall_at_4"),
             c("heldout_pairwise_family_accuracy"),
             c("heldout_probability_mass_correct_family")))
print("")
print("PER-SEED held-out results at the largest completed scale (N=%d):" % have[-1])
for x in d["symbolic"][str(have[-1])]:
    print("  V4 seed %d: P=%.2f R=%.2f exact=%.2f  rules=%d  train_seconds=%d"
          % (x["seed"], x["precision"], x["recall"], x["exact_set_accuracy"],
             x["active_rules"], x["train_seconds"]))
for x in d["transformer"][str(have[-1])]:
    print("  TF seed %d: top4=%.2f R@4=%.2f pair=%.2f  families_recovered=%d"
          % (x["seed"], x["top4_exact_family_accuracy"], x["recall_at_4"],
             x["pairwise_family_accuracy"],
             x["probe"]["families_fully_recovered"]))
print("")
sym_ex = [avg(d["symbolic"][str(N)], "exact_set_accuracy") for N in have]
sym_pr = [avg(d["symbolic"][str(N)], "precision") for N in have]
tf_pair = [avg(d["transformer"][str(N)], "pairwise_family_accuracy") for N in have]
tf_r4 = [avg(d["transformer"][str(N)], "recall_at_4") for N in have]
tf_t4 = [avg(d["transformer"][str(N)], "top4_exact_family_accuracy") for N in have]
sym_same = [ps(d["symbolic"][str(N)], "same_family_pairs_grouped") /
            float(ps(d["symbolic"][str(N)], "same_family_pairs_total")) for N in have]
sym_diff = [ps(d["symbolic"][str(N)], "different_family_pairs_grouped") /
            float(ps(d["symbolic"][str(N)], "different_family_pairs_total")) for N in have]
tf_same = [ps(d["transformer"][str(N)], "same_family_pairs_grouped") /
           float(ps(d["transformer"][str(N)], "same_family_pairs_total")) for N in have]
tf_diff = [ps(d["transformer"][str(N)], "different_family_pairs_grouped") /
           float(ps(d["transformer"][str(N)], "different_family_pairs_total")) for N in have]
tf_fam = [pav(d["transformer"][str(N)], "families_fully_recovered") for N in have]
sym_fam = [pav(d["symbolic"][str(N)], "families_fully_recovered") for N in have]
print("INTERPRETATION:")
print("  scales:                          %s" % have)
print("  V4 precision:                    %s" % ["%.3f" % x for x in sym_pr])
print("  V4 exact-set:                    %s" % ["%.3f" % x for x in sym_ex])
print("  V4 same-family / diff-family:    %s / %s"
      % (["%.2f" % x for x in sym_same], ["%.2f" % x for x in sym_diff]))
print("  V4 full families recovered:      %s" % ["%.1f" % x for x in sym_fam])
print("  TF top-4 exact:                  %s" % ["%.3f" % x for x in tf_t4])
print("  TF Recall@4:                     %s" % ["%.3f" % x for x in tf_r4])
print("  TF pairwise:                     %s" % ["%.3f" % x for x in tf_pair])
print("  TF same-family / diff-family:    %s / %s"
      % (["%.2f" % x for x in tf_same], ["%.2f" % x for x in tf_diff]))
print("  TF full families recovered:      %s" % ["%.1f" % x for x in tf_fam])
sym_gen = sym_ex[-1] > 0.25 or (sym_pr[-1] - sym_pr[0]) > 0.25
tf_gen = tf_pair[-1] > 0.65 or tf_r4[-1] > 0.55
sym_fam_up = sym_fam[-1] >= sym_fam[0] + 1.0
tf_fam_up = tf_fam[-1] >= tf_fam[0] + 1.0 or (tf_same[-1] - tf_same[0]) > 0.15
if sym_gen and tf_gen:
    v = "D. Both improve with scale."
elif tf_gen and not sym_gen:
    v = "B. Transformer begins recovering generalization with scale, symbolic V4 does not."
elif sym_gen and not tf_gen:
    v = "C. Symbolic V4 begins recovering generalization with scale, Transformer does not."
elif sym_fam_up or tf_fam_up:
    v = ("E. Subject families are recovered internally (increasingly with scale) "
         "but the held-out subject+verb relation still fails.")
else:
    v = "A. Both models remain flat at chance even at N=%d." % have[-1]
print("  %s" % v)
print("")
viol = list(integ["violations"])
for N in have:
    for r in d["symbolic"][str(N)]:
        viol += r["cheat_control_problems"]
print("CHEAT-CONTROL VIOLATIONS: %d" % len(viol))
for x in viol[:10]:
    print("   !! %s" % x)
print("=============================================================")
