#!/usr/bin/env python3
"""
Per-aptamer AF3 confidence extraction for the cocaine / morphine aptamer study.
Reviewer-hardened rewrite.

Metric definitions (verified against the AF3 data.json/summary structure):
    pTM            = global 'ptm'
    iPTM           = chain_pair_iptm[0][1]      (chain 0 = DNA aptamer, chain 1 = ligand)
    min-chain PAE  = chain_pair_pae_min[0][1]   (MINIMUM inter-chain PAE; exploratory)

Levels:
    top-ranked AF3 model = the single top-level *_summary_confidences.json
                           (AF3's overall ranking; NOT the max-iPTM model)
    prediction-level     = the seed-sample predictions in seed-<S>_sample-<Y>/ ,
                           pooled (9 seeds x 5 samples). Reported as prediction-level.
    seed-grouped         = aggregate samples within each seed first, then across seeds.

Every prediction is stored as ONE intact record with its real seed, sample and source
file. Metrics are validated (finite + range). Duplicates, missing results, malformed
directories, incomplete conditions, and chain-order violations are fatal in --strict
(default); --allow-partial downgrades them to reported warnings.

Outputs (in --outdir, default ~):
  per_aptamer_per_condition.csv  paired wide-format: one row per (aptamer, condition),
        cocaine & morphine side by side, top-ranked + prediction-level pTM/iPTM/PAE
        (mean +- sd), three-state win flags, n_pred, source dirs.
  per_prediction_long.csv        true long-format: one row per (aptamer, condition,
        target, seed, sample, metric) -- for plotting / statistical modelling.
  per_prediction_values.csv      one row per prediction with real seed, sample, source.
  run_metadata.json              module path + sha256, args, completeness, counts.
Prints iPTM (primary) / pTM (secondary) / min-PAE (exploratory) discrimination tables
at top-ranked, prediction-level, and seed-grouped levels, with ties reported.
"""
import os, sys, csv, re, json, math, argparse, hashlib, datetime
from collections import defaultdict, namedtuple
from pathlib import Path

SEED_RE = re.compile(r"seed-(\d+)_sample-(\d+)")
VALID_TARGETS = {"cocaine", "morphine"}
TARGET_CCD = {"cocaine": "COC", "morphine": "MOI"}  # morphine is MOI in this dataset (CLI-overridable)
RANGES = {"ptm": (0.0, 1.0), "iptm": (0.0, 1.0), "pae": (0.0, 1e4)}

Pred = namedtuple("Pred", "condition aptamer target seed sample ptm iptm pae source")


class DataError(Exception):
    pass


def finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def read_summary(path):
    """Return validated (ptm, iptm, pae) from a summary_confidences.json, else DataError."""
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    for k in ("ptm", "chain_pair_iptm", "chain_pair_pae_min"):
        if k not in d:
            raise DataError(f"{path}: missing key '{k}'")
    try:
        ptm = d["ptm"]
        iptm = d["chain_pair_iptm"][0][1]
        pae = d["chain_pair_pae_min"][0][1]
    except (IndexError, TypeError):
        raise DataError(f"{path}: chain_pair matrix too small for [0][1]")
    vals = {"ptm": ptm, "iptm": iptm, "pae": pae}
    for name, v in vals.items():
        lo, hi = RANGES[name]
        if not finite(v):
            raise DataError(f"{path}: {name} not finite ({v!r})")
        if not (lo - 1e-6 <= v <= hi + 1e-6):
            raise DataError(f"{path}: {name}={v} outside [{lo},{hi}]")
    return float(ptm), float(iptm), float(pae)


def verify_chain_order(data_json, target):
    """Assert sequences[0] is DNA and sequences[1] is the expected ligand CCD."""
    with open(data_json, encoding="utf-8") as f:
        d = json.load(f)
    seqs = d.get("sequences", [])
    if len(seqs) < 2 or "dna" not in seqs[0] or "ligand" not in seqs[1]:
        raise DataError(f"{data_json}: chains are not [dna, ligand]")
    ccd = seqs[1]["ligand"].get("ccdCodes", [])
    if ccd[:1] != [TARGET_CCD[target]]:
        raise DataError(f"{data_json}: chain-1 ligand {ccd} != expected {TARGET_CCD[target]}")


def one(glob_list, what, where):
    """Require exactly one match."""
    g = sorted(glob_list)
    if len(g) == 0:
        raise DataError(f"{where}: no {what}")
    if len(g) > 1:
        raise DataError(f"{where}: {len(g)} {what} (ambiguous): {[p.name for p in g]}")
    return g[0]


def collect(archive, conditions, parse, strict, chain_check):
    top = defaultdict(lambda: defaultdict(dict))       # [cond][apt][tgt] = (ptm,iptm,pae)
    top_src = defaultdict(lambda: defaultdict(dict))
    preds = []
    warnings = []

    def problem(msg):
        if strict:
            raise DataError(msg)
        warnings.append(msg)

    for cond, dirs in conditions:
        for dname, restrict in dirs:
            base = archive / dname
            if not base.is_dir():
                problem(f"missing condition directory: {base}")
                continue
            for sub in sorted(base.iterdir()):
                if not sub.is_dir():
                    continue
                apt, tgt = parse(sub.name)
                if not apt or not tgt:
                    continue
                apt, tgt = apt.strip().upper(), tgt.strip().lower()
                if tgt not in VALID_TARGETS:
                    continue
                if restrict and tgt != restrict:
                    continue
                if tgt in top[cond][apt]:
                    raise DataError(f"duplicate {cond}/{apt}/{tgt}:\n  {top_src[cond][apt][tgt]}\n  {sub}")
                if chain_check:
                    verify_chain_order(one(sub.glob("*_data.json"), "*_data.json", sub), tgt)
                try:
                    tfile = one((f for f in sub.glob("*_summary_confidences.json") if "seed-" not in f.name),
                                "top-level summary", sub)
                except DataError as e:
                    problem(str(e))
                    continue
                top[cond][apt][tgt] = read_summary(tfile)
                top_src[cond][apt][tgt] = str(sub)
                seen = set()
                for sd in sorted(sub.glob("seed-*_sample-*")):
                    if not sd.is_dir():
                        continue
                    m = SEED_RE.search(sd.name)
                    if not m:
                        continue
                    seed, sample = int(m.group(1)), int(m.group(2))
                    if (seed, sample) in seen:
                        raise DataError(f"{sub}: duplicate seed-sample {seed}_{sample}")
                    seen.add((seed, sample))
                    sfile = one(sd.glob("*_summary_confidences.json"), "seed summary", sd)
                    p, i, a = read_summary(sfile)
                    preds.append(Pred(cond, apt, tgt, seed, sample, p, i, a, str(sfile)))
    return top, top_src, preds, warnings


def cmp3(cv, mv, higher_better=True):
    if cv is None or mv is None:
        return ""
    if cv == mv:
        return "tie"
    coc = (cv > mv) if higher_better else (cv < mv)
    return "cocaine" if coc else "morphine"


def disc_table(title, cond_order, getter, higher_better):
    """getter(cond, apt, tgt) -> value or None. Independent denominators + ties."""
    print(f"\n{title}")
    print(f"{'Condition':<20}{'cocaine wins':>16}{'ties':>8}{'n':>6}")
    print("-" * 50)
    for cond in cond_order:
        win = tie = tot = 0
        apts = set()
        for (c, a, t) in getter.keys_for(cond):
            apts.add(a)
        for a in sorted(apts):
            cv = getter(cond, a, "cocaine")
            mv = getter(cond, a, "morphine")
            if cv is None or mv is None:
                continue
            tot += 1
            if cv == mv:
                tie += 1
            elif ((cv > mv) if higher_better else (cv < mv)):
                win += 1
        pct = f"{100*win/tot:.1f}%" if tot else "n/a"
        print(f"{cond:<20}{f'{win}/{tot} ({pct})':>16}{tie:>8}{tot:>6}")


class Getter:
    """Callable value lookup with key enumeration, for disc_table."""
    def __init__(self, store, fn):
        self._store = store      # {cond: {apt: {tgt: payload}}}
        self._fn = fn            # payload -> value or None
    def __call__(self, cond, apt, tgt):
        p = self._store.get(cond, {}).get(apt, {}).get(tgt)
        return self._fn(p) if p is not None else None
    def keys_for(self, cond):
        for apt, tg in self._store.get(cond, {}).items():
            for tgt in tg:
                yield (cond, apt, tgt)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--archive", default="~/af_data_archive")
    ap.add_argument("--home", default="~", help="dir containing analyze_af3_outputs.py")
    ap.add_argument("--sequences", default="~/aptamer_sequences.csv")
    ap.add_argument("--mg-levels", type=int, nargs="+", default=[0, 2, 4, 6, 8, 10])
    ap.add_argument("--outdir", default="~")
    ap.add_argument("--expected", type=int, default=71, help="expected aptamers per condition per target")
    ap.add_argument("--allow-partial", action="store_true", help="downgrade fatal data issues to warnings")
    ap.add_argument("--no-chain-check", action="store_true", help="skip per-aptamer chain-order verification")
    ap.add_argument("--dry-run", action="store_true",
                    help="list the exact directories that will be read vs ignored, then exit")
    ap.add_argument("--cocaine-ccd", default="COC", help="CCD code expected at chain 1 in cocaine complexes")
    ap.add_argument("--morphine-ccd", default="MOI", help="CCD code expected at chain 1 in morphine complexes")
    args = ap.parse_args()
    strict = not args.allow_partial
    TARGET_CCD["cocaine"] = args.cocaine_ccd.strip().upper()
    TARGET_CCD["morphine"] = args.morphine_ccd.strip().upper()

    archive = Path(args.archive).expanduser().resolve()
    home = Path(args.home).expanduser().resolve()
    seqfile = Path(args.sequences).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    if not archive.is_dir():
        sys.exit(f"archive not found: {archive}")
    if not seqfile.is_file():
        sys.exit(f"sequences file not found: {seqfile}")

    sys.path.insert(0, str(home))
    import analyze_af3_outputs as A
    parse = A.parse_aptamer_target_from_dir
    mod_path = os.path.abspath(A.__file__)
    mod_sha = hashlib.sha256(open(mod_path, "rb").read()).hexdigest()
    print(f"Using parser from: {mod_path}\n  sha256={mod_sha[:16]}")

    conditions = [("buffer_optimized", [("afoutput_buffer", None)]),
                  ("buffer_proportional", [("afoutput_proportional", None)])]
    for n in args.mg_levels:
        conditions.append((f"Mg{n}", [
            (f"afoutput_Mg{n}_cocaine_target_and_aptamer_with_weights", "cocaine"),
            (f"afoutput_Mg{n}_Morphine_target_cocaine_aptamer_with_weights", "morphine")]))
    cond_order = [c for c, _ in conditions]

    if args.dry_run:
        target_names = [dn for _, dirs in conditions for dn, _ in dirs]
        tset = set(target_names)
        print(f"\nArchive: {archive}")
        print(f"Will READ these {len(target_names)} directories (explicit allowlist; "
              f"the archive is NOT scanned for any others):")
        for dn in target_names:
            print(f"  [{'OK     ' if (archive / dn).is_dir() else 'MISSING'}] {dn}")
        present = sorted(d.name for d in archive.iterdir() if d.is_dir())
        ignored = [d for d in present if d not in tset]
        print(f"\nWill IGNORE these {len(ignored)} other folder(s) present in the archive:")
        for d in ignored:
            print(f"  [ignored] {d}")
        sys.exit(0)

    top, top_src, preds, warns = collect(archive, conditions, parse, strict, not args.no_chain_check)

    # prediction-level pooled records (intact, aligned)
    pe = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for p in preds:
        pe[p.condition][p.aptamer][p.target].append(p)

    # ---- completeness validation ----
    comp_problems = []
    sets_per_cond = {}
    for cond in cond_order:
        coc = {a for a in top[cond] if "cocaine" in top[cond][a]}
        mor = {a for a in top[cond] if "morphine" in top[cond][a]}
        sets_per_cond[cond] = coc & mor
        if len(coc) != args.expected:
            comp_problems.append(f"{cond}: {len(coc)} cocaine (expected {args.expected})")
        if len(mor) != args.expected:
            comp_problems.append(f"{cond}: {len(mor)} morphine (expected {args.expected})")
        if coc != mor:
            comp_problems.append(f"{cond}: cocaine and morphine aptamer sets differ "
                                 f"(coc-only={sorted(coc-mor)[:5]}, mor-only={sorted(mor-coc)[:5]})")
    base_set = sets_per_cond[cond_order[0]] if cond_order else set()
    for cond in cond_order[1:]:
        if sets_per_cond[cond] != base_set:
            comp_problems.append(f"{cond}: aptamer set differs from {cond_order[0]}")
    if comp_problems:
        msg = "COMPLETENESS PROBLEMS:\n  " + "\n  ".join(comp_problems)
        if strict:
            sys.exit(msg + "\n(use --allow-partial to proceed anyway)")
        warns.append(msg)
    partial = bool(comp_problems or warns)

    # ---- statistics helpers ----
    def mean_sd(xs):
        xs = list(xs)
        if not xs:
            return (None, None, 0)
        import statistics
        return (statistics.mean(xs), statistics.stdev(xs) if len(xs) > 1 else 0.0, len(xs))

    def pred_stat(cond, apt, tgt, metric):
        ps = pe.get(cond, {}).get(apt, {}).get(tgt, [])
        return mean_sd(getattr(p, metric) for p in ps)

    def seedgrouped_mean(cond, apt, tgt, metric):
        ps = pe.get(cond, {}).get(apt, {}).get(tgt, [])
        if not ps:
            return None
        byseed = defaultdict(list)
        for p in ps:
            byseed[p.seed].append(getattr(p, metric))
        import statistics
        per_seed = [statistics.mean(v) for v in byseed.values()]
        return statistics.mean(per_seed)

    # ---- discrimination tables (independent denominators + ties) ----
    idx = {"ptm": 0, "iptm": 1, "pae": 2}
    g_top = {m: Getter(top, (lambda j: (lambda p: p[j]))(idx[m])) for m in idx}

    # prediction-level getter needs pe; build a thin wrapper store
    class PredGetter(Getter):
        def __init__(self, metric, fn):
            self.metric = metric; self.fn = fn
        def __call__(self, cond, apt, tgt):
            ps = pe.get(cond, {}).get(apt, {}).get(tgt, [])
            return self.fn(cond, apt, tgt, self.metric) if ps else None
        def keys_for(self, cond):
            for apt, tg in pe.get(cond, {}).items():
                for tgt in tg:
                    yield (cond, apt, tgt)

    print("\n================  TOP-RANKED AF3 MODEL  ================")
    disc_table("iPTM cocaine>morphine  [PRIMARY]", cond_order, g_top["iptm"], True)
    disc_table("pTM  cocaine>morphine  [secondary, global]", cond_order, g_top["ptm"], True)
    disc_table("min inter-chain PAE cocaine<morphine  [exploratory]", cond_order, g_top["pae"], False)

    print("\n================  PREDICTION-LEVEL (pooled 9x5)  ================")
    disc_table("iPTM cocaine>morphine  [PRIMARY]", cond_order,
               PredGetter("iptm", lambda c, a, t, m: pred_stat(c, a, t, m)[0]), True)
    disc_table("pTM  cocaine>morphine  [secondary]", cond_order,
               PredGetter("ptm", lambda c, a, t, m: pred_stat(c, a, t, m)[0]), True)
    disc_table("min inter-chain PAE cocaine<morphine  [exploratory]", cond_order,
               PredGetter("pae", lambda c, a, t, m: pred_stat(c, a, t, m)[0]), False)

    print("\n================  SEED-GROUPED (mean per seed, then across seeds)  ================")
    disc_table("iPTM cocaine>morphine  [PRIMARY]", cond_order,
               PredGetter("iptm", lambda c, a, t, m: seedgrouped_mean(c, a, t, m)), True)

    # ---- sequences (only for aptamers actually in this analysis) ----
    needed = {a for cond in top for a in top[cond]}
    seq = {}
    with open(seqfile, encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        if "aptamer_id" not in rdr.fieldnames or "sequence" not in rdr.fieldnames:
            sys.exit(f"{seqfile}: needs 'aptamer_id' and 'sequence' columns; has {rdr.fieldnames}")
        for row in rdr:
            aid = (row.get("aptamer_id") or "").strip().upper()
            if aid not in needed:
                continue
            s = (row.get("sequence") or "").strip()
            if not s:
                continue
            if aid in seq and seq[aid] != s:
                warns.append(f"sequence CSV: {aid} has conflicting sequences "
                             f"(len {len(seq[aid])} vs {len(s)}); keeping first, check {seqfile.name}")
                continue
            seq.setdefault(aid, s)

    def r4(x):
        return round(x, 4) if isinstance(x, (int, float)) else ""

    # ---- per_aptamer_per_condition.csv (paired wide-format) ----
    cols = ["aptamer_id", "sequence", "condition",
            "coc_iptm_top", "mor_iptm_top", "coc_ptm_top", "mor_ptm_top", "coc_minpae_top", "mor_minpae_top",
            "coc_iptm_pred_mean", "coc_iptm_pred_sd", "mor_iptm_pred_mean", "mor_iptm_pred_sd",
            "coc_ptm_pred_mean", "coc_ptm_pred_sd", "mor_ptm_pred_mean", "mor_ptm_pred_sd",
            "coc_minpae_pred_mean", "coc_minpae_pred_sd", "mor_minpae_pred_mean", "mor_minpae_pred_sd",
            "top_model_iptm_winner", "top_model_ptm_winner", "top_model_minpae_winner",
            "pred_iptm_winner", "pred_ptm_winner", "pred_minpae_winner",
            "n_pred_coc", "n_pred_mor", "coc_source", "mor_source"]
    p1 = outdir / "per_aptamer_per_condition.csv"
    with open(p1, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for cond in cond_order:
            apts = {a for a in top[cond] if "cocaine" in top[cond][a] and "morphine" in top[cond][a]}
            for a in sorted(apts):
                ct, mt = top[cond][a]["cocaine"], top[cond][a]["morphine"]
                ci, cis, cn = pred_stat(cond, a, "cocaine", "iptm")
                mi, mis, mn = pred_stat(cond, a, "morphine", "iptm")
                cp, cps, _ = pred_stat(cond, a, "cocaine", "ptm")
                mp, mps, _ = pred_stat(cond, a, "morphine", "ptm")
                ca, cas, _ = pred_stat(cond, a, "cocaine", "pae")
                ma, mas, _ = pred_stat(cond, a, "morphine", "pae")
                w.writerow([a, seq.get(a, ""), cond,
                            r4(ct[1]), r4(mt[1]), r4(ct[0]), r4(mt[0]), r4(ct[2]), r4(mt[2]),
                            r4(ci), r4(cis), r4(mi), r4(mis), r4(cp), r4(cps), r4(mp), r4(mps),
                            r4(ca), r4(cas), r4(ma), r4(mas),
                            cmp3(ct[1], mt[1], True), cmp3(ct[0], mt[0], True), cmp3(ct[2], mt[2], False),
                            cmp3(ci, mi, True), cmp3(cp, mp, True), cmp3(ca, ma, False),
                            cn, mn, top_src[cond][a]["cocaine"], top_src[cond][a]["morphine"]])

    # ---- per_prediction_long.csv (true long-format) ----
    p2 = outdir / "per_prediction_long.csv"
    nlong = 0
    with open(p2, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["aptamer_id", "condition", "target", "seed", "sample", "metric", "value", "source"])
        for p in sorted(preds, key=lambda p: (p.condition, p.aptamer, p.target, p.seed, p.sample)):
            for metric, label in (("ptm", "ptm"), ("iptm", "iptm"), ("pae", "minpae")):
                w.writerow([p.aptamer, p.condition, p.target, p.seed, p.sample, label, r4(getattr(p, metric)), p.source])
                nlong += 1

    # ---- per_prediction_values.csv (wide per prediction, with real ids) ----
    p3 = outdir / "per_prediction_values.csv"
    with open(p3, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["aptamer_id", "condition", "target", "seed", "sample", "ptm", "iptm", "minpae", "source"])
        for p in sorted(preds, key=lambda p: (p.condition, p.aptamer, p.target, p.seed, p.sample)):
            w.writerow([p.aptamer, p.condition, p.target, p.seed, p.sample, r4(p.ptm), r4(p.iptm), r4(p.pae), p.source])

    # ---- metadata ----
    meta = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "argv": sys.argv,
        "archive": str(archive), "module_path": mod_path, "module_sha256": mod_sha,
        "conditions": cond_order, "expected_per_target": args.expected,
        "strict": strict, "chain_check": not args.no_chain_check,
        "complete": not partial, "warnings": warns,
        "n_predictions": len(preds),
        "aptamers_per_condition": {c: len(sets_per_cond.get(c, set())) for c in cond_order},
    }
    with open(outdir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\nWrote:\n  {p1}\n  {p2}  ({nlong} rows)\n  {p3}\n  {outdir/'run_metadata.json'}")
    print(f"Predictions: {len(preds)} | complete: {not partial}")
    if warns:
        print("WARNINGS:\n  " + "\n  ".join(warns))


if __name__ == "__main__":
    try:
        main()
    except DataError as e:
        sys.exit(f"\nDATA ERROR (halting; nothing written): {e}")
