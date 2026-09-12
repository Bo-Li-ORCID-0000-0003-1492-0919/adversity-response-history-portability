#!/usr/bin/env python3
"""Run the frozen NMH post-confirmatory measurement adjudication.

This program implements only the plan frozen in
results/NMH_measurement_adjudication/00_measurement_adjudication_freeze.md.
It reuses the confirmatory two-event samples and reads only prespecified
GHQ-12/CES-D item columns from the local raw files.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "NMH_measurement_adjudication"
DP = ROOT / "data_processed"
FREEZE = OUT / "00_measurement_adjudication_freeze.md"
PRED_SAMPLE = DP / "NMH_cross_event_prediction_samples.parquet"
UK_EPISODES = DP / "NMH_UKHLS_event_episodes.parquet"
HRS_EPISODES = DP / "NMH_HRS_event_episodes.parquet"
UK_RAW = ROOT / "raw_data" / "UKDA-6614-stata" / "stata" / "stata14_se" / "ukhls"
HRS_RAW = ROOT / "raw_data" / "HRS" / "randhrs1992_2022v1_STATA" / "randhrs1992_2022v1.dta"
EU_SOURCE = ROOT / "results" / "NMH_confirmatory" / "05_model_prediction_metrics.csv"
ITEM_OUT = DP / "NMH_two_event_item_level_scores.parquet"

SEED = 20260903
BOOT_SUCCESS = 500
BOOT_MAX_ATTEMPTS = 5000
PRIMARY_TYPES = ["widowhood", "unemployment", "caregiving", "health"]
EDUCATION_LEVELS = ["bachelor_plus", "other_qualification", "no_qualification", "unknown"]
METRICS = [
    "RMSE", "MAE", "predictive_R2", "mean_log_predictive_density",
    "calibration_intercept", "calibration_slope",
]
RIDGE_GRID = [0.0, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
GHQ_CONCEPTS = [
    "concentration", "loss_of_sleep", "useful_role", "making_decisions",
    "under_strain", "overcoming_difficulties", "enjoy_activities",
    "face_problems", "unhappy_depressed", "losing_confidence",
    "worthless", "general_happiness",
]
GHQ_HALF_A = {1, 2, 3, 5, 7, 9}
CESD_STEMS = ["depres", "effort", "sleepr", "whappy", "flone", "enlife", "fsad", "going"]
CESD_CONCEPTS = [
    "felt_depressed", "everything_an_effort", "restless_sleep", "happy",
    "felt_lonely", "enjoyed_life", "felt_sad", "could_not_get_going",
]
CESD_HALF_A = {1, 2, 4, 5}


def note(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_num(x: pd.Series) -> pd.Series:
    return pd.to_numeric(x, errors="coerce")


def md_value(x) -> str:
    if pd.isna(x):
        return "NA"
    if isinstance(x, (float, np.floating)):
        if abs(float(x)) >= 10000:
            return f"{float(x):,.0f}"
        return f"{float(x):.4f}"
    if isinstance(x, (int, np.integer)):
        return f"{int(x):,}"
    return str(x).replace("|", "\\|").replace("\n", " ")


def md_table(frame: pd.DataFrame, columns: list[str], max_rows: int | None = None) -> str:
    d = frame.loc[:, [c for c in columns if c in frame.columns]].copy()
    if max_rows is not None:
        d = d.head(max_rows)
    if d.empty:
        return "No eligible rows."
    header = "| " + " | ".join(d.columns) + " |"
    rule = "|" + "|".join(["---"] * len(d.columns)) + "|"
    body = ["| " + " | ".join(md_value(v) for v in row) + " |" for row in d.itertuples(index=False, name=None)]
    return "\n".join([header, rule, *body])


def write_md(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def simple_group_folds(ids: pd.Series, k: int, seed: int) -> dict:
    values = np.asarray(pd.unique(ids))
    rng = np.random.default_rng(seed)
    rng.shuffle(values)
    return {v: i % k for i, v in enumerate(values)}


def validation_splits(d: pd.DataFrame, validation: str) -> tuple[list[tuple[int, pd.Series, pd.Series]], object]:
    if validation == "grouped_10fold":
        assignment = simple_group_folds(d.person_id, min(10, max(2, len(d))), SEED)
        splits = []
        for fold in sorted(set(assignment.values())):
            test = d.person_id.map(assignment).eq(fold)
            splits.append((fold, ~test, test))
        return splits, pd.NaT
    dates = d.later_event_date.dropna().sort_values()
    cutoff = dates.iloc[max(0, int(math.floor(0.70 * len(dates))) - 1)]
    train = d.later_event_date.le(cutoff)
    test = d.later_event_date.gt(cutoff)
    return [(0, train, test)], cutoff


def load_primary_history() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    sample = pd.read_parquet(PRED_SAMPLE)
    sample = sample.loc[sample.cohort_type.eq("two_event")].copy()
    episode_files = {"UKHLS": UK_EPISODES, "HRS": HRS_EPISODES}
    episodes = {cohort: pd.read_parquet(path) for cohort, path in episode_files.items()}
    pieces = []
    join_fields = [
        "episode_id", "person_id", "event_wave", "event_date", "event_year",
        "primary_event_type", "pre1_wave", "year2_wave", "outcome_pre1_raw",
        "outcome_event_raw", "outcome_year2_raw", "age_at_event", "sex",
        "education", "preceding_primary_episode_count",
    ]
    for cohort, d in sample.groupby("cohort", sort=False):
        e = episodes[cohort][join_fields].copy()
        early = e.add_prefix("earlier_").rename(columns={"earlier_episode_id": "first_episode_id"})
        later = e.add_prefix("later_").rename(columns={"later_episode_id": "second_episode_id"})
        q = d.merge(early, on="first_episode_id", how="left", validate="one_to_one")
        q = q.merge(later, on="second_episode_id", how="left", validate="one_to_one")
        if not q.person_id.eq(q.earlier_person_id).all() or not q.person_id.eq(q.later_person_id).all():
            raise AssertionError(f"{cohort}: episode-to-person join failed")
        q["earlier_acute_raw"] = q.earlier_outcome_event_raw - q.earlier_outcome_pre1_raw
        q["earlier_persistence_raw"] = q.earlier_outcome_year2_raw - q.earlier_outcome_pre1_raw
        q["later_acute_raw"] = q.later_outcome_event_raw - q.later_outcome_pre1_raw
        q["later_persistence_raw"] = q.later_outcome_year2_raw - q.later_outcome_pre1_raw
        pieces.append(q)
    out = pd.concat(pieces, ignore_index=True)
    expected = {"UKHLS": 1956, "HRS": 1166}
    counts = out.groupby("cohort").size().to_dict()
    if counts != expected or out[["cohort", "person_id"]].duplicated().any():
        raise AssertionError(f"Frozen sample mismatch: {counts}")
    if not out.earlier_primary_event_type.ne(out.later_primary_event_type).all():
        raise AssertionError("Same-type pair entered frozen sample")
    if not out.later_event_date.gt(out.earlier_event_date).all():
        raise AssertionError("Event ordering failed")
    return out.sort_values(["cohort", "person_id"]).reset_index(drop=True), episodes


def extract_ukhls_items(history: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    h = history.loc[history.cohort.eq("UKHLS")].copy()
    required = []
    for position in ["earlier", "later"]:
        episode_col = "first_episode_id" if position == "earlier" else "second_episode_id"
        for role, suffix in [("pre1", "pre1_wave"), ("event", "event_wave"), ("year2", "year2_wave")]:
            x = h[["cohort", "person_id", episode_col, f"{position}_{suffix}"]].copy()
            x.columns = ["cohort", "person_id", "episode_id", "wave"]
            x["episode_position"] = position
            x["time_role"] = role
            required.append(x)
    required = pd.concat(required, ignore_index=True)
    required["wave"] = required.wave.astype(int)
    metadata, waves = [], []
    for wave in sorted(required.wave.unique()):
        prefix = chr(ord("a") + wave - 1)
        path = UK_RAW / f"{prefix}_indresp.dta"
        cols = ["pidp"] + [f"{prefix}_scghq{s}" for s in "abcdefghijkl"] + [f"{prefix}_scghq1_dv"]
        reader = pd.read_stata(path, iterator=True, convert_categoricals=False)
        labels = reader.variable_labels()
        value_labels = reader.value_labels()
        for i, source in enumerate(cols[1:13], start=1):
            metadata.append({
                "record_type": "item_metadata", "cohort": "UKHLS", "instrument": "GHQ-12",
                "wave": wave, "item_position": i, "item_concept": GHQ_CONCEPTS[i - 1],
                "half": "A" if i in GHQ_HALF_A else "B", "source_variable": source,
                "variable_label": labels.get(source, ""), "raw_valid_codes": "1,2,3,4",
                "missing_codes": "-9,-8,-7,-2,-1", "scoring": "raw minus 1; 0-3 distress orientation",
                "reverse_coded": i in {1, 3, 4, 7, 8, 12}, "available": source in labels,
                "notes": json.dumps({str(int(k)): str(v).strip() for k, v in value_labels.get(source, {}).items()}, ensure_ascii=False),
            })
        d = pd.read_stata(path, columns=cols, convert_categoricals=False)
        d = d.loc[d.pidp.isin(required.loc[required.wave.eq(wave), "person_id"])].copy()
        d = d.rename(columns={"pidp": "person_id", f"{prefix}_scghq1_dv": "source_total"})
        item_cols = []
        for i, letter in enumerate("abcdefghijkl", start=1):
            source = f"{prefix}_scghq{letter}"
            target = f"item_{i:02d}"
            value = safe_num(d[source]).where(safe_num(d[source]).between(1, 4))
            d[target] = value - 1
            item_cols.append(target)
        d["source_total"] = safe_num(d.source_total).where(safe_num(d.source_total).between(0, 36))
        d["wave"] = wave
        d["source_variables"] = json.dumps(cols[1:13])
        waves.append(d[["person_id", "wave", "source_total", "source_variables", *item_cols]])
    long = required.merge(pd.concat(waves, ignore_index=True), on=["person_id", "wave"], how="left", validate="many_to_one")
    item_cols = [f"item_{i:02d}" for i in range(1, 13)]
    long["instrument"] = "GHQ-12"
    long["n_items_expected"] = 12
    long["n_items_observed"] = long[item_cols].notna().sum(axis=1)
    long["full_item_score"] = long[item_cols].sum(axis=1, min_count=12)
    half_a = [f"item_{i:02d}" for i in sorted(GHQ_HALF_A)]
    half_b = [c for c in item_cols if c not in half_a]
    long["half_a_score"] = long[half_a].sum(axis=1, min_count=len(half_a))
    long["half_b_score"] = long[half_b].sum(axis=1, min_count=len(half_b))
    long["item_complete"] = long.n_items_observed.eq(12)
    long["sum_matches_source_total"] = long.full_item_score.eq(long.source_total) & long.item_complete & long.source_total.notna()
    for i in range(13, 13):
        long[f"item_{i:02d}"] = np.nan
    return long, metadata


def extract_hrs_items(history: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    h = history.loc[history.cohort.eq("HRS")].copy()
    required = []
    for position in ["earlier", "later"]:
        episode_col = "first_episode_id" if position == "earlier" else "second_episode_id"
        for role, suffix in [("pre1", "pre1_wave"), ("event", "event_wave"), ("year2", "year2_wave")]:
            x = h[["cohort", "person_id", episode_col, f"{position}_{suffix}"]].copy()
            x.columns = ["cohort", "person_id", "episode_id", "wave"]
            x["episode_position"] = position
            x["time_role"] = role
            required.append(x)
    required = pd.concat(required, ignore_index=True)
    required["wave"] = required.wave.astype(int)
    reader = pd.read_stata(HRS_RAW, iterator=True, convert_categoricals=False)
    labels = reader.variable_labels()
    columns = ["hhidpn"]
    metadata = []
    for wave in sorted(required.wave.unique()):
        for i, stem in enumerate(CESD_STEMS, start=1):
            source = f"r{wave}{stem}"
            columns.append(source)
            metadata.append({
                "record_type": "item_metadata", "cohort": "HRS", "instrument": "CES-D 8",
                "wave": wave, "item_position": i, "item_concept": CESD_CONCEPTS[i - 1],
                "half": "A" if i in CESD_HALF_A else "B", "source_variable": source,
                "variable_label": labels.get(source, ""), "raw_valid_codes": "0,1",
                "missing_codes": "Stata/system missing or values outside 0/1",
                "scoring": "binary symptom; happy/enjoyed life reverse scored as 1-raw",
                "reverse_coded": stem in {"whappy", "enlife"}, "available": source in labels,
                "notes": "RAND item indicator; no attached Stata value label",
            })
        columns.append(f"r{wave}cesd")
    columns = list(dict.fromkeys(columns))
    wide = pd.read_stata(HRS_RAW, columns=columns, convert_categoricals=False)
    wide = wide.loc[wide.hhidpn.isin(required.person_id)].copy().rename(columns={"hhidpn": "person_id"})
    pieces = []
    for wave in sorted(required.wave.unique()):
        d = wide[["person_id", *[f"r{wave}{s}" for s in CESD_STEMS], f"r{wave}cesd"]].copy()
        d = d.rename(columns={f"r{wave}cesd": "source_total"})
        for i, stem in enumerate(CESD_STEMS, start=1):
            value = safe_num(d[f"r{wave}{stem}"])
            value = value.where(value.isin([0, 1]))
            if stem in {"whappy", "enlife"}:
                value = 1 - value
            d[f"item_{i:02d}"] = value
        d["source_total"] = safe_num(d.source_total).where(safe_num(d.source_total).between(0, 8))
        d["wave"] = wave
        d["source_variables"] = json.dumps([f"r{wave}{s}" for s in CESD_STEMS])
        pieces.append(d[["person_id", "wave", "source_total", "source_variables", *[f"item_{i:02d}" for i in range(1, 9)]]])
    long = required.merge(pd.concat(pieces, ignore_index=True), on=["person_id", "wave"], how="left", validate="many_to_one")
    item_cols = [f"item_{i:02d}" for i in range(1, 9)]
    long["instrument"] = "CES-D 8"
    long["n_items_expected"] = 8
    long["n_items_observed"] = long[item_cols].notna().sum(axis=1)
    long["full_item_score"] = long[item_cols].sum(axis=1, min_count=8)
    half_a = [f"item_{i:02d}" for i in sorted(CESD_HALF_A)]
    half_b = [c for c in item_cols if c not in half_a]
    long["half_a_score"] = long[half_a].sum(axis=1, min_count=len(half_a))
    long["half_b_score"] = long[half_b].sum(axis=1, min_count=len(half_b))
    long["item_complete"] = long.n_items_observed.eq(8)
    long["sum_matches_source_total"] = long.full_item_score.eq(long.source_total) & long.item_complete & long.source_total.notna()
    for i in range(9, 13):
        long[f"item_{i:02d}"] = np.nan
    return long, metadata


def cronbach_alpha(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x).all(axis=1)]
    if len(x) < 3 or x.shape[1] < 2:
        return np.nan
    item_var = np.var(x, axis=0, ddof=1)
    total_var = np.var(x.sum(axis=1), ddof=1)
    if total_var <= 0:
        return np.nan
    p = x.shape[1]
    return float(p / (p - 1) * (1 - item_var.sum() / total_var))


def omega_total(x: np.ndarray) -> tuple[float, bool, str]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x).all(axis=1)]
    if len(x) < max(20, 3 * x.shape[1]):
        return np.nan, False, "insufficient complete cases"
    sd = x.std(axis=0, ddof=1)
    if np.any(sd <= 0):
        return np.nan, False, "zero-variance item"
    r = np.corrcoef(x, rowvar=False)
    p = r.shape[0]
    eigvals, eigvecs = np.linalg.eigh(r)
    loading0 = eigvecs[:, -1] * math.sqrt(max(float(eigvals[-1] - 1), 0.1))
    if loading0.sum() < 0:
        loading0 *= -1
    loading0 = np.clip(loading0, 0.05, 0.8)
    unique0 = np.clip(1 - loading0**2, 0.1, 0.95)

    def unpack(theta):
        loading = 0.99 * np.tanh(theta[:p])
        unique = 0.001 + 0.999 / (1 + np.exp(-theta[p:]))
        return loading, unique

    def objective(theta):
        loading, unique = unpack(theta)
        sigma = np.outer(loading, loading) + np.diag(unique)
        sign, logdet = np.linalg.slogdet(sigma)
        if sign <= 0:
            return 1e12
        try:
            inv = np.linalg.inv(sigma)
        except np.linalg.LinAlgError:
            return 1e12
        return float(logdet + np.trace(inv @ r))

    theta0 = np.r_[np.arctanh(loading0 / 0.99), np.log((unique0 - 0.001) / (1 - unique0))]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = minimize(objective, theta0, method="L-BFGS-B", options={"maxiter": 2000, "ftol": 1e-12})
    loading, unique = unpack(fit.x)
    if loading.sum() < 0:
        loading *= -1
    denom = float(loading.sum() ** 2 + unique.sum())
    omega = float(loading.sum() ** 2 / denom) if denom > 0 else np.nan
    admissible = bool(fit.success and np.isfinite(omega) and np.all(unique > 0) and np.all(unique <= 1))
    return (omega if admissible else np.nan), admissible, str(fit.message)


def reliability_rows(items: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    audit_rows, split_rows = [], []
    for (cohort, position, role), g in items.groupby(["cohort", "episode_position", "time_role"], sort=False):
        nitems = int(g.n_items_expected.iloc[0])
        full_cols = [f"item_{i:02d}" for i in range(1, nitems + 1)]
        half_a_index = GHQ_HALF_A if cohort == "UKHLS" else CESD_HALF_A
        half_a_cols = [f"item_{i:02d}" for i in sorted(half_a_index)]
        half_b_cols = [c for c in full_cols if c not in half_a_cols]
        for scale, cols in [("full", full_cols), ("half_A", half_a_cols), ("half_B", half_b_cols)]:
            x = g[cols].to_numpy(dtype=float)
            complete = np.isfinite(x).all(axis=1)
            alpha = cronbach_alpha(x)
            omega, converged, message = omega_total(x)
            for statistic, estimate, status, notes in [
                ("alpha", alpha, True, "complete-case covariance"),
                ("omega_total", omega, converged, message),
            ]:
                audit_rows.append({
                    "record_type": "reliability", "cohort": cohort,
                    "instrument": g.instrument.iloc[0], "episode_position": position,
                    "time_role": role, "scale": scale, "statistic": statistic,
                    "estimate": estimate, "n": int(complete.sum()), "converged": status,
                    "notes": notes,
                })
        pair = g[["half_a_score", "half_b_score"]].dropna()
        r = pair.corr().iloc[0, 1] if len(pair) >= 3 else np.nan
        sb = 2 * r / (1 + r) if np.isfinite(r) and r > -1 else np.nan
        split_rows += [
            {"record_type": "split_reliability", "cohort": cohort, "episode_position": position,
             "time_role": role, "outcome_metric": "level", "direction": "A_vs_B",
             "statistic": "split_half_correlation", "estimate": r, "n": len(pair)},
            {"record_type": "split_reliability", "cohort": cohort, "episode_position": position,
             "time_role": role, "outcome_metric": "level", "direction": "A_vs_B",
             "statistic": "spearman_brown", "estimate": sb, "n": len(pair)},
        ]
    return pd.DataFrame(audit_rows), pd.DataFrame(split_rows)


def build_change_reliability(items: pd.DataFrame, reliability: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, change_wide = [], []
    level_rel = reliability.loc[reliability.record_type.eq("reliability")].copy()
    for (cohort, position), g in items.groupby(["cohort", "episode_position"], sort=False):
        wide = g.pivot(index=["person_id", "episode_id"], columns="time_role", values=["full_item_score", "half_a_score", "half_b_score"])
        wide.columns = [f"{a}_{b}" for a, b in wide.columns]
        wide = wide.reset_index()
        wide["cohort"] = cohort
        wide["episode_position"] = position
        for metric, end in [("acute", "event"), ("persistence", "year2")]:
            for scale, stem in [("full", "full_item_score"), ("half_A", "half_a_score"), ("half_B", "half_b_score")]:
                pre = wide[f"{stem}_pre1"]
                post = wide[f"{stem}_{end}"]
                diff = post - pre
                ok = pre.notna() & post.notna()
                var_pre = float(pre[ok].var(ddof=1)) if ok.sum() > 2 else np.nan
                var_post = float(post[ok].var(ddof=1)) if ok.sum() > 2 else np.nan
                var_diff = float(diff[ok].var(ddof=1)) if ok.sum() > 2 else np.nan
                for base_stat in ["alpha", "omega_total"]:
                    def rel_at(role):
                        q = level_rel.loc[
                            level_rel.cohort.eq(cohort) & level_rel.episode_position.eq(position)
                            & level_rel.time_role.eq(role) & level_rel.scale.eq(scale)
                            & level_rel.statistic.eq(base_stat), "estimate"
                        ]
                        return float(q.iloc[0]) if len(q) else np.nan
                    rp, rq = rel_at("pre1"), rel_at(end)
                    if all(np.isfinite(v) for v in [var_pre, var_post, var_diff, rp, rq]) and var_diff > 0:
                        change_rel = 1 - ((1 - rp) * var_pre + (1 - rq) * var_post) / var_diff
                    else:
                        change_rel = np.nan
                    rows.append({
                        "record_type": "change_reliability", "cohort": cohort,
                        "episode_position": position, "time_role": f"pre1_to_{end}",
                        "outcome_metric": metric, "scale": scale,
                        "statistic": f"{base_stat}_derived_change_reliability",
                        "estimate": change_rel, "n": int(ok.sum()),
                        "notes": "negative estimates retained; no truncation",
                    })
            a = wide[f"half_a_score_{end}"] - wide["half_a_score_pre1"]
            b = wide[f"half_b_score_{end}"] - wide["half_b_score_pre1"]
            pair = pd.DataFrame({"a": a, "b": b}).dropna()
            r = pair.corr().iloc[0, 1] if len(pair) >= 3 else np.nan
            sb = 2 * r / (1 + r) if np.isfinite(r) and r > -1 else np.nan
            rows += [
                {"record_type": "change_reliability", "cohort": cohort, "episode_position": position,
                 "time_role": f"pre1_to_{end}", "outcome_metric": metric, "scale": "split_halves",
                 "statistic": "change_split_half_correlation", "estimate": r, "n": len(pair), "notes": "observed"},
                {"record_type": "change_reliability", "cohort": cohort, "episode_position": position,
                 "time_role": f"pre1_to_{end}", "outcome_metric": metric, "scale": "split_halves",
                 "statistic": "change_spearman_brown", "estimate": sb, "n": len(pair), "notes": "2r/(1+r)"},
            ]
            wide[f"half_a_{metric}"] = a
            wide[f"half_b_{metric}"] = b
        change_wide.append(wide)
    return pd.DataFrame(rows), pd.concat(change_wide, ignore_index=True)


def fold_outcome_scale(cohort: str, train_people: set, cutoff, episodes: dict[str, pd.DataFrame]) -> tuple[float, float]:
    e = episodes[cohort]
    mask = e.person_id.isin(train_people) & e.n_primary_labels.gt(0)
    if pd.notna(cutoff):
        mask &= e.event_date.le(cutoff)
    ref = safe_num(e.loc[mask, "outcome_pre1_raw"]).dropna()
    mu, sd = float(ref.mean()), float(ref.std(ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        raise AssertionError(f"{cohort}: invalid training outcome scale")
    return mu, sd


def prepare_fold(tr: pd.DataFrame, te: pd.DataFrame, outcome: str, formulation: str, mu: float, sd: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    end = "event" if outcome == "acute" else "year2"
    prepared = []
    for d in [tr.copy(), te.copy()]:
        d["earlier_pre_z"] = (safe_num(d.earlier_outcome_pre1_raw) - mu) / sd
        d["later_pre_z"] = (safe_num(d.later_outcome_pre1_raw) - mu) / sd
        d["prior_acute_z"] = safe_num(d.earlier_acute_raw) / sd
        d["prior_persistence_z"] = safe_num(d.earlier_persistence_raw) / sd
        d["target_response_z"] = safe_num(d[f"later_{outcome}_raw"]) / sd
        if formulation == "change":
            d["analysis_outcome"] = d.target_response_z
        else:
            d["analysis_outcome"] = (safe_num(d[f"later_outcome_{end}_raw"]) - mu) / sd
        prepared.append(d)
    return prepared[0], prepared[1]


def design_matrix(d: pd.DataFrame, extras: list[str], centers: dict | None = None, names: list[str] | None = None) -> tuple[np.ndarray, list[str], dict]:
    if centers is None:
        centers = {
            "later_age": float(safe_num(d.later_age_at_event).mean()),
            "later_year": float(safe_num(d.later_event_year).mean()),
            "later_preceding": float(safe_num(d.later_preceding_primary_episode_count).mean()),
        }
    data: dict[str, np.ndarray] = {"Intercept": np.ones(len(d), dtype=float)}
    for event in PRIMARY_TYPES[1:]:
        data[f"later_event[{event}]"] = d.later_primary_event_type.eq(event).to_numpy(dtype=float)
    data["later_age_centered"] = safe_num(d.later_age_at_event).to_numpy(dtype=float) - centers["later_age"]
    data["sex[female]"] = d.later_sex.eq("female").to_numpy(dtype=float)
    for level in EDUCATION_LEVELS[1:]:
        data[f"education[{level}]"] = d.later_education.fillna("unknown").eq(level).to_numpy(dtype=float)
    data["later_year_centered"] = safe_num(d.later_event_year).to_numpy(dtype=float) - centers["later_year"]
    data["preceding_events_centered"] = safe_num(d.later_preceding_primary_episode_count).to_numpy(dtype=float) - centers["later_preceding"]
    for extra in extras:
        data[extra] = safe_num(d[extra]).to_numpy(dtype=float)
    if names is None:
        names = list(data)
    x = np.zeros((len(d), len(names)), dtype=float)
    for j, name in enumerate(names):
        if name in data:
            x[:, j] = data[name]
    return x, names, centers


def residualize_prior(tr: pd.DataFrame, te: pd.DataFrame, response: str) -> tuple[np.ndarray, np.ndarray, dict]:
    centers = {
        "age": float(safe_num(tr.earlier_age_at_event).mean()),
        "year": float(safe_num(tr.earlier_event_year).mean()),
        "preceding": float(safe_num(tr.earlier_preceding_primary_episode_count).mean()),
    }

    def xmat(d):
        data = {"Intercept": np.ones(len(d), dtype=float)}
        for event in PRIMARY_TYPES[1:]:
            data[f"earlier_event[{event}]"] = d.earlier_primary_event_type.eq(event).to_numpy(dtype=float)
        data["earlier_pre_z"] = safe_num(d.earlier_pre_z).to_numpy(dtype=float)
        data["earlier_age_centered"] = safe_num(d.earlier_age_at_event).to_numpy(dtype=float) - centers["age"]
        data["earlier_year_centered"] = safe_num(d.earlier_event_year).to_numpy(dtype=float) - centers["year"]
        data["earlier_preceding_centered"] = safe_num(d.earlier_preceding_primary_episode_count).to_numpy(dtype=float) - centers["preceding"]
        return np.column_stack(list(data.values()))

    xtr, xte = xmat(tr), xmat(te)
    ytr = safe_num(tr[response]).to_numpy(dtype=float)
    yte = safe_num(te[response]).to_numpy(dtype=float)
    beta = np.linalg.pinv(xtr) @ ytr
    return ytr - xtr @ beta, yte - xte @ beta, {"rank": int(np.linalg.matrix_rank(xtr)), "n_columns": int(xtr.shape[1])}


def fit_linear(x: np.ndarray, y: np.ndarray, penalty: float = 0.0, penalized: list[int] | None = None) -> tuple[np.ndarray, float]:
    if penalty <= 0 or not penalized:
        beta = np.linalg.pinv(x) @ y
    else:
        p = np.zeros(x.shape[1])
        p[penalized] = penalty
        beta = np.linalg.pinv(x.T @ x + np.diag(p)) @ (x.T @ y)
    resid = y - x @ beta
    variance = float(np.mean(resid**2))
    return beta, variance


def select_ridge_alpha(d: pd.DataFrame, y: np.ndarray, extras: list[str], penalized_names: list[str], centers: dict) -> float:
    assignment = simple_group_folds(d.person_id, min(5, max(2, len(d))), SEED + 17)
    losses = {alpha: [] for alpha in RIDGE_GRID}
    for fold in sorted(set(assignment.values())):
        valid = d.person_id.map(assignment).eq(fold).to_numpy()
        train = ~valid
        if train.sum() < 5 or valid.sum() < 1:
            continue
        xtr, names, _ = design_matrix(d.loc[train], extras, centers=centers)
        xva, _, _ = design_matrix(d.loc[valid], extras, centers=centers, names=names)
        indices = [names.index(name) for name in penalized_names if name in names]
        for alpha in RIDGE_GRID:
            beta, _ = fit_linear(xtr, y[train], alpha, indices)
            losses[alpha].append(float(np.mean((y[valid] - xva @ beta) ** 2)))
    eligible = [(float(np.mean(v)), alpha) for alpha, v in losses.items() if v]
    return min(eligible)[1] if eligible else 0.0


def metric_values(y, pred, variance, weights=None) -> dict:
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    variance = np.asarray(variance, dtype=float)
    if variance.ndim == 0:
        variance = np.full(len(y), float(variance))
    if weights is None:
        weights = np.ones(len(y), dtype=float)
    weights = np.asarray(weights, dtype=float)
    good = np.isfinite(y) & np.isfinite(pred) & np.isfinite(weights) & (weights > 0)
    y, pred, weights, variance = y[good], pred[good], weights[good], variance[good]
    wsum = float(weights.sum())
    err = y - pred
    mse = float(np.sum(weights * err**2) / wsum)
    mae = float(np.sum(weights * np.abs(err)) / wsum)
    ybar = float(np.sum(weights * y) / wsum)
    denom = float(np.sum(weights * (y - ybar) ** 2))
    r2 = 1 - float(np.sum(weights * err**2)) / denom if denom > 0 else np.nan
    z = np.column_stack([np.ones(len(pred)), pred])
    zw = z * np.sqrt(weights)[:, None]
    yw = y * np.sqrt(weights)
    calibration = np.linalg.pinv(zw) @ yw
    if not np.all(np.isfinite(variance) & (variance > 0)):
        raise ValueError(
            "Predictive density requires finite positive training residual variance; "
            "evaluation-set MSE must not replace training variance."
        )
    valid_var = variance
    lpd = float(np.sum(weights * (-0.5 * (np.log(2 * np.pi * valid_var) + err**2 / valid_var))) / wsum)
    return {
        "RMSE": math.sqrt(mse), "MAE": mae, "predictive_R2": r2,
        "mean_log_predictive_density": lpd,
        "calibration_intercept": float(calibration[0]),
        "calibration_slope": float(calibration[1]),
    }


def comparison_improvements(base: dict, candidate: dict) -> dict:
    return {
        "RMSE": base["RMSE"] - candidate["RMSE"],
        "MAE": base["MAE"] - candidate["MAE"],
        "predictive_R2": candidate["predictive_R2"] - base["predictive_R2"],
        "mean_log_predictive_density": candidate["mean_log_predictive_density"] - base["mean_log_predictive_density"],
    }


def bootstrap_bundle_comparison(bundle: pd.DataFrame, base: str, candidate: str, seed_offset: int = 0) -> tuple[dict, dict, int, int]:
    y = bundle.observed.to_numpy(dtype=float)
    pb = bundle[f"pred::{base}"].to_numpy(dtype=float)
    pc = bundle[f"pred::{candidate}"].to_numpy(dtype=float)
    vb = bundle[f"var::{base}"].to_numpy(dtype=float)
    vc = bundle[f"var::{candidate}"].to_numpy(dtype=float)
    base_point = metric_values(y, pb, vb)
    candidate_point = metric_values(y, pc, vc)
    point = comparison_improvements(base_point, candidate_point)
    people = pd.unique(bundle.person_id)
    rng = np.random.default_rng(SEED + seed_offset)
    values = {m: [] for m in point}
    attempts = 0
    while len(values["RMSE"]) < BOOT_SUCCESS and attempts < BOOT_MAX_ATTEMPTS:
        attempts += 1
        counts = pd.Series(rng.multinomial(len(people), np.full(len(people), 1 / len(people))), index=people)
        w = bundle.person_id.map(counts).to_numpy(dtype=float)
        mb = metric_values(y, pb, vb, w)
        mc = metric_values(y, pc, vc, w)
        gain = comparison_improvements(mb, mc)
        if all(np.isfinite(list(gain.values()))):
            for metric, value in gain.items():
                values[metric].append(value)
    intervals = {
        metric: (float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975)), float(np.std(v, ddof=1)))
        if v else (np.nan, np.nan, np.nan)
        for metric, v in values.items()
    }
    return point, intervals, len(values["RMSE"]), attempts


def fit_named_model(tr: pd.DataFrame, te: pd.DataFrame, extras: list[str], ridge: bool = False) -> tuple[np.ndarray, float, float, int]:
    xtr, names, centers = design_matrix(tr, extras)
    xte, _, _ = design_matrix(te, extras, centers=centers, names=names)
    ytr = safe_num(tr.analysis_outcome).to_numpy(dtype=float)
    prior_names = [n for n in extras if n.startswith("prior_") or n.startswith("resid_")]
    alpha = select_ridge_alpha(tr, ytr, extras, prior_names, centers) if ridge else 0.0
    indices = [names.index(n) for n in prior_names if n in names]
    beta, variance = fit_linear(xtr, ytr, alpha, indices)
    return xte @ beta, variance, alpha, int(np.linalg.matrix_rank(xtr))


def run_baseline_predictions(history: pd.DataFrame, episodes: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows, all_bundles = [], []
    counter = 0
    for cohort, d0 in history.groupby("cohort", sort=False):
        for validation in ["grouped_10fold", "temporal_70_30"]:
            splits, cutoff = validation_splits(d0, validation)
            for outcome in ["acute", "persistence"]:
                prior = f"prior_{outcome}_z"
                for formulation in ["change", "ancova_level"]:
                    output = []
                    for fold, train_mask, test_mask in splits:
                        tr0, te0 = d0.loc[train_mask].copy(), d0.loc[test_mask].copy()
                        mu, sd = fold_outcome_scale(cohort, set(tr0.person_id), cutoff, episodes)
                        tr, te = prepare_fold(tr0, te0, outcome, formulation, mu, sd)
                        mandatory = ["later_pre_z"] if formulation == "ancova_level" else []
                        specifications = {
                            "B0": mandatory,
                            "B1": list(dict.fromkeys([*mandatory, "earlier_pre_z"])),
                            "B2": list(dict.fromkeys([*mandatory, prior])),
                            "B3": list(dict.fromkeys([*mandatory, "later_pre_z"])),
                            "B4": list(dict.fromkeys([*mandatory, "later_pre_z", prior])),
                        }
                        frame = te[["cohort", "person_id", "sample_id"]].copy()
                        frame["observed"] = te.analysis_outcome.to_numpy(dtype=float)
                        frame["fold"] = fold
                        frame["cutoff"] = cutoff
                        frame["n_train"] = len(tr)
                        for model, extras in specifications.items():
                            pred, variance, alpha, rank = fit_named_model(tr, te, extras)
                            frame[f"pred::{model}"] = pred
                            frame[f"var::{model}"] = variance
                            frame[f"rank::{model}"] = rank
                        output.append(frame)
                    bundle = pd.concat(output, ignore_index=True)
                    bundle["validation"] = validation
                    bundle["outcome_metric"] = outcome
                    bundle["outcome_formulation"] = formulation
                    all_bundles.append(bundle)
                    n_persons = int(bundle.person_id.nunique())
                    for model in ["B0", "B1", "B2", "B3", "B4"]:
                        values = metric_values(bundle.observed, bundle[f"pred::{model}"], bundle[f"var::{model}"])
                        for metric, estimate in values.items():
                            metric_rows.append({
                                "record_type": "model_metric", "cohort": cohort,
                                "validation": validation, "outcome_metric": outcome,
                                "outcome_formulation": formulation, "model": model,
                                "base_model": "", "candidate_model": "", "comparison": "",
                                "metric": metric, "estimate": estimate, "ci_lower": np.nan,
                                "ci_upper": np.nan, "bootstrap_se": np.nan,
                                "n_persons": n_persons, "n_train_min": int(bundle.n_train.min()),
                                "n_train_max": int(bundle.n_train.max()), "temporal_cutoff": cutoff,
                                "bootstrap_success": np.nan, "bootstrap_attempts": np.nan,
                                "notes": "",
                            })
                    for base, candidate in [("B0", "B1"), ("B0", "B2"), ("B0", "B3"), ("B3", "B4")]:
                        counter += 1
                        point, intervals, success, attempts = bootstrap_bundle_comparison(bundle, base, candidate, counter)
                        for metric, estimate in point.items():
                            lo, hi, se = intervals[metric]
                            metric_rows.append({
                                "record_type": "comparison", "cohort": cohort,
                                "validation": validation, "outcome_metric": outcome,
                                "outcome_formulation": formulation, "model": "",
                                "base_model": base, "candidate_model": candidate,
                                "comparison": f"{candidate}_minus_{base}", "metric": metric,
                                "estimate": estimate, "ci_lower": lo, "ci_upper": hi,
                                "bootstrap_se": se, "n_persons": n_persons,
                                "n_train_min": int(bundle.n_train.min()), "n_train_max": int(bundle.n_train.max()),
                                "temporal_cutoff": cutoff, "bootstrap_success": success,
                                "bootstrap_attempts": attempts,
                                "notes": "structural identity under mandatory ANCOVA baseline" if formulation == "ancova_level" and (base, candidate) == ("B0", "B3") else "",
                            })
    return pd.DataFrame(metric_rows), pd.concat(all_bundles, ignore_index=True)


def run_representation_predictions(history: pd.DataFrame, episodes: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, bundles = [], []
    counter = 500
    for cohort, d0 in history.groupby("cohort", sort=False):
        # Keep every representation comparison paired on the same held-out
        # persons.  One UKHLS row lacks age at the earlier event and is therefore
        # excluded from this residualization-required block only.
        residualization_required = [
            "earlier_outcome_pre1_raw", "earlier_age_at_event", "earlier_event_year",
            "earlier_preceding_primary_episode_count", "earlier_acute_raw",
            "earlier_persistence_raw",
        ]
        d0 = d0.dropna(subset=residualization_required).copy()
        for validation in ["grouped_10fold", "temporal_70_30"]:
            splits, cutoff = validation_splits(d0, validation)
            for outcome in ["acute", "persistence"]:
                for formulation in ["change", "ancova_level"]:
                    scopes = [("context", []) , ("current_state", ["later_pre_z"])] if formulation == "change" else [("ancova_current_state", ["later_pre_z"])]
                    for scope, base_extras in scopes:
                        output = []
                        for fold, train_mask, test_mask in splits:
                            tr0, te0 = d0.loc[train_mask].copy(), d0.loc[test_mask].copy()
                            mu, sd = fold_outcome_scale(cohort, set(tr0.person_id), cutoff, episodes)
                            tr, te = prepare_fold(tr0, te0, outcome, formulation, mu, sd)
                            rtr_a, rte_a, _ = residualize_prior(tr, te, "prior_acute_z")
                            rtr_p, rte_p, _ = residualize_prior(tr, te, "prior_persistence_z")
                            tr["resid_acute"] = rtr_a; te["resid_acute"] = rte_a
                            tr["resid_persistence"] = rtr_p; te["resid_persistence"] = rte_p
                            specs = {
                                "BASE": (base_extras, False),
                                "R1_prior_acute": ([*base_extras, "prior_acute_z"], False),
                                "R2_prior_persistence": ([*base_extras, "prior_persistence_z"], False),
                                "R3_joint_raw": ([*base_extras, "prior_acute_z", "prior_persistence_z"], True),
                                "R4_acute_residual": ([*base_extras, "resid_acute"], False),
                                "R4_persistence_residual": ([*base_extras, "resid_persistence"], False),
                                "R4_joint_residual": ([*base_extras, "resid_acute", "resid_persistence"], True),
                            }
                            frame = te[["cohort", "person_id", "sample_id"]].copy()
                            frame["observed"] = te.analysis_outcome.to_numpy(dtype=float)
                            frame["fold"] = fold; frame["cutoff"] = cutoff; frame["n_train"] = len(tr)
                            for model, (extras, ridge) in specs.items():
                                pred, variance, alpha, rank = fit_named_model(tr, te, extras, ridge)
                                frame[f"pred::{model}"] = pred
                                frame[f"var::{model}"] = variance
                                frame[f"alpha::{model}"] = alpha
                                frame[f"rank::{model}"] = rank
                            output.append(frame)
                        bundle = pd.concat(output, ignore_index=True)
                        bundle["validation"] = validation; bundle["outcome_metric"] = outcome
                        bundle["outcome_formulation"] = formulation; bundle["baseline_scope"] = scope
                        bundles.append(bundle)
                        n_persons = int(bundle.person_id.nunique())
                        models = ["BASE", "R1_prior_acute", "R2_prior_persistence", "R3_joint_raw", "R4_acute_residual", "R4_persistence_residual", "R4_joint_residual"]
                        for model in models:
                            values = metric_values(bundle.observed, bundle[f"pred::{model}"], bundle[f"var::{model}"])
                            alpha_values = bundle[f"alpha::{model}"].dropna()
                            for metric, estimate in values.items():
                                rows.append({
                                    "record_type": "model_metric", "cohort": cohort, "validation": validation,
                                    "outcome_metric": outcome, "outcome_formulation": formulation,
                                    "baseline_scope": scope, "model": model, "base_model": "",
                                    "candidate_model": "", "comparison": "", "metric": metric,
                                    "estimate": estimate, "ci_lower": np.nan, "ci_upper": np.nan,
                                    "bootstrap_se": np.nan, "n_persons": n_persons,
                                    "ridge_alpha_min": float(alpha_values.min()) if len(alpha_values) else np.nan,
                                    "ridge_alpha_median": float(alpha_values.median()) if len(alpha_values) else np.nan,
                                    "ridge_alpha_max": float(alpha_values.max()) if len(alpha_values) else np.nan,
                                    "temporal_cutoff": cutoff, "bootstrap_success": np.nan,
                                    "bootstrap_attempts": np.nan, "notes": "",
                                })
                        for candidate in models[1:]:
                            counter += 1
                            point, intervals, success, attempts = bootstrap_bundle_comparison(bundle, "BASE", candidate, counter)
                            for metric, estimate in point.items():
                                lo, hi, se = intervals[metric]
                                rows.append({
                                    "record_type": "comparison", "cohort": cohort, "validation": validation,
                                    "outcome_metric": outcome, "outcome_formulation": formulation,
                                    "baseline_scope": scope, "model": "", "base_model": "BASE",
                                    "candidate_model": candidate, "comparison": f"{candidate}_minus_BASE",
                                    "metric": metric, "estimate": estimate, "ci_lower": lo,
                                    "ci_upper": hi, "bootstrap_se": se, "n_persons": n_persons,
                                    "ridge_alpha_min": np.nan, "ridge_alpha_median": np.nan,
                                    "ridge_alpha_max": np.nan, "temporal_cutoff": cutoff,
                                    "bootstrap_success": success, "bootstrap_attempts": attempts,
                                    "notes": "paired person-cluster bootstrap",
                                })
    return pd.DataFrame(rows), pd.concat(bundles, ignore_index=True)


def make_item_person_history(history: pd.DataFrame, change_wide: pd.DataFrame) -> pd.DataFrame:
    early = change_wide.loc[change_wide.episode_position.eq("earlier")].drop(columns=["cohort", "episode_position"])
    later = change_wide.loc[change_wide.episode_position.eq("later")].drop(columns=["cohort", "episode_position"])
    early = early.add_prefix("item_earlier_").rename(columns={"item_earlier_person_id": "person_id", "item_earlier_episode_id": "first_episode_id"})
    later = later.add_prefix("item_later_").rename(columns={"item_later_person_id": "person_id", "item_later_episode_id": "second_episode_id"})
    out = history.merge(early, on=["person_id", "first_episode_id"], how="left", validate="one_to_one")
    out = out.merge(later, on=["person_id", "second_episode_id"], how="left", validate="one_to_one")
    return out


def item_prepare_fold(tr: pd.DataFrame, te: pd.DataFrame, outcome: str, formulation: str, predictor_half: str, outcome_half: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    end = "event" if outcome == "acute" else "year2"
    prior_pre = f"item_earlier_half_{predictor_half}_score_pre1"
    target_pre = f"item_later_half_{outcome_half}_score_pre1"
    prior_response = f"item_earlier_half_{predictor_half}_{outcome}"
    target_response = f"item_later_half_{outcome_half}_{outcome}"
    target_level = f"item_later_half_{outcome_half}_score_{end}"
    mu_prior, sd_prior = float(safe_num(tr[prior_pre]).mean()), float(safe_num(tr[prior_pre]).std(ddof=1))
    mu_target, sd_target = float(safe_num(tr[target_pre]).mean()), float(safe_num(tr[target_pre]).std(ddof=1))
    if min(sd_prior, sd_target) <= 0 or not np.isfinite(sd_prior + sd_target):
        raise AssertionError("Invalid split-half training scale")
    prepared = []
    for d in [tr.copy(), te.copy()]:
        d["prior_item_response"] = safe_num(d[prior_response]) / sd_prior
        d["later_item_pre"] = (safe_num(d[target_pre]) - mu_target) / sd_target
        if formulation == "change":
            d["analysis_outcome"] = safe_num(d[target_response]) / sd_target
        else:
            d["analysis_outcome"] = (safe_num(d[target_level]) - mu_target) / sd_target
        prepared.append(d)
    return prepared[0], prepared[1]


def bootstrap_average_directions(direction_bundles: dict[str, pd.DataFrame], seed_offset: int) -> tuple[dict, dict, int, int]:
    directions = list(direction_bundles)
    common = set(direction_bundles[directions[0]].person_id)
    for d in directions[1:]:
        common &= set(direction_bundles[d].person_id)
    common = sorted(common)
    point_by_direction = {}
    for direction, b in direction_bundles.items():
        b = b.loc[b.person_id.isin(common)]
        mb = metric_values(b.observed, b["pred::BASE"], b["var::BASE"])
        mc = metric_values(b.observed, b["pred::SPLIT_PRIOR"], b["var::SPLIT_PRIOR"])
        point_by_direction[direction] = comparison_improvements(mb, mc)
    point = {m: float(np.mean([v[m] for v in point_by_direction.values()])) for m in point_by_direction[directions[0]]}
    rng = np.random.default_rng(SEED + seed_offset)
    values = {m: [] for m in point}
    attempts = 0
    while len(values["RMSE"]) < BOOT_SUCCESS and attempts < BOOT_MAX_ATTEMPTS:
        attempts += 1
        counts = pd.Series(rng.multinomial(len(common), np.full(len(common), 1 / len(common))), index=common)
        gains = []
        for direction, b in direction_bundles.items():
            b = b.loc[b.person_id.isin(common)]
            w = b.person_id.map(counts).to_numpy(dtype=float)
            mb = metric_values(b.observed, b["pred::BASE"], b["var::BASE"], w)
            mc = metric_values(b.observed, b["pred::SPLIT_PRIOR"], b["var::SPLIT_PRIOR"], w)
            gains.append(comparison_improvements(mb, mc))
        avg = {m: float(np.mean([g[m] for g in gains])) for m in point}
        if all(np.isfinite(list(avg.values()))):
            for m, value in avg.items():
                values[m].append(value)
    intervals = {
        m: (float(np.quantile(v, .025)), float(np.quantile(v, .975)), float(np.std(v, ddof=1))) if v else (np.nan, np.nan, np.nan)
        for m, v in values.items()
    }
    return point, intervals, len(values["RMSE"]), attempts


def run_split_predictions(item_history: pd.DataFrame) -> pd.DataFrame:
    rows = []
    counter = 1000
    for cohort, d0 in item_history.groupby("cohort", sort=False):
        for validation in ["grouped_10fold", "temporal_70_30"]:
            splits, cutoff = validation_splits(d0, validation)
            for outcome in ["acute", "persistence"]:
                for formulation in ["change", "ancova_level"]:
                    scopes = [("context", []) , ("current_state", ["later_item_pre"])] if formulation == "change" else [("ancova_current_state", ["later_item_pre"])]
                    for scope, base_extras in scopes:
                        direction_bundles = {}
                        for direction, predictor_half, outcome_half in [("A_to_B", "a", "b"), ("B_to_A", "b", "a")]:
                            output = []
                            for fold, train_mask, test_mask in splits:
                                tr0, te0 = d0.loc[train_mask].copy(), d0.loc[test_mask].copy()
                                tr, te = item_prepare_fold(tr0, te0, outcome, formulation, predictor_half, outcome_half)
                                needed = ["analysis_outcome", "prior_item_response", *base_extras]
                                tr = tr.dropna(subset=needed); te = te.dropna(subset=needed)
                                if len(tr) < 10 or len(te) < 2:
                                    continue
                                frame = te[["cohort", "person_id", "sample_id"]].copy()
                                frame["observed"] = te.analysis_outcome.to_numpy(dtype=float)
                                frame["fold"] = fold; frame["cutoff"] = cutoff; frame["n_train"] = len(tr)
                                for model, extras in [("BASE", base_extras), ("SPLIT_PRIOR", [*base_extras, "prior_item_response"])]:
                                    pred, variance, _, _ = fit_named_model(tr, te, extras)
                                    frame[f"pred::{model}"] = pred
                                    frame[f"var::{model}"] = variance
                                output.append(frame)
                            if output:
                                direction_bundles[direction] = pd.concat(output, ignore_index=True)
                        if len(direction_bundles) != 2:
                            rows.append({
                                "record_type": "availability", "cohort": cohort, "validation": validation,
                                "outcome_metric": outcome, "outcome_formulation": formulation,
                                "baseline_scope": scope, "direction": "average", "statistic": "split_prediction",
                                "notes": "one or both frozen directions unavailable",
                            })
                            continue
                        for direction, bundle in direction_bundles.items():
                            for model in ["BASE", "SPLIT_PRIOR"]:
                                values = metric_values(bundle.observed, bundle[f"pred::{model}"], bundle[f"var::{model}"])
                                for metric, estimate in values.items():
                                    rows.append({
                                        "record_type": "model_metric", "cohort": cohort, "validation": validation,
                                        "outcome_metric": outcome, "outcome_formulation": formulation,
                                        "baseline_scope": scope, "direction": direction, "model": model,
                                        "base_model": "", "candidate_model": "", "comparison": "",
                                        "statistic": metric, "estimate": estimate, "ci_lower": np.nan,
                                        "ci_upper": np.nan, "bootstrap_se": np.nan,
                                        "n": int(bundle.person_id.nunique()), "bootstrap_success": np.nan,
                                        "bootstrap_attempts": np.nan, "notes": "",
                                    })
                        counter += 1
                        point, intervals, success, attempts = bootstrap_average_directions(direction_bundles, counter)
                        common_n = len(set(direction_bundles["A_to_B"].person_id) & set(direction_bundles["B_to_A"].person_id))
                        for metric, estimate in point.items():
                            lo, hi, se = intervals[metric]
                            rows.append({
                                "record_type": "comparison", "cohort": cohort, "validation": validation,
                                "outcome_metric": outcome, "outcome_formulation": formulation,
                                "baseline_scope": scope, "direction": "average",
                                "model": "", "base_model": "BASE", "candidate_model": "SPLIT_PRIOR",
                                "comparison": "SPLIT_PRIOR_minus_BASE", "statistic": metric,
                                "estimate": estimate, "ci_lower": lo, "ci_upper": hi,
                                "bootstrap_se": se, "n": common_n, "bootstrap_success": success,
                                "bootstrap_attempts": attempts,
                                "notes": "equal-weight average of A-to-B and B-to-A gains",
                            })
    return pd.DataFrame(rows)


def observed_split_correlations(item_history: pd.DataFrame, change_reliability: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cohort, d in item_history.groupby("cohort", sort=False):
        for metric in ["acute", "persistence"]:
            directional = []
            for direction, p_half, t_half in [("A_to_B", "a", "b"), ("B_to_A", "b", "a")]:
                x = safe_num(d[f"item_earlier_half_{p_half}_{metric}"])
                y = safe_num(d[f"item_later_half_{t_half}_{metric}"])
                pair = pd.DataFrame({"x": x, "y": y}).dropna()
                r = pair.corr().iloc[0, 1] if len(pair) >= 3 else np.nan
                def rel(position, half):
                    q = change_reliability.loc[
                        change_reliability.cohort.eq(cohort)
                        & change_reliability.episode_position.eq(position)
                        & change_reliability.outcome_metric.eq(metric)
                        & change_reliability.scale.eq(f"half_{half.upper()}")
                        & change_reliability.statistic.eq("alpha_derived_change_reliability"), "estimate"
                    ]
                    return float(q.iloc[0]) if len(q) else np.nan
                rp, rt = rel("earlier", p_half), rel("later", t_half)
                dis = r / math.sqrt(rp * rt) if np.isfinite(r) and rp > 0 and rt > 0 else np.nan
                directional.append((r, dis))
                rows += [
                    {"record_type": "observed_correlation", "cohort": cohort, "outcome_metric": metric,
                     "direction": direction, "statistic": "observed_cross_form_correlation",
                     "estimate": r, "n": len(pair), "notes": "sensitivity; no hypothesis test"},
                    {"record_type": "disattenuated_correlation", "cohort": cohort, "outcome_metric": metric,
                     "direction": direction, "statistic": "alpha_disattenuated_cross_form_correlation",
                     "estimate": dis, "n": len(pair), "notes": "sensitivity only; not substituted for observed prediction"},
                ]
            rows += [
                {"record_type": "observed_correlation", "cohort": cohort, "outcome_metric": metric,
                 "direction": "average", "statistic": "observed_cross_form_correlation",
                 "estimate": float(np.nanmean([x[0] for x in directional])), "n": len(d), "notes": "equal-weight directional average"},
                {"record_type": "disattenuated_correlation", "cohort": cohort, "outcome_metric": metric,
                 "direction": "average", "statistic": "alpha_disattenuated_cross_form_correlation",
                 "estimate": float(np.nanmean([x[1] for x in directional])), "n": len(d), "notes": "equal-weight directional average; sensitivity only"},
            ]
    return pd.DataFrame(rows)


def assemble_item_audit(items: pd.DataFrame, metadata: pd.DataFrame, reliability: pd.DataFrame) -> pd.DataFrame:
    rows = [metadata]
    completeness = []
    for (cohort, position, role), g in items.groupby(["cohort", "episode_position", "time_role"], sort=False):
        complete = int(g.item_complete.sum())
        comparable = g.item_complete & g.source_total.notna()
        matches = int(g.loc[comparable, "sum_matches_source_total"].sum())
        completeness += [
            {"record_type": "completeness", "cohort": cohort, "instrument": g.instrument.iloc[0],
             "episode_position": position, "time_role": role, "eligible_n": len(g),
             "item_complete_n": complete, "item_complete_pct": complete / len(g),
             "n": len(g), "notes": "all frozen two-event persons at the specified occasion"},
            {"record_type": "scoring_check", "cohort": cohort, "instrument": g.instrument.iloc[0],
             "episode_position": position, "time_role": role, "eligible_n": int(comparable.sum()),
             "item_complete_n": matches, "item_complete_pct": matches / comparable.sum() if comparable.sum() else np.nan,
             "n": int(comparable.sum()), "statistic": "reconstructed_total_exact_match",
             "estimate": matches / comparable.sum() if comparable.sum() else np.nan,
             "notes": "item sum compared with existing harmonized total when both observed"},
        ]
    rows += [pd.DataFrame(completeness), reliability]
    return pd.concat(rows, ignore_index=True, sort=False)


def practical_benchmarks(baseline: pd.DataFrame, representations: pd.DataFrame, split: pd.DataFrame) -> pd.DataFrame:
    sources = []
    b = baseline.loc[
        baseline.record_type.eq("comparison") & baseline.metric.eq("predictive_R2")
        & baseline.comparison.isin(["B2_minus_B0", "B4_minus_B3"])
    ].copy()
    b["source_table"] = "baseline_information_decomposition"
    b["baseline_scope"] = b.comparison.map({"B2_minus_B0": "context", "B4_minus_B3": "current_state"})
    sources.append(b.rename(columns={"metric": "statistic"}))
    r = representations.loc[
        representations.record_type.eq("comparison") & representations.metric.eq("predictive_R2")
    ].copy()
    r["source_table"] = "prior_response_representation"
    sources.append(r.rename(columns={"metric": "statistic"}))
    s = split.loc[
        split.record_type.eq("comparison") & split.statistic.eq("predictive_R2") & split.direction.eq("average")
    ].copy()
    s["source_table"] = "split_half_cross_form"
    s["n_persons"] = s["n"]
    sources.append(s)
    gains = pd.concat(sources, ignore_index=True, sort=False)
    eu = pd.read_csv(EU_SOURCE)
    eu = eu.loc[
        eu.record_type.eq("comparison") & eu.comparison.eq("E_minus_U")
        & eu.metric.eq("predictive_R2"), ["cohort", "validation", "improvement"]
    ].rename(columns={"improvement": "eu_predictive_R2_gain"})
    gains = gains.merge(eu, on=["cohort", "validation"], how="left", validate="many_to_one")
    out = []
    for row in gains.to_dict("records"):
        estimate, lo, hi = row.get("estimate"), row.get("ci_lower"), row.get("ci_upper")
        eu_gain = row.get("eu_predictive_R2_gain")
        ratio = estimate / abs(eu_gain) if pd.notna(estimate) and pd.notna(eu_gain) and eu_gain != 0 else np.nan
        same_order = bool(np.isfinite(ratio) and 0.1 <= abs(ratio) <= 10)
        for threshold in [0.0025, 0.005, 0.010]:
            out.append({
                "source_table": row.get("source_table"), "cohort": row.get("cohort"),
                "validation": row.get("validation"), "outcome_metric": row.get("outcome_metric"),
                "outcome_formulation": row.get("outcome_formulation"),
                "baseline_scope": row.get("baseline_scope", ""), "comparison": row.get("comparison"),
                "estimate": estimate, "ci_lower": lo, "ci_upper": hi,
                "benchmark": threshold, "upper_below": bool(pd.notna(hi) and hi < threshold),
                "interval_crosses": bool(pd.notna(lo) and pd.notna(hi) and lo <= threshold <= hi),
                "lower_above": bool(pd.notna(lo) and lo > threshold),
                "eu_predictive_R2_gain": eu_gain, "gain_to_abs_eu_ratio": ratio,
                "same_order_of_magnitude_as_eu": same_order,
                "n_persons": row.get("n_persons", row.get("n")),
                "notes": "benchmarks are predictive-statistical, not clinical",
            })
    return pd.DataFrame(out)


def cross_cohort_summary(baseline: pd.DataFrame, representations: pd.DataFrame, split: pd.DataFrame) -> pd.DataFrame:
    frames = []
    b = baseline.loc[
        baseline.record_type.eq("comparison") & baseline.metric.eq("predictive_R2")
        & baseline.comparison.isin(["B2_minus_B0", "B4_minus_B3"])
    ].copy()
    b["comparison_key"] = b.comparison
    b["baseline_scope"] = b.comparison.map({"B2_minus_B0": "context", "B4_minus_B3": "current_state"})
    frames.append(b)
    r = representations.loc[
        representations.record_type.eq("comparison") & representations.metric.eq("predictive_R2")
        & representations.candidate_model.isin(["R3_joint_raw", "R4_joint_residual"])
    ].copy()
    r["comparison_key"] = r.candidate_model + "_minus_BASE"
    frames.append(r)
    s = split.loc[
        split.record_type.eq("comparison") & split.statistic.eq("predictive_R2")
        & split.direction.eq("average")
    ].copy()
    s["comparison_key"] = "split_cross_form_minus_BASE"
    s["n_persons"] = s["n"]
    frames.append(s)
    source = pd.concat(frames, ignore_index=True, sort=False)
    keys = ["validation", "outcome_metric", "outcome_formulation", "baseline_scope", "comparison_key"]
    rows = []
    for _, row in source.iterrows():
        rows.append({
            "record_type": "cohort", "cohort": row.cohort,
            "validation": row.validation, "outcome_metric": row.outcome_metric,
            "outcome_formulation": row.outcome_formulation,
            "baseline_scope": row.baseline_scope, "comparison_key": row.comparison_key,
            "estimate": row.estimate, "ci_lower": row.ci_lower, "ci_upper": row.ci_upper,
            "bootstrap_se": row.bootstrap_se, "n_persons": row.get("n_persons", row.get("n")),
            "pool_method": "", "Q": np.nan, "I2": np.nan,
            "notes": "cohort-specific person-bootstrap estimate",
        })
    for group_key, g in source.groupby(keys, dropna=False, sort=False):
        valid = g.loc[safe_num(g.bootstrap_se).gt(0) & safe_num(g.estimate).notna()].copy()
        if len(valid) < 2:
            continue
        weights = 1 / safe_num(valid.bootstrap_se).to_numpy(dtype=float) ** 2
        estimates = safe_num(valid.estimate).to_numpy(dtype=float)
        pooled = float(np.sum(weights * estimates) / np.sum(weights))
        se = float(math.sqrt(1 / weights.sum()))
        q = float(np.sum(weights * (estimates - pooled) ** 2))
        df = len(valid) - 1
        i2 = float(max(0, (q - df) / q) * 100) if q > 0 else 0.0
        values = dict(zip(keys, group_key))
        rows.append({
            "record_type": "descriptive_pool", "cohort": "UKHLS+HRS",
            **values, "estimate": pooled, "ci_lower": pooled - 1.96 * se,
            "ci_upper": pooled + 1.96 * se, "bootstrap_se": se,
            "n_persons": int(safe_num(valid.n_persons).sum()),
            "pool_method": "inverse_variance_fixed_effect_descriptive",
            "Q": q, "I2": i2,
            "notes": "two cohorts only; not a definitive meta-analysis",
        })
    return pd.DataFrame(rows)


def adjudicate_pattern(summary: pd.DataFrame) -> tuple[str, list[str]]:
    pooled = summary.loc[
        summary.record_type.eq("descriptive_pool")
        & summary.validation.eq("grouped_10fold")
        & summary.outcome_formulation.eq("change")
    ].copy()

    def get(outcome, key, scope):
        q = pooled.loc[
            pooled.outcome_metric.eq(outcome) & pooled.comparison_key.eq(key)
            & pooled.baseline_scope.eq(scope)
        ]
        return q.iloc[0] if len(q) else None

    a_ok, b_ok = True, True
    reasons = []
    for outcome in ["acute", "persistence"]:
        b2 = get(outcome, "B2_minus_B0", "context")
        b4 = get(outcome, "B4_minus_B3", "current_state")
        cond_a = b2 is not None and b2.ci_lower > 0 and b2.estimate >= 0.0025 and b4 is not None and b4.ci_upper < 0.0025
        a_ok &= cond_a
        if not cond_a:
            reasons.append(f"Pattern A condition not met for {outcome}")
        checks_b = [b2, b4, get(outcome, "R3_joint_raw_minus_BASE", "current_state"),
                    get(outcome, "R4_joint_residual_minus_BASE", "current_state"),
                    get(outcome, "split_cross_form_minus_BASE", "current_state")]
        cond_b = all(x is not None and x.ci_upper < 0.0025 for x in checks_b)
        b_ok &= cond_b
        if not cond_b:
            reasons.append(f"Pattern B condition not met for {outcome}")
    if a_ok:
        return "Pattern A", ["Prior response meets the frozen useful-without-current-state and sub-benchmark-with-current-state criteria for both outcomes."]
    if b_ok:
        return "Pattern B", ["All frozen observed, joint, residualized, and split-half upper bounds are below 0.0025 for both outcomes."]

    c_any = False
    for outcome in ["acute", "persistence"]:
        keys = [
            ("B4_minus_B3", "current_state"),
            ("R4_joint_residual_minus_BASE", "current_state"),
            ("split_cross_form_minus_BASE", "current_state"),
        ]
        primary = [get(outcome, key, scope) for key, scope in keys]
        primary_ok = all(x is not None and x.ci_lower > 0 and x.estimate >= 0.0025 for x in primary)
        direction_ok = True
        for key, scope in keys:
            q = summary.loc[
                summary.record_type.eq("cohort") & summary.outcome_metric.eq(outcome)
                & summary.outcome_formulation.eq("change") & summary.baseline_scope.eq(scope)
                & summary.comparison_key.eq(key)
            ]
            direction_ok &= len(q) >= 4 and bool((q.estimate >= 0).all()) and set(q.validation) == {"grouped_10fold", "temporal_70_30"}
        c_any |= primary_ok and direction_ok
    if c_any:
        return "Pattern C", ["The frozen independent-current-state, residualized, split-half, cross-cohort, and temporal direction criteria are jointly met for at least one outcome."]
    return "Mixed or inconclusive", list(dict.fromkeys(reasons))


def data_dictionary_table() -> pd.DataFrame:
    descriptions = {
        "cohort": "Cohort identifier (UKHLS or HRS).",
        "person_id": "Existing harmonized person identifier (pidp/HHIDPN).",
        "episode_id": "Existing confirmatory episode identifier.",
        "wave": "Survey wave supplying the item responses.",
        "episode_position": "Earlier or later event in the frozen two-event pair.",
        "time_role": "pre1, event, or year2 occasion relative to the episode.",
        "instrument": "GHQ-12 or CES-D 8.",
        "source_total": "Existing harmonized total score at the occasion.",
        "source_variables": "JSON list of exact raw item variable names.",
        "item_01-item_12": "Distress-oriented item scores; unused HRS item slots 09-12 are missing.",
        "n_items_expected": "Number of items expected for the cohort instrument.",
        "n_items_observed": "Number of valid observed items.",
        "full_item_score": "Complete-case sum of all distress-oriented items.",
        "half_a_score": "Complete-case sum of the frozen Half A items.",
        "half_b_score": "Complete-case sum of the frozen Half B items.",
        "item_complete": "True when every expected item is valid.",
        "sum_matches_source_total": "True when reconstructed and existing totals exactly agree.",
    }
    return pd.DataFrame([{"variable": k, "description": v} for k, v in descriptions.items()])


def render_reports(history: pd.DataFrame, item_data: pd.DataFrame, item_audit: pd.DataFrame,
                   baseline: pd.DataFrame, representations: pd.DataFrame,
                   split: pd.DataFrame, benchmarks: pd.DataFrame,
                   summary: pd.DataFrame, pattern: str, pattern_notes: list[str], freeze_hash: str) -> None:
    sample = history.groupby("cohort").agg(persons=("person_id", "nunique"), rows=("person_id", "size")).reset_index()
    baseline_key = baseline.loc[baseline.record_type.eq("comparison") & baseline.metric.eq("predictive_R2")].copy()
    write_md(OUT / "01_baseline_information_decomposition.md", [
        "# Baseline information decomposition", "",
        "## Frozen sample", "", md_table(sample, ["cohort", "persons", "rows"]), "",
        "The primary unit is one person in the fixed two-event sample. Acute response and two-year persistence were evaluated separately under change-score and ANCOVA level formulations. The ANCOVA formulation necessarily includes the later pre-event score in every model; B3 therefore duplicates B0 and B4 duplicates B2 by design.", "",
        "## Prespecified predictive-R-squared increments", "",
        md_table(baseline_key, ["cohort", "validation", "outcome_metric", "outcome_formulation", "comparison", "estimate", "ci_lower", "ci_upper", "bootstrap_se", "n_persons", "bootstrap_success"]), "",
        "## Model sequence", "",
        "B0 contains later event type, age, sex, education, calendar year, and preceding-event count; B1 adds earlier pre-event mental health; B2 adds the corresponding earlier adversity response; B3 adds later current pre-event mental health; B4 adds earlier response to B3. All transformations and fits were training-only. No hypothesis tests or p values were produced.", "",
        f"Freeze SHA-256: `{freeze_hash}`.",
    ])

    rep_key = representations.loc[representations.record_type.eq("comparison") & representations.metric.eq("predictive_R2")].copy()
    ridge = representations.loc[
        representations.record_type.eq("model_metric") & representations.model.isin(["R3_joint_raw", "R4_joint_residual"])
        & representations.metric.eq("RMSE")
    ].copy()
    write_md(OUT / "02_prior_response_representation.md", [
        "# Prior-response representation", "",
        "R1 uses prior acute response, R2 prior two-year persistence, R3 both raw responses with training-selected ridge shrinkage, and R4 uses training-fold residualization on earlier event type, earlier baseline mental health, earlier age, earlier calendar year, and preceding-event count. R4 joint uses ridge on the two residualized responses.", "",
        "## Incremental predictive R-squared", "", md_table(rep_key, ["cohort", "validation", "outcome_metric", "outcome_formulation", "baseline_scope", "comparison", "estimate", "ci_lower", "ci_upper", "bootstrap_se", "n_persons"]), "",
        "## Ridge selections across outer folds", "", md_table(ridge, ["cohort", "validation", "outcome_metric", "outcome_formulation", "baseline_scope", "model", "ridge_alpha_min", "ridge_alpha_median", "ridge_alpha_max"]), "",
        "Residualization and ridge selection used training data only; later outcomes were not used in residualization.", "",
        f"Freeze SHA-256: `{freeze_hash}`.",
    ])

    item_meta = item_audit.loc[item_audit.record_type.eq("item_metadata")].copy()
    item_map = item_meta.groupby(["cohort", "instrument", "item_position", "item_concept", "half"], as_index=False).agg(
        wave_min=("wave", "min"), wave_max=("wave", "max"),
        exact_source_variables=("source_variable", lambda x: ", ".join(x.astype(str))),
        variable_label=("variable_label", "first"), raw_valid_codes=("raw_valid_codes", "first"),
        scoring=("scoring", "first"), reverse_coded=("reverse_coded", "first"),
    )
    completeness = item_audit.loc[item_audit.record_type.isin(["completeness", "scoring_check"])].copy()
    reliability = item_audit.loc[item_audit.record_type.eq("reliability") & item_audit.scale.eq("full")].copy()
    dictionary = data_dictionary_table()
    duplicate_keys = int(item_data[["cohort", "person_id", "episode_id", "time_role"]].duplicated().sum())
    cohort_rows = item_data.groupby("cohort").size().rename("rows").reset_index()
    write_md(OUT / "03_item_level_measurement_audit.md", [
        "# Item-level measurement audit", "",
        "## Feasibility", "",
        "Item-level reconstruction was feasible in both cohorts for every occasion required by the fixed two-event samples. UKHLS used waves 1-15 (prefixes a-o); HRS used waves 2-15. Only the raw item columns and person identifier were read.", "",
        "## Exact item mapping and frozen halves", "", md_table(item_map, ["cohort", "instrument", "item_position", "item_concept", "half", "wave_min", "wave_max", "exact_source_variables", "variable_label", "raw_valid_codes", "scoring", "reverse_coded"]), "",
        "UKHLS raw response codes 1-4 are already ordered from least to most distress for both positive- and negative-worded questions; subtracting one yields the published Likert item orientation and exactly reconstructs the 0-36 score. The positive-worded semantic reversal is therefore embodied in the source response ordering. HRS happy and enjoyed-life indicators were explicitly transformed as 1 minus raw; the other six indicators were retained. No wording or scoring change affecting the mapping was detected across eligible waves; UKHLS label capitalization/spacing varies only cosmetically.", "",
        "## Eligible item completeness and total-score reconstruction", "", md_table(completeness, ["record_type", "cohort", "episode_position", "time_role", "eligible_n", "item_complete_n", "item_complete_pct", "statistic", "estimate"]), "",
        "## Full-scale reliability", "", md_table(reliability, ["cohort", "episode_position", "time_role", "statistic", "estimate", "n", "converged", "notes"]), "",
        "## Item-level data QA", "", md_table(cohort_rows, ["cohort", "rows"]), "",
        f"Declared unique key: `cohort + person_id + episode_id + time_role`; duplicate keys: {duplicate_keys:,}. Total rows: {len(item_data):,}. Source two-event rows: UKHLS 1,956 and HRS 1,166 persons, each with two episodes and three item occasions.", "",
        "## Item-level data dictionary", "", md_table(dictionary, ["variable", "description"]), "",
        "Missing items were not imputed or prorated. UKHLS was item-complete at every frozen occasion. In HRS, the existing RAND total was present for the fixed sample even in a small number of records with one or more unavailable raw item indicators; those records remain in the row-count audit but are excluded from item-level reliability and cross-form analyses. Reliability estimates use complete cases. Omega failures, if any, remain missing with their optimizer message in the CSV.", "",
        f"Freeze SHA-256: `{freeze_hash}`.",
    ])

    rel = split.loc[split.record_type.isin(["split_reliability", "change_reliability", "observed_correlation", "disattenuated_correlation"])].copy()
    sp = split.loc[split.record_type.eq("comparison") & split.statistic.eq("predictive_R2")].copy()
    write_md(OUT / "04_split_half_reliability_and_prediction.md", [
        "# Split-half reliability and cross-form prediction", "",
        "The item halves were fixed before item metadata were inspected. Cross-form prediction was run in both A-to-B and B-to-A directions; reported comparison rows average the two directions equally. Disattenuated correlations are sensitivity quantities only and were not substituted for observed predictive results.", "",
        "## Reliability and observed/disattenuated correlations", "", md_table(rel, ["record_type", "cohort", "episode_position", "time_role", "outcome_metric", "scale", "direction", "statistic", "estimate", "n", "notes"]), "",
        "## Average cross-form predictive-R-squared increments", "", md_table(sp, ["cohort", "validation", "outcome_metric", "outcome_formulation", "baseline_scope", "direction", "estimate", "ci_lower", "ci_upper", "bootstrap_se", "n", "bootstrap_success"]), "",
        "All half-score scaling and model fitting were training-only. Item-complete cases were required; no imputation was used.", "",
        f"Freeze SHA-256: `{freeze_hash}`.",
    ])

    b025 = benchmarks.loc[benchmarks.benchmark.eq(0.0025)].copy()
    write_md(OUT / "05_practical_equivalence_bounds.md", [
        "# Practical predictive-effect bounds", "",
        "Every prior-response predictive-R-squared gain was compared with frozen thresholds 0.0025, 0.005, and 0.010. These are transparent predictive-statistical reference points, not clinical thresholds. The table below shows the smallest threshold; the CSV contains all three.", "",
        md_table(b025, ["source_table", "cohort", "validation", "outcome_metric", "outcome_formulation", "baseline_scope", "comparison", "estimate", "ci_lower", "ci_upper", "benchmark", "upper_below", "interval_crosses", "lower_above", "eu_predictive_R2_gain", "gain_to_abs_eu_ratio", "same_order_of_magnitude_as_eu"]), "",
        "The E-U benchmark is copied without re-estimation from the existing confirmatory trajectory-prediction result for the matching cohort and validation design. Same order of magnitude means an absolute gain ratio from 0.1 through 10.", "",
        f"Freeze SHA-256: `{freeze_hash}`.",
    ])

    pooled = summary.loc[summary.record_type.eq("descriptive_pool")].copy()
    write_md(OUT / "06_cross_cohort_measurement_summary.md", [
        "# Cross-cohort measurement summary", "",
        "Cohort-specific estimates precede descriptive inverse-variance summaries in the CSV. The pooled rows below use cohort person-bootstrap standard errors. With two cohorts, these are descriptive summaries and not a definitive meta-analysis.", "",
        md_table(pooled, ["validation", "outcome_metric", "outcome_formulation", "baseline_scope", "comparison_key", "estimate", "ci_lower", "ci_upper", "bootstrap_se", "n_persons", "Q", "I2"]), "",
        f"Freeze SHA-256: `{freeze_hash}`.",
    ])

    source_paths = [PRED_SAMPLE, UK_EPISODES, HRS_EPISODES, HRS_RAW, EU_SOURCE]
    source_paths += [UK_RAW / f"{prefix}_indresp.dta" for prefix in "abcdefghijklmno"]
    provenance = pd.DataFrame([
        {"source": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in source_paths
    ])
    requested_outputs = [ITEM_OUT, *sorted(OUT.glob("*.csv"))]
    output_provenance = pd.DataFrame([
        {"output": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in requested_outputs
    ])
    software = pd.DataFrame([
        {"component": "Python", "version": platform.python_version()},
        {"component": "pandas", "version": pd.__version__},
        {"component": "NumPy", "version": np.__version__},
        {"component": "SciPy", "version": __import__("scipy").__version__},
        {"component": "platform", "version": platform.platform()},
    ])
    write_md(OUT / "07_measurement_adjudication_final_summary.md", [
        "# Measurement adjudication final summary", "",
        "## Frozen evidence category", "", f"**{pattern}**", "",
        *[f"- {x}" for x in pattern_notes], "",
        "This category follows the result-blind rules in the freeze and concerns incremental out-of-sample predictive information only. It does not establish the existence of resilience, a psychological trait, causality, or a general theoretical model.", "",
        "## Scope and exclusions", "",
        "- Primary samples remained UKHLS 1,956 and HRS 1,166; no person or event was added.",
        "- Existing three-event samples (UKHLS 35; HRS 47) were not analysed as primary evidence and remain labelled insufficient sample.",
        "- Acute response and two-year persistence, change-score and ANCOVA level formulations, grouped ten-fold and temporal holdout validations, raw/joint/residualized prior representations, and cross-form item sensitivities were all retained.",
        "- Missing data were handled by analysis-specific complete cases. There was no imputation, outlier deletion, event recoding, or post-result restriction.",
        "- No hypothesis tests, p values, causal interpretations, new theory, subgroup analysis, moderator, or mediator were introduced.", "",
        "## Provenance and QA", "",
        f"Freeze SHA-256: `{freeze_hash}`. Item-level Parquet rows: {len(item_data):,}; duplicate declared keys: {duplicate_keys:,}. All six requested CSV files were generated from the frozen samples and are separately structurally validated.", "",
        "### Input sources", "", md_table(provenance, ["source", "bytes", "sha256"]), "",
        "### Requested data outputs", "", md_table(output_provenance, ["output", "bytes", "sha256"]), "",
        "### Software", "", md_table(software, ["component", "version"]),
    ])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    freeze_hash = sha256(FREEZE)
    expected_hash = "7f8110766bbb3e5027410cf3f24bbebe78cadfda160e4f8f3f8ee258f921ca2d"
    if freeze_hash != expected_hash:
        raise AssertionError("Measurement-adjudication freeze changed after creation")

    note("Loading frozen two-event samples and existing episode files")
    history, episodes = load_primary_history()
    note("Reading only required UKHLS GHQ-12 and HRS CES-D item columns")
    uk_items, uk_meta = extract_ukhls_items(history)
    hrs_items, hrs_meta = extract_hrs_items(history)
    item_data = pd.concat([uk_items, hrs_items], ignore_index=True, sort=False)
    item_data = item_data[[
        "cohort", "person_id", "episode_id", "wave", "episode_position", "time_role",
        "instrument", "source_total", "source_variables", *[f"item_{i:02d}" for i in range(1, 13)],
        "n_items_expected", "n_items_observed", "full_item_score", "half_a_score",
        "half_b_score", "item_complete", "sum_matches_source_total",
    ]].sort_values(["cohort", "person_id", "episode_position", "time_role"]).reset_index(drop=True)
    if item_data[["cohort", "person_id", "episode_id", "time_role"]].duplicated().any():
        raise AssertionError("Item-level output key is not unique")
    expected_rows = 1956 * 2 * 3 + 1166 * 2 * 3
    if len(item_data) != expected_rows:
        raise AssertionError(f"Item-level row count mismatch: {len(item_data)} != {expected_rows}")
    comparable = item_data.item_complete & item_data.source_total.notna()
    if not item_data.loc[comparable, "sum_matches_source_total"].all():
        raise AssertionError("Complete-item reconstruction is inconsistent with source totals")
    if item_data.groupby("cohort").item_complete.sum().eq(0).any():
        raise AssertionError("A cohort has no item-complete records")
    item_data.to_parquet(ITEM_OUT, index=False)

    reliability, split_level = reliability_rows(item_data)
    change_rel, change_wide = build_change_reliability(item_data, reliability)
    item_audit = assemble_item_audit(item_data, pd.DataFrame(uk_meta + hrs_meta), reliability)
    item_history = make_item_person_history(history, change_wide)

    note("Running frozen baseline information decomposition")
    baseline, _ = run_baseline_predictions(history, episodes)
    note("Running frozen raw, joint, and residualized prior-response representations")
    representations, _ = run_representation_predictions(history, episodes)
    note("Running frozen cross-form split-half prediction")
    split_prediction = run_split_predictions(item_history)
    split_corr = observed_split_correlations(item_history, change_rel)
    split = pd.concat([split_level, change_rel, split_corr, split_prediction], ignore_index=True, sort=False)

    benchmarks = practical_benchmarks(baseline, representations, split)
    summary = cross_cohort_summary(baseline, representations, split)
    pattern, pattern_notes = adjudicate_pattern(summary)

    note("Writing frozen result tables and audit records")
    baseline.to_csv(OUT / "01_baseline_information_decomposition.csv", index=False)
    representations.to_csv(OUT / "02_prior_response_representation.csv", index=False)
    item_audit.to_csv(OUT / "03_item_level_measurement_audit.csv", index=False)
    split.to_csv(OUT / "04_split_half_reliability_and_prediction.csv", index=False)
    benchmarks.to_csv(OUT / "05_practical_equivalence_bounds.csv", index=False)
    summary.to_csv(OUT / "06_cross_cohort_measurement_summary.csv", index=False)
    render_reports(history, item_data, item_audit, baseline, representations, split, benchmarks, summary, pattern, pattern_notes, freeze_hash)

    manifest = {
        "freeze_sha256": freeze_hash,
        "python": sys.version,
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scipy": __import__("scipy").__version__,
        "inputs": {str(p.relative_to(ROOT)): sha256(p) for p in [PRED_SAMPLE, UK_EPISODES, HRS_EPISODES, HRS_RAW, EU_SOURCE]},
        "outputs": {str(p.relative_to(ROOT)): sha256(p) for p in [ITEM_OUT, *sorted(OUT.glob("*.csv")), *sorted(OUT.glob("*.md"))]},
        "row_counts": {
            "history": len(history), "item_level": len(item_data), "baseline_csv": len(baseline),
            "representation_csv": len(representations), "item_audit_csv": len(item_audit),
            "split_csv": len(split), "benchmark_csv": len(benchmarks), "cross_cohort_csv": len(summary),
        },
        "unique_key_qa": {
            "history_cohort_person_duplicates": int(history[["cohort", "person_id"]].duplicated().sum()),
            "item_level_declared_key_duplicates": int(item_data[["cohort", "person_id", "episode_id", "time_role"]].duplicated().sum()),
        },
    }
    # Included inside the final markdown instead of creating an additional requested output.
    note(json.dumps(manifest["row_counts"], sort_keys=True))


if __name__ == "__main__":
    main()
