#!/usr/bin/env python3
"""Frozen NMH confirmatory rebuild for UKHLS and HRS.

This script only implements the analysis plan recorded in
results/NMH_confirmatory/00_NMH_confirmatory_freeze.md.  It reuses the
previously constructed UKHLS person-wave/event data and reads selected,
prespecified columns from the harmonized HRS files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import chi2


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "NMH_confirmatory"
DP = ROOT / "data_processed"
UK_EVENTS = ROOT / "results" / "person_event_response_long.parquet"
UK_PANEL = DP / "ukhls_adult_core_panel.csv.gz"
HRS_CORE = ROOT / "raw_data" / "HRS" / "randhrs1992_2022v1_STATA" / "randhrs1992_2022v1.dta"
HRS_FAMILY = ROOT / "raw_data" / "HRS" / "randhrsfam1992_2022v1_STATA" / "randhrsfamr1992_2022v1.dta"

UK_OUT = DP / "NMH_UKHLS_event_episodes.parquet"
HRS_OUT = DP / "NMH_HRS_event_episodes.parquet"
PRED_OUT = DP / "NMH_cross_event_prediction_samples.parquet"

SEED = 20260903
BOOT_SUCCESS = 500
PRIMARY = ["widowhood", "unemployment", "caregiving", "health"]
TIME_LEVELS = ["pre2", "pre1", "event", "year2", "year4"]
MONTH_DAYS = 30.4375

UK_PRIMARY_MAP = {
    "A_unemployment_onset": "unemployment",
    "B_caregiving_onset": "caregiving",
    "C_persistent_illness_disability_onset": "health",
    "E_widowhood": "widowhood",
}
UK_SECONDARY_MAP = {
    "D_separation_divorce": "separation_divorce",
    "F_severe_financial_strain_onset": "financial_strain",
}


def note(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def valid_range(s: pd.Series, low: float, high: float) -> pd.Series:
    x = safe_num(s)
    return x.where(x.between(low, high))


def date_from_stata(s: pd.Series) -> pd.Series:
    # Extended Stata missing values arrive as very large finite doubles when
    # categorical conversion is disabled; remove them before date conversion.
    x = safe_num(s).where(lambda z: z.between(0, 50000)).astype("float64")
    values = x.to_numpy(dtype="float64")
    ok = np.isfinite(values)
    converted = np.full(len(values), np.datetime64("NaT", "D"), dtype="datetime64[D]")
    converted[ok] = np.datetime64("1960-01-01") + values[ok].astype("timedelta64[D]")
    return pd.Series(converted.astype("datetime64[ns]"), index=s.index)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def nearest_by_months(
    dates: pd.Series,
    anchor: pd.Timestamp,
    target: float,
    low: float,
    high: float,
    direction: str,
) -> int | None:
    if pd.isna(anchor) or dates.empty:
        return None
    d = pd.to_datetime(dates)
    months = (d - anchor).dt.days / MONTH_DAYS
    if direction == "before":
        elapsed = -months
    else:
        elapsed = months
    ok = elapsed.between(low, high)
    if not ok.any():
        return None
    candidates = elapsed[ok]
    return int((candidates - target).abs().idxmin())


def uk_education(code: object) -> str:
    if pd.isna(code):
        return "unknown"
    c = int(code)
    if c in {1, 2, 20, 21}:
        return "degree"
    if c in {9, 96}:
        return "no_qualification"
    if c > 0:
        return "other_qualification"
    return "unknown"


def hrs_education(code: object) -> str:
    if pd.isna(code):
        return "unknown"
    c = int(code)
    if c == 5:
        return "degree"
    if c == 1:
        return "no_qualification"
    if c in {2, 3, 4}:
        return "other_qualification"
    return "unknown"


def uk_socio(code: object) -> str:
    if pd.isna(code):
        return "unknown"
    c = int(code)
    mapping = {
        1: "employed",
        2: "self_employed",
        3: "unemployed",
        4: "retired",
        5: "family_care",
        6: "student",
        7: "long_term_sick",
        8: "other",
        9: "other",
        10: "other",
        11: "other",
    }
    return mapping.get(c, "unknown")


def hrs_socio(code: object) -> str:
    if pd.isna(code):
        return "unknown"
    c = int(code)
    mapping = {
        1: "employed",
        2: "employed",
        3: "unemployed",
        4: "partly_retired",
        5: "retired",
        6: "disabled",
        7: "not_labor_force",
    }
    return mapping.get(c, "unknown")


def collapse_event_rows(rows: pd.DataFrame, cohort: str) -> pd.DataFrame:
    """Collapse simultaneous status transitions to one person-episode."""
    key = ["person_id", "event_wave"]
    records: list[dict] = []
    for (person, wave), g in rows.groupby(key, sort=False, observed=True):
        primary_labels = sorted(set(g.loc[g.event_role.eq("primary"), "event_type"]))
        secondary_labels = sorted(set(g.loc[g.event_role.ne("primary"), "event_type"]))
        all_labels = sorted(set(g.event_type))
        record = {
            "cohort": cohort,
            "person_id": int(person),
            "event_wave": int(wave),
            "event_date": pd.to_datetime(g.event_date.iloc[0], errors="coerce"),
            "primary_event_labels": "|".join(primary_labels),
            "secondary_event_labels": "|".join(secondary_labels),
            "all_event_labels": "|".join(all_labels),
            "n_primary_labels": len(primary_labels),
            "n_secondary_labels": len(secondary_labels),
            "n_all_labels": len(all_labels),
            "compound_primary": len(primary_labels) >= 2,
            "isolated_primary": len(primary_labels) == 1,
            "primary_event_type": (
                primary_labels[0]
                if len(primary_labels) == 1
                else ("compound_primary" if len(primary_labels) >= 2 else "")
            ),
        }
        records.append(record)
    out = pd.DataFrame(records)
    out["compound_transition"] = out["compound_primary"]
    out = out.sort_values(["person_id", "event_date", "event_wave"], na_position="last").reset_index(drop=True)
    out["episode_id"] = [f"{cohort}_{i + 1:08d}" for i in range(len(out))]
    return out


def select_uk_timepoints(episodes: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.sort_values(["pidp", "interview_date", "wave"]).copy()
    grouped = {int(k): v for k, v in panel.groupby("pidp", sort=False)}
    selected: list[dict] = []
    for row in episodes.itertuples(index=False):
        p = grouped.get(int(row.person_id))
        if p is None:
            selected.append({})
            continue
        event_matches = p.loc[p.wave.eq(row.event_wave)]
        pre_matches = p.loc[p.wave.eq(row.event_wave - 1)]
        ev = event_matches.iloc[-1] if not event_matches.empty else None
        pre = pre_matches.iloc[-1] if not pre_matches.empty else None
        event_date = pd.Timestamp(row.event_date) if pd.notna(row.event_date) else (ev.interview_date if ev is not None else pd.NaT)
        pre_date = pre.interview_date if pre is not None else pd.NaT

        pre2_idx = nearest_by_months(p.interview_date, pre_date, 12, 8, 18, "before")
        y2_idx = nearest_by_months(p.interview_date, event_date, 24, 18, 30, "after")
        y4_idx = nearest_by_months(p.interview_date, event_date, 48, 42, 54, "after")
        pre2 = p.loc[pre2_idx] if pre2_idx is not None else None
        y2 = p.loc[y2_idx] if y2_idx is not None else None
        y4 = p.loc[y4_idx] if y4_idx is not None else None

        def ghq(x: pd.Series | None) -> float:
            if x is None:
                return np.nan
            v = float(x.scghq1_dv)
            return v if 0 <= v <= 36 else np.nan

        rec = {
            "pre2_wave": pre2.wave if pre2 is not None else np.nan,
            "pre1_wave": pre.wave if pre is not None else np.nan,
            "year2_wave": y2.wave if y2 is not None else np.nan,
            "year4_wave": y4.wave if y4 is not None else np.nan,
            "pre2_date": pre2.interview_date if pre2 is not None else pd.NaT,
            "pre1_date": pre_date,
            "year2_date": y2.interview_date if y2 is not None else pd.NaT,
            "year4_date": y4.interview_date if y4 is not None else pd.NaT,
            "outcome_pre2_raw": ghq(pre2),
            "outcome_pre1_raw": ghq(pre),
            "outcome_event_raw": ghq(ev),
            "outcome_year2_raw": ghq(y2),
            "outcome_year4_raw": ghq(y4),
            # Frozen covariate timing is pre1.  The historical column name is
            # retained for compatibility, but its value is age at pre1.
            "age_at_event": float(pre.age_dv) if pre is not None and 16 <= float(pre.age_dv) <= 110 else np.nan,
            "sex_code": float(ev.sex_dv) if ev is not None and float(ev.sex_dv) in {1, 2} else np.nan,
            "baseline_socio_code": float(pre.jbstat) if pre is not None and float(pre.jbstat) > 0 else np.nan,
            "baseline_marital_code": float(pre.marstat_dv) if pre is not None and float(pre.marstat_dv) > 0 else np.nan,
            "baseline_health_code": float(pre.health) if pre is not None and float(pre.health) > 0 else np.nan,
            "time_selection_method": "actual_interview_date",
            "event_self_report": bool(ev is not None and int(ev.ivfio) in {1, 3}),
            "pre1_self_report": bool(pre is not None and int(pre.ivfio) in {1, 3}),
        }
        # Most recent valid education at or before the pre-event interview.
        hist = p.loc[(p.wave <= row.event_wave - 1) & (safe_num(p.qfhigh) > 0), ["wave", "qfhigh"]]
        if hist.empty:
            rec["education_code"] = np.nan
            rec["education_source_wave"] = np.nan
        else:
            last = hist.sort_values("wave").iloc[-1]
            rec["education_code"] = float(last.qfhigh)
            rec["education_source_wave"] = float(last.wave)
        selected.append(rec)
    return pd.concat([episodes.reset_index(drop=True), pd.DataFrame(selected)], axis=1)


def apply_history_and_censor(episodes: pd.DataFrame) -> pd.DataFrame:
    out = episodes.copy()
    out["outcome_year2_raw_uncensored"] = out["outcome_year2_raw"]
    out["outcome_year4_raw_uncensored"] = out["outcome_year4_raw"]
    out["outcome_year2_raw_uncensored"] = out["outcome_year2_raw"]
    out["outcome_year4_raw_uncensored"] = out["outcome_year4_raw"]
    out["preceding_primary_episode_count"] = 0
    out["subsequent_primary_event_date"] = pd.NaT
    out["year2_censored_by_later_primary"] = False
    out["year4_censored_by_later_primary"] = False
    for person, idx in out.groupby("person_id", sort=False).groups.items():
        loc = out.loc[idx].sort_values(["event_date", "event_wave"])
        primary = loc.loc[loc.n_primary_labels.gt(0)]
        primary_dates = list(primary.event_date)
        primary_waves = list(primary.event_wave)
        for ridx, r in loc.iterrows():
            earlier = sum(
                (pd.notna(d) and pd.notna(r.event_date) and d < r.event_date)
                or (pd.isna(d) and w < r.event_wave)
                for d, w in zip(primary_dates, primary_waves)
            )
            out.at[ridx, "preceding_primary_episode_count"] = int(earlier)
            later = [
                (d, w)
                for d, w in zip(primary_dates, primary_waves)
                if (pd.notna(d) and pd.notna(r.event_date) and d > r.event_date)
                or (pd.isna(d) and w > r.event_wave)
            ]
            if later:
                later.sort(key=lambda x: (pd.Timestamp.max if pd.isna(x[0]) else x[0], x[1]))
                next_date, next_wave = later[0]
                out.at[ridx, "subsequent_primary_event_date"] = next_date
                y2_after = (
                    pd.notna(next_date)
                    and pd.notna(r.year2_date)
                    and r.year2_date >= next_date
                ) or (pd.isna(next_date) and pd.notna(r.year2_wave) and r.year2_wave >= next_wave)
                y4_after = (
                    pd.notna(next_date)
                    and pd.notna(r.year4_date)
                    and r.year4_date >= next_date
                ) or (pd.isna(next_date) and pd.notna(r.year4_wave) and r.year4_wave >= next_wave)
                if y2_after:
                    out.at[ridx, "year2_censored_by_later_primary"] = True
                    out.at[ridx, "outcome_year2_raw"] = np.nan
                if y4_after:
                    out.at[ridx, "year4_censored_by_later_primary"] = True
                    out.at[ridx, "outcome_year4_raw"] = np.nan
    return out


def standardize_and_derive(episodes: pd.DataFrame) -> pd.DataFrame:
    out = episodes.copy()
    ref = out.loc[out.n_primary_labels.gt(0), "outcome_pre1_raw"].dropna()
    mu = float(ref.mean())
    sd = float(ref.std(ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        raise AssertionError("Invalid pre-event outcome SD")
    out["standardization_mean"] = mu
    out["standardization_sd"] = sd
    for tp in TIME_LEVELS:
        out[f"outcome_{tp}_z"] = (out[f"outcome_{tp}_raw"] - mu) / sd
    out["outcome_year2_z_uncensored"] = (out.outcome_year2_raw_uncensored - mu) / sd
    out["outcome_year4_z_uncensored"] = (out.outcome_year4_raw_uncensored - mu) / sd
    out["acute_response_z"] = out.outcome_event_z - out.outcome_pre1_z
    out["persistence_2y_z"] = out.outcome_year2_z - out.outcome_pre1_z
    out["persistence_4y_z"] = out.outcome_year4_z - out.outcome_pre1_z
    out["early_recovery_z"] = out.outcome_year2_z - out.outcome_event_z
    out["long_recovery_z"] = out.outcome_year4_z - out.outcome_event_z
    out["prior_response_history_z"] = np.nan
    for _, idx in out.groupby("person_id", sort=False).groups.items():
        loc = out.loc[idx].sort_values(["event_date", "event_wave", "episode_id"])
        history: list[float] = []
        for event_date, same_date in loc.groupby(["event_date", "event_wave"], sort=False, dropna=False):
            prior_mean = float(np.mean(history)) if history else np.nan
            out.loc[same_date.index, "prior_response_history_z"] = prior_mean
            eligible_history = same_date.loc[
                same_date.n_primary_labels.gt(0) & same_date.acute_response_z.notna(),
                "acute_response_z",
            ].tolist()
            history.extend(float(v) for v in eligible_history)
    out["primary_trajectory_eligible"] = (
        out.isolated_primary
        & out.primary_event_type.isin(PRIMARY)
        & out.outcome_pre1_raw.notna()
        & out.outcome_event_raw.notna()
        & out.event_self_report
        & out.pre1_self_report
    )
    return out


def build_ukhls() -> pd.DataFrame:
    note("Building UKHLS episodes from frozen event and adult panel files")
    ev = pd.read_parquet(UK_EVENTS)
    ev["person_id"] = safe_num(ev.pidp).astype("int64")
    ev["event_wave"] = safe_num(ev.event_wave).astype("int16")
    ev["event_date"] = pd.to_datetime(ev.event_date_proxy, errors="coerce")
    ev["event_type"] = ev.event_type.map({**UK_PRIMARY_MAP, **UK_SECONDARY_MAP})
    ev["event_role"] = np.where(ev.event_type.isin(PRIMARY), "primary", "secondary")
    ev = ev.dropna(subset=["event_type", "event_date"]).copy()
    event_rows = ev[["person_id", "event_wave", "event_date", "event_type", "event_role"]].drop_duplicates()
    episodes = collapse_event_rows(event_rows, "UKHLS")

    usecols = [
        "pidp", "wave", "interview_date", "scghq1_dv", "ivfio", "age_dv",
        "sex_dv", "qfhigh", "jbstat", "marstat_dv", "health",
    ]
    panel = pd.read_csv(UK_PANEL, compression="gzip", usecols=usecols, low_memory=False)
    panel["pidp"] = safe_num(panel.pidp).astype("int64")
    panel["wave"] = safe_num(panel.wave).astype("int16")
    panel["interview_date"] = pd.to_datetime(panel.interview_date, errors="coerce")
    panel = panel.loc[safe_num(panel.ivfio).isin([1, 3])].copy()
    panel = panel.sort_values(["pidp", "wave", "interview_date"]).drop_duplicates(["pidp", "wave"], keep="last")
    episodes = select_uk_timepoints(episodes, panel)
    episodes["event_year"] = pd.to_datetime(episodes.event_date).dt.year.astype("Int16")
    episodes["sex"] = episodes.sex_code.map({1.0: "male", 2.0: "female"}).fillna("unknown")
    episodes["education"] = episodes.education_code.map(uk_education)
    episodes["baseline_socioeconomic"] = episodes.baseline_socio_code.map(uk_socio)
    episodes["official_event_weight"] = np.nan
    episodes["official_year2_weight"] = np.nan
    episodes["official_year4_weight"] = np.nan
    episodes = apply_history_and_censor(episodes)
    return standardize_and_derive(episodes)


def hrs_columns() -> tuple[list[str], list[str]]:
    common = ["hhidpn", "ragender", "raeduc", "raedyrs"]
    core = list(common)
    chronic = ["diabe", "cancre", "lunge", "hearte", "stroke", "arthre"]
    for w in range(1, 17):
        for stem in ["iwmid", "iwstat", "proxy", "agey_e", "wtresp", "mstat", "lbrf", "shlt", "sayret"]:
            core.append(f"r{w}{stem}")
        core.append(f"h{w}inpov")
        if w >= 2:
            core.extend([f"r{w}cesd", f"r{w}adl5a"])
        core.extend([f"r{w}{stem}" for stem in chronic])
    fam = ["hhidpn"] + [f"r{w}ppcr" for w in range(1, 17)]
    return core, fam


def available_stata_columns(path: Path) -> set[str]:
    with pd.io.stata.StataReader(path, convert_categoricals=False) as reader:
        return set(reader.variable_labels().keys())


def build_hrs_long() -> tuple[pd.DataFrame, pd.DataFrame]:
    core_requested, fam_requested = hrs_columns()
    core_avail = available_stata_columns(HRS_CORE)
    fam_avail = available_stata_columns(HRS_FAMILY)
    core_cols = [c for c in core_requested if c in core_avail]
    fam_cols = [c for c in fam_requested if c in fam_avail]
    note(f"Reading {len(core_cols)} selected RAND HRS columns and {len(fam_cols)} family columns")
    core = pd.read_stata(HRS_CORE, columns=core_cols, convert_categoricals=False)
    fam = pd.read_stata(HRS_FAMILY, columns=fam_cols, convert_categoricals=False)
    core["hhidpn"] = safe_num(core.hhidpn).astype("int64")
    fam["hhidpn"] = safe_num(fam.hhidpn).astype("int64")
    wide = core.merge(fam, on="hhidpn", how="left", validate="one_to_one")
    chunks: list[pd.DataFrame] = []
    chronic = ["diabe", "cancre", "lunge", "hearte", "stroke", "arthre"]
    for w in range(1, 17):
        d = pd.DataFrame({
            "person_id": wide.hhidpn,
            "wave": w,
            "interview_date": date_from_stata(wide.get(f"r{w}iwmid", pd.Series(np.nan, index=wide.index))),
            "iwstat": safe_num(wide.get(f"r{w}iwstat", pd.Series(np.nan, index=wide.index))),
            "proxy": safe_num(wide.get(f"r{w}proxy", pd.Series(np.nan, index=wide.index))),
            "outcome_raw": valid_range(wide.get(f"r{w}cesd", pd.Series(np.nan, index=wide.index)), 0, 8),
            "age": valid_range(wide.get(f"r{w}agey_e", pd.Series(np.nan, index=wide.index)), 16, 110),
            "weight": safe_num(wide.get(f"r{w}wtresp", pd.Series(np.nan, index=wide.index))).where(lambda x: x > 0),
            "mstat": safe_num(wide.get(f"r{w}mstat", pd.Series(np.nan, index=wide.index))),
            "lbrf": safe_num(wide.get(f"r{w}lbrf", pd.Series(np.nan, index=wide.index))),
            "adl": safe_num(wide.get(f"r{w}adl5a", pd.Series(np.nan, index=wide.index))),
            "shlt": safe_num(wide.get(f"r{w}shlt", pd.Series(np.nan, index=wide.index))),
            "sayret": safe_num(wide.get(f"r{w}sayret", pd.Series(np.nan, index=wide.index))),
            "poverty": safe_num(wide.get(f"h{w}inpov", pd.Series(np.nan, index=wide.index))),
            "ppcr": safe_num(wide.get(f"r{w}ppcr", pd.Series(np.nan, index=wide.index))),
            "sex_code": safe_num(wide.ragender),
            "education_code": safe_num(wide.raeduc),
            "education_years": safe_num(wide.raedyrs),
        })
        for stem in chronic:
            d[stem] = safe_num(wide.get(f"r{w}{stem}", pd.Series(np.nan, index=wide.index)))
        d["self_report"] = d.iwstat.isin([1, 2]) & d.proxy.eq(0)
        d.loc[~d.self_report, "outcome_raw"] = np.nan
        chunks.append(d)
    long = pd.concat(chunks, ignore_index=True)
    long = long.sort_values(["person_id", "wave"]).reset_index(drop=True)
    return wide, long


def build_hrs_events(long: pd.DataFrame) -> pd.DataFrame:
    idx = long.set_index(["person_id", "wave"])
    records: list[pd.DataFrame] = []
    chronic = ["diabe", "cancre", "lunge", "hearte", "stroke", "arthre"]
    for w in range(3, 15):
        cur = idx.xs(w, level="wave").copy()
        pre = idx.xs(w - 1, level="wave").reindex(cur.index)
        nxt = idx.xs(w + 1, level="wave").reindex(cur.index)
        date = cur.interview_date

        specs: list[tuple[str, str, pd.Series]] = []
        widow = pre.mstat.isin([1, 2, 3, 4, 5, 6, 8]) & cur.mstat.eq(7)
        specs.append(("widowhood", "primary", widow))
        unemp = pre.lbrf.isin([1, 2, 4, 5, 6, 7]) & cur.lbrf.eq(3)
        specs.append(("unemployment", "primary", unemp))
        care = pre.ppcr.eq(0) & cur.ppcr.between(1, 8)
        specs.append(("caregiving", "primary", care))
        health = pre.adl.eq(0) & cur.adl.between(1, 5) & nxt.adl.between(1, 5)
        specs.append(("health", "primary", health))

        pre_chronic_valid = pre[chronic].isin([0, 1]).all(axis=1)
        cur_chronic_valid = cur[chronic].isin([0, 1]).all(axis=1)
        chronic_onset = pre_chronic_valid & cur_chronic_valid & pre[chronic].eq(0).all(axis=1) & cur[chronic].eq(1).any(axis=1)
        specs.append(("health_chronic_onset", "health_sensitivity", chronic_onset))
        srh = pre.shlt.between(1, 3) & cur.shlt.between(4, 5)
        specs.append(("health_self_rated_decline", "health_sensitivity", srh))
        poverty = pre.poverty.eq(0) & cur.poverty.eq(1)
        specs.append(("poverty_entry", "secondary", poverty))
        retirement = pre.sayret.isin([0, 2]) & cur.sayret.eq(1)
        specs.append(("retirement_comparator", "neutral_comparator", retirement))

        for event_type, role, mask in specs:
            ids = cur.index[mask.fillna(False)]
            if len(ids) == 0:
                continue
            records.append(pd.DataFrame({
                "person_id": ids.astype("int64"),
                "event_wave": w,
                "event_date": date.reindex(ids).to_numpy(),
                "event_type": event_type,
                "event_role": role,
            }))
    return pd.concat(records, ignore_index=True)


def select_hrs_timepoints(episodes: pd.DataFrame, long: pd.DataFrame) -> pd.DataFrame:
    grouped = {int(k): v.set_index("wave", drop=False) for k, v in long.groupby("person_id", sort=False)}
    selected: list[dict] = []
    for row in episodes.itertuples(index=False):
        p = grouped[int(row.person_id)]
        ev = p.loc[row.event_wave]
        pre = p.loc[row.event_wave - 1]
        pre2 = p.loc[row.event_wave - 2] if row.event_wave - 2 in p.index else None
        event_date = ev.interview_date
        actual_usable = pd.notna(event_date)

        pself = p.loc[p.self_report]
        y2_idx = nearest_by_months(pself.interview_date, event_date, 24, 18, 30, "after") if actual_usable else None
        y4_idx = nearest_by_months(pself.interview_date, event_date, 48, 42, 54, "after") if actual_usable else None
        method = "actual_interview_date"
        if actual_usable:
            # The frozen rule gives actual interview dates priority and forbids
            # jumping outside the 18-30 / 42-54 month windows.
            y2 = pself.loc[y2_idx] if y2_idx is not None else None
            y4 = pself.loc[y4_idx] if y4_idx is not None else None
        else:
            y2 = None
            y4 = None
            if row.event_wave + 1 in p.index:
                candidate = p.loc[row.event_wave + 1]
                y2 = candidate if bool(candidate.self_report) else None
            if row.event_wave + 2 in p.index:
                candidate = p.loc[row.event_wave + 2]
                y4 = candidate if bool(candidate.self_report) else None
            method = "nominal_biennial_fallback"

        def outcome(x: pd.Series | None) -> float:
            if x is None or not bool(x.self_report) or pd.isna(x.outcome_raw):
                return np.nan
            return float(x.outcome_raw)

        selected.append({
            "pre2_wave": row.event_wave - 2 if pre2 is not None else np.nan,
            "pre1_wave": row.event_wave - 1,
            "year2_wave": y2.wave if y2 is not None else np.nan,
            "year4_wave": y4.wave if y4 is not None else np.nan,
            "pre2_date": pre2.interview_date if pre2 is not None else pd.NaT,
            "pre1_date": pre.interview_date,
            "year2_date": y2.interview_date if y2 is not None else pd.NaT,
            "year4_date": y4.interview_date if y4 is not None else pd.NaT,
            "outcome_pre2_raw": outcome(pre2),
            "outcome_pre1_raw": outcome(pre),
            "outcome_event_raw": outcome(ev),
            "outcome_year2_raw": outcome(y2),
            "outcome_year4_raw": outcome(y4),
            "age_at_event": float(pre.age) if pd.notna(pre.age) else np.nan,
            "sex_code": float(ev.sex_code) if ev.sex_code in {1, 2} else np.nan,
            "education_code": float(ev.education_code) if pd.notna(ev.education_code) else np.nan,
            "education_source_wave": 0,
            "education_years": float(ev.education_years) if pd.notna(ev.education_years) else np.nan,
            "baseline_socio_code": float(pre.lbrf) if pd.notna(pre.lbrf) else np.nan,
            "baseline_marital_code": float(pre.mstat) if pd.notna(pre.mstat) else np.nan,
            "baseline_health_code": float(pre.shlt) if pd.notna(pre.shlt) else np.nan,
            "time_selection_method": method,
            "event_self_report": bool(ev.self_report),
            "pre1_self_report": bool(pre.self_report),
            "official_event_weight": float(ev.weight) if pd.notna(ev.weight) else np.nan,
            "official_year2_weight": float(y2.weight) if y2 is not None and pd.notna(y2.weight) else np.nan,
            "official_year4_weight": float(y4.weight) if y4 is not None and pd.notna(y4.weight) else np.nan,
        })
    return pd.concat([episodes.reset_index(drop=True), pd.DataFrame(selected)], axis=1)


def build_hrs() -> pd.DataFrame:
    note("Building HRS harmonized episodes")
    _, long = build_hrs_long()
    rows = build_hrs_events(long)
    episodes = collapse_event_rows(rows, "HRS")
    episodes = select_hrs_timepoints(episodes, long)
    episodes["event_year"] = pd.to_datetime(episodes.event_date).dt.year
    nominal_year = 1990 + 2 * episodes.event_wave
    episodes["event_year"] = episodes.event_year.fillna(nominal_year).astype("Int16")
    episodes["sex"] = episodes.sex_code.map({1.0: "male", 2.0: "female"}).fillna("unknown")
    episodes["education"] = episodes.education_code.map(hrs_education)
    episodes["baseline_socioeconomic"] = episodes.baseline_socio_code.map(hrs_socio)
    episodes = apply_history_and_censor(episodes)
    return standardize_and_derive(episodes)


def dictionary_for(df: pd.DataFrame, cohort: str, source: str) -> pd.DataFrame:
    descriptions = {
        "episode_id": "Unique person-transition episode after simultaneous event labels were collapsed.",
        "person_id": "Cohort-specific persistent person identifier (UKHLS pidp; HRS hhidpn).",
        "primary_event_type": "Single primary event family, compound_primary, or blank for nonprimary episode.",
        "event_date": "Observed event-wave interview date; interval-censored proxy for adversity timing.",
        "outcome_pre1_raw": "Immediate pre-event self-report outcome (UKHLS GHQ-12 or HRS CES-D 8).",
        "outcome_event_raw": "Event-wave self-report outcome.",
        "outcome_year2_raw": "Closest eligible self-report outcome 18-30 months after event, censored at later primary event.",
        "outcome_year4_raw": "Closest eligible self-report outcome 42-54 months after event, censored at later primary event.",
        "acute_response_z": "Cohort-standardized event minus pre1 response.",
        "persistence_2y_z": "Cohort-standardized year2 minus pre1 response.",
        "persistence_4y_z": "Cohort-standardized year4 minus pre1 response.",
        "age_at_event": "Age measured at the immediate pre-event interview (legacy-compatible column name).",
        "prior_response_history_z": "Mean acute response across strictly earlier observed primary episodes; missing when no prior response is available.",
    }
    rows = []
    for col in df.columns:
        rows.append({
            "cohort": cohort,
            "variable": col,
            "dtype": str(df[col].dtype),
            "description": descriptions.get(col, col.replace("_", " ").capitalize() + "."),
            "source": source,
            "derivation_plan": "results/NMH_confirmatory/00_NMH_confirmatory_freeze.md",
        })
    return pd.DataFrame(rows)


def qa_episode_data(df: pd.DataFrame, cohort: str) -> dict:
    if df.episode_id.duplicated().any():
        raise AssertionError(f"{cohort}: episode_id is not unique")
    if df[["person_id", "event_wave"]].duplicated().any():
        raise AssertionError(f"{cohort}: person-wave transition not unique after collapse")
    if (df.n_primary_labels >= 2).ne(df.compound_primary).any():
        raise AssertionError(f"{cohort}: compound-primary flag mismatch")
    if (df.n_primary_labels == 1).ne(df.isolated_primary).any():
        raise AssertionError(f"{cohort}: isolated-primary flag mismatch")
    return {
        "cohort": cohort,
        "rows": int(len(df)),
        "unique_episode_id": int(df.episode_id.nunique()),
        "unique_persons": int(df.person_id.nunique()),
        "unique_person_wave": int(df[["person_id", "event_wave"]].drop_duplicates().shape[0]),
        "primary_episodes": int(df.n_primary_labels.gt(0).sum()),
        "isolated_primary_episodes": int(df.isolated_primary.sum()),
        "compound_primary_episodes": int(df.compound_primary.sum()),
        "eligible_primary_episodes": int(df.primary_trajectory_eligible.sum()),
    }


def make_long(episodes: pd.DataFrame, include_compound: bool = False) -> pd.DataFrame:
    mask = episodes.primary_trajectory_eligible.copy()
    if include_compound:
        mask = (
            episodes.n_primary_labels.gt(0)
            & episodes.outcome_pre1_raw.notna()
            & episodes.outcome_event_raw.notna()
            & episodes.event_self_report
            & episodes.pre1_self_report
        )
    base = episodes.loc[mask].copy()
    base = base.loc[base.age_at_event.notna() & base.sex.isin(["male", "female"])].copy()
    fields = [
        "cohort", "episode_id", "person_id", "event_date", "event_year",
        "primary_event_type", "age_at_event", "sex", "education",
        "baseline_socioeconomic", "baseline_marital_code", "baseline_health_code",
        "standardization_mean", "standardization_sd",
    ]
    pieces = []
    for tp in TIME_LEVELS:
        x = base[fields].copy()
        x["time_point"] = tp
        x["outcome_raw"] = base[f"outcome_{tp}_raw"].to_numpy()
        x["outcome_z"] = base[f"outcome_{tp}_z"].to_numpy()
        pieces.append(x)
    out = pd.concat(pieces, ignore_index=True).dropna(subset=["outcome_raw", "outcome_z"])
    out["calendar_time"] = safe_num(out.event_year)
    return out.sort_values(["person_id", "episode_id", "time_point"]).reset_index(drop=True)


def fixed_design(
    long: pd.DataFrame,
    model: str,
    centers: dict[str, float] | None = None,
    include_covariates: bool = True,
) -> tuple[np.ndarray, list[str], dict[str, float]]:
    if centers is None:
        centers = {
            "age": float(long.age_at_event.mean()),
            "calendar": float(long.calendar_time.mean()),
        }
    data: dict[str, np.ndarray] = {"Intercept": np.ones(len(long), dtype=float)}
    for tp in ["pre2", "event", "year2", "year4"]:
        data[f"time[{tp}]"] = long.time_point.eq(tp).to_numpy(dtype=float)
    for event in ["unemployment", "caregiving", "health"]:
        data[f"event[{event}]"] = long.primary_event_type.eq(event).to_numpy(dtype=float)
    if model == "E":
        for tp in ["pre2", "event", "year2", "year4"]:
            for event in ["unemployment", "caregiving", "health"]:
                data[f"time[{tp}]:event[{event}]"] = (
                    long.time_point.eq(tp) & long.primary_event_type.eq(event)
                ).to_numpy(dtype=float)
    if include_covariates:
        data["age_centered"] = safe_num(long.age_at_event).to_numpy(dtype=float) - centers["age"]
        data["calendar_centered"] = safe_num(long.calendar_time).to_numpy(dtype=float) - centers["calendar"]
        data["sex[female]"] = long.sex.eq("female").to_numpy(dtype=float)
        for level in ["other_qualification", "no_qualification", "unknown"]:
            data[f"education[{level}]"] = long.education.eq(level).to_numpy(dtype=float)
        socio_levels = sorted(set(long.baseline_socioeconomic.dropna().astype(str)) - {"employed"})
        for level in socio_levels:
            data[f"socio[{level}]"] = long.baseline_socioeconomic.eq(level).to_numpy(dtype=float)
    names = list(data)
    x = np.column_stack([data[n] for n in names])
    # Frozen reference levels occasionally yield an all-zero column in a fold.
    # Preserve estimability by removing only columns with no variation in the
    # training data; the prediction design is aligned by name later.
    keep = np.ones(x.shape[1], dtype=bool)
    for j in range(1, x.shape[1]):
        if np.allclose(x[:, j], x[0, j]):
            keep[j] = False
    x = x[:, keep]
    names = [n for n, k in zip(names, keep) if k]
    if np.linalg.matrix_rank(x) < x.shape[1]:
        # Deterministic QR-free pruning, used only for redundant dummy columns.
        accepted: list[int] = []
        rank = 0
        for j in range(x.shape[1]):
            trial = x[:, accepted + [j]]
            r = np.linalg.matrix_rank(trial)
            if r > rank:
                accepted.append(j)
                rank = r
        x = x[:, accepted]
        names = [names[j] for j in accepted]
    return x, names, centers


def align_design(
    long: pd.DataFrame,
    model: str,
    names: list[str],
    centers: dict[str, float],
    include_covariates: bool = True,
) -> np.ndarray:
    x, current, _ = fixed_design(long, model, centers, include_covariates)
    pos = {n: j for j, n in enumerate(current)}
    out = np.zeros((len(long), len(names)), dtype=float)
    for j, name in enumerate(names):
        if name in pos:
            out[:, j] = x[:, pos[name]]
    return out


@dataclass
class NestedFit:
    beta: np.ndarray
    names: list[str]
    person_variance: float
    episode_variance: float
    residual_variance: float
    fixed_variance: float
    log_likelihood: float
    aic: float
    bic: float
    marginal_r2: float
    conditional_r2: float
    converged: bool
    optimizer: str
    message: str
    log_variances: np.ndarray


class NestedRandomInterceptML:
    """Exact Gaussian ML for nested episode and person random intercepts."""

    def __init__(self, y: np.ndarray, x: np.ndarray, person: np.ndarray, episode: np.ndarray):
        self.y = np.asarray(y, dtype=float)
        self.x = np.asarray(x, dtype=float)
        self.n_obs, self.p = self.x.shape
        self.ep_codes, ep_unique = pd.factorize(episode, sort=False)
        self.person_codes, person_unique = pd.factorize(person, sort=False)
        ep_person_raw = pd.Series(person).groupby(pd.Series(episode), sort=False).first()
        person_map = {v: i for i, v in enumerate(person_unique)}
        self.ep_person = np.array([person_map[ep_person_raw.loc[e]] for e in ep_unique], dtype=int)
        if np.any(pd.Series(person).groupby(pd.Series(episode)).nunique().to_numpy() != 1):
            raise AssertionError("Episode spans multiple persons")
        g = len(ep_unique)
        q = len(person_unique)
        self.n_ep = np.bincount(self.ep_codes, minlength=g).astype(float)
        self.sy_ep = np.bincount(self.ep_codes, weights=self.y, minlength=g)
        self.sx_ep = np.zeros((g, self.p), dtype=float)
        np.add.at(self.sx_ep, self.ep_codes, self.x)
        self.xtx = self.x.T @ self.x
        self.xty = self.x.T @ self.y
        self.yty = float(self.y @ self.y)
        self.n_person = q

    def evaluate(self, log_vars: np.ndarray, details: bool = False):
        r, e, pvar = np.exp(np.asarray(log_vars, dtype=float))
        denom = r + self.n_ep * e
        c = e / (r * denom)
        xtvix = self.xtx / r - self.sx_ep.T @ (c[:, None] * self.sx_ep)
        xtviy = self.xty / r - self.sx_ep.T @ (c * self.sy_ep)
        ytviy = self.yty / r - float(np.sum(c * self.sy_ep**2))
        u = np.zeros((self.n_person, self.p), dtype=float)
        t = np.zeros(self.n_person, dtype=float)
        s = np.zeros(self.n_person, dtype=float)
        np.add.at(u, self.ep_person, self.sx_ep / denom[:, None])
        np.add.at(t, self.ep_person, self.sy_ep / denom)
        np.add.at(s, self.ep_person, self.n_ep / denom)
        k = pvar / (1.0 + pvar * s)
        xtvix -= u.T @ (k[:, None] * u)
        xtviy -= u.T @ (k * t)
        ytviy -= float(np.sum(k * t**2))
        sign, _ = np.linalg.slogdet(xtvix)
        if sign <= 0 or not np.isfinite(xtvix).all():
            return (1e100, None) if details else 1e100
        try:
            beta = np.linalg.solve(xtvix, xtviy)
        except np.linalg.LinAlgError:
            return (1e100, None) if details else 1e100
        quad = ytviy - float(xtviy @ beta)
        if not np.isfinite(quad) or quad <= 0:
            return (1e100, None) if details else 1e100
        logdet_a = float(np.sum((self.n_ep - 1) * math.log(r) + np.log(denom)))
        logdet = logdet_a + float(np.log1p(pvar * s).sum())
        ll = -0.5 * (self.n_obs * math.log(2 * math.pi) + logdet + quad)
        if details:
            return -ll, {"beta": beta, "ll": ll, "quad": quad}
        return -ll

    def fit(self, names: list[str]) -> NestedFit:
        variance = max(float(np.var(self.y, ddof=0)), 1e-4)
        starts = [
            [variance * 0.20, variance * 0.20, variance * 0.60],
            [variance * 0.05, variance * 0.20, variance * 0.75],
            [variance * 0.30, variance * 0.05, variance * 0.65],
            [variance * 0.10, variance * 0.40, variance * 0.50],
            [variance * 0.40, variance * 0.10, variance * 0.50],
        ]
        candidates = []
        bounds = [(-14, 6), (-14, 6), (-14, 6)]
        for start in starts:
            fit = minimize(
                self.evaluate,
                np.log(np.maximum(start, 1e-7)),
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 300, "ftol": 1e-10},
            )
            if np.isfinite(fit.fun):
                candidates.append(fit)
        if not candidates:
            raise RuntimeError("All frozen optimizer starts failed")
        best = min(candidates, key=lambda z: z.fun)
        optimizer = "L-BFGS-B (best of five frozen starts)"
        if not best.success:
            powell = minimize(
                self.evaluate,
                best.x,
                method="Powell",
                bounds=bounds,
                options={"maxiter": 1200, "xtol": 1e-8, "ftol": 1e-8},
            )
            if np.isfinite(powell.fun) and powell.fun <= best.fun:
                best = powell
                optimizer = "Powell fallback after five L-BFGS-B starts"
        objective, detail = self.evaluate(best.x, details=True)
        if detail is None:
            raise RuntimeError("Final mixed-model evaluation failed")
        r, e, pvar = np.exp(best.x)
        fixed_values = self.x @ detail["beta"]
        fixed_var = float(np.var(fixed_values, ddof=0))
        total = fixed_var + pvar + e + r
        k = self.p + 3
        ll = float(detail["ll"])
        return NestedFit(
            beta=detail["beta"], names=list(names), person_variance=float(pvar),
            episode_variance=float(e), residual_variance=float(r), fixed_variance=fixed_var,
            log_likelihood=ll, aic=-2 * ll + 2 * k,
            bic=-2 * ll + math.log(self.n_obs) * k,
            marginal_r2=fixed_var / total, conditional_r2=(fixed_var + pvar + e) / total,
            converged=bool(best.success), optimizer=optimizer, message=str(best.message),
            log_variances=np.asarray(best.x, dtype=float),
        )


def model_record(cohort: str, model: str, formula: str, long: pd.DataFrame, fit: NestedFit) -> dict:
    return {
        "cohort": cohort,
        "record_type": "model",
        "model": model,
        "formula": formula,
        "n_episodes": int(long.episode_id.nunique()),
        "n_persons": int(long.person_id.nunique()),
        "n_observations": int(len(long)),
        "n_fixed_effects": len(fit.beta),
        "n_parameters": len(fit.beta) + 3,
        "log_likelihood": fit.log_likelihood,
        "AIC": fit.aic,
        "BIC": fit.bic,
        "fixed_variance": fit.fixed_variance,
        "person_variance": fit.person_variance,
        "episode_variance": fit.episode_variance,
        "residual_variance": fit.residual_variance,
        "marginal_R2": fit.marginal_r2,
        "conditional_R2": fit.conditional_r2,
        "converged": fit.converged,
        "optimizer": fit.optimizer,
        "optimizer_message": fit.message,
    }


def comparison_record(cohort: str, u: NestedFit, e: NestedFit) -> dict:
    df = len(e.beta) - len(u.beta)
    lr = 2 * (e.log_likelihood - u.log_likelihood)
    return {
        "cohort": cohort,
        "record_type": "comparison",
        "model": "U_vs_E",
        "formula": "Nested ML likelihood-ratio comparison on identical observations",
        "n_fixed_effects": df,
        "log_likelihood": lr,
        "AIC": e.aic - u.aic,
        "BIC": e.bic - u.bic,
        "marginal_R2": e.marginal_r2 - u.marginal_r2,
        "conditional_R2": e.conditional_r2 - u.conditional_r2,
        "lrt_df": df,
        "lrt_statistic": lr,
        "lrt_p_value": float(chi2.sf(max(lr, 0), df)) if df > 0 else np.nan,
        "converged": bool(u.converged and e.converged),
    }


def design_row(names: list[str], time_point: str, event_type: str) -> np.ndarray:
    values = np.zeros(len(names), dtype=float)
    for i, name in enumerate(names):
        if name == "Intercept":
            values[i] = 1
        elif name == f"time[{time_point}]":
            values[i] = 1
        elif name == f"event[{event_type}]":
            values[i] = 1
        elif name == f"time[{time_point}]:event[{event_type}]":
            values[i] = 1
    return values


def trajectory_contrasts(fit: NestedFit) -> list[dict]:
    out = []
    contrast_specs = [
        ("pretrend_pre2_to_pre1", "pre2", "pre1"),
        ("acute_pre1_to_event", "pre1", "event"),
        ("early_recovery_event_to_year2", "event", "year2"),
        ("long_recovery_event_to_year4", "event", "year4"),
        ("persistence_pre1_to_year2", "pre1", "year2"),
        ("persistence_pre1_to_year4", "pre1", "year4"),
    ]
    for event in PRIMARY:
        for tp in TIME_LEVELS:
            row = design_row(fit.names, tp, event)
            out.append({
                "record_type": "estimated_mean",
                "event_type": event,
                "time_point": tp,
                "contrast": "",
                "estimate": float(row @ fit.beta),
                "contrast_vector": row,
            })
        for label, start, end in contrast_specs:
            vec = design_row(fit.names, end, event) - design_row(fit.names, start, event)
            out.append({
                "record_type": "contrast",
                "event_type": event,
                "time_point": "",
                "contrast": label,
                "estimate": float(vec @ fit.beta),
                "contrast_vector": vec,
            })
    return out


def person_gls_contributions(long: pd.DataFrame, x: np.ndarray, y: np.ndarray, fit: NestedFit):
    people = pd.unique(long.person_id)
    person_pos = {p: i for i, p in enumerate(people)}
    p = x.shape[1]
    a = np.zeros((len(people), p, p), dtype=float)
    b = np.zeros((len(people), p), dtype=float)
    grouped = long.groupby("person_id", sort=False).indices
    for person, indices in grouped.items():
        ii = np.asarray(indices, dtype=int)
        xp = x[ii]
        yp = y[ii]
        ep = long.iloc[ii].episode_id.to_numpy()
        same = ep[:, None] == ep[None, :]
        v = (
            fit.residual_variance * np.eye(len(ii))
            + fit.episode_variance * same.astype(float)
            + fit.person_variance * np.ones((len(ii), len(ii)))
        )
        try:
            vx = np.linalg.solve(v, xp)
            vy = np.linalg.solve(v, yp)
        except np.linalg.LinAlgError:
            vx = np.linalg.pinv(v) @ xp
            vy = np.linalg.pinv(v) @ yp
        j = person_pos[person]
        a[j] = xp.T @ vx
        b[j] = xp.T @ vy
    return people, a, b


def bootstrap_fixed_effects(
    long: pd.DataFrame,
    x: np.ndarray,
    y: np.ndarray,
    fit: NestedFit,
    success_target: int = BOOT_SUCCESS,
    seed: int = SEED,
) -> tuple[np.ndarray, int]:
    people, a, b = person_gls_contributions(long, x, y, fit)
    n = len(people)
    rng = np.random.default_rng(seed)
    betas = []
    attempts = 0
    while len(betas) < success_target and attempts < success_target * 3:
        attempts += 1
        counts = rng.multinomial(n, np.full(n, 1 / n))
        aa = np.tensordot(counts, a, axes=(0, 0))
        bb = counts @ b
        try:
            beta = np.linalg.solve(aa, bb)
        except np.linalg.LinAlgError:
            continue
        if np.isfinite(beta).all():
            betas.append(beta)
    if len(betas) < success_target:
        raise RuntimeError("Could not obtain required successful person bootstraps")
    return np.vstack(betas), attempts


def add_bootstrap_cis(records: list[dict], boot_beta: np.ndarray) -> None:
    for row in records:
        vec = row.pop("contrast_vector")
        values = boot_beta @ vec
        row["ci_lower"] = float(np.quantile(values, 0.025))
        row["ci_upper"] = float(np.quantile(values, 0.975))
        row["bootstrap_success"] = int(len(values))


def metrics(y: np.ndarray, pred: np.ndarray, weights: np.ndarray | None = None, variance: float | None = None) -> dict:
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    if weights is None:
        weights = np.ones(len(y), dtype=float)
    weights = np.asarray(weights, dtype=float)
    good = np.isfinite(y) & np.isfinite(pred) & np.isfinite(weights) & (weights > 0)
    y, pred, weights = y[good], pred[good], weights[good]
    wsum = weights.sum()
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
    if variance is None or not np.isfinite(variance) or variance <= 0:
        raise ValueError(
            "Predictive density requires finite positive training residual variance; "
            "evaluation-set MSE must not replace training variance."
        )
    lpd = float(np.sum(weights * (-0.5 * (np.log(2 * np.pi * variance) + err**2 / variance))) / wsum)
    return {
        "RMSE": math.sqrt(mse), "MAE": mae, "predictive_R2": r2,
        "mean_log_predictive_density": lpd,
        "calibration_intercept": float(calibration[0]),
        "calibration_slope": float(calibration[1]),
        "n_observations": int(len(y)),
    }


def balanced_person_folds(long: pd.DataFrame, k: int, seed: int) -> dict[object, int]:
    tab = long.groupby(["person_id", "primary_event_type"], observed=True).size().unstack(fill_value=0)
    for event in PRIMARY:
        if event not in tab:
            tab[event] = 0
    tab = tab[PRIMARY]
    total = tab.sum(axis=1).to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    jitter = rng.random(len(tab))
    order = np.lexsort((jitter, -total))
    fold_load = np.zeros((k, 1 + len(PRIMARY)), dtype=float)
    assignment: dict[object, int] = {}
    target = np.r_[total.sum(), tab.sum(axis=0).to_numpy(dtype=float)] / k
    target[target == 0] = 1
    for j in order:
        vec = np.r_[total[j], tab.iloc[j].to_numpy(dtype=float)]
        scores = [np.sum(((fold_load[f] + vec - target) / target) ** 2) for f in range(k)]
        f = int(np.argmin(scores))
        assignment[tab.index[j]] = f
        fold_load[f] += vec
    return assignment


def validation_bundle(
    long: pd.DataFrame,
    episodes: pd.DataFrame,
    validation: str,
    seed: int,
) -> pd.DataFrame:
    outputs = []
    if validation == "grouped_10fold":
        assignment = balanced_person_folds(long, 10, seed)
        splits = []
        for fold in range(10):
            test_people = {p for p, f in assignment.items() if f == fold}
            train_mask = ~long.person_id.isin(test_people)
            test_mask = long.person_id.isin(test_people)
            splits.append((fold, train_mask, test_mask))
        temporal_cutoff = None
    elif validation == "temporal_70_30":
        dates = long[["episode_id", "event_date"]].drop_duplicates().event_date.dropna().sort_values()
        cutoff = dates.iloc[max(0, int(math.floor(0.70 * len(dates))) - 1)]
        train_mask = long.event_date <= cutoff
        train_people = set(long.loc[train_mask, "person_id"])
        test_mask = (long.event_date > cutoff) & ~long.person_id.isin(train_people)
        overlap_excluded = (long.event_date > cutoff) & long.person_id.isin(train_people)
        splits = [(0, train_mask, test_mask)]
        temporal_cutoff = cutoff
    else:
        raise ValueError(validation)

    for fold, train_mask, test_mask in splits:
        train = long.loc[train_mask].copy()
        test = long.loc[test_mask].copy()
        if train.empty or test.empty:
            continue
        # Cohort standardization is recalculated within training persons.
        train_people = set(train.person_id)
        ref_mask = episodes.person_id.isin(train_people) & episodes.n_primary_labels.gt(0)
        if temporal_cutoff is not None:
            ref_mask &= episodes.event_date.le(temporal_cutoff)
        pre_values = episodes.loc[ref_mask, "outcome_pre1_raw"].dropna()
        mu, sd = float(pre_values.mean()), float(pre_values.std(ddof=1))
        train["analysis_y"] = (train.outcome_raw - mu) / sd
        test["analysis_y"] = (test.outcome_raw - mu) / sd
        model_preds = {}
        model_vars = {}
        for model in ["U", "E"]:
            xtr, names, centers = fixed_design(train, model)
            xte = align_design(test, model, names, centers)
            fitted = NestedRandomInterceptML(
                train.analysis_y.to_numpy(), xtr,
                train.person_id.to_numpy(), train.episode_id.to_numpy(),
            ).fit(names)
            model_preds[model] = xte @ fitted.beta
            model_vars[model] = fitted.residual_variance
        frame = test[["person_id", "episode_id", "event_date", "primary_event_type", "time_point"]].copy()
        frame["observed"] = test.analysis_y.to_numpy()
        frame["pred_U"] = model_preds["U"]
        frame["pred_E"] = model_preds["E"]
        frame["variance_U"] = model_vars["U"]
        frame["variance_E"] = model_vars["E"]
        frame["fold"] = fold
        frame["validation"] = validation
        frame["training_event_cutoff"] = temporal_cutoff
        frame["n_train_persons"] = int(train.person_id.nunique())
        frame["n_train_episodes"] = int(train.episode_id.nunique())
        frame["n_temporal_overlap_rows_excluded"] = (
            int(overlap_excluded.sum()) if validation == "temporal_70_30" else 0
        )
        frame["n_temporal_overlap_episodes_excluded"] = (
            int(long.loc[overlap_excluded, "episode_id"].nunique()) if validation == "temporal_70_30" else 0
        )
        outputs.append(frame)
    if not outputs:
        return pd.DataFrame()
    return pd.concat(outputs, ignore_index=True)


def prediction_metric_rows(cohort: str, bundle: pd.DataFrame) -> list[dict]:
    rows = []
    for validation, d in bundle.groupby("validation", sort=False):
        validation_meta = {
            "training_event_cutoff": d.training_event_cutoff.dropna().iloc[0] if d.training_event_cutoff.notna().any() else pd.NaT,
            "n_train_persons": int(d.n_train_persons.max()),
            "n_train_episodes": int(d.n_train_episodes.max()),
            "n_temporal_overlap_rows_excluded": int(d.n_temporal_overlap_rows_excluded.max()),
            "n_temporal_overlap_episodes_excluded": int(d.n_temporal_overlap_episodes_excluded.max()),
        }
        point = {}
        for model in ["U", "E"]:
            m = metrics(d.observed.to_numpy(), d[f"pred_{model}"].to_numpy(), variance=float(d[f"variance_{model}"].mean()))
            point[model] = m
            rows.append({
                "cohort": cohort, "validation": validation, "record_type": "model_metric",
                "model": model, "comparison": "", "n_persons": int(d.person_id.nunique()),
                "n_episodes": int(d.episode_id.nunique()), **m, **validation_meta,
            })
        rng = np.random.default_rng(SEED + (0 if validation == "grouped_10fold" else 17))
        people = pd.unique(d.person_id)
        diffs = defaultdict(list)
        attempts = 0
        while len(diffs["RMSE"]) < BOOT_SUCCESS:
            attempts += 1
            counts = pd.Series(rng.multinomial(len(people), np.full(len(people), 1 / len(people))), index=people)
            w = d.person_id.map(counts).to_numpy(dtype=float)
            if w.sum() == 0:
                continue
            mu = metrics(d.observed, d.pred_U, w, float(d.variance_U.mean()))
            me = metrics(d.observed, d.pred_E, w, float(d.variance_E.mean()))
            diffs["RMSE"].append(mu["RMSE"] - me["RMSE"])
            diffs["MAE"].append(mu["MAE"] - me["MAE"])
            diffs["predictive_R2"].append(me["predictive_R2"] - mu["predictive_R2"])
            diffs["mean_log_predictive_density"].append(me["mean_log_predictive_density"] - mu["mean_log_predictive_density"])
            diffs["calibration_intercept_closeness"].append(abs(mu["calibration_intercept"]) - abs(me["calibration_intercept"]))
            diffs["calibration_slope_closeness"].append(abs(mu["calibration_slope"] - 1) - abs(me["calibration_slope"] - 1))
        for metric_name, values in diffs.items():
            if metric_name in {"RMSE", "MAE"}:
                point_improvement = point["U"][metric_name] - point["E"][metric_name]
                denom = point["U"][metric_name]
                relative_values = np.asarray(values) / denom if denom != 0 else np.full(len(values), np.nan)
                relative_point = point_improvement / denom if denom != 0 else np.nan
            elif metric_name in {"predictive_R2", "mean_log_predictive_density"}:
                point_improvement = point["E"][metric_name] - point["U"][metric_name]
                denom = abs(point["U"][metric_name])
                report_relative = metric_name == "predictive_R2" and point["U"][metric_name] > 0
                relative_values = np.asarray(values) / denom if report_relative and denom != 0 else np.full(len(values), np.nan)
                relative_point = point_improvement / denom if report_relative and denom != 0 else np.nan
            elif metric_name == "calibration_intercept_closeness":
                point_improvement = abs(point["U"]["calibration_intercept"]) - abs(point["E"]["calibration_intercept"])
                relative_values = np.full(len(values), np.nan); relative_point = np.nan
            else:
                point_improvement = abs(point["U"]["calibration_slope"] - 1) - abs(point["E"]["calibration_slope"] - 1)
                relative_values = np.full(len(values), np.nan); relative_point = np.nan
            rows.append({
                "cohort": cohort, "validation": validation, "record_type": "comparison",
                "model": "", "comparison": "E_minus_U" if metric_name not in {"RMSE", "MAE"} else "U_minus_E",
                "metric": metric_name, "improvement": float(point_improvement),
                "ci_lower": float(np.quantile(values, 0.025)),
                "ci_upper": float(np.quantile(values, 0.975)),
                "relative_improvement": float(relative_point) if np.isfinite(relative_point) else np.nan,
                "relative_ci_lower": float(np.nanquantile(relative_values, .025)) if np.isfinite(relative_values).any() else np.nan,
                "relative_ci_upper": float(np.nanquantile(relative_values, .975)) if np.isfinite(relative_values).any() else np.nan,
                "bootstrap_success": BOOT_SUCCESS, "bootstrap_attempts": attempts,
                "n_persons": int(len(people)), "n_episodes": int(d.episode_id.nunique()),
                **validation_meta,
            })
    return rows


def build_cross_event_samples(episodes: pd.DataFrame) -> pd.DataFrame:
    d = episodes.loc[
        episodes.primary_trajectory_eligible
        & episodes.acute_response_z.notna()
        & episodes.persistence_2y_z.notna()
    ].copy()
    d = d.sort_values(["person_id", "event_date", "event_wave", "episode_id"])
    records = []
    for person, g in d.groupby("person_id", sort=False):
        g = g.reset_index(drop=True)
        # Frozen two-event cohort: earliest qualifying episode and first later,
        # different-family episode occurring after the earlier 2-y outcome.
        if len(g) >= 2:
            pair = None
            for i in range(len(g)):
                first = g.iloc[i]
                choices = g.loc[
                    (g.index > i)
                    & g.primary_event_type.ne(first.primary_event_type)
                    & g.event_date.gt(first.year2_date)
                ]
                if not choices.empty:
                    pair = (first, choices.iloc[0])
                    break
            if pair is not None:
                records.append(cross_record(pair[0], pair[1], None, "two_event"))

        # Frozen three-event cohort: the first qualifying temporally separated
        # sequence of three distinct families.  Earlier 2-y outcomes must precede
        # the next event so the prior response is observable prospectively.
        found = None
        for i in range(len(g)):
            a = g.iloc[i]
            bchoices = g.loc[
                (g.index > i)
                & g.primary_event_type.ne(a.primary_event_type)
                & g.event_date.gt(a.year2_date)
            ]
            for j, b in bchoices.iterrows():
                cchoices = g.loc[
                    (g.index > j)
                    & ~g.primary_event_type.isin([a.primary_event_type, b.primary_event_type])
                    & g.event_date.gt(b.year2_date)
                ]
                if not cchoices.empty:
                    found = (a, b, cchoices.iloc[0])
                    break
            if found is not None:
                break
        if found is not None:
            records.append(cross_record(found[0], found[1], found[2], "three_event"))
    out = pd.DataFrame(records)
    if out.empty:
        return out
    out["sample_id"] = [f"{c}_{i + 1:08d}" for i, c in enumerate(out.cohort)]
    if out.sample_id.duplicated().any() or out[["cohort", "cohort_type", "person_id"]].duplicated().any():
        raise AssertionError("Cross-event prediction sample key is not unique")
    return out


def cross_record(first: pd.Series, second: pd.Series, third: pd.Series | None, cohort_type: str) -> dict:
    target = second if third is None else third
    prior_rows = [first] if third is None else [first, second]
    rec = {
        "cohort": first.cohort,
        "cohort_type": cohort_type,
        "person_id": int(first.person_id),
        "first_episode_id": first.episode_id,
        "second_episode_id": second.episode_id,
        "target_episode_id": target.episode_id,
        "first_event_type": first.primary_event_type,
        "second_event_type": second.primary_event_type,
        "target_event_type": target.primary_event_type,
        "target_event_date": target.event_date,
        "target_event_year": int(target.event_year),
        "target_pre_raw": float(target.outcome_pre1_raw),
        "target_age": float(target.age_at_event),
        "target_sex": target.sex,
        "target_education": target.education,
        "target_socio": target.baseline_socioeconomic,
        "target_number_preceding_primary_events": int(target.preceding_primary_episode_count),
        "target_acute_raw": float(target.outcome_event_raw - target.outcome_pre1_raw),
        "target_persistence_raw": float(target.outcome_year2_raw - target.outcome_pre1_raw),
        "prior_acute_raw": float(np.mean([r.outcome_event_raw - r.outcome_pre1_raw for r in prior_rows])),
        "prior_persistence_raw": float(np.mean([r.outcome_year2_raw - r.outcome_pre1_raw for r in prior_rows])),
        "prior_acute_sd_raw": float(np.std([r.outcome_event_raw - r.outcome_pre1_raw for r in prior_rows], ddof=1)) if len(prior_rows) > 1 else np.nan,
        "prior_persistence_sd_raw": float(np.std([r.outcome_year2_raw - r.outcome_pre1_raw for r in prior_rows], ddof=1)) if len(prior_rows) > 1 else np.nan,
        "prior_event_types": "|".join([r.primary_event_type for r in prior_rows]),
        "prior_type_combination": "|".join(sorted(r.primary_event_type for r in prior_rows)),
        "standardization_mean": float(target.standardization_mean),
        "standardization_sd": float(target.standardization_sd),
    }
    for i, prior in enumerate(prior_rows, start=1):
        rec[f"prior{i}_event_type"] = prior.primary_event_type
        rec[f"prior{i}_acute_raw"] = float(prior.outcome_event_raw - prior.outcome_pre1_raw)
        rec[f"prior{i}_persistence_raw"] = float(prior.outcome_year2_raw - prior.outcome_pre1_raw)
    return rec


def training_eb_signature(train: pd.DataFrame, test: pd.DataFrame, metric: str) -> tuple[bool, np.ndarray, np.ndarray, dict]:
    """Training-only EB person intercept from two prior different-event responses."""
    if not train.cohort_type.eq("three_event").all():
        return False, np.array([]), np.array([]), {"reason": "not_three_event"}
    y1 = train[f"prior1_{metric}_raw"].to_numpy(dtype=float)
    y2 = train[f"prior2_{metric}_raw"].to_numpy(dtype=float)
    types = pd.concat([train.prior1_event_type, train.prior2_event_type], ignore_index=True)
    responses = np.r_[y1, y2]
    design = {"Intercept": np.ones(len(responses))}
    for event in ["unemployment", "caregiving", "health"]:
        design[f"event[{event}]"] = types.eq(event).to_numpy(dtype=float)
    x = np.column_stack(list(design.values()))
    keep, rank = [], 0
    for j in range(x.shape[1]):
        r = np.linalg.matrix_rank(x[:, keep + [j]])
        if r > rank:
            keep.append(j); rank = r
    x = x[:, keep]
    names = [list(design)[j] for j in keep]
    beta = np.linalg.pinv(x) @ responses

    def event_mean(values: pd.Series) -> np.ndarray:
        out = np.full(len(values), beta[names.index("Intercept")])
        for event in ["unemployment", "caregiving", "health"]:
            name = f"event[{event}]"
            if name in names:
                out += values.eq(event).to_numpy(dtype=float) * beta[names.index(name)]
        return out

    r1 = y1 - event_mean(train.prior1_event_type)
    r2 = y2 - event_mean(train.prior2_event_type)
    sigma2 = float(np.mean((r1 - r2) ** 2) / 2)
    person_means = (r1 + r2) / 2
    tau2 = max(float(np.var(person_means, ddof=1)) - sigma2 / 2, 0.0)
    if not np.isfinite(tau2) or tau2 <= 1e-10:
        return False, np.array([]), np.array([]), {"reason": "person_variance_boundary", "tau2": tau2, "sigma2": sigma2}
    shrink = tau2 / (tau2 + sigma2 / 2)
    train_eb = shrink * person_means
    test_r1 = test[f"prior1_{metric}_raw"].to_numpy(dtype=float) - event_mean(test.prior1_event_type)
    test_r2 = test[f"prior2_{metric}_raw"].to_numpy(dtype=float) - event_mean(test.prior2_event_type)
    test_eb = shrink * (test_r1 + test_r2) / 2
    return True, train_eb, test_eb, {"tau2": tau2, "sigma2": sigma2, "shrinkage": shrink}


def cross_design(
    d: pd.DataFrame,
    model: str,
    metric: str,
    mu: float,
    sd: float,
    centers: dict | None = None,
    names: list[str] | None = None,
) -> tuple[np.ndarray, list[str], dict]:
    if centers is None:
        centers = {"age": float(d.target_age.mean()), "year": float(d.target_event_year.mean())}
    baseline_z = (d.target_pre_raw.to_numpy(dtype=float) - mu) / sd
    prior = d[f"prior_{metric}_raw"].to_numpy(dtype=float) / sd
    prior_eb = d.get(f"prior_{metric}_eb_raw", d[f"prior_{metric}_raw"]).to_numpy(dtype=float) / sd
    data: dict[str, np.ndarray] = {
        "Intercept": np.ones(len(d)),
        "baseline_z": baseline_z,
        "age_centered": d.target_age.to_numpy(dtype=float) - centers["age"],
        "calendar_centered": d.target_event_year.to_numpy(dtype=float) - centers["year"],
        "sex[female]": d.target_sex.eq("female").to_numpy(dtype=float),
        "number_preceding_primary_events": d.target_number_preceding_primary_events.to_numpy(dtype=float),
    }
    for level in ["other_qualification", "no_qualification", "unknown"]:
        data[f"education[{level}]"] = d.target_education.eq(level).to_numpy(dtype=float)
    socio_levels = sorted(set(d.target_socio.astype(str)) - {"employed"})
    for level in socio_levels:
        data[f"socio[{level}]"] = d.target_socio.eq(level).to_numpy(dtype=float)
    for event in ["unemployment", "caregiving", "health"]:
        data[f"target_event[{event}]"] = d.target_event_type.eq(event).to_numpy(dtype=float)
    if model in {"P1", "P2"}:
        data["prior_response"] = prior
    if model in {"P1_EB", "P2_EB"}:
        data["prior_response_EB"] = prior_eb
    if model in {"P2", "P2_EB"}:
        if d.cohort_type.eq("two_event").all():
            for event in ["unemployment", "caregiving", "health"]:
                data[f"prior_event[{event}]"] = d.prior1_event_type.eq(event).to_numpy(dtype=float)
        else:
            for combo in sorted(set(d.prior_type_combination.astype(str))):
                data[f"prior_type_combination[{combo}]"] = d.prior_type_combination.eq(combo).to_numpy(dtype=float)
        response_name = "prior_response_EB" if model == "P2_EB" else "prior_response"
        response = data[response_name]
        for event in ["unemployment", "caregiving", "health"]:
            data[f"{response_name}:target_event[{event}]"] = response * d.target_event_type.eq(event).to_numpy(dtype=float)
    if names is None:
        full_names = list(data)
        x = np.column_stack([data[n] for n in full_names])
        keep, rank = [], 0
        for j in range(x.shape[1]):
            r = np.linalg.matrix_rank(x[:, keep + [j]])
            if r > rank:
                keep.append(j)
                rank = r
        return x[:, keep], [full_names[j] for j in keep], centers
    x = np.zeros((len(d), len(names)))
    for j, name in enumerate(names):
        if name in data:
            x[:, j] = data[name]
    return x, names, centers


def simple_group_folds(ids: pd.Series, k: int, seed: int) -> dict:
    values = np.asarray(pd.unique(ids))
    rng = np.random.default_rng(seed)
    rng.shuffle(values)
    return {v: i % k for i, v in enumerate(values)}


def cross_prediction_bundle(
    d: pd.DataFrame,
    metric: str,
    validation: str,
    standardization_episodes: pd.DataFrame,
) -> pd.DataFrame:
    d = d.copy().reset_index(drop=True)
    if validation == "grouped_10fold":
        assn = simple_group_folds(d.person_id, min(10, max(2, len(d))), SEED)
        splits = []
        for fold in sorted(set(assn.values())):
            test = d.person_id.map(assn).eq(fold)
            splits.append((fold, ~test, test))
        temporal_cutoff = None
    else:
        dates = d.target_event_date.dropna().sort_values()
        if len(dates) < 2:
            return pd.DataFrame()
        cutoff = dates.iloc[max(0, int(math.floor(0.70 * len(dates))) - 1)]
        train = d.target_event_date <= cutoff
        test = d.target_event_date > cutoff
        splits = [(0, train, test)]
        temporal_cutoff = cutoff
    outputs = []
    eb_available_all = True
    for fold, train_mask, test_mask in splits:
        tr, te = d.loc[train_mask].copy(), d.loc[test_mask].copy()
        if len(tr) < 10 or len(te) < 2:
            continue
        train_people = set(tr.person_id)
        ref_mask = standardization_episodes.person_id.isin(train_people) & standardization_episodes.n_primary_labels.gt(0)
        if temporal_cutoff is not None:
            ref_mask &= standardization_episodes.event_date.le(temporal_cutoff)
        ref = standardization_episodes.loc[ref_mask, "outcome_pre1_raw"].dropna()
        mu, sd = float(ref.mean()), float(ref.std(ddof=1))
        if not np.isfinite(sd) or sd <= 0:
            raise AssertionError("Invalid training-fold outcome scale")
        models = ["P0", "P1", "P2"]
        if d.cohort_type.eq("three_event").all():
            eb_ok, eb_train, eb_test, eb_info = training_eb_signature(tr, te, metric)
            if eb_ok:
                tr[f"prior_{metric}_eb_raw"] = eb_train
                te[f"prior_{metric}_eb_raw"] = eb_test
                models += ["P1_EB", "P2_EB"]
            else:
                eb_available_all = False
        ytr = tr[f"target_{metric}_raw"].to_numpy(dtype=float) / sd
        yte = te[f"target_{metric}_raw"].to_numpy(dtype=float) / sd
        frame = te[["cohort", "cohort_type", "person_id", "sample_id", "target_event_date", "target_event_type"]].copy()
        frame["observed"] = yte
        for model in models:
            xtr, names, centers = cross_design(tr, model, metric, mu, sd)
            xte, _, _ = cross_design(te, model, metric, mu, sd, centers, names)
            beta = np.linalg.pinv(xtr) @ ytr
            pred = xte @ beta
            frame[f"pred_{model}"] = pred
            frame[f"residvar_{model}"] = float(np.mean((ytr - xtr @ beta) ** 2))
        frame["fold"] = fold
        frame["validation"] = validation
        frame["outcome_metric"] = metric
        frame["training_event_cutoff"] = temporal_cutoff
        frame["n_train_persons"] = int(tr.person_id.nunique())
        frame["eb_available"] = "P1_EB" in models
        if "P1_EB" in models:
            frame["eb_tau2"] = eb_info["tau2"]
            frame["eb_sigma2"] = eb_info["sigma2"]
            frame["eb_shrinkage"] = eb_info["shrinkage"]
        outputs.append(frame)
    result = pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()
    if not eb_available_all and not result.empty:
        result = result.drop(columns=[c for c in result if c.startswith("pred_P1_EB") or c.startswith("pred_P2_EB") or c.startswith("residvar_P1_EB") or c.startswith("residvar_P2_EB")], errors="ignore")
        result["eb_available"] = False
    return result


def cross_metric_rows(bundle: pd.DataFrame) -> list[dict]:
    rows = []
    if bundle.empty:
        return rows
    for keys, g in bundle.groupby(["cohort", "cohort_type", "validation", "outcome_metric"], sort=False):
        cohort, cohort_type, validation, outcome_metric = keys
        models = sorted(
            c[5:] for c in g.columns
            if c.startswith("pred_") and g[c].notna().any()
        )
        point = {}
        validation_meta = {
            "training_event_cutoff": g.training_event_cutoff.dropna().iloc[0] if g.training_event_cutoff.notna().any() else pd.NaT,
            "n_train_persons": int(g.n_train_persons.max()),
        }
        if cohort_type == "three_event" and ("eb_available" not in g or not g.eb_available.all()):
            rows.append({
                "cohort": cohort, "cohort_type": cohort_type, "validation": validation,
                "outcome_metric": outcome_metric, "record_type": "availability",
                "model": "P1_EB/P2_EB", "comparison": "", "n_persons": int(g.person_id.nunique()),
                "notes": "EB sensitivity unavailable because at least one training fold placed person variance on the prespecified reporting boundary.",
            })
        for model in models:
            m = metrics(g.observed, g[f"pred_{model}"], variance=float(g[f"residvar_{model}"].mean()))
            point[model] = m
            rows.append({
                "cohort": cohort, "cohort_type": cohort_type, "validation": validation,
                "outcome_metric": outcome_metric, "record_type": "model_metric", "model": model,
                "comparison": "", "n_persons": int(g.person_id.nunique()), **m,
                **validation_meta,
            })
        comparisons = [("P0", "P1"), ("P1", "P2")]
        if "P1_EB" in models:
            comparisons += [("P0", "P1_EB"), ("P1_EB", "P2_EB")]
        rng = np.random.default_rng(SEED + len(rows))
        people = pd.unique(g.person_id)
        for base, candidate in comparisons:
            if base not in models or candidate not in models:
                continue
            values = defaultdict(list)
            for _ in range(BOOT_SUCCESS):
                counts = pd.Series(rng.multinomial(len(people), np.full(len(people), 1 / len(people))), index=people)
                w = g.person_id.map(counts).to_numpy(dtype=float)
                mb = metrics(g.observed, g[f"pred_{base}"], w, float(g[f"residvar_{base}"].mean()))
                mc = metrics(g.observed, g[f"pred_{candidate}"], w, float(g[f"residvar_{candidate}"].mean()))
                values["RMSE"].append(mb["RMSE"] - mc["RMSE"])
                values["MAE"].append(mb["MAE"] - mc["MAE"])
                values["predictive_R2"].append(mc["predictive_R2"] - mb["predictive_R2"])
                values["mean_log_predictive_density"].append(mc["mean_log_predictive_density"] - mb["mean_log_predictive_density"])
                values["calibration_intercept_closeness"].append(abs(mb["calibration_intercept"]) - abs(mc["calibration_intercept"]))
                values["calibration_slope_closeness"].append(abs(mb["calibration_slope"] - 1) - abs(mc["calibration_slope"] - 1))
            for metric_name, v in values.items():
                if metric_name in {"RMSE", "MAE"}:
                    point_improvement = point[base][metric_name] - point[candidate][metric_name]
                elif metric_name in {"predictive_R2", "mean_log_predictive_density"}:
                    point_improvement = point[candidate][metric_name] - point[base][metric_name]
                elif metric_name == "calibration_intercept_closeness":
                    point_improvement = abs(point[base]["calibration_intercept"]) - abs(point[candidate]["calibration_intercept"])
                else:
                    point_improvement = abs(point[base]["calibration_slope"] - 1) - abs(point[candidate]["calibration_slope"] - 1)
                rows.append({
                    "cohort": cohort, "cohort_type": cohort_type, "validation": validation,
                    "outcome_metric": outcome_metric, "record_type": "comparison",
                    "model": "", "comparison": f"{candidate}_minus_{base}", "metric": metric_name,
                    "improvement": float(point_improvement), "ci_lower": float(np.quantile(v, .025)),
                    "ci_upper": float(np.quantile(v, .975)), "bootstrap_success": BOOT_SUCCESS,
                    "n_persons": int(len(people)),
                    **validation_meta,
                })
    return rows


def logistic_fit(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, bool]:
    y = np.asarray(y, dtype=float)
    def objective(beta):
        eta = np.clip(x @ beta, -30, 30)
        return float(np.sum(np.logaddexp(0, eta) - y * eta) + 1e-8 * np.sum(beta[1:] ** 2))
    def grad(beta):
        return x.T @ (expit(np.clip(x @ beta, -30, 30)) - y) + np.r_[0, 2e-8 * beta[1:]]
    fit = minimize(objective, np.zeros(x.shape[1]), jac=grad, method="L-BFGS-B", options={"maxiter": 600})
    return np.asarray(fit.x), bool(fit.success)


def attrition_design(d: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    data = {"Intercept": np.ones(len(d))}
    for event in ["unemployment", "caregiving", "health"]:
        data[f"event[{event}]"] = d.primary_event_type.eq(event).to_numpy(dtype=float)
    data["age_centered"] = d.age_at_event.to_numpy(dtype=float) - float(d.age_at_event.mean())
    data["sex[female]"] = d.sex.eq("female").to_numpy(dtype=float)
    for level in ["other_qualification", "no_qualification", "unknown"]:
        data[f"education[{level}]"] = d.education.eq(level).to_numpy(dtype=float)
    data["calendar_centered"] = safe_num(d.event_year).to_numpy(dtype=float) - float(safe_num(d.event_year).mean())
    data["pre1_outcome_z"] = d.outcome_pre1_z.to_numpy(dtype=float)
    prior = safe_num(d.prior_response_history_z)
    data["prior_response_history_z"] = prior.fillna(0).to_numpy(dtype=float)
    data["no_prior_response_history"] = prior.isna().to_numpy(dtype=float)
    data["baseline_health"] = safe_num(d.baseline_health_code).fillna(safe_num(d.baseline_health_code).median()).to_numpy(dtype=float)
    data["baseline_marital"] = safe_num(d.baseline_marital_code).fillna(safe_num(d.baseline_marital_code).median()).to_numpy(dtype=float)
    for level in sorted(set(d.baseline_socioeconomic.astype(str)) - {"employed"}):
        data[f"socio[{level}]"] = d.baseline_socioeconomic.eq(level).to_numpy(dtype=float)
    names = list(data)
    x = np.column_stack([data[n] for n in names])
    keep, rank = [], 0
    for j in range(x.shape[1]):
        r = np.linalg.matrix_rank(x[:, keep + [j]])
        if r > rank:
            keep.append(j); rank = r
    return x[:, keep], [names[j] for j in keep]


def weighted_summary(d: pd.DataFrame, outcome: str, weights: np.ndarray, analysis: str) -> list[dict]:
    rows = []
    for event, gidx in d.groupby("primary_event_type", sort=False).groups.items():
        ii = d.index.get_indexer(gidx)
        y = d.loc[gidx, outcome].to_numpy(dtype=float)
        w = weights[ii]
        ok = np.isfinite(y) & np.isfinite(w) & (w > 0)
        if not ok.any():
            continue
        estimate = float(np.sum(w[ok] * y[ok]) / np.sum(w[ok]))
        rows.append({
            "record_type": "weighted_estimate", "analysis": analysis,
            "event_type": event, "outcome": outcome, "estimate": estimate,
            "n_events": int(ok.sum()), "sum_weights": float(w[ok].sum()),
            "effective_sample_size": float(w[ok].sum() ** 2 / np.sum(w[ok] ** 2)),
        })
    return rows


def attrition_rows(episodes: pd.DataFrame) -> list[dict]:
    d = episodes.loc[
        episodes.primary_trajectory_eligible
        & episodes.age_at_event.notna()
        & episodes.sex.isin(["male", "female"])
    ].copy().reset_index(drop=True)
    rows = []
    official_available = d.official_event_weight.notna().mean()
    rows.append({
        "record_type": "official_weight_audit", "analysis": "event_wave_weight",
        "n_events": len(d), "observed_proportion": official_available,
        "notes": (
            "Wave-specific HRS respondent weight is available but does not directly match the longitudinal event-follow-up estimand."
            if d.cohort.iloc[0] == "HRS" else
            "No harmonized event-specific longitudinal weight was present in the reused UKHLS panel."
        ),
    })
    for horizon, outcome in [("year2", "persistence_2y_z"), ("year4", "persistence_4y_z")]:
        observed = d[outcome].notna().to_numpy(dtype=float)
        x, names = attrition_design(d)
        beta, converged = logistic_fit(x, observed)
        phat = np.clip(expit(x @ beta), .01, .99)
        numerator = float(np.mean(observed))
        w = np.where(observed == 1, numerator / phat, np.nan)
        raw = w[np.isfinite(w)]
        lo, hi = np.quantile(raw, [.01, .99]) if len(raw) else (np.nan, np.nan)
        trunc = np.where(np.isfinite(w), np.clip(w, lo, hi), np.nan)
        rows.append({
            "record_type": "attrition_model", "analysis": horizon,
            "n_events": len(d), "n_persons": int(d.person_id.nunique()),
            "observed_events": int(observed.sum()), "observed_proportion": numerator,
            "model_converged": converged, "n_predictors": len(names),
            "weight_min": float(np.nanmin(trunc)), "weight_p01": float(lo),
            "weight_median": float(np.nanmedian(trunc)), "weight_p99": float(hi),
            "weight_max": float(np.nanmax(trunc)),
            "untruncated_weight_min": float(np.nanmin(raw)),
            "untruncated_weight_p01": float(np.nanquantile(raw, .01)),
            "untruncated_weight_median": float(np.nanmedian(raw)),
            "untruncated_weight_p99": float(np.nanquantile(raw, .99)),
            "untruncated_weight_max": float(np.nanmax(raw)),
            "untruncated_weight_mean": float(np.nanmean(raw)),
            "untruncated_weight_sd": float(np.nanstd(raw, ddof=1)),
            "untruncated_weight_sum": float(np.nansum(raw)),
            "truncated_weight_mean": float(np.nanmean(trunc)),
            "truncated_weight_sd": float(np.nanstd(trunc, ddof=1)),
            "truncated_weight_sum": float(np.nansum(trunc)),
            "effective_sample_size": float(np.nansum(trunc) ** 2 / np.nansum(trunc**2)),
        })
        rows += weighted_summary(d, outcome, np.ones(len(d)), f"{horizon}_unweighted")
        rows += weighted_summary(d, outcome, trunc, f"{horizon}_IPOW_truncated_1_99")
        complete = d[["outcome_pre1_raw", "outcome_event_raw", "outcome_year2_raw", "outcome_year4_raw"]].notna().all(axis=1).to_numpy()
        rows += weighted_summary(d, outcome, np.where(complete, 1.0, np.nan), f"{horizon}_complete_four_points")
    return rows


def sample_flow_rows(episodes_by_cohort: dict[str, pd.DataFrame]) -> list[dict]:
    rows = []
    for cohort, d in episodes_by_cohort.items():
        stages = [
            ("all_collapsed_candidate_episodes", pd.Series(True, index=d.index)),
            ("any_primary_episode", d.n_primary_labels.gt(0)),
            ("isolated_primary_episode", d.isolated_primary),
            ("valid_pre1_and_event_self_report", d.primary_trajectory_eligible),
            ("valid_age_and_binary_sex", d.primary_trajectory_eligible & d.age_at_event.notna() & d.sex.isin(["male", "female"])),
            ("has_pre2", d.primary_trajectory_eligible & d.outcome_pre2_raw.notna()),
            ("has_uncensored_year2", d.primary_trajectory_eligible & d.outcome_year2_raw.notna()),
            ("has_uncensored_year4", d.primary_trajectory_eligible & d.outcome_year4_raw.notna()),
            ("complete_four_timepoints", d.primary_trajectory_eligible & d[["outcome_pre1_raw", "outcome_event_raw", "outcome_year2_raw", "outcome_year4_raw"]].notna().all(axis=1)),
        ]
        for stage, mask in stages:
            x = d.loc[mask]
            rows.append({
                "cohort": cohort, "record_type": "sample_flow", "stage": stage,
                "event_type": "all", "event_role": "primary",
                "n_events": int(len(x)), "n_persons": int(x.person_id.nunique()),
                "n_long_observations": int(sum(x[f"outcome_{t}_raw"].notna().sum() for t in TIME_LEVELS)),
                "notes": "Frozen, collapsed episode counts.",
            })
        for event in PRIMARY:
            x = d.loc[d.primary_event_type.eq(event)]
            eligible = x.loc[x.primary_trajectory_eligible]
            rows.append({
                "cohort": cohort, "record_type": "event_family", "stage": "all_then_eligible",
                "event_type": event, "event_role": "primary",
                "n_events": int(len(x)), "n_persons": int(x.person_id.nunique()),
                "eligible_events": int(len(eligible)), "eligible_persons": int(eligible.person_id.nunique()),
                "pre2_complete": int(eligible.outcome_pre2_raw.notna().sum()),
                "year2_complete": int(eligible.outcome_year2_raw.notna().sum()),
                "year4_complete": int(eligible.outcome_year4_raw.notna().sum()),
            })
        timing_specs = [
            ("pre2_to_pre1_months", (pd.to_datetime(d.pre1_date) - pd.to_datetime(d.pre2_date)).dt.days / MONTH_DAYS),
            ("event_to_year2_months", (pd.to_datetime(d.year2_date) - pd.to_datetime(d.event_date)).dt.days / MONTH_DAYS),
            ("event_to_year4_months", (pd.to_datetime(d.year4_date) - pd.to_datetime(d.event_date)).dt.days / MONTH_DAYS),
        ]
        for label, values in timing_specs:
            values = values.loc[d.n_primary_labels.gt(0)].dropna()
            rows.append({
                "cohort": cohort, "record_type": "timing_qa", "stage": label,
                "event_type": "all", "event_role": "primary", "n_events": int(len(values)),
                "timing_min_months": float(values.min()) if len(values) else np.nan,
                "timing_p25_months": float(values.quantile(.25)) if len(values) else np.nan,
                "timing_median_months": float(values.median()) if len(values) else np.nan,
                "timing_p75_months": float(values.quantile(.75)) if len(values) else np.nan,
                "timing_max_months": float(values.max()) if len(values) else np.nan,
            })
        secondary_labels = sorted(set("|".join(d.secondary_event_labels.fillna("")).split("|")) - {""})
        for event in secondary_labels:
            x = d.loc[d.all_event_labels.str.split("|").apply(lambda z: event in z)]
            rows.append({
                "cohort": cohort, "record_type": "event_family", "stage": "secondary_or_sensitivity",
                "event_type": event, "event_role": "secondary_or_sensitivity",
                "n_events": int(len(x)), "n_persons": int(x.person_id.nunique()),
                "eligible_events": int((x.outcome_pre1_raw.notna() & x.outcome_event_raw.notna()).sum()),
            })
    return rows


def episode_count_rows(episodes_by_cohort: dict[str, pd.DataFrame]) -> list[dict]:
    rows = []
    for cohort, d in episodes_by_cohort.items():
        definitions = {
            "isolated_primary": d.isolated_primary,
            "compound_primary": d.compound_primary,
            "primary_with_any_secondary_concurrent": d.n_primary_labels.gt(0) & d.n_secondary_labels.gt(0),
            "year2_censored_by_later_primary": d.year2_censored_by_later_primary,
            "year4_censored_by_later_primary": d.year4_censored_by_later_primary,
        }
        for label, mask in definitions.items():
            x = d.loc[mask]
            rows.append({
                "cohort": cohort, "record_type": "episode_status", "category": label,
                "event_type": "all", "n_episodes": int(len(x)), "n_persons": int(x.person_id.nunique()),
            })
        for count, x in d.loc[d.n_primary_labels.gt(0)].groupby("preceding_primary_episode_count"):
            rows.append({
                "cohort": cohort, "record_type": "prior_primary_count", "category": str(int(count)),
                "event_type": "all", "n_episodes": int(len(x)), "n_persons": int(x.person_id.nunique()),
            })
        for event in PRIMARY:
            x = d.loc[d.primary_event_type.eq(event)]
            rows.append({
                "cohort": cohort, "record_type": "event_status", "category": "isolated_primary",
                "event_type": event, "n_episodes": int(len(x)), "n_persons": int(x.person_id.nunique()),
                "year2_censored": int(x.year2_censored_by_later_primary.sum()),
                "year4_censored": int(x.year4_censored_by_later_primary.sum()),
                "with_preceding_primary": int(x.preceding_primary_episode_count.gt(0).sum()),
            })
    return rows


def sensitivity_rows(episodes_by_cohort: dict[str, pd.DataFrame]) -> list[dict]:
    specs = [
        ("UKHLS", "compound_primary", "compound_primary", "primary_compound_single_episode"),
        ("UKHLS", "separation_divorce", "separation_divorce", "UKHLS_secondary_separation_divorce"),
        ("UKHLS", "financial_strain", "financial_strain", "UKHLS_secondary_financial_strain"),
        ("HRS", "health_chronic_onset", "health_chronic_onset", "HRS_health_incident_chronic"),
        ("HRS", "health_self_rated_decline", "health_self_rated_decline", "HRS_health_self_rated_decline"),
        ("HRS", "poverty_entry", "poverty_entry", "HRS_secondary_poverty_entry"),
        ("HRS", "retirement_comparator", "retirement_comparator", "HRS_neutral_retirement_comparator"),
    ]
    rows = []
    for cohort, token, output_event, analysis in specs:
        d = episodes_by_cohort[cohort]
        if token == "compound_primary":
            x = d.loc[d.compound_primary].copy()
        else:
            x = d.loc[d.all_event_labels.fillna("").str.split("|").apply(lambda z: token in z)].copy()
        x = x.loc[x.outcome_pre1_raw.notna() & x.outcome_event_raw.notna()]
        if x.empty:
            rows.append({"cohort": cohort, "analysis_sample": analysis, "event_type": output_event, "n_episodes": 0})
            continue
        metrics_map = {
            "acute_pre1_to_event": x.outcome_event_z - x.outcome_pre1_z,
            "persistence_pre1_to_year2": x.outcome_year2_z - x.outcome_pre1_z,
            "persistence_pre1_to_year4": x.outcome_year4_z - x.outcome_pre1_z,
        }
        for contrast, values in metrics_map.items():
            rows.append({
                "cohort": cohort, "record_type": "fixed_sensitivity", "analysis_sample": analysis,
                "event_type": output_event, "contrast": contrast,
                "estimate": float(values.mean()) if values.notna().any() else np.nan,
                "n_episodes": int(values.notna().sum()),
                "n_persons": int(x.loc[values.notna(), "person_id"].nunique()),
                "notes": "Descriptive frozen sensitivity; no hypothesis test.",
            })
    # Separate frozen no-censoring sensitivity, restoring raw outcomes captured
    # before subsequent-event censoring.
    for cohort, d in episodes_by_cohort.items():
        x = d.loc[d.primary_trajectory_eligible].copy()
        for event, g in x.groupby("primary_event_type", sort=False):
            for horizon in ["year2", "year4"]:
                val = (g[f"outcome_{horizon}_raw_uncensored"] - g.outcome_pre1_raw) / g.standardization_sd
                rows.append({
                    "cohort": cohort, "record_type": "fixed_sensitivity", "analysis_sample": "no_subsequent_event_censoring",
                    "event_type": event, "contrast": f"persistence_pre1_to_{horizon}",
                    "estimate": float(val.mean()), "n_episodes": int(val.notna().sum()),
                    "n_persons": int(g.loc[val.notna(), "person_id"].nunique()),
                    "notes": "Uses the presaved uncensored follow-up value; primary analysis remains censored.",
                })
        # Raw-scale and complete-four-point frozen sensitivities.
        complete = x[["outcome_pre1_raw", "outcome_event_raw", "outcome_year2_raw", "outcome_year4_raw"]].notna().all(axis=1)
        for event, g in x.groupby("primary_event_type", sort=False):
            raw_metrics = {
                "acute_pre1_to_event": g.outcome_event_raw - g.outcome_pre1_raw,
                "persistence_pre1_to_year2": g.outcome_year2_raw - g.outcome_pre1_raw,
                "persistence_pre1_to_year4": g.outcome_year4_raw - g.outcome_pre1_raw,
            }
            for contrast, values in raw_metrics.items():
                rows.append({
                    "cohort": cohort, "record_type": "fixed_sensitivity", "analysis_sample": "raw_outcome_scale",
                    "event_type": event, "contrast": contrast, "estimate": float(values.mean()),
                    "n_episodes": int(values.notna().sum()), "n_persons": int(g.loc[values.notna(), "person_id"].nunique()),
                    "notes": "UKHLS GHQ-12 raw units or HRS CES-D 8 raw units.",
                })
            gc = g.loc[complete.reindex(g.index).fillna(False)]
            complete_metrics = {
                "acute_pre1_to_event": gc.outcome_event_z - gc.outcome_pre1_z,
                "persistence_pre1_to_year2": gc.outcome_year2_z - gc.outcome_pre1_z,
                "persistence_pre1_to_year4": gc.outcome_year4_z - gc.outcome_pre1_z,
            }
            for contrast, values in complete_metrics.items():
                rows.append({
                    "cohort": cohort, "record_type": "fixed_sensitivity", "analysis_sample": "complete_four_timepoints",
                    "event_type": event, "contrast": contrast, "estimate": float(values.mean()) if len(values) else np.nan,
                    "n_episodes": int(values.notna().sum()), "n_persons": int(gc.loc[values.notna(), "person_id"].nunique()),
                    "notes": "Requires pre1, event, 2-year, and 4-year; pre2 remains optional.",
                })

        # All unique primary episodes, including one-row compound episodes,
        # adjusted for preceding-event count and any concurrent event label.
        all_ep = d.loc[
            d.n_primary_labels.gt(0) & d.outcome_pre1_raw.notna() & d.outcome_event_raw.notna()
            & d.event_self_report & d.pre1_self_report
        ].copy()
        all_ep["analysis_event"] = all_ep.primary_event_type
        levels = [e for e in PRIMARY + ["compound_primary"] if e in set(all_ep.analysis_event)]
        for contrast, values in {
            "acute_pre1_to_event": all_ep.outcome_event_z - all_ep.outcome_pre1_z,
            "persistence_pre1_to_year2": all_ep.outcome_year2_z - all_ep.outcome_pre1_z,
            "persistence_pre1_to_year4": all_ep.outcome_year4_z - all_ep.outcome_pre1_z,
        }.items():
            valid = values.notna()
            g = all_ep.loc[valid].copy()
            y = values.loc[valid].to_numpy(dtype=float)
            if len(g) == 0:
                continue
            names = ["Intercept"] + [f"event[{e}]" for e in levels if e != "widowhood"] + ["preceding_count", "concurrent_indicator"]
            cols = [np.ones(len(g))]
            cols += [g.analysis_event.eq(e).to_numpy(dtype=float) for e in levels if e != "widowhood"]
            cols += [g.preceding_primary_episode_count.to_numpy(dtype=float), g.n_all_labels.gt(1).to_numpy(dtype=float)]
            design = np.column_stack(cols)
            beta = np.linalg.pinv(design) @ y
            for event in levels:
                estimate = float(beta[0])
                name = f"event[{event}]"
                if name in names:
                    estimate += float(beta[names.index(name)])
                ge = g.loc[g.analysis_event.eq(event)]
                rows.append({
                    "cohort": cohort, "record_type": "fixed_sensitivity",
                    "analysis_sample": "all_unique_primary_episodes_preceding_concurrent_adjusted",
                    "event_type": event, "contrast": contrast, "estimate": estimate,
                    "n_episodes": int(len(ge)), "n_persons": int(ge.person_id.nunique()),
                    "notes": "One row per episode; adjusted to preceding count=0 and concurrent indicator=0.",
                })
    return rows


def heterogeneity_row(
    domain: str,
    contrast: str,
    uk: dict,
    hrs: dict,
    uk_boot: np.ndarray,
    hrs_boot: np.ndarray,
) -> dict:
    uk_se = float(np.std(uk_boot, ddof=1))
    hrs_se = float(np.std(hrs_boot, ddof=1))
    variances = np.array([uk_se**2, hrs_se**2])
    estimates = np.array([uk["estimate"], hrs["estimate"]])
    weights = 1 / variances
    pooled = float(np.sum(weights * estimates) / np.sum(weights))
    q = float(np.sum(weights * (estimates - pooled) ** 2))
    i2 = max(0.0, (q - 1) / q) if q > 0 else 0.0
    diff = float(uk["estimate"] - hrs["estimate"])
    diff_boot = np.asarray(uk_boot) - np.asarray(hrs_boot)
    return {
        "record_type": "trajectory_contrast", "domain": domain, "contrast": contrast,
        "UKHLS_estimate": uk["estimate"], "UKHLS_ci_lower": uk["ci_lower"], "UKHLS_ci_upper": uk["ci_upper"],
        "HRS_estimate": hrs["estimate"], "HRS_ci_lower": hrs["ci_lower"], "HRS_ci_upper": hrs["ci_upper"],
        "direction_agreement": bool(np.sign(uk["estimate"]) == np.sign(hrs["estimate"])),
        "UKHLS_minus_HRS": diff,
        "difference_ci_lower": float(np.quantile(diff_boot, .025)),
        "difference_ci_upper": float(np.quantile(diff_boot, .975)),
        "difference_bootstrap_success": int(len(diff_boot)), "Q": q, "I2": i2,
        "replication_type": "harmonized_domain" if domain == "widowhood" else "conceptual",
    }


def cross_cohort_rows(
    traj: pd.DataFrame,
    pred: pd.DataFrame,
    cross: pd.DataFrame,
    trajectory_boot_values: dict[tuple[str, str, str], np.ndarray],
) -> list[dict]:
    rows = []
    t = traj.loc[traj.record_type.eq("contrast") & traj.analysis_sample.eq("primary_isolated")]
    for domain in PRIMARY:
        for contrast in ["acute_pre1_to_event", "persistence_pre1_to_year2", "persistence_pre1_to_year4"]:
            uk = t.loc[t.cohort.eq("UKHLS") & t.event_type.eq(domain) & t.contrast.eq(contrast)]
            hrs = t.loc[t.cohort.eq("HRS") & t.event_type.eq(domain) & t.contrast.eq(contrast)]
            if len(uk) == 1 and len(hrs) == 1:
                rows.append(heterogeneity_row(
                    domain, contrast, uk.iloc[0].to_dict(), hrs.iloc[0].to_dict(),
                    trajectory_boot_values[("UKHLS", domain, contrast)],
                    trajectory_boot_values[("HRS", domain, contrast)],
                ))
    for validation in ["grouped_10fold", "temporal_70_30"]:
        for metric_name in ["RMSE", "predictive_R2"]:
            rec = {"record_type": "model_gain", "domain": "all_primary", "contrast": f"E_vs_U_{validation}_{metric_name}"}
            for cohort in ["UKHLS", "HRS"]:
                x = pred.loc[
                    pred.cohort.eq(cohort) & pred.validation.eq(validation)
                    & pred.record_type.eq("comparison") & pred.metric.eq(metric_name)
                ]
                if not x.empty:
                    rec[f"{cohort}_estimate"] = float(x.iloc[0].improvement)
                    rec[f"{cohort}_ci_lower"] = float(x.iloc[0].ci_lower)
                    rec[f"{cohort}_ci_upper"] = float(x.iloc[0].ci_upper)
            if "UKHLS_estimate" in rec and "HRS_estimate" in rec:
                rec["direction_agreement"] = bool(np.sign(rec["UKHLS_estimate"]) == np.sign(rec["HRS_estimate"]))
            rows.append(rec)
    for cohort_type in ["two_event", "three_event"]:
        for outcome in ["acute", "persistence"]:
            for validation in ["grouped_10fold", "temporal_70_30"]:
                rec = {"record_type": "prior_response_gain", "domain": "all_primary", "contrast": f"P1_vs_P0_{cohort_type}_{outcome}_{validation}"}
                for cohort in ["UKHLS", "HRS"]:
                    x = cross.loc[
                        cross.cohort.eq(cohort) & cross.cohort_type.eq(cohort_type)
                        & cross.validation.eq(validation) & cross.outcome_metric.eq(outcome)
                        & cross.record_type.eq("comparison") & cross.comparison.eq("P1_minus_P0")
                        & cross.metric.eq("predictive_R2")
                    ]
                    if not x.empty:
                        rec[f"{cohort}_estimate"] = float(x.iloc[0].improvement)
                        rec[f"{cohort}_ci_lower"] = float(x.iloc[0].ci_lower)
                        rec[f"{cohort}_ci_upper"] = float(x.iloc[0].ci_upper)
                if "UKHLS_estimate" in rec and "HRS_estimate" in rec:
                    rec["direction_agreement"] = bool(np.sign(rec["UKHLS_estimate"]) == np.sign(rec["HRS_estimate"]))
                rows.append(rec)
    return rows


def fmt(x: object) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)) or pd.isna(x):
        return "NA"
    if isinstance(x, (int, np.integer)):
        return f"{int(x):,}"
    if isinstance(x, (float, np.floating)):
        return f"{float(x):.4f}"
    return str(x)


def md_table(df: pd.DataFrame, columns: list[str], max_rows: int | None = None) -> str:
    x = df.loc[:, [c for c in columns if c in df.columns]].copy()
    if max_rows is not None:
        x = x.head(max_rows)
    cols = list(x.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in x.iterrows():
        lines.append("| " + " | ".join(fmt(row[c]).replace("|", "\\|") for c in cols) + " |")
    return "\n".join(lines)


def write_md(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def provenance_document(episodes_by_cohort: dict[str, pd.DataFrame], pred_samples: pd.DataFrame, qa: list[dict]) -> None:
    lines = [
        "# NMH derived-data documentation", "",
        "## Provenance", "",
        f"- Frozen plan: `results/NMH_confirmatory/00_NMH_confirmatory_freeze.md` (SHA-256 `{sha256(OUT / '00_NMH_confirmatory_freeze.md')}`).",
        f"- Script: `scripts/run_nmh_confirmatory_rebuild.py` (SHA-256 `{sha256(Path(__file__))}`).",
        f"- UKHLS sources: `{UK_EVENTS.relative_to(ROOT)}` and `{UK_PANEL.relative_to(ROOT)}`.",
        f"- HRS sources: `{HRS_CORE.relative_to(ROOT)}` and `{HRS_FAMILY.relative_to(ROOT)}`; only prespecified columns were read.",
        f"- Generated: {pd.Timestamp.now(tz='Asia/Shanghai').isoformat()}.",
        f"- Runtime: Python {platform.python_version()}, pandas {pd.__version__}, NumPy {np.__version__}.",
        "", "## Unique-key and row-count QA", "", md_table(pd.DataFrame(qa), list(pd.DataFrame(qa).columns)), "",
    ]
    for cohort, d in episodes_by_cohort.items():
        lines += [f"## {cohort} episode data dictionary", "", md_table(dictionary_for(d, cohort, "See provenance above"), ["variable", "dtype", "description"]), ""]
    if not pred_samples.empty:
        lines += ["## Cross-event prediction sample data dictionary", "", md_table(dictionary_for(pred_samples, "UKHLS/HRS", "Derived from the two episode datasets"), ["variable", "dtype", "description"]), ""]
    write_md(DP / "NMH_derived_data_documentation.md", "\n".join(lines))


def build_data_stage() -> tuple[dict[str, pd.DataFrame], list[dict]]:
    OUT.mkdir(parents=True, exist_ok=True)
    DP.mkdir(parents=True, exist_ok=True)
    uk = build_ukhls()
    hrs = build_hrs()
    qa = [qa_episode_data(uk, "UKHLS"), qa_episode_data(hrs, "HRS")]
    uk.to_parquet(UK_OUT, index=False)
    hrs.to_parquet(HRS_OUT, index=False)
    note(f"Saved {UK_OUT.name} ({len(uk):,} rows) and {HRS_OUT.name} ({len(hrs):,} rows)")
    return {"UKHLS": uk, "HRS": hrs}, qa


def run_analysis(episodes_by_cohort: dict[str, pd.DataFrame], qa: list[dict]) -> None:
    flow = pd.DataFrame(sample_flow_rows(episodes_by_cohort))
    episode_counts = pd.DataFrame(episode_count_rows(episodes_by_cohort))
    sensitivity = pd.DataFrame(sensitivity_rows(episodes_by_cohort))
    model_rows: list[dict] = []
    trajectory_rows: list[dict] = []
    prediction_rows: list[dict] = []
    validation_bundles: list[pd.DataFrame] = []
    attrition_all: list[pd.DataFrame] = []
    trajectory_boot_values: dict[tuple[str, str, str], np.ndarray] = {}

    for cohort, episodes in episodes_by_cohort.items():
        note(f"Fitting frozen U/E models for {cohort}")
        long = make_long(episodes)
        if not set(PRIMARY).issubset(set(long.primary_event_type)):
            raise AssertionError(f"{cohort}: one or more common primary domains absent")
        xu, names_u, centers_u = fixed_design(long, "U")
        xe, names_e, centers_e = fixed_design(long, "E")
        fit_u = NestedRandomInterceptML(
            long.outcome_z.to_numpy(), xu, long.person_id.to_numpy(), long.episode_id.to_numpy()
        ).fit(names_u)
        fit_e = NestedRandomInterceptML(
            long.outcome_z.to_numpy(), xe, long.person_id.to_numpy(), long.episode_id.to_numpy()
        ).fit(names_e)
        model_rows.append(model_record(
            cohort, "U",
            "z_outcome ~ time_point + event_type + calendar_time + baseline covariates + (1|person) + (1|episode)",
            long, fit_u,
        ))
        model_rows.append(model_record(
            cohort, "E",
            "z_outcome ~ time_point * event_type + calendar_time + baseline covariates + (1|person) + (1|episode)",
            long, fit_e,
        ))
        model_rows.append(comparison_record(cohort, fit_u, fit_e))
        for model_name, fitted in [("U", fit_u), ("E", fit_e)]:
            for name, estimate in zip(fitted.names, fitted.beta):
                model_rows.append({
                    "cohort": cohort, "record_type": "fixed_effect", "model": model_name,
                    "coefficient": name, "estimate": float(estimate),
                    "n_episodes": int(long.episode_id.nunique()),
                    "n_persons": int(long.person_id.nunique()),
                    "n_observations": int(len(long)), "converged": fitted.converged,
                })

        note(f"Computing {BOOT_SUCCESS} successful person-cluster bootstrap fits for {cohort}")
        boot_beta, attempts = bootstrap_fixed_effects(
            long, xe, long.outcome_z.to_numpy(), fit_e, BOOT_SUCCESS,
            SEED + (0 if cohort == "UKHLS" else 1000),
        )
        recs = trajectory_contrasts(fit_e)
        for rec in recs:
            if rec["record_type"] == "contrast":
                trajectory_boot_values[(cohort, rec["event_type"], rec["contrast"])] = boot_beta @ rec["contrast_vector"]
        add_bootstrap_cis(recs, boot_beta)
        for row in recs:
            row.update({
                "cohort": cohort, "analysis_sample": "primary_isolated", "outcome_scale": "cohort_pre1_SD",
                "n_episodes": int(long.episode_id.nunique()), "n_persons": int(long.person_id.nunique()),
                "n_observations": int(len(long)), "bootstrap_attempts": attempts, "model": "E",
            })
        trajectory_rows += recs
        # Fixed-effect coefficient ledger, with the same cluster-bootstrap CIs.
        for j, name in enumerate(names_e):
            values = boot_beta[:, j]
            trajectory_rows.append({
                "cohort": cohort, "analysis_sample": "primary_isolated", "outcome_scale": "cohort_pre1_SD",
                "record_type": "fixed_effect", "event_type": "", "time_point": "", "contrast": name,
                "estimate": float(fit_e.beta[j]), "ci_lower": float(np.quantile(values, .025)),
                "ci_upper": float(np.quantile(values, .975)), "n_episodes": int(long.episode_id.nunique()),
                "n_persons": int(long.person_id.nunique()), "n_observations": int(len(long)),
                "bootstrap_success": BOOT_SUCCESS, "bootstrap_attempts": attempts, "model": "E",
            })

        note(f"Running frozen grouped and temporal trajectory validation for {cohort}")
        cohort_bundles = []
        for validation in ["grouped_10fold", "temporal_70_30"]:
            b = validation_bundle(long, episodes, validation, SEED + (0 if cohort == "UKHLS" else 2000))
            b["cohort"] = cohort
            cohort_bundles.append(b)
        bundle = pd.concat(cohort_bundles, ignore_index=True)
        validation_bundles.append(bundle)
        prediction_rows += prediction_metric_rows(cohort, bundle)

        a = pd.DataFrame(attrition_rows(episodes))
        a["cohort"] = cohort
        attrition_all.append(a)

    trajectory = pd.concat([pd.DataFrame(trajectory_rows), sensitivity], ignore_index=True, sort=False)
    models = pd.DataFrame(model_rows)
    predictions = pd.DataFrame(prediction_rows)
    attrition = pd.concat(attrition_all, ignore_index=True, sort=False)

    note("Constructing frozen two-event and three-event prospective samples")
    pred_samples = pd.concat(
        [build_cross_event_samples(d) for d in episodes_by_cohort.values()],
        ignore_index=True, sort=False,
    )
    pred_samples.to_parquet(PRED_OUT, index=False)
    pred_qa = {
        "cohort": "UKHLS/HRS cross-event",
        "rows": int(len(pred_samples)),
        "unique_episode_id": int(pred_samples.sample_id.nunique()),
        "unique_persons": int(pred_samples[["cohort", "person_id"]].drop_duplicates().shape[0]),
        "unique_person_wave": int(pred_samples[["cohort", "cohort_type", "person_id"]].drop_duplicates().shape[0]),
        "primary_episodes": int(pred_samples.cohort_type.eq("two_event").sum()),
        "isolated_primary_episodes": int(pred_samples.cohort_type.eq("three_event").sum()),
        "compound_primary_episodes": 0,
        "eligible_primary_episodes": int(len(pred_samples)),
    }
    provenance_document(episodes_by_cohort, pred_samples, qa + [pred_qa])

    cross_bundles = []
    for (cohort, cohort_type), d in pred_samples.groupby(["cohort", "cohort_type"], sort=False):
        note(f"Cross-adversity prediction: {cohort} {cohort_type} (n={len(d):,})")
        for metric_name in ["acute", "persistence"]:
            for validation in ["grouped_10fold", "temporal_70_30"]:
                b = cross_prediction_bundle(
                    d, metric_name, validation, episodes_by_cohort[cohort]
                )
                if not b.empty:
                    cross_bundles.append(b)
    cross_bundle = pd.concat(cross_bundles, ignore_index=True, sort=False) if cross_bundles else pd.DataFrame()
    cross_metrics = pd.DataFrame(cross_metric_rows(cross_bundle))
    two = cross_metrics.loc[cross_metrics.cohort_type.eq("two_event")].copy() if not cross_metrics.empty else pd.DataFrame()
    three = cross_metrics.loc[cross_metrics.cohort_type.eq("three_event")].copy() if not cross_metrics.empty else pd.DataFrame()
    cross_cohort = pd.DataFrame(cross_cohort_rows(
        trajectory, predictions, cross_metrics, trajectory_boot_values
    ))

    # Nine listed result CSVs; the freeze is Markdown-only.
    flow.to_csv(OUT / "01_event_harmonization_and_sample_flow.csv", index=False)
    episode_counts.to_csv(OUT / "02_event_episode_counts.csv", index=False)
    trajectory.to_csv(OUT / "03_trajectory_estimates.csv", index=False)
    models.to_csv(OUT / "04_corrected_model_comparison.csv", index=False)
    predictions.to_csv(OUT / "05_model_prediction_metrics.csv", index=False)
    two.to_csv(OUT / "06_two_event_prediction.csv", index=False)
    three.to_csv(OUT / "06_three_event_prediction.csv", index=False)
    attrition.to_csv(OUT / "07_weighting_results.csv", index=False)
    cross_cohort.to_csv(OUT / "08_cross_cohort_results.csv", index=False)

    write_reports(
        flow, episode_counts, trajectory, models, predictions, two, three,
        attrition, cross_cohort, episodes_by_cohort, qa + [pred_qa],
    )
    note("All frozen confirmatory outputs written")


def write_reports(
    flow: pd.DataFrame,
    episode_counts: pd.DataFrame,
    trajectory: pd.DataFrame,
    models: pd.DataFrame,
    predictions: pd.DataFrame,
    two: pd.DataFrame,
    three: pd.DataFrame,
    attrition: pd.DataFrame,
    cross_cohort: pd.DataFrame,
    episodes_by_cohort: dict[str, pd.DataFrame],
    qa: list[dict],
) -> None:
    write_md(OUT / "01_event_harmonization_and_sample_flow.md", "\n".join([
        "# Event harmonization and sample flow", "",
        "All counts use one collapsed person-transition episode. The primary domains are widowhood, unemployment, caregiving, and health. Separation/divorce, financial strain, poverty entry, and the retirement comparator remain outside the primary sample.", "",
        "## Derived-data QA", "", md_table(pd.DataFrame(qa), list(pd.DataFrame(qa).columns)), "",
        "## Primary sample flow", "", md_table(flow.loc[flow.record_type.eq("sample_flow")], ["cohort", "stage", "n_events", "n_persons", "n_long_observations"]), "",
        "## Event-family availability", "", md_table(flow.loc[flow.record_type.eq("event_family")], ["cohort", "event_type", "event_role", "n_events", "n_persons", "eligible_events", "eligible_persons", "pre2_complete", "year2_complete", "year4_complete"]), "",
        "## Actual timing QA", "", md_table(flow.loc[flow.record_type.eq("timing_qa")], ["cohort", "stage", "n_events", "timing_min_months", "timing_p25_months", "timing_median_months", "timing_p75_months", "timing_max_months"]), "",
        "## Harmonization record", "",
        "- UKHLS outcome: adult self-report GHQ-12 Likert 0-36.",
        "- HRS outcome: RAND harmonized CES-D 8-item score 0-8, self-report only.",
        "- HRS caregiving: parental or parent-in-law personal-care onset only; HRS unemployment: valid non-unemployed labour-force state to unemployed; HRS primary health: ADL 0 to at least 1 with next-wave persistence.",
        "- Outcome standardization: within cohort, one immediate pre-event observation per unique primary episode; no wave-specific standardization.",
        "- Follow-ups: closest eligible interview in 18-30 months and 42-54 months; no observations outside those windows.",
        "- Full row-level counts and provenance are in the paired CSV and `data_processed/NMH_derived_data_documentation.md`.",
    ]))

    write_md(OUT / "02_event_episode_and_contamination_audit.md", "\n".join([
        "# Event episode and contamination audit", "",
        "Simultaneous primary labels were collapsed to one episode. The primary analysis uses isolated primary episodes. A subsequent primary episode censors any index-episode follow-up at or after its transition date.", "",
        "## Episode status", "", md_table(episode_counts.loc[episode_counts.record_type.eq("episode_status")], ["cohort", "category", "n_episodes", "n_persons"]), "",
        "## Event-specific censoring and history", "", md_table(episode_counts.loc[episode_counts.record_type.eq("event_status")], ["cohort", "event_type", "n_episodes", "n_persons", "year2_censored", "year4_censored", "with_preceding_primary"]), "",
        "## Number of preceding primary episodes", "", md_table(episode_counts.loc[episode_counts.record_type.eq("prior_primary_count")], ["cohort", "category", "n_episodes", "n_persons"]), "",
        "No duplicated outcome rows were created for compound episodes.",
    ]))

    primary_traj = trajectory.loc[trajectory.analysis_sample.eq("primary_isolated") & trajectory.record_type.isin(["estimated_mean", "contrast"])]
    sensitivity = trajectory.loc[trajectory.record_type.eq("fixed_sensitivity")]
    write_md(OUT / "03_pretrend_and_harmonized_trajectory_analysis.md", "\n".join([
        "# Pretrend and harmonized trajectory analysis", "",
        "Outcome values are cohort-standardized using the frozen immediate pre-event reference distribution. Estimates are from model E with person and episode random intercepts. Intervals use 500 successful person-cluster bootstrap resamples; variance components were fixed at the primary ML estimate while fixed effects were re-estimated within each resample.", "",
        "## Estimated trajectories", "", md_table(primary_traj.loc[primary_traj.record_type.eq("estimated_mean")], ["cohort", "event_type", "time_point", "estimate", "ci_lower", "ci_upper", "n_episodes", "n_persons"]), "",
        "## Pretrend, acute response, recovery, and persistence contrasts", "", md_table(primary_traj.loc[primary_traj.record_type.eq("contrast")], ["cohort", "event_type", "contrast", "estimate", "ci_lower", "ci_upper", "bootstrap_success"]), "",
        "## Prespecified sensitivities", "", md_table(sensitivity, ["cohort", "analysis_sample", "event_type", "contrast", "estimate", "n_episodes", "n_persons", "notes"]), "",
        "No additional event family, outcome transformation, or post-result model change was introduced.",
    ]))

    write_md(OUT / "04_corrected_competitive_models.md", "\n".join([
        "# Corrected competitive trajectory models", "",
        "Both models were fitted by Gaussian maximum likelihood on identical observed rows. U shares one trajectory across event types; E includes the frozen time-by-event interaction. Both include calendar time, age, sex, education, cohort-specific socioeconomic status, and person/episode random intercepts.", "",
        "## Model results", "", md_table(models.loc[models.record_type.eq("model")], ["cohort", "model", "n_episodes", "n_persons", "n_observations", "log_likelihood", "AIC", "BIC", "marginal_R2", "conditional_R2", "person_variance", "episode_variance", "residual_variance", "converged"]), "",
        "## Frozen comparison", "", md_table(models.loc[models.record_type.eq("comparison")], ["cohort", "model", "lrt_statistic", "lrt_df", "lrt_p_value", "AIC", "BIC", "marginal_R2", "conditional_R2"]), "",
        "## Fixed-effect coefficients", "", md_table(models.loc[models.record_type.eq("fixed_effect")], ["cohort", "model", "coefficient", "estimate", "n_episodes", "n_persons", "n_observations", "converged"]), "",
        "`AIC` and `BIC` in comparison rows are E minus U; `marginal_R2` and `conditional_R2` are E minus U.",
    ]))

    write_md(OUT / "05_out_of_sample_trajectory_prediction.md", "\n".join([
        "# Out-of-sample trajectory prediction", "",
        "Validation used person-grouped 10-fold splits and an early-70%/late-30% temporal split with late-period people excluded from training. Cohort outcome standardization was recalculated from training persons. Positive U-minus-E RMSE/MAE and positive E-minus-U predictive R2/log density indicate better E prediction.", "",
        "## Predictive metrics", "", md_table(predictions.loc[predictions.record_type.eq("model_metric")], ["cohort", "validation", "model", "n_train_persons", "n_train_episodes", "n_persons", "n_episodes", "n_observations", "training_event_cutoff", "n_temporal_overlap_episodes_excluded", "RMSE", "MAE", "predictive_R2", "mean_log_predictive_density", "calibration_intercept", "calibration_slope"]), "",
        "## E versus U bootstrap improvements", "", md_table(predictions.loc[predictions.record_type.eq("comparison")], ["cohort", "validation", "metric", "comparison", "improvement", "ci_lower", "ci_upper", "relative_improvement", "relative_ci_lower", "relative_ci_upper", "bootstrap_success"]), "",
    ]))

    for name, frame in [("two-event", two), ("three-event", three)]:
        if frame.empty:
            table = "No qualifying sample under the frozen prospective timing rules."
        else:
            table = md_table(frame, ["cohort", "validation", "outcome_metric", "record_type", "model", "comparison", "metric", "n_persons", "RMSE", "MAE", "predictive_R2", "improvement", "ci_lower", "ci_upper"])
        if name == "two-event":
            two_text = table
        else:
            three_text = table
    write_md(OUT / "06_cross_adversity_response_prediction.md", "\n".join([
        "# Cross-adversity response prediction", "",
        "Two-event samples use the earliest complete primary episode and the first later, different-family episode occurring after the earlier 2-year outcome. Three-event samples require three distinct families in a prospectively separated sequence; the first two responses form the prior signature and the third is the target.", "",
        "P0 contains target baseline/event characteristics. P1 adds the prior response. P2 adds prior event type and prior-response-by-target-type terms. Three-event rows also report empirical-Bayes-shrunk prior-response variants when estimable. All comparisons use person-grouped or temporal out-of-sample predictions.", "",
        "## Two-event prediction", "", two_text, "",
        "## Three-event prediction", "", three_text, "",
        "Positive RMSE/MAE improvements are base-minus-candidate; positive predictive-R2/log-density improvements are candidate-minus-base.",
    ]))

    write_md(OUT / "07_attrition_and_weighting_sensitivity.md", "\n".join([
        "# Attrition and weighting sensitivity", "",
        "Separate frozen observation models were fitted for 2-year and 4-year availability. Stabilized inverse-probability-of-observation weights were truncated at the 1st and 99th percentiles. Official wave weights were audited separately and were not treated as event-follow-up weights when they did not match the estimand.", "",
        "## Availability and weight diagnostics", "", md_table(attrition.loc[attrition.record_type.isin(["official_weight_audit", "attrition_model"])], ["cohort", "record_type", "analysis", "n_events", "n_persons", "observed_events", "observed_proportion", "model_converged", "untruncated_weight_min", "untruncated_weight_p01", "untruncated_weight_median", "untruncated_weight_p99", "untruncated_weight_max", "untruncated_weight_mean", "untruncated_weight_sd", "untruncated_weight_sum", "weight_min", "weight_median", "weight_max", "truncated_weight_mean", "truncated_weight_sd", "truncated_weight_sum", "effective_sample_size", "notes"]), "",
        "## Unweighted, IPOW, and complete-case estimates", "", md_table(attrition.loc[attrition.record_type.eq("weighted_estimate")], ["cohort", "analysis", "event_type", "outcome", "estimate", "n_events", "effective_sample_size"]), "",
    ]))

    write_md(OUT / "08_cross_cohort_replication.md", "\n".join([
        "# Cross-cohort replication", "",
        "Comparisons use within-cohort standardized outcomes and the four frozen common domains. HRS unemployment, caregiving, and persistent-ADL health are conceptual replications because their operational definitions are not identical to UKHLS; widowhood is the harmonized-domain comparison.", "",
        "## Trajectory contrast agreement and heterogeneity", "", md_table(cross_cohort.loc[cross_cohort.record_type.eq("trajectory_contrast")], ["domain", "contrast", "UKHLS_estimate", "UKHLS_ci_lower", "UKHLS_ci_upper", "HRS_estimate", "HRS_ci_lower", "HRS_ci_upper", "direction_agreement", "UKHLS_minus_HRS", "difference_ci_lower", "difference_ci_upper", "difference_bootstrap_success", "Q", "I2", "replication_type"]), "",
        "## Model and prior-response predictive gains", "", md_table(cross_cohort.loc[cross_cohort.record_type.ne("trajectory_contrast")], ["record_type", "contrast", "UKHLS_estimate", "UKHLS_ci_lower", "UKHLS_ci_upper", "HRS_estimate", "HRS_ci_lower", "HRS_ci_upper", "direction_agreement"]), "",
        "No pooled stacked model was fitted because it was frozen as optional and unnecessary for the requested cohort-specific adjudication.",
    ]))

    summary_flow = flow.loc[flow.stage.eq("valid_age_and_binary_sex"), ["cohort", "n_events", "n_persons", "n_long_observations"]]
    model_comp = models.loc[models.record_type.eq("comparison"), ["cohort", "lrt_statistic", "lrt_df", "AIC", "BIC", "marginal_R2", "conditional_R2", "converged"]]
    pred_comp = predictions.loc[predictions.record_type.eq("comparison") & predictions.metric.isin(["RMSE", "predictive_R2"])]
    write_md(OUT / "09_NMH_confirmatory_summary.md", "\n".join([
        "# NMH confirmatory rebuild summary", "",
        "## Analysis sample", "", md_table(summary_flow, list(summary_flow.columns)), "",
        "## Corrected in-sample model comparison", "", md_table(model_comp, list(model_comp.columns)), "",
        "## Out-of-sample E versus U", "", md_table(pred_comp, ["cohort", "validation", "metric", "comparison", "improvement", "ci_lower", "ci_upper"]), "",
        "## Cross-adversity prospective sample sizes", "", md_table(pd.DataFrame({
            "sample": ["UKHLS two-event", "UKHLS three-event", "HRS two-event", "HRS three-event"],
            "n_persons": [
                int(two.loc[two.cohort.eq("UKHLS"), "n_persons"].max()) if not two.loc[two.cohort.eq("UKHLS")].empty else 0,
                int(three.loc[three.cohort.eq("UKHLS"), "n_persons"].max()) if not three.loc[three.cohort.eq("UKHLS")].empty else 0,
                int(two.loc[two.cohort.eq("HRS"), "n_persons"].max()) if not two.loc[two.cohort.eq("HRS")].empty else 0,
                int(three.loc[three.cohort.eq("HRS"), "n_persons"].max()) if not three.loc[three.cohort.eq("HRS")].empty else 0,
            ],
        }), ["sample", "n_persons"]), "",
        "## Cross-adversity predictive increments", "", md_table(
            pd.concat([two, three], ignore_index=True, sort=False).loc[
                lambda z: z.record_type.eq("comparison")
                & z.comparison.isin(["P1_minus_P0", "P2_minus_P1"])
                & z.metric.isin(["RMSE", "predictive_R2"])
            ],
            ["cohort", "cohort_type", "validation", "outcome_metric", "comparison", "metric", "improvement", "ci_lower", "ci_upper", "bootstrap_success"],
        ), "",
        "## Frozen sensitivity inventory", "", md_table(
            trajectory.loc[trajectory.record_type.eq("fixed_sensitivity")]
            .groupby(["cohort", "analysis_sample"], as_index=False)
            .agg(n_rows=("estimate", "size"), n_estimable=("estimate", "count")),
            ["cohort", "analysis_sample", "n_rows", "n_estimable"],
        ), "",
        "## Output integrity", "",
        f"- Freeze SHA-256: `{sha256(OUT / '00_NMH_confirmatory_freeze.md')}`.",
        f"- UKHLS episode rows: {len(episodes_by_cohort['UKHLS']):,}.",
        f"- HRS episode rows: {len(episodes_by_cohort['HRS']):,}.",
        "- Bootstrap target: 500 successful person-cluster resamples for all reported primary trajectory intervals and predictive improvements.",
        "- All secondary definitions and comparators remain separated from the primary four-domain analysis.",
    ]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["build", "analysis", "all"], default="all")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage in {"build", "all"}:
        episodes_by_cohort, qa = build_data_stage()
    else:
        episodes_by_cohort = {
            "UKHLS": pd.read_parquet(UK_OUT),
            "HRS": pd.read_parquet(HRS_OUT),
        }
        qa = [qa_episode_data(d, c) for c, d in episodes_by_cohort.items()]
    if args.stage in {"analysis", "all"}:
        run_analysis(episodes_by_cohort, qa)


if __name__ == "__main__":
    main()
