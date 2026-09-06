#!/usr/bin/env python3
"""Validate longitudinal identity and construct linkage panels for UKHLS Waves 1-15.

The person panel uses xwaveid for complete issue/outcome information and merges
actual interview dates plus relationship pointers from each indall file. Household
continuity is inferred only from shared pidp members; wave-specific hidp values are
never treated as longitudinal identifiers.
"""

from __future__ import annotations

import gzip
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
UKHLS = ROOT / "raw_data" / "UKDA-6614-stata" / "stata" / "stata14_se" / "ukhls"
TABLES = ROOT / "tables"
PROCESSED = ROOT / "data_processed"
PREFIXES = list("abcdefghijklmno")
WAVE_OF = {prefix: i for i, prefix in enumerate(PREFIXES, start=1)}
ADULT_SELF_CODES = {1, 3}
ADULT_PROXY_CODES = {2}
YOUTH_CODES = {21}


def positive(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").gt(0)


def write_table(df: pd.DataFrame, name: str) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLES / name, index=False, encoding="utf-8-sig")


def write_processed_gzip(df: pd.DataFrame, name: str) -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED / name, index=False, compression="gzip", encoding="utf-8")


def load_person_wave_from_xwaveid() -> pd.DataFrame:
    stable = ["pidp", "sex", "birthy", "hhorig", "memorig", "sampst", "psmwave", "quarter", "dcsedfl_dv", "dcsedw_dv"]
    wave_cols = [f"{p}_{root}" for p in PREFIXES for root in ("hidp", "pno", "ivfio", "ivfho", "month")]
    wide = pd.read_stata(UKHLS / "xwaveid.dta", columns=stable + wave_cols, convert_categoricals=False)
    frames: list[pd.DataFrame] = []
    for prefix in PREFIXES:
        wave = WAVE_OF[prefix]
        cols = stable + [f"{prefix}_{root}" for root in ("hidp", "pno", "ivfio", "ivfho", "month")]
        part = wide[cols].copy()
        part = part.rename(columns={f"{prefix}_{root}": root for root in ("hidp", "pno", "ivfio", "ivfho", "month")})
        part.insert(1, "wave", wave)
        # Keep every record that has meaningful wave issue/enumeration information.
        meaningful = positive(part["hidp"]) | (~part["ivfio"].isin([-9, -8]) & part["ivfio"].notna())
        frames.append(part.loc[meaningful])
    panel = pd.concat(frames, ignore_index=True)
    # A positive hidp in xwaveid can identify an issued/traced household even when
    # the person was not enumerated in that wave's indall file. Actual enumeration
    # is therefore assigned only after merging the indall presence flag below.
    panel["enumerated"] = False
    panel["adult_self_interview"] = panel["ivfio"].isin(ADULT_SELF_CODES)
    panel["adult_proxy_interview"] = panel["ivfio"].isin(ADULT_PROXY_CODES)
    panel["youth_interview"] = panel["ivfio"].isin(YOUTH_CODES)
    panel["productive_interview"] = panel[["adult_self_interview", "adult_proxy_interview", "youth_interview"]].any(axis=1)
    panel["reported_dead_this_wave"] = panel["ivfio"].eq(99) | panel["dcsedw_dv"].eq(panel["wave"])
    return panel


def load_indall_supplement() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    roots = [
        "hidp", "pno", "age_dv", "sex_dv", "intdatd_dv", "intdatm_dv", "intdaty_dv",
        "ppid", "sppid", "fnpid", "mnpid", "hhsize", "ivfio", "hhorig", "memorig", "sampst",
    ]
    for prefix in PREFIXES:
        path = UKHLS / f"{prefix}_indall.dta"
        available = pd.read_stata(path, iterator=True).variable_labels().keys()
        selected = ["pidp"] + [f"{prefix}_{root}" for root in roots if f"{prefix}_{root}" in available]
        data = pd.read_stata(path, columns=selected, convert_categoricals=False)
        rename = {f"{prefix}_{root}": root for root in roots if f"{prefix}_{root}" in selected}
        data = data.rename(columns=rename)
        data.insert(1, "wave", WAVE_OF[prefix])
        data["indall_enumerated"] = True
        keep = [
            "pidp", "wave", "age_dv", "sex_dv", "intdatd_dv", "intdatm_dv", "intdaty_dv",
            "ppid", "sppid", "fnpid", "mnpid", "hhsize", "indall_enumerated",
        ]
        for column in keep:
            if column not in data:
                data[column] = np.nan
        frames.append(data[keep])
        print(f"Loaded indall wave {WAVE_OF[prefix]}", flush=True)
    return pd.concat(frames, ignore_index=True)


def add_actual_dates(panel: pd.DataFrame) -> pd.DataFrame:
    valid = (
        panel["intdaty_dv"].between(2008, 2030)
        & panel["intdatm_dv"].between(1, 12)
        & panel["intdatd_dv"].between(1, 31)
    )
    panel["interview_date"] = pd.NaT
    panel.loc[valid, "interview_date"] = pd.to_datetime(
        dict(
            year=panel.loc[valid, "intdaty_dv"],
            month=panel.loc[valid, "intdatm_dv"],
            day=panel.loc[valid, "intdatd_dv"],
        ),
        errors="coerce",
    )
    panel = panel.sort_values(["pidp", "wave"])
    panel["months_since_prior_interview"] = (
        panel.groupby("pidp")["interview_date"].diff().dt.days / 30.4375
    )
    panel["interview_date"] = panel["interview_date"].dt.strftime("%Y-%m-%d")
    return panel


def person_flow(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.sort_values(["pidp", "wave"]).copy()
    all_enumerated_by_wave = {w: set(panel.loc[(panel.wave == w) & panel.enumerated, "pidp"]) for w in range(1, 16)}
    self_by_wave = {w: set(panel.loc[(panel.wave == w) & panel.adult_self_interview, "pidp"]) for w in range(1, 16)}
    youth_by_wave = {w: set(panel.loc[(panel.wave == w) & panel.youth_interview, "pidp"]) for w in range(1, 16)}
    rows: list[dict] = []
    ever_enumerated: set = set()
    ever_self: set = set()
    for wave in range(1, 16):
        part = panel[panel.wave == wave]
        current_enum = all_enumerated_by_wave[wave]
        current_self = self_by_wave[wave]
        prev_enum = all_enumerated_by_wave.get(wave - 1, set())
        prev_self = self_by_wave.get(wave - 1, set())
        next_enum = all_enumerated_by_wave.get(wave + 1, set())
        new_enum = current_enum - ever_enumerated
        returned = (current_enum & ever_enumerated) - prev_enum
        resumed_self = (current_self & ever_self) - prev_self
        rows.append({
            "wave": wave,
            "person_issue_records": len(part),
            "enumerated_persons": len(current_enum),
            "enumerated_households": part.loc[part.enumerated & positive(part.hidp), "hidp"].nunique(),
            "adult_self_interviews": int(part.adult_self_interview.sum()),
            "adult_proxy_interviews": int(part.adult_proxy_interview.sum()),
            "youth_interviews": int(part.youth_interview.sum()),
            "reported_dead": int(part.reported_dead_this_wave.sum()),
            "newly_enumerated": len(new_enum),
            "returned_after_enumeration_gap": len(returned),
            "resumed_adult_self_after_response_gap": len(resumed_self),
            "enumerated_retained_from_prior_wave": len(current_enum & prev_enum) if wave > 1 else np.nan,
            "enumerated_retained_to_next_wave": len(current_enum & next_enum) if wave < 15 else np.nan,
            "adult_self_retained_from_prior_wave": len(current_self & prev_self) if wave > 1 else np.nan,
            "youth_to_adult_interview_from_prior_wave": len(youth_by_wave.get(wave - 1, set()) & (current_self | set(part.loc[part.adult_proxy_interview, "pidp"]))) if wave > 1 else np.nan,
        })
        ever_enumerated |= current_enum
        ever_self |= current_self
    return pd.DataFrame(rows)


def household_wave_and_flow(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    enumerated = panel[panel.enumerated & positive(panel.hidp)].copy()
    hh = enumerated.groupby(["wave", "hidp"], as_index=False).agg(
        members=("pidp", "nunique"),
        adult_self_interviews=("adult_self_interview", "sum"),
        adult_proxy_interviews=("adult_proxy_interview", "sum"),
        youth_interviews=("youth_interview", "sum"),
        household_outcome=("ivfho", "first"),
    )
    rows: list[dict] = []
    for wave in range(1, 15):
        left = enumerated[enumerated.wave == wave][["pidp", "hidp"]].rename(columns={"hidp": "hidp_prev"})
        right = enumerated[enumerated.wave == wave + 1][["pidp", "hidp"]].rename(columns={"hidp": "hidp_next"})
        overlap = left.merge(right, on="pidp", how="inner")
        edges = overlap.groupby(["hidp_prev", "hidp_next"]).size().rename("shared_members").reset_index()
        prev_degree = edges.groupby("hidp_prev")["hidp_next"].nunique()
        next_degree = edges.groupby("hidp_next")["hidp_prev"].nunique()
        stable = edges[
            edges["hidp_prev"].map(prev_degree).eq(1)
            & edges["hidp_next"].map(next_degree).eq(1)
        ]
        rows.append({
            "from_wave": wave,
            "to_wave": wave + 1,
            "households_from": left.hidp_prev.nunique(),
            "households_to": right.hidp_next.nunique(),
            "persons_enumerated_both_waves": overlap.pidp.nunique(),
            "observed_household_transition_edges": len(edges),
            "observed_split_households": int((prev_degree > 1).sum()),
            "observed_merge_households": int((next_degree > 1).sum()),
            "one_to_one_continuity_edges": len(stable),
            "persons_in_one_to_one_continuity": int(stable.shared_members.sum()),
            "interpretation": "Continuity inferred from shared pidp; hidp itself is wave-specific",
        })
    return hh, pd.DataFrame(rows)


def attrition_flow(panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    self_counts = panel.groupby("pidp")["adult_self_interview"].sum()
    ever_self = int((self_counts >= 1).sum())
    for threshold in (1, 2, 3, 5, 10, 15):
        numerator = int((self_counts >= threshold).sum())
        rows.append({
            "metric_group": "adult_self_interview_depth",
            "wave_or_threshold": f"{threshold}+",
            "numerator_persons": numerator,
            "denominator_persons": ever_self,
            "percent": 100 * numerator / ever_self if ever_self else np.nan,
            "notes": "ivfio=1 full self interview or ivfio=3 telephone interview; proxies excluded",
        })
    wave1 = set(panel.loc[(panel.wave == 1) & panel.adult_self_interview, "pidp"])
    for wave in range(1, 16):
        current = set(panel.loc[(panel.wave == wave) & panel.adult_self_interview, "pidp"])
        numerator = len(wave1 & current)
        rows.append({
            "metric_group": "wave1_adult_self_cohort_observed",
            "wave_or_threshold": str(wave),
            "numerator_persons": numerator,
            "denominator_persons": len(wave1),
            "percent": 100 * numerator / len(wave1) if wave1 else np.nan,
            "notes": "Nonmonotone: permits missed intervening waves; proxies excluded",
        })
    death = panel.groupby("pidp")["dcsedfl_dv"].max().eq(1)
    rows.append({
        "metric_group": "reported_death",
        "wave_or_threshold": "by_release",
        "numerator_persons": int(death.sum()),
        "denominator_persons": int(panel.pidp.nunique()),
        "percent": 100 * death.mean(),
        "notes": "Cross-wave death flag among persons with at least one UKHLS wave record",
    })
    return pd.DataFrame(rows)


def relationship_panels(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    couple_frames: list[pd.DataFrame] = []
    pc_frames: list[pd.DataFrame] = []
    metrics: list[dict] = []
    enumerated = panel[panel.enumerated].copy()
    for wave in range(1, 16):
        data = enumerated[enumerated.wave == wave].copy()
        by_pid = data.drop_duplicates("pidp").set_index("pidp")
        pointed = data[positive(data.ppid)].copy()
        partner_exists = pointed.ppid.isin(by_pid.index)
        same_hh = pd.Series(False, index=pointed.index)
        reciprocal = pd.Series(False, index=pointed.index)
        valid_idx = pointed.index[partner_exists]
        if len(valid_idx):
            partner_rows = by_pid.loc[pointed.loc[valid_idx, "ppid"]]
            same_hh.loc[valid_idx] = partner_rows.hidp.to_numpy() == pointed.loc[valid_idx, "hidp"].to_numpy()
            reciprocal.loc[valid_idx] = partner_rows.ppid.to_numpy() == pointed.loc[valid_idx, "pidp"].to_numpy()
        pointed["partner_enumerated_same_wave"] = partner_exists.to_numpy()
        pointed["same_household"] = same_hh.to_numpy()
        pointed["reciprocal_partner_pointer"] = reciprocal.to_numpy()
        if len(pointed):
            pointed["pidp_low"] = pointed[["pidp", "ppid"]].min(axis=1)
            pointed["pidp_high"] = pointed[["pidp", "ppid"]].max(axis=1)
            pairs = pointed.drop_duplicates(["pidp_low", "pidp_high"])[
                ["wave", "hidp", "pidp_low", "pidp_high", "same_household", "reciprocal_partner_pointer"]
            ].copy()
            self_map = by_pid["adult_self_interview"]
            pairs["both_adult_self_interviews"] = pairs.pidp_low.map(self_map).fillna(False) & pairs.pidp_high.map(self_map).fillna(False)
            couple_frames.append(pairs)
        for relation, source in (("natural_father", "fnpid"), ("natural_mother", "mnpid")):
            linked = data[positive(data[source])][["wave", "hidp", "pidp", source]].copy()
            linked = linked.rename(columns={"pidp": "child_pidp", source: "parent_pidp"})
            linked["relationship"] = relation
            linked["parent_enumerated_same_wave"] = linked.parent_pidp.isin(by_pid.index)
            linked["parent_same_household"] = False
            idx = linked.index[linked.parent_enumerated_same_wave]
            if len(idx):
                linked.loc[idx, "parent_same_household"] = (
                    by_pid.loc[linked.loc[idx, "parent_pidp"], "hidp"].to_numpy() == linked.loc[idx, "hidp"].to_numpy()
                )
            pc_frames.append(linked)
        metrics.append({
            "scope": f"wave_{wave}",
            "partner_pointers": len(pointed),
            "partner_pointer_target_enumerated_n": int(partner_exists.sum()),
            "partner_pointer_target_enumerated_pct": 100 * partner_exists.mean() if len(pointed) else np.nan,
            "reciprocal_partner_pointer_n": int(reciprocal.sum()),
            "reciprocal_partner_pointer_pct": 100 * reciprocal.mean() if len(pointed) else np.nan,
            "unique_couple_waves": len(couple_frames[-1]) if couple_frames else 0,
            "couples_both_adult_self": int(couple_frames[-1].both_adult_self_interviews.sum()) if couple_frames else 0,
            "natural_parent_child_links": sum(len(frame) for frame in pc_frames[-2:]),
            "notes": "Wave-specific co-resident pointers from indall; adult self excludes proxies",
        })
    couples = pd.concat(couple_frames, ignore_index=True) if couple_frames else pd.DataFrame()
    parent_child = pd.concat(pc_frames, ignore_index=True) if pc_frames else pd.DataFrame()
    return couples, parent_child, pd.DataFrame(metrics)


def static_family_linkage_metrics() -> tuple[pd.DataFrame, dict]:
    path = UKHLS / "xhhrel.dta"
    columns = ["pidp", "ptx_N", "bpx_N", "bpx_ef", "bpx_rr_ef", "gpx_rr_ef", "mr_ef"]
    columns += [f"ptx_pidp_{i}" for i in range(1, 7)]
    columns += [f"ptx_cr_{i}" for i in range(1, 7)]
    columns += [f"bpx_pidp_{i}" for i in range(1, 5)]
    data = pd.read_stata(path, columns=columns, convert_categoricals=False)
    partner_edges = []
    parent_edges = []
    for i in range(1, 7):
        x = data.loc[positive(data[f"ptx_pidp_{i}"]), ["pidp", f"ptx_pidp_{i}", f"ptx_cr_{i}"]].copy()
        x.columns = ["pidp", "relative_pidp", "current_wave15_flag"]
        x["relationship"] = "partner"
        partner_edges.append(x)
    for i in range(1, 5):
        x = data.loc[positive(data[f"bpx_pidp_{i}"]), ["pidp", f"bpx_pidp_{i}"]].copy()
        x.columns = ["pidp", "relative_pidp"]
        x["current_wave15_flag"] = np.nan
        x["relationship"] = "biological_parent"
        parent_edges.append(x)
    edges = pd.concat(partner_edges + parent_edges, ignore_index=True)
    partner = edges[edges.relationship == "partner"]
    partner_set = set(map(tuple, partner[["pidp", "relative_pidp"]].itertuples(index=False, name=None)))
    reciprocal = sum((b, a) in partner_set for a, b in partner_set)
    summary = {
        "scope": "xhhrel_static",
        "partner_pointers": len(partner),
        "partner_pointer_target_enumerated_n": int(partner.relative_pidp.isin(data.pidp).sum()),
        "partner_pointer_target_enumerated_pct": 100 * partner.relative_pidp.isin(data.pidp).mean() if len(partner) else np.nan,
        "reciprocal_partner_pointer_n": reciprocal,
        "reciprocal_partner_pointer_pct": 100 * reciprocal / len(partner) if len(partner) else np.nan,
        "unique_couple_waves": np.nan,
        "couples_both_adult_self": np.nan,
        "natural_parent_child_links": int((edges.relationship == "biological_parent").sum()),
        "notes": (
            f"Static cross-wave family matrix; no relationship start/end dates. "
            f"Flags: >2 biological parents={(data.bpx_ef == 1).sum()}, inverse biological mismatch={(data.bpx_rr_ef == 1).sum()}, "
            f"inverse grandparent mismatch={(data.gpx_rr_ef == 1).sum()}, multiple relationship reports={(data.mr_ef == 1).sum()}"
        ),
    }
    return edges, summary


def main() -> None:
    panel = load_person_wave_from_xwaveid()
    supplement = load_indall_supplement()
    panel = panel.merge(supplement, on=["pidp", "wave"], how="left", validate="one_to_one")
    panel["enumerated"] = panel["indall_enumerated"].fillna(False).astype(bool)
    panel = add_actual_dates(panel)
    pflow = person_flow(panel)
    household_wave, hflow = household_wave_and_flow(panel)
    aflow = attrition_flow(panel)
    couples, parent_child, relationship_metrics = relationship_panels(panel)
    family_edges, family_summary = static_family_linkage_metrics()
    relationship_metrics = pd.concat([relationship_metrics, pd.DataFrame([family_summary])], ignore_index=True)

    write_processed_gzip(panel, "ukhls_person_wave_linkage.csv.gz")
    write_processed_gzip(household_wave, "ukhls_household_wave_linkage.csv.gz")
    write_processed_gzip(couples, "ukhls_couple_wave_linkage.csv.gz")
    write_processed_gzip(parent_child, "ukhls_parent_child_wave_linkage.csv.gz")
    write_processed_gzip(family_edges, "ukhls_family_relationship_static.csv.gz")
    write_table(pflow, "ukhls_person_flow.csv")
    write_table(hflow, "ukhls_household_flow.csv")
    write_table(aflow, "ukhls_attrition_flow.csv")
    write_table(relationship_metrics, "ukhls_relationship_linkage.csv")
    print(
        f"Person-wave rows={len(panel):,}; household-wave rows={len(household_wave):,}; "
        f"couple-wave rows={len(couples):,}; parent-child-wave rows={len(parent_child):,}"
    )


if __name__ == "__main__":
    main()
