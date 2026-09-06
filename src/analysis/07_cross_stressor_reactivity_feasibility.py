#!/usr/bin/env python3
"""Local-only feasibility counts for cross-stressor GHQ reactivity stability.

This script reuses the existing self-report adult person-wave panel.  It does
not read raw Stata files, fit models, test hypotheses, or calculate p-values.
The only scientific output is the requested Markdown feasibility report.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "data_processed" / "ukhls_adult_core_panel.csv.gz"
OUTPUT = ROOT / "results" / "cross_stressor_reactivity_feasibility.md"

FAMILIES = [
    "A_job_loss",
    "B_caregiving_onset",
    "C_persistent_illness_onset",
    "D_separation_divorce",
    "E_widowhood",
    "F_financial_strain_onset",
]

DISPLAY = {
    "A_job_loss": "A Employment→unemployment",
    "B_caregiving_onset": "B Caregiving onset",
    "C_persistent_illness_onset": "C Persistent illness/disability onset",
    "C_any_illness_onset": "C sensitivity: illness onset without persistence",
    "D_separation_divorce": "D Separation/divorce",
    "E_widowhood": "E Widowhood",
    "F_financial_strain_onset": "F Severe financial-strain onset",
}


def nfmt(value) -> str:
    if pd.isna(value):
        return "—"
    return f"{int(value):,}"


def ffmt(value, digits: int = 2) -> str:
    if pd.isna(value):
        return "—"
    return f"{float(value):.{digits}f}"


def pct(num: int, den: int) -> str:
    return "—" if den == 0 else f"{100 * num / den:.1f}%"


def valid_ghq(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    return x.where(x.between(0, 36))


def describe_delta(values: pd.Series) -> dict:
    x = pd.to_numeric(values, errors="coerce").dropna()
    if len(x) == 0:
        return {"n": 0, "mean": np.nan, "sd": np.nan, "median": np.nan, "q1": np.nan, "q3": np.nan}
    return {
        "n": len(x),
        "mean": x.mean(),
        "sd": x.std(ddof=1) if len(x) > 1 else np.nan,
        "median": x.median(),
        "q1": x.quantile(0.25),
        "q3": x.quantile(0.75),
    }


def prepare_transitions() -> pd.DataFrame:
    core = pd.read_csv(PANEL, low_memory=False)
    assert not core.duplicated(["pidp", "wave"]).any(), "adult core panel key is not unique"
    core["interview_date"] = pd.to_datetime(core.interview_date, errors="coerce")
    core["ghq"] = valid_ghq(core.scghq1_dv)
    core = core.sort_values(["pidp", "wave"]).reset_index(drop=True)
    group = core.groupby("pidp", sort=False)
    previous = group.shift(1)
    following = group.shift(-1)

    t = core.copy()
    t["pre_wave"] = previous.wave
    t["pre_date"] = previous.interview_date
    t["pre_ghq"] = previous.ghq
    t["post_ghq"] = t.ghq
    t["delta_ghq"] = t.post_ghq - t.pre_ghq
    t["gap_months"] = (t.interview_date - t.pre_date).dt.days / 30.4375
    t["consecutive"] = t.wave.sub(t.pre_wave).eq(1)
    t["common_window"] = t.consecutive & t.gap_months.between(8, 18)

    t["event_A"] = previous.jbstat.isin([1, 2]) & t.jbstat.eq(3)
    t["event_B"] = previous.caregiver.eq(0) & t.caregiver.eq(1)
    t["event_C_any"] = previous.health.eq(2) & t.health.eq(1)
    t["event_D"] = previous.partnered.eq(1) & t.partnered.eq(0) & t.mastat_dv.isin([4, 5, 7, 8])
    t["event_E"] = previous.partnered.eq(1) & t.partnered.eq(0) & t.mastat_dv.isin([6, 9])
    t["event_F"] = previous.finnow.isin([1, 2, 3]) & t.finnow.isin([4, 5])

    t["next_wave"] = following.wave
    t["next_date"] = following.interview_date
    t["next_health"] = following.health
    t["next_gap_months"] = (t.next_date - t.interview_date).dt.days / 30.4375
    t["persistence_observable"] = t.next_wave.sub(t.wave).eq(1) & t.next_gap_months.between(8, 18) & t.next_health.isin([1, 2])
    t["event_C_persistent"] = t.event_C_any & t.persistence_observable & t.next_health.eq(1)

    return t


def occurrences(transitions: pd.DataFrame, illness_persistent: bool = True, include_financial: bool = True) -> pd.DataFrame:
    mapping = {
        "A_job_loss": "event_A",
        "B_caregiving_onset": "event_B",
        "C_persistent_illness_onset": "event_C_persistent" if illness_persistent else "event_C_any",
        "D_separation_divorce": "event_D",
        "E_widowhood": "event_E",
    }
    if include_financial:
        mapping["F_financial_strain_onset"] = "event_F"
    frames = []
    base_columns = ["pidp", "wave", "pre_wave", "pre_date", "interview_date", "gap_months", "pre_ghq", "post_ghq", "delta_ghq"]
    for family, flag in mapping.items():
        x = transitions.loc[transitions.common_window & transitions[flag], base_columns].copy()
        x["family"] = family
        x["pre_ghq_available"] = x.pre_ghq.notna()
        x["post_ghq_available"] = x.post_ghq.notna()
        x["complete_ghq"] = x.pre_ghq_available & x.post_ghq_available
        frames.append(x)
    out = pd.concat(frames, ignore_index=True)
    order = {family: i for i, family in enumerate(FAMILIES)}
    out["family_order"] = out.family.map(order)
    return out.sort_values(["pidp", "interview_date", "wave", "family_order"]).reset_index(drop=True)


def family_completeness(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for family, x in events.groupby("family", sort=False):
        d = describe_delta(x.loc[x.complete_ghq, "delta_ghq"])
        rows.append({
            "family": family,
            "events": len(x),
            "persons": x.pidp.nunique(),
            "pre_n": int(x.pre_ghq_available.sum()),
            "post_n": int(x.post_ghq_available.sum()),
            "complete_n": int(x.complete_ghq.sum()),
            "complete_persons": x.loc[x.complete_ghq, "pidp"].nunique(),
            **{f"delta_{key}": value for key, value in d.items()},
        })
    return pd.DataFrame(rows)


def sample_counts(events: pd.DataFrame, label: str) -> dict:
    q = events[events.complete_ghq].copy()
    per_person = q.groupby("pidp").agg(events=("family", "size"), families=("family", "nunique"))
    distinct_transitions = q.groupby("pidp").wave.nunique()
    same_counts = q.groupby(["pidp", "family"]).size()
    same_pairs = (same_counts * (same_counts - 1) // 2)
    family_counts = q.groupby(["pidp", "family"]).size().unstack(fill_value=0)
    cross_pairs_by_person = pd.Series(0, index=family_counts.index, dtype="int64")
    for f1, f2 in combinations(family_counts.columns, 2):
        cross_pairs_by_person += family_counts[f1] * family_counts[f2]
    return {
        "scope": label,
        "qualifying_events": len(q),
        "persons_1plus": int((per_person.events >= 1).sum()),
        "persons_2plus_events": int((per_person.events >= 2).sum()),
        "persons_2plus_families": int((per_person.families >= 2).sum()),
        "persons_3plus_events": int((per_person.events >= 3).sum()),
        "persons_3plus_families": int((per_person.families >= 3).sum()),
        "persons_2plus_distinct_transitions": int((distinct_transitions >= 2).sum()),
        "persons_same_family_repeat": int((same_counts >= 2).groupby(level=0).any().sum()),
        "same_family_event_pairs": int(same_pairs.sum()),
        "persons_cross_family": int((per_person.families >= 2).sum()),
        "cross_family_event_pairs": int(cross_pairs_by_person.sum()),
        "persons_both_pair_types": int(((same_counts >= 2).groupby(level=0).any().reindex(per_person.index, fill_value=False) & (per_person.families >= 2)).sum()),
    }


def build_event_pairs(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pidp, group in events.groupby("pidp", sort=False):
        records = group.sort_values(["interview_date", "wave", "family_order"]).to_dict("records")
        for i, j in combinations(range(len(records)), 2):
            a, b = records[i], records[j]
            f1, f2 = sorted([a["family"], b["family"]], key=FAMILIES.index)
            rows.append({
                "pidp": pidp,
                "family_1": f1,
                "family_2": f2,
                "same_family": f1 == f2,
                "same_transition": a["wave"] == b["wave"],
                "both_pre": bool(a["pre_ghq_available"] and b["pre_ghq_available"]),
                "both_post": bool(a["post_ghq_available"] and b["post_ghq_available"]),
                "both_complete": bool(a["complete_ghq"] and b["complete_ghq"]),
                "delta_1": a["delta_ghq"],
                "delta_2": b["delta_ghq"],
            })
    return pd.DataFrame(rows)


def pair_summary(pairs: pd.DataFrame, families: list[str]) -> pd.DataFrame:
    rows = []
    for i, f1 in enumerate(families):
        for f2 in families[i:]:
            x = pairs[(pairs.family_1 == f1) & (pairs.family_2 == f2)]
            complete = x[x.both_complete]
            pooled = pd.concat([complete.delta_1, complete.delta_2], ignore_index=True)
            d = describe_delta(pooled)
            rows.append({
                "family_1": f1,
                "family_2": f2,
                "pair_type": "same-family repeated" if f1 == f2 else "different-family",
                "raw_pairs": len(x),
                "raw_persons": x.pidp.nunique(),
                "both_pre_n": int(x.both_pre.sum()) if len(x) else 0,
                "both_post_n": int(x.both_post.sum()) if len(x) else 0,
                "both_complete_n": int(x.both_complete.sum()) if len(x) else 0,
                "complete_persons": complete.pidp.nunique(),
                "same_transition_complete_pairs": int((complete.same_transition).sum()),
                **{f"pooled_delta_{key}": value for key, value in d.items()},
            })
    return pd.DataFrame(rows)


def matrix_lines(pair_stats: pd.DataFrame, families: list[str]) -> list[str]:
    lookup = {(r.family_1, r.family_2): r for r in pair_stats.itertuples(index=False)}
    short = {family: family.split("_", 1)[0] for family in families}
    lines = ["| Family | " + " | ".join(short[f] for f in families) + " |", "|---|" + "---:|" * len(families)]
    for i, f1 in enumerate(families):
        cells = []
        for j, f2 in enumerate(families):
            a, b = (f1, f2) if i <= j else (f2, f1)
            r = lookup[(a, b)]
            cells.append(f"{nfmt(r.complete_persons)} / {nfmt(r.both_complete_n)}")
        lines.append(f"| {short[f1]} | " + " | ".join(cells) + " |")
    return lines


def timing_summary(events: pd.DataFrame, scope: str) -> dict:
    q = events[events.complete_ghq].copy()
    record_counts = q.groupby("pidp").size()
    q = q.sort_values(["pidp", "interview_date", "wave", "family_order"])
    distinct = q.drop_duplicates(["pidp", "wave"]).copy()
    transition_counts = distinct.groupby("pidp").size()
    eligible_people = transition_counts[transition_counts >= 2].index
    first_two = distinct[distinct.pidp.isin(eligible_people)].groupby("pidp", sort=False).head(2)
    wide = first_two.groupby("pidp").agg(
        first_date=("interview_date", "first"), second_date=("interview_date", "last"),
        first_wave=("wave", "first"), second_wave=("wave", "last"),
    )
    wide["months"] = (wide.second_date - wide.first_date).dt.days / 30.4375
    x = wide.months.dropna()
    return {
        "scope": scope,
        "persons_2plus_event_records": int((record_counts >= 2).sum()),
        "persons_2plus_distinct_transitions": len(wide),
        "persons_only_one_distinct_transition_despite_2plus_records": int(((record_counts >= 2) & transition_counts.reindex(record_counts.index, fill_value=0).eq(1)).sum()),
        "median": x.median(), "q1": x.quantile(.25), "q3": x.quantile(.75),
        "min": x.min(), "max": x.max(), "p1": x.quantile(.01), "p99": x.quantile(.99),
        "ge12": int((x >= 12).sum()), "ge24": int((x >= 24).sum()),
        "ge36": int((x >= 36).sum()), "ge60": int((x >= 60).sum()),
        "adjacent_wave": int(wide.second_wave.sub(wide.first_wave).eq(1).sum()),
    }


def overlap_audit(events: pd.DataFrame, scope: str) -> tuple[pd.DataFrame, dict]:
    q = events[events.complete_ghq].copy()
    transition_size = q.groupby(["pidp", "wave"]).family.transform("nunique")
    q["single_transition"] = transition_size.eq(1)
    rows = []
    for family, x in q.groupby("family", sort=False):
        rows.append({
            "scope": scope, "family": family, "qualifying_events": len(x),
            "single_event_transition": int(x.single_transition.sum()),
            "multi_event_transition": int((~x.single_transition).sum()),
            "persons_with_multi_event_occurrence": x.loc[~x.single_transition, "pidp"].nunique(),
        })
    transition = q.groupby(["pidp", "wave"]).agg(n_families=("family", "nunique")).reset_index()
    single = q[q.single_transition]
    single_family_person = single.groupby("pidp").family.nunique()
    summary = {
        "scope": scope,
        "unique_qualifying_transitions": len(transition),
        "single_transitions": int((transition.n_families == 1).sum()),
        "multi_transitions": int((transition.n_families >= 2).sum()),
        "persons_2plus_families_both_on_single_transitions": int((single_family_person >= 2).sum()),
        "multi_2families": int((transition.n_families == 2).sum()),
        "multi_3plusfamilies": int((transition.n_families >= 3).sum()),
    }
    return pd.DataFrame(rows), summary


def repeat_onset_audit(transitions: pd.DataFrame, family: str) -> dict:
    if family == "illness":
        yes, no, current = 1, 2, transitions.health
        onset_flag = transitions.event_C_any
        persistent_flag = transitions.event_C_persistent
    elif family == "caregiving":
        yes, no, current = 1, 0, transitions.caregiver
        onset_flag = transitions.event_B
        persistent_flag = pd.Series(False, index=transitions.index)
    else:
        raise ValueError(family)

    x = transitions.copy()
    group = x.groupby("pidp", sort=False)
    lag1_state, lag2_state, lag3_state = group[current.name].shift(1), group[current.name].shift(2), group[current.name].shift(3)
    lag1_wave, lag2_wave, lag3_wave = group.wave.shift(1), group.wave.shift(2), group.wave.shift(3)
    lag1_date, lag2_date, lag3_date = group.interview_date.shift(1), group.interview_date.shift(2), group.interview_date.shift(3)
    gap1 = (x.interview_date - lag1_date).dt.days / 30.4375
    gap2 = (lag1_date - lag2_date).dt.days / 30.4375
    gap3 = (lag2_date - lag3_date).dt.days / 30.4375
    rapid = (
        current.eq(yes) & lag1_state.eq(no) & lag2_state.eq(yes) & lag3_state.eq(no)
        & x.wave.sub(lag1_wave).eq(1) & lag1_wave.sub(lag2_wave).eq(1) & lag2_wave.sub(lag3_wave).eq(1)
        & gap1.between(8, 18) & gap2.between(8, 18) & gap3.between(8, 18)
    )
    event_mask = x.common_window & onset_flag
    event_rows = x[event_mask].copy()
    event_rows["onset_order"] = event_rows.groupby("pidp").cumcount() + 1
    repeated = event_rows[event_rows.onset_order >= 2]
    complete = event_rows.pre_ghq.notna() & event_rows.post_ghq.notna()
    rapid_events = x[x.common_window & onset_flag & rapid]
    rapid_complete = rapid_events.pre_ghq.notna() & rapid_events.post_ghq.notna()
    result = {
        "family": family,
        "all_onsets": len(event_rows),
        "all_onset_persons": event_rows.pidp.nunique(),
        "persons_2plus_onsets": int((event_rows.groupby("pidp").size() >= 2).sum()),
        "second_or_later_onsets": len(repeated),
        "second_or_later_qualifying": int((repeated.pre_ghq.notna() & repeated.post_ghq.notna()).sum()),
        "rapid_no_yes_no_yes_second_onsets": len(rapid_events),
        "rapid_pattern_persons": rapid_events.pidp.nunique(),
        "rapid_pattern_qualifying_second_onsets": int(rapid_complete.sum()),
        "all_qualifying_onsets": int(complete.sum()),
    }
    if family == "illness":
        observable = event_rows.persistence_observable
        persistent = event_rows.event_C_persistent
        persistent_rows = event_rows[persistent].copy()
        persistent_rows["persistent_order"] = persistent_rows.groupby("pidp").cumcount() + 1
        result.update({
            "persistence_observable_onsets": int(observable.sum()),
            "persistent_onsets": int(persistent.sum()),
            "persistent_onset_persons": event_rows.loc[persistent, "pidp"].nunique(),
            "persistent_qualifying_onsets": int((persistent & complete).sum()),
            "persons_2plus_persistent_onsets": int((persistent_rows.groupby("pidp").size() >= 2).sum()),
            "second_or_later_persistent_onsets": int((persistent_rows.persistent_order >= 2).sum()),
        })
    return result


def validate_results(
    transitions: pd.DataFrame,
    all_persistent: pd.DataFrame,
    core_persistent: pd.DataFrame,
    count_rows: list[dict],
    pair_core_stats: pd.DataFrame,
    pair_all_stats: pd.DataFrame,
    overlap_core: pd.DataFrame,
    overlap_all: pd.DataFrame,
    overlap_core_summary: dict,
    overlap_all_summary: dict,
) -> None:
    """Fail fast if the independently rendered summaries do not reconcile."""
    assert transitions.loc[transitions.event_C_persistent, "event_C_any"].all()
    assert all_persistent.gap_months.between(8, 18).all()
    assert all_persistent.pre_wave.add(1).eq(all_persistent.wave).all()
    assert not all_persistent.duplicated(["pidp", "wave", "family"]).any()
    complete = all_persistent[all_persistent.complete_ghq]
    assert complete.pre_ghq.between(0, 36).all() and complete.post_ghq.between(0, 36).all()
    assert np.allclose(complete.delta_ghq, complete.post_ghq - complete.pre_ghq)

    persistent_rows = all_persistent[all_persistent.family == "C_persistent_illness_onset"]
    persistent_keys = pd.MultiIndex.from_frame(persistent_rows[["pidp", "wave"]])
    transition_lookup = transitions.set_index(["pidp", "wave"])
    matched = transition_lookup.loc[persistent_keys]
    assert matched.event_C_persistent.all()
    assert matched.next_health.eq(1).all() and matched.next_gap_months.between(8, 18).all()

    # Counts are nested as the event and family requirements become stricter.
    for row in count_rows:
        assert row["persons_3plus_families"] <= row["persons_3plus_events"] <= row["persons_2plus_events"] <= row["persons_1plus"]
        assert row["persons_3plus_families"] <= row["persons_2plus_families"] <= row["persons_2plus_events"]
        assert row["persons_2plus_distinct_transitions"] <= row["persons_2plus_events"]
    for core, expanded in [(count_rows[0], count_rows[1]), (count_rows[2], count_rows[3])]:
        for field in ["qualifying_events", "persons_1plus", "persons_2plus_events", "persons_2plus_families", "persons_3plus_events", "persons_3plus_families"]:
            assert core[field] <= expanded[field]

    assert int(pair_core_stats.both_complete_n.sum()) == count_rows[0]["same_family_event_pairs"] + count_rows[0]["cross_family_event_pairs"]
    assert int(pair_all_stats.both_complete_n.sum()) == count_rows[1]["same_family_event_pairs"] + count_rows[1]["cross_family_event_pairs"]
    for stats in [pair_core_stats, pair_all_stats]:
        assert (stats.raw_pairs >= stats.both_complete_n).all()
        assert (stats.raw_persons >= stats.complete_persons).all()
        assert (stats.same_transition_complete_pairs <= stats.both_complete_n).all()
        assert stats.loc[stats.family_1.eq(stats.family_2), "same_transition_complete_pairs"].eq(0).all()

    for events, audit, summary in [
        (core_persistent, overlap_core, overlap_core_summary),
        (all_persistent, overlap_all, overlap_all_summary),
    ]:
        q = events[events.complete_ghq]
        assert int(audit.qualifying_events.sum()) == len(q)
        assert (audit.qualifying_events == audit.single_event_transition + audit.multi_event_transition).all()
        assert summary["unique_qualifying_transitions"] == summary["single_transitions"] + summary["multi_transitions"]
        assert summary["multi_transitions"] == summary["multi_2families"] + summary["multi_3plusfamilies"]


def render_family_table(summary: pd.DataFrame) -> list[str]:
    lines = [
        "| Event family | Status-defined events | Persons | Pre GHQ | Post GHQ | Complete pre/post | Qualifying persons | ΔGHQ mean (SD) | Median [IQR] |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary.itertuples(index=False):
        lines.append(
            f"| {DISPLAY[r.family]} | {nfmt(r.events)} | {nfmt(r.persons)} | "
            f"{nfmt(r.pre_n)} ({pct(r.pre_n, r.events)}) | {nfmt(r.post_n)} ({pct(r.post_n, r.events)}) | "
            f"{nfmt(r.complete_n)} ({pct(r.complete_n, r.events)}) | {nfmt(r.complete_persons)} | "
            f"{ffmt(r.delta_mean)} ({ffmt(r.delta_sd)}) | {ffmt(r.delta_median)} [{ffmt(r.delta_q1)}, {ffmt(r.delta_q3)}] |"
        )
    return lines


def render_pair_completeness(stats: pd.DataFrame) -> list[str]:
    lines = [
        "| Pair | Type | Raw pairs | Persons | Both pre | Both post | Both complete | Complete persons | Same-transition complete pairs | Pooled ΔGHQ mean (SD) | Median [IQR] |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in stats.itertuples(index=False):
        pair = f"{r.family_1.split('_', 1)[0]}×{r.family_2.split('_', 1)[0]}"
        lines.append(
            f"| {pair} | {r.pair_type} | {nfmt(r.raw_pairs)} | {nfmt(r.raw_persons)} | "
            f"{nfmt(r.both_pre_n)} ({pct(r.both_pre_n, r.raw_pairs)}) | "
            f"{nfmt(r.both_post_n)} ({pct(r.both_post_n, r.raw_pairs)}) | "
            f"{nfmt(r.both_complete_n)} ({pct(r.both_complete_n, r.raw_pairs)}) | {nfmt(r.complete_persons)} | "
            f"{nfmt(r.same_transition_complete_pairs)} | {ffmt(r.pooled_delta_mean)} ({ffmt(r.pooled_delta_sd)}) | "
            f"{ffmt(r.pooled_delta_median)} [{ffmt(r.pooled_delta_q1)}, {ffmt(r.pooled_delta_q3)}] |"
        )
    return lines


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    transitions = prepare_transitions()

    all_persistent = occurrences(transitions, illness_persistent=True, include_financial=True)
    core_persistent = all_persistent[all_persistent.family != "F_financial_strain_onset"].copy()
    all_any_illness = occurrences(transitions, illness_persistent=False, include_financial=True)
    core_any_illness = all_any_illness[all_any_illness.family != "F_financial_strain_onset"].copy()

    # Family completeness: primary six rows plus non-persistent illness sensitivity.
    family_primary = family_completeness(all_persistent)
    illness_any = all_any_illness[all_any_illness.family == "C_persistent_illness_onset"].copy()
    illness_any["family"] = "C_any_illness_onset"
    family_all = pd.concat([family_primary, family_completeness(illness_any)], ignore_index=True)
    family_order = {family: i for i, family in enumerate(FAMILIES)} | {"C_any_illness_onset": 2.5}
    family_all["order"] = family_all.family.map(family_order)
    family_all = family_all.sort_values("order")

    count_rows = [
        sample_counts(core_persistent, "A–E, persistent C"),
        sample_counts(all_persistent, "A–F, persistent C"),
        sample_counts(core_any_illness, "A–E, C without persistence sensitivity"),
        sample_counts(all_any_illness, "A–F, C without persistence sensitivity"),
    ]

    pairs_core = build_event_pairs(core_persistent)
    pairs_all = build_event_pairs(all_persistent)
    pair_core_stats = pair_summary(pairs_core, FAMILIES[:5])
    pair_all_stats = pair_summary(pairs_all, FAMILIES)

    timing_rows = [timing_summary(core_persistent, "A–E"), timing_summary(all_persistent, "A–F")]
    overlap_core, overlap_core_summary = overlap_audit(core_persistent, "A–E")
    overlap_all, overlap_all_summary = overlap_audit(all_persistent, "A–F")
    overlap = pd.concat([overlap_core, overlap_all], ignore_index=True)

    illness_repeat = repeat_onset_audit(transitions, "illness")
    care_repeat = repeat_onset_audit(transitions, "caregiving")

    validate_results(
        transitions, all_persistent, core_persistent, count_rows,
        pair_core_stats, pair_all_stats, overlap_core, overlap_all,
        overlap_core_summary, overlap_all_summary,
    )

    lines = [
        "# Cross-stressor reactivity feasibility audit", "",
        "This is a local data-feasibility audit only. It reuses `data_processed/ukhls_adult_core_panel.csv.gz`; no raw-file rescan, web search, regression, fixed-effects model, hypothesis test, p-value, reaction correlation, candidate ranking, or scientific recommendation was performed.", "",
        "## 1. Data definition", "",
        "- Unit of an event record: the same `pidp` observed at adjacent UKHLS waves t−1 and t in the existing adult self-interview panel.",
        "- Psychological result: released GHQ-12 Likert score 0–36. `pre_ghq = GHQ_(t-1)`, `post_ghq = GHQ_t`, `ΔGHQ = post - pre`; higher positive change means more distress.",
        "- A **qualifying event** has valid self-report GHQ at both interviews and an actual interview gap of 8–18 months. Proxy interviews are absent from the reused adult core panel.",
        "- Event timing remains interval-censored: the event is known only to have occurred between the two interviews. Dates below are interview dates at which the new state is observed, not exact event dates.",
        "- A employment→unemployment: `jbstat` 1/2 at t−1 and 3 at t.",
        "- B caregiving onset: no caregiving at t−1 and caregiving at t from the existing combined `aidhh/aidxhh` state.",
        "- C primary persistent illness/disability onset: `health` 2 (no) at t−1, 1 (yes) at t, plus the same self respondent observed at t+1 with `health=1`; both t−1→t and t→t+1 interview gaps must be 8–18 months. Counts without this persistence condition are reported separately.",
        "- D separation/divorce: co-resident partner at t−1, no co-resident partner at t, and de facto marital-status code 4/5/7/8.",
        "- E widowhood: co-resident partner at t−1, no co-resident partner at t, and de facto status code 6/9.",
        "- F severe financial-strain onset (secondary): `finnow` 1/2/3 at t−1 to 4/5 at t. F is never folded into an A–E total without a separately labelled A–F result.", "",
        "## 2. Total feasibility counts", "",
        "The primary rows use persistent C. The two sensitivity rows replace C with every no→yes illness onset while leaving all other definitions unchanged.", "",
        "| Scope | Qualifying events | ≥1 event persons | ≥2 event persons | ≥2 families persons | ≥3 event persons | ≥3 families persons | ≥2 distinct event transitions |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in count_rows:
        lines.append(
            f"| {r['scope']} | {nfmt(r['qualifying_events'])} | {nfmt(r['persons_1plus'])} | {nfmt(r['persons_2plus_events'])} | "
            f"{nfmt(r['persons_2plus_families'])} | {nfmt(r['persons_3plus_events'])} | {nfmt(r['persons_3plus_families'])} | "
            f"{nfmt(r['persons_2plus_distinct_transitions'])} |"
        )

    lines += ["", "## 3. Event-family counts and GHQ availability", ""]
    lines += render_family_table(family_all)
    lines += [
        "", "`Status-defined events` already require adjacent observations and an 8–18-month pre/post interview gap, but not GHQ completeness. The ΔGHQ columns use complete pre/post events only and are descriptive.", "",
        "## 4. Qualifying event-pair matrix", "",
        "Each cell is **unique persons / number of qualifying event pairs**. Diagonal cells are repeated events from the same family; off-diagonal cells are different-family pairs. Event pairs are combinatorial within person, so one person with several events may contribute more than one pair.", "",
        "### A–E only", "",
    ]
    lines += matrix_lines(pair_core_stats, FAMILIES[:5])
    lines += ["", "### A–F including secondary financial strain", ""]
    lines += matrix_lines(pair_all_stats, FAMILIES)

    lines += ["", "## 5. Same-family versus cross-family availability", "",
              "| Scope | Persons with a same-family repeat | Same-family event pairs | Persons with different families | Different-family event pairs | Persons with both pair types |",
              "|---|---:|---:|---:|---:|---:|",
    ]
    for r in count_rows[:2]:
        lines.append(
            f"| {r['scope']} | {nfmt(r['persons_same_family_repeat'])} | {nfmt(r['same_family_event_pairs'])} | "
            f"{nfmt(r['persons_cross_family'])} | {nfmt(r['cross_family_event_pairs'])} | {nfmt(r['persons_both_pair_types'])} |"
        )

    lines += ["", "## 6. Time between the first two distinct event transitions", "",
              "Simultaneous multi-family labels at one t−1→t transition are collapsed before calculating time. Thus a person whose only two event records occur on the same transition does not receive an artificial zero-month first→second interval.", "",
              "| Scope | Persons with ≥2 event records | Persons with ≥2 distinct transitions | ≥2 records but only 1 transition | Median months | IQR | Observed min–max | P1–P99 | Adjacent-wave first/second |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in timing_rows:
        lines.append(
            f"| {r['scope']} | {nfmt(r['persons_2plus_event_records'])} | {nfmt(r['persons_2plus_distinct_transitions'])} | "
            f"{nfmt(r['persons_only_one_distinct_transition_despite_2plus_records'])} | {ffmt(r['median'])} | "
            f"{ffmt(r['q1'])}–{ffmt(r['q3'])} | {ffmt(r['min'])}–{ffmt(r['max'])} | {ffmt(r['p1'])}–{ffmt(r['p99'])} | {nfmt(r['adjacent_wave'])} |"
        )
    lines += ["", "| Scope | ≥12 months | ≥24 months | ≥36 months | ≥60 months |", "|---|---:|---:|---:|---:|"]
    for r in timing_rows:
        denominator = r["persons_2plus_distinct_transitions"]
        lines.append(
            f"| {r['scope']} | {nfmt(r['ge12'])} ({pct(r['ge12'], denominator)}) | {nfmt(r['ge24'])} ({pct(r['ge24'], denominator)}) | "
            f"{nfmt(r['ge36'])} ({pct(r['ge36'], denominator)}) | {nfmt(r['ge60'])} ({pct(r['ge60'], denominator)}) |"
        )

    lines += ["", "## 7. Overlapping-event audit", "",
              "An occurrence is `multi-event` when at least one other A–E (or A–F for that scope) family is coded on the same t−1→t transition. No occurrence is deleted.", "",
              "| Scope | Family | Qualifying events | Single-event transition | Multi-event transition | Persons with a multi-event occurrence |",
              "|---|---|---:|---:|---:|---:|",
    ]
    for r in overlap.itertuples(index=False):
        lines.append(
            f"| {r.scope} | {DISPLAY[r.family]} | {nfmt(r.qualifying_events)} | {nfmt(r.single_event_transition)} | "
            f"{nfmt(r.multi_event_transition)} | {nfmt(r.persons_with_multi_event_occurrence)} |"
        )
    lines += ["", "| Scope | Unique qualifying transitions | Single-event transitions | Multi-event transitions | Exactly 2 families | 3+ families | Persons with ≥2 different families, each on single-event transitions |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for r in [overlap_core_summary, overlap_all_summary]:
        lines.append(
            f"| {r['scope']} | {nfmt(r['unique_qualifying_transitions'])} | {nfmt(r['single_transitions'])} | {nfmt(r['multi_transitions'])} | "
            f"{nfmt(r['multi_2families'])} | {nfmt(r['multi_3plusfamilies'])} | {nfmt(r['persons_2plus_families_both_on_single_transitions'])} |"
        )

    lines += ["", "## 8. GHQ completeness and descriptive reactions for every A–F pair", "",
              "`Raw pairs` are formed from status-defined 8–18-month event occurrences before GHQ filtering. `Both complete` is the number of pairs for which both event reactions have valid pre/post GHQ. Pooled ΔGHQ combines the two reactions in each complete pair; simultaneous pair labels therefore contribute the same transition-level ΔGHQ twice and are separately counted.", ""]
    lines += render_pair_completeness(pair_all_stats)

    lines += ["", "## 9. Repeated-onset identifiability and persistent-illness sensitivity", "",
              "The rapid pattern below requires four consecutive self-interview waves, each adjacent gap 8–18 months, with state sequence no→yes→no→yes. Its final no→yes is counted as a potential reporting-fluctuation second onset; it is not interpreted as an independent new event.", "",
              "| Audit item | Illness/disability | Caregiving |", "|---|---:|---:|",
              f"| All status-defined onsets | {nfmt(illness_repeat['all_onsets'])} | {nfmt(care_repeat['all_onsets'])} |",
              f"| Persons with any onset | {nfmt(illness_repeat['all_onset_persons'])} | {nfmt(care_repeat['all_onset_persons'])} |",
              f"| Persons with ≥2 onsets | {nfmt(illness_repeat['persons_2plus_onsets'])} | {nfmt(care_repeat['persons_2plus_onsets'])} |",
              f"| Second-or-later onset occurrences | {nfmt(illness_repeat['second_or_later_onsets'])} | {nfmt(care_repeat['second_or_later_onsets'])} |",
              f"| Second-or-later onsets with complete pre/post GHQ | {nfmt(illness_repeat['second_or_later_qualifying'])} | {nfmt(care_repeat['second_or_later_qualifying'])} |",
              f"| Rapid no→yes→no→yes second onsets | {nfmt(illness_repeat['rapid_no_yes_no_yes_second_onsets'])} | {nfmt(care_repeat['rapid_no_yes_no_yes_second_onsets'])} |",
              f"| Persons with rapid pattern | {nfmt(illness_repeat['rapid_pattern_persons'])} | {nfmt(care_repeat['rapid_pattern_persons'])} |",
              f"| Rapid-pattern second onsets with complete GHQ | {nfmt(illness_repeat['rapid_pattern_qualifying_second_onsets'])} | {nfmt(care_repeat['rapid_pattern_qualifying_second_onsets'])} |", "",
              "Persistent-illness sensitivity:", "",
              f"- Illness no→yes onsets with the t+1 persistence status observable: **{nfmt(illness_repeat['persistence_observable_onsets'])}** of {nfmt(illness_repeat['all_onsets'])}.",
              f"- Strict persistent onsets (t+1=yes, both adjacent gaps 8–18 months): **{nfmt(illness_repeat['persistent_onsets'])} events among {nfmt(illness_repeat['persistent_onset_persons'])} persons**; {nfmt(illness_repeat['persistent_qualifying_onsets'])} have complete acute pre/post GHQ.",
              f"- Persons with at least two strict persistent illness onsets: **{nfmt(illness_repeat['persons_2plus_persistent_onsets'])}**; second-or-later strict persistent occurrences: **{nfmt(illness_repeat['second_or_later_persistent_onsets'])}**.", "",
              "## 10. Data-level limitations", "",
              "- Event occurrence is inferred from annual state transitions, not an exact event date. Time-between-event statistics are therefore distances between the interviews at which new states are first observed.",
              "- Requiring t+1 illness persistence selects people with another timely self interview and valid health status; the persistent-C sample is not simply a cleaner random subset.",
              "- Long-standing illness/disability is a broad self-reported state. No→yes→no→yes sequences can reflect reporting fluctuation, changing interpretation, remission/recurrence, or genuinely distinct conditions.",
              "- Caregiving onset does not identify the recipient or whether a later onset concerns the same recipient; exits can reflect recovery, death, institutionalisation, household change, or reporting variation.",
              "- Employment→unemployment is a status transition and does not by itself distinguish redundancy, dismissal, contract ending, voluntary exit, or events occurring more than once inside the interview interval.",
              "- Separation/divorce and widowhood use loss of a co-resident partner pointer plus de facto marital status. Relationship breakdown, residential separation, legal divorce and bereavement dates need not coincide.",
              "- Severe financial strain is subjective and is therefore kept secondary throughout.",
              "- Multi-event transitions may be one crisis expressed in several domains. They remain counted but are explicitly separated from single-event transitions.",
              "- Pair counts are not independent observations: a person with multiple events contributes combinatorial pairs, and a simultaneous multi-family transition can enter an off-diagonal cell.",
              "- ΔGHQ summaries are raw descriptions. They do not estimate event effects and can contain secular change, regression to the mean, measurement error, anticipation, recovery and other co-occurring changes.",
              "- No first–second reaction correlation was calculated. This report answers only how many locally observable persons, events, transitions and event pairs satisfy the stated rules.",
    ]

    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
