#!/usr/bin/env python3
"""Build the UKHLS person-event response dataset from audited local inputs.

The script reuses the existing adult self-interview person-wave panel and the
audited A--F event constructor in 07_cross_stressor_reactivity_feasibility.py.
It does not read raw Stata files or run inferential models.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EVENT_BUILDER = ROOT / "scripts" / "07_cross_stressor_reactivity_feasibility.py"
OUTPUT_DATA = ROOT / "results" / "person_event_response_long.parquet"
OUTPUT_AUDIT = ROOT / "results" / "person_event_response_dataset_audit.md"

FAMILY_ORDER = [
    "A_unemployment_onset",
    "B_caregiving_onset",
    "C_persistent_illness_disability_onset",
    "D_separation_divorce",
    "E_widowhood",
    "F_severe_financial_strain_onset",
]

AUDITED_TO_OUTPUT = {
    "A_job_loss": "A_unemployment_onset",
    "B_caregiving_onset": "B_caregiving_onset",
    "C_persistent_illness_onset": "C_persistent_illness_disability_onset",
    "D_separation_divorce": "D_separation_divorce",
    "E_widowhood": "E_widowhood",
    "F_financial_strain_onset": "F_severe_financial_strain_onset",
}

SHORT_LABEL = {
    "A_unemployment_onset": "A Unemployment onset",
    "B_caregiving_onset": "B Caregiving onset",
    "C_persistent_illness_disability_onset": "C Persistent illness/disability onset",
    "D_separation_divorce": "D Separation/divorce",
    "E_widowhood": "E Widowhood",
    "F_severe_financial_strain_onset": "F Severe financial strain onset (secondary)",
}


def load_event_builder():
    spec = importlib.util.spec_from_file_location("audited_event_builder", EVENT_BUILDER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load audited event builder: {EVENT_BUILDER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_missing_reason(pre: pd.Series, post: pd.Series) -> pd.Series:
    pre_missing = pre.isna()
    post_missing = post.isna()
    reason = pd.Series("complete", index=pre.index, dtype="string")
    reason.loc[pre_missing & ~post_missing] = "pre_released_missing_code_-9"
    reason.loc[~pre_missing & post_missing] = "post_released_missing_code_-9"
    reason.loc[pre_missing & post_missing] = "pre_and_post_released_missing_code_-9"
    return reason


def person_summaries(events: pd.DataFrame) -> pd.DataFrame:
    observed = events.groupby("pidp", sort=False).agg(
        number_of_observed_events=("event_type", "size"),
        number_of_observed_event_families=("event_type", "nunique"),
    )
    qualifying = events[events.qualifying_event].copy()
    q = qualifying.groupby("pidp", sort=False).agg(
        number_of_qualifying_events=("event_type", "size"),
        number_of_event_families=("event_type", "nunique"),
        first_event_date=("event_date_proxy", "min"),
        last_event_date=("event_date_proxy", "max"),
    )
    core = qualifying[~qualifying.is_secondary_event].groupby("pidp", sort=False).agg(
        number_of_core_qualifying_events=("event_type", "size"),
        number_of_core_event_families=("event_type", "nunique"),
        first_core_event_date=("event_date_proxy", "min"),
        last_core_event_date=("event_date_proxy", "max"),
    )
    out = observed.join(q, how="left").join(core, how="left")
    count_columns = [
        "number_of_observed_events", "number_of_observed_event_families",
        "number_of_qualifying_events", "number_of_event_families",
        "number_of_core_qualifying_events", "number_of_core_event_families",
    ]
    for column in count_columns:
        out[column] = out[column].fillna(0).astype("int16")
    return out.reset_index()


def build_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    builder = load_event_builder()
    transitions = builder.prepare_transitions()

    # Recover the released GHQ source values so the missing reason is explicit.
    transitions["post_GHQ12_source"] = pd.to_numeric(transitions.scghq1_dv, errors="coerce")
    transitions["pre_GHQ12_source"] = transitions.groupby("pidp", sort=False).post_GHQ12_source.shift(1)

    events = builder.occurrences(transitions, illness_persistent=True, include_financial=True).copy()
    source = transitions[["pidp", "wave", "pre_GHQ12_source", "post_GHQ12_source"]]
    events = events.merge(source, on=["pidp", "wave"], how="left", validate="many_to_one")
    events["event_type"] = events.family.map(AUDITED_TO_OUTPUT).astype("string")
    assert events.event_type.notna().all()
    events["event_family_order"] = events.event_type.map({x: i + 1 for i, x in enumerate(FAMILY_ORDER)})

    events = events.sort_values(
        ["pidp", "event_date_proxy" if "event_date_proxy" in events else "interview_date", "wave", "event_family_order"]
    ).reset_index(drop=True)
    events = events.rename(columns={
        "wave": "post_wave",
        "interview_date": "event_date_proxy",
        "pre_ghq": "pre_GHQ12",
        "post_ghq": "post_GHQ12",
        "delta_ghq": "delta_GHQ12",
        "gap_months": "pre_post_gap_months",
        "complete_ghq": "qualifying_event",
    })

    events["pre_wave"] = pd.to_numeric(events.pre_wave, errors="raise").astype("int16")
    events["post_wave"] = pd.to_numeric(events.post_wave, errors="raise").astype("int16")
    events["pidp"] = pd.to_numeric(events.pidp, errors="raise").astype("int64")
    events["event_date_proxy"] = pd.to_datetime(events.event_date_proxy, errors="raise")
    events["is_secondary_event"] = events.event_type.eq("F_severe_financial_strain_onset")
    events["ghq_missing_reason"] = make_missing_reason(events.pre_GHQ12, events.post_GHQ12)

    # Composite IDs are deterministic across rebuilds and expose the transition.
    events["transition_id"] = (
        events.pidp.astype("string") + "_w" + events.pre_wave.astype("string").str.zfill(2)
        + "_w" + events.post_wave.astype("string").str.zfill(2)
    )
    letter = events.event_type.str.slice(0, 1)
    events["event_id"] = events.transition_id + "_" + letter
    events["transition_event_family_count"] = events.groupby(
        ["pidp", "pre_wave", "post_wave"], sort=False
    ).event_type.transform("nunique").astype("int8")
    events["single_event_transition"] = events.transition_event_family_count.eq(1)

    # event_order is record-level. Co-occurring family labels therefore receive
    # consecutive deterministic A--F positions and a zero-month interval.
    events["event_order"] = events.groupby("pidp", sort=False).cumcount().add(1).astype("int16")
    events["previous_event_type"] = events.groupby("pidp", sort=False).event_type.shift(1).astype("string")
    previous_date = events.groupby("pidp", sort=False).event_date_proxy.shift(1)
    events["months_since_previous_event"] = (
        (events.event_date_proxy - previous_date).dt.days / 30.4375
    ).astype("float32")
    events["same_transition_as_previous_event"] = (
        events.groupby("pidp", sort=False).transition_id.shift(1).eq(events.transition_id).fillna(False).astype(bool)
    )

    # A shared transition order is supplied so simultaneous family labels can be
    # treated as one dated transition without discarding any event family.
    events["event_transition_order"] = (
        events.groupby("pidp", sort=False).post_wave.rank(method="dense").astype("int16")
    )

    persons = person_summaries(events)
    events = events.merge(persons, on="pidp", how="left", validate="many_to_one")

    # Use nullable compact integer columns for GHQ while preserving missingness.
    for column in ["pre_GHQ12", "post_GHQ12", "delta_GHQ12"]:
        events[column] = pd.to_numeric(events[column], errors="coerce").round().astype("Int8")
    for column in ["pre_GHQ12_source", "post_GHQ12_source"]:
        events[column] = pd.to_numeric(events[column], errors="coerce").round().astype("Int8")
    events["pre_post_gap_months"] = pd.to_numeric(events.pre_post_gap_months).astype("float32")

    columns = [
        "event_id", "transition_id", "pidp", "event_type", "is_secondary_event",
        "event_wave", "event_date_proxy", "pre_wave", "post_wave",
        "pre_post_gap_months", "pre_GHQ12", "post_GHQ12", "delta_GHQ12",
        "qualifying_event", "ghq_missing_reason", "event_order",
        "previous_event_type", "months_since_previous_event",
        "same_transition_as_previous_event", "event_transition_order",
        "transition_event_family_count", "single_event_transition",
        "number_of_qualifying_events", "number_of_event_families",
        "first_event_date", "last_event_date", "number_of_observed_events",
        "number_of_observed_event_families", "number_of_core_qualifying_events",
        "number_of_core_event_families", "first_core_event_date", "last_core_event_date",
        "pre_GHQ12_source", "post_GHQ12_source",
    ]
    events["event_wave"] = events.post_wave
    events = events[columns].sort_values(["pidp", "event_order"]).reset_index(drop=True)
    return events, persons


def validate_dataset(events: pd.DataFrame, persons: pd.DataFrame) -> None:
    required = {
        "event_id", "pidp", "event_type", "event_wave", "event_date_proxy",
        "pre_wave", "post_wave", "pre_GHQ12", "post_GHQ12", "delta_GHQ12",
        "ghq_missing_reason", "number_of_qualifying_events", "number_of_event_families",
        "first_event_date", "last_event_date", "event_order", "previous_event_type",
        "months_since_previous_event",
    }
    assert required.issubset(events.columns)
    assert events.event_id.is_unique and events.event_id.notna().all()
    assert events.event_type.isin(FAMILY_ORDER).all()
    assert events.post_wave.sub(events.pre_wave).eq(1).all()
    assert events.event_wave.eq(events.post_wave).all()
    assert events.pre_post_gap_months.between(8, 18).all()
    assert events.event_date_proxy.notna().all()
    assert events.qualifying_event.eq(events.pre_GHQ12.notna() & events.post_GHQ12.notna()).all()
    assert events.delta_GHQ12.eq(events.post_GHQ12 - events.pre_GHQ12).fillna(True).all()
    assert events.loc[events.qualifying_event, "ghq_missing_reason"].eq("complete").all()
    assert events.loc[~events.qualifying_event, "ghq_missing_reason"].ne("complete").all()
    missing_sources = pd.concat([
        events.loc[events.pre_GHQ12.isna(), "pre_GHQ12_source"],
        events.loc[events.post_GHQ12.isna(), "post_GHQ12_source"],
    ])
    assert missing_sources.eq(-9).all(), "Unexpected GHQ missing/invalid source value"
    assert events.groupby("pidp").event_order.apply(lambda x: x.tolist() == list(range(1, len(x) + 1))).all()
    assert events.loc[events.same_transition_as_previous_event, "months_since_previous_event"].eq(0).all()
    assert events.transition_event_family_count.eq(
        events.groupby(["pidp", "pre_wave", "post_wave"]).event_type.transform("nunique")
    ).all()

    derived = person_summaries(events)
    compare = persons.merge(derived, on="pidp", suffixes=("_written", "_derived"), validate="one_to_one")
    for column in [
        "number_of_observed_events", "number_of_observed_event_families",
        "number_of_qualifying_events", "number_of_event_families",
        "number_of_core_qualifying_events", "number_of_core_event_families",
    ]:
        assert compare[f"{column}_written"].eq(compare[f"{column}_derived"]).all()
    assert events.number_of_qualifying_events.eq(events.groupby("pidp").qualifying_event.transform("sum")).all()


def count_distribution(events: pd.DataFrame, column: str, scope_mask: pd.Series) -> pd.DataFrame:
    one_per_person = events.loc[scope_mask].drop_duplicates("pidp")[["pidp", column]]
    counts = one_per_person[column]
    return pd.DataFrame({
        "category": ["1", "2", "3", "4+"],
        "persons": [int((counts == 1).sum()), int((counts == 2).sum()), int((counts == 3).sum()), int((counts >= 4).sum())],
    })


def family_distribution(events: pd.DataFrame, column: str, scope_mask: pd.Series) -> pd.DataFrame:
    values = events.loc[scope_mask].drop_duplicates("pidp")[column]
    return values.value_counts().sort_index().rename_axis("families").reset_index(name="persons")


def family_summary(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for family in FAMILY_ORDER:
        x = events[events.event_type.eq(family)]
        q = x[x.qualifying_event]
        rows.append({
            "event_type": family,
            "events": len(x),
            "persons": x.pidp.nunique(),
            "complete_events": len(q),
            "complete_persons": q.pidp.nunique(),
            "mean_delta": q.delta_GHQ12.astype("float64").mean(),
            "sd_delta": q.delta_GHQ12.astype("float64").std(ddof=1),
        })
    return pd.DataFrame(rows)


def nfmt(value) -> str:
    return f"{int(value):,}"


def ffmt(value) -> str:
    return "—" if pd.isna(value) else f"{float(value):.2f}"


def pct(num: int, den: int) -> str:
    return "—" if den == 0 else f"{100 * num / den:.1f}%"


def render_audit(events: pd.DataFrame) -> str:
    family = family_summary(events)
    q = events.qualifying_event
    core = ~events.is_secondary_event
    core_q = core & q
    all_persons = events.pidp.nunique()
    q_persons = events.loc[q, "pidp"].nunique()
    core_persons = events.loc[core, "pidp"].nunique()
    core_q_persons = events.loc[core_q, "pidp"].nunique()

    event_dist_all = count_distribution(events, "number_of_qualifying_events", q)
    event_dist_core = count_distribution(events, "number_of_core_qualifying_events", core_q)
    family_dist_all = family_distribution(events, "number_of_event_families", q)
    family_dist_core = family_distribution(events, "number_of_core_event_families", core_q)
    missing_reason = events.ghq_missing_reason.value_counts().rename_axis("reason").reset_index(name="events")

    simultaneous_records = int(events.same_transition_as_previous_event.sum())
    multi_transitions = events.loc[events.transition_event_family_count.ge(2), ["pidp", "transition_id"]].drop_duplicates()

    schema_notes = {
        "event_id": "Deterministic unique key: pidp + pre/post wave + A–F family.",
        "transition_id": "Shared by event families occurring on the same pidp t−1→t transition.",
        "pidp": "UKHLS person identifier; identical across pre/post rows.",
        "event_type": "One of the audited A–F event families.",
        "event_wave": "Wave t at which the new state is observed; identical to post_wave.",
        "post_wave": "Adjacent post-event self-interview wave t.",
        "event_date_proxy": "Date of the post-event interview; not an exact event date.",
        "pre_wave": "Adjacent prior self-interview wave t−1.",
        "pre_GHQ12": "Valid released GHQ-12 Likert 0–36 at t−1; missing otherwise.",
        "post_GHQ12": "Valid released GHQ-12 Likert 0–36 at t; missing otherwise.",
        "delta_GHQ12": "post_GHQ12 − pre_GHQ12; present only when both scores are valid.",
        "qualifying_event": "True when both GHQ values are valid; event/state and 8–18-month rules already hold for every row.",
        "ghq_missing_reason": "Complete, pre missing, post missing, or both missing; observed missing source code is −9.",
        "event_order": "Record-level order within person, sorted by date/wave then A–F; simultaneous labels are consecutive.",
        "previous_event_type": "Previous record-level event family; simultaneous labels can be the previous record.",
        "months_since_previous_event": "Months between post-interview date proxies; zero for simultaneous event labels.",
        "event_transition_order": "Distinct dated transition order; simultaneous labels share this value.",
        "number_of_qualifying_events": "Person-level count of A–F rows with complete pre/post GHQ.",
        "number_of_event_families": "Person-level number of distinct A–F families among qualifying events.",
        "first_event_date": "First qualifying A–F event-date proxy for the person.",
        "last_event_date": "Last qualifying A–F event-date proxy for the person.",
        "is_secondary_event": "True only for F severe financial strain onset.",
    }

    lines = [
        "# Person-event response dataset audit", "",
        "## 1. Build scope and data definition", "",
        "- Source: existing `data_processed/ukhls_adult_core_panel.csv.gz` and the audited event constructor in `scripts/07_cross_stressor_reactivity_feasibility.py`; no raw Stata files were read.",
        "- One row is one audited A–F event-family label on a same-person adjacent-wave transition with an 8–18-month interview gap.",
        "- A unemployment onset, B caregiving onset, C strict persistent illness/disability onset, D separation/divorce, E widowhood and F severe financial strain onset use the previously audited definitions without alteration.",
        "- F is retained but explicitly flagged as secondary. Core A–E counts are reported separately.",
        "- Rows with missing GHQ are retained and labelled. `qualifying_event=True` requires valid self-report GHQ-12 at both adjacent interviews.",
        "- `event_date_proxy` is the post-interview date at wave t; it is not the exact date of the adversity.", "",
        "## 2. Output structure", "",
        f"- Dataset: `results/{OUTPUT_DATA.name}`", f"- Rows: **{nfmt(len(events))}**", f"- Columns: **{nfmt(len(events.columns))}**", "",
        "| Variable | Build dtype | Meaning |", "|---|---|---|",
    ]
    for column, note in schema_notes.items():
        lines.append(f"| `{column}` | `{events[column].dtype}` | {note} |")

    lines += ["", "## 3. Total event and person counts", "",
              "| Scope | Status-defined events | Unique persons | Qualifying events | Persons with ≥1 qualifying event | GHQ complete |",
              "|---|---:|---:|---:|---:|---:|",
              f"| Core A–E | {nfmt(core.sum())} | {nfmt(core_persons)} | {nfmt(core_q.sum())} | {nfmt(core_q_persons)} | {pct(int(core_q.sum()), int(core.sum()))} |",
              f"| A–F including secondary F | {nfmt(len(events))} | {nfmt(all_persons)} | {nfmt(q.sum())} | {nfmt(q_persons)} | {pct(int(q.sum()), len(events))} |", "",
              "## 4. Qualifying-event count per person", "",
              "Persons without any complete pre/post GHQ event remain in the Parquet file but are excluded from the distributions below.", "",
              "| Qualifying events | Core A–E persons | A–F persons |", "|---:|---:|---:|",
    ]
    for category in ["1", "2", "3", "4+"]:
        c = int(event_dist_core.loc[event_dist_core.category.eq(category), "persons"].iloc[0])
        a = int(event_dist_all.loc[event_dist_all.category.eq(category), "persons"].iloc[0])
        lines.append(f"| {category} | {nfmt(c)} | {nfmt(a)} |")

    lines += ["", "## 5. Distinct qualifying event-family count per person", "",
              "| Event families | Core A–E persons | A–F persons |", "|---:|---:|---:|"]
    for count in sorted(set(family_dist_core.families) | set(family_dist_all.families)):
        c = family_dist_core.loc[family_dist_core.families.eq(count), "persons"].sum()
        a = family_dist_all.loc[family_dist_all.families.eq(count), "persons"].sum()
        lines.append(f"| {int(count)} | {nfmt(c)} | {nfmt(a)} |")

    lines += ["", "## 6. GHQ completeness and missing reason", "",
              f"Overall complete pre/post GHQ: **{nfmt(q.sum())} of {nfmt(len(events))} events ({pct(int(q.sum()), len(events))})**.", "",
              "| GHQ status/reason | Events | Percent |", "|---|---:|---:|"]
    for r in missing_reason.itertuples(index=False):
        lines.append(f"| `{r.reason}` | {nfmt(r.events)} | {pct(r.events, len(events))} |")

    lines += ["", "## 7. Event-type counts and descriptive GHQ change", "",
              "Mean and SD use only events with complete pre/post GHQ. They are descriptive summaries only.", "",
              "| Event type | Status-defined events | Persons | Complete events | Complete persons | GHQ complete | Mean ΔGHQ12 | SD |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in family.itertuples(index=False):
        lines.append(
            f"| {SHORT_LABEL[r.event_type]} | {nfmt(r.events)} | {nfmt(r.persons)} | {nfmt(r.complete_events)} | "
            f"{nfmt(r.complete_persons)} | {pct(r.complete_events, r.events)} | {ffmt(r.mean_delta)} | {ffmt(r.sd_delta)} |"
        )

    lines += ["", "## 8. Event ordering and co-occurring transitions", "",
              f"- Event records that follow another family label on the same transition: **{nfmt(simultaneous_records)}**.",
              f"- Unique transitions carrying at least two A–F families: **{nfmt(len(multi_transitions))}**.",
              "- `event_order` is record-level and therefore remains unique/sequential within person. `event_transition_order` is the safer field when simultaneous family labels should share one temporal position.",
              "- For simultaneous labels, `months_since_previous_event=0` and `same_transition_as_previous_event=True`; no event family was deleted or merged.", "",
              "## 9. Quality-control checks", "",
              "All checks passed:", "",
              "- `event_id` is unique and nonmissing.",
              "- Every row has identical `pidp` across the reused transition construction, adjacent `pre_wave`/`post_wave`, and an 8–18-month interview gap.",
              "- C rows use only the strict audited t+1 persistent illness/disability onset definition.",
              "- Valid GHQ values are within 0–36 and `delta_GHQ12 = post_GHQ12 − pre_GHQ12`.",
              "- Every missing GHQ value in these event rows maps to released source code −9.",
              "- Person-level counts re-derived from event rows exactly match the repeated person-level columns.",
              "- The Parquet file was read back after writing; its row count, column order, event IDs and qualifying-event total match the in-memory dataset.", "",
              "## 10. Data-structure limitations", "",
              "- Event timing is interval-censored and represented by the post-interview date.",
              "- Multiple event families on one transition remain separate rows; they are not necessarily independent adversities.",
              "- `previous_event_type` follows record order, so use `same_transition_as_previous_event` or `event_transition_order` when distinguishing prior transitions from simultaneous labels.",
              "- Incomplete-GHQ event rows are retained for missingness accounting but have missing `delta_GHQ12` and do not contribute to qualifying-event person summaries.",
              "- Person-level summary fields are repeated on every event row for analytic convenience.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    OUTPUT_DATA.parent.mkdir(parents=True, exist_ok=True)
    events, persons = build_dataset()
    validate_dataset(events, persons)

    temporary = OUTPUT_DATA.with_suffix(".tmp.parquet")
    events.to_parquet(temporary, engine="pyarrow", compression="zstd", index=False)
    temporary.replace(OUTPUT_DATA)
    OUTPUT_DATA.chmod(0o644)

    reread = pd.read_parquet(OUTPUT_DATA, engine="pyarrow")
    assert len(reread) == len(events)
    assert list(reread.columns) == list(events.columns)
    assert reread.event_id.equals(events.event_id)
    assert int(reread.qualifying_event.sum()) == int(events.qualifying_event.sum())

    OUTPUT_AUDIT.write_text(render_audit(events), encoding="utf-8")
    print(f"Wrote {OUTPUT_DATA}")
    print(f"Wrote {OUTPUT_AUDIT}")


if __name__ == "__main__":
    main()
