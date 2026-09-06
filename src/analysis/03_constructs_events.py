#!/usr/bin/env python3
"""Build the psychological construct, event, and relationship opportunity matrices."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
UKHLS = ROOT / "raw_data" / "UKDA-6614-stata" / "stata" / "stata14_se" / "ukhls"
TABLES = ROOT / "tables"
PROCESSED = ROOT / "data_processed"
PREFIXES = list("abcdefghijklmno")
WAVE_OF = {p: i for i, p in enumerate(PREFIXES, start=1)}
SELF_CODES = {1, 3}


def C(name, domain, family, roots, **kwargs):
    row = {
        "construct_name": name,
        "domain": domain,
        "dataset_family": family,
        "roots": roots,
        "score_root": kwargs.pop("score_root", roots[0] if len(roots) == 1 else None),
        "items": kwargs.pop("items", []),
        "custom": kwargs.pop("custom", None),
        "respondent": kwargs.pop("respondent", "Adult self-respondent (16+); proxy excluded" if family == "indresp" else "Youth self-respondent"),
        "number_of_items": kwargs.pop("number_of_items", len(kwargs.get("items", [])) or 1),
        "response_scale": kwargs.pop("response_scale", "See official value labels; negative codes are missing/proxy/inapplicable"),
        "scale_scoring": kwargs.pop("scale_scoring", "Use released item/derived score as documented"),
        "reverse_items": kwargs.pop("reverse_items", []),
        "same_wording": kwargs.pop("same_wording", "Yes across listed waves, subject to questionnaire verification"),
        "comparability": kwargs.pop("comparability", "Strong item-name and label continuity; distribution audit still required"),
        "proxy_allowed": kwargs.pop("proxy_allowed", "No"),
        "reliability_note": kwargs.pop("reliability_note", "Not applicable (single item)"),
        "priority": kwargs.pop("priority", "core"),
        "notes": kwargs.pop("notes", ""),
    }
    if kwargs:
        raise ValueError(f"Unknown construct options for {name}: {kwargs}")
    return row


GHQ_ITEMS = [f"scghq{x}" for x in "abcdefghijkl"]
WEMWBS_ITEMS = [f"scwemwb{x}" for x in "abcdefg"]
LONELY_ITEMS = ["sclonely", "scleftout", "scisolate"]
FRIEND_SUPPORT = ["scfrely", "scfopenup", "scfundstnd", "scfcritic", "scfletdwn", "scfannoy"]
FAMILY_SUPPORT = ["scrrely", "scrundstnd", "scrcritic", "scrletdwn"]
PARTNER_SUPPORT = ["scprely", "scpundstnd", "scpcritic", "scpletdwn"]
SELF_ESTEEM = ["ypesta", "ypestb", "ypestc", "ypeste", "ypestf", "ypesti", "ypestj", "ypestk"]

CONSTRUCTS = [
    C("GHQ-12 psychological distress (Likert)", "mental health", "indresp", ["scghq1_dv"], items=GHQ_ITEMS,
      number_of_items=12, response_scale="12 items coded 1-4; released Likert total 0-36, higher=more distress",
      scale_scoring="Primary candidate score: released scghq1_dv (Likert 0-36); do not switch to caseness after results",
      reliability_note="Cronbach alpha recomputed by wave from 12 aligned items"),
    C("GHQ-12 caseness", "mental health", "indresp", ["scghq2_dv"], items=GHQ_ITEMS,
      number_of_items=12, response_scale="Released GHQ caseness total 0-12",
      scale_scoring="Alternative scoring only; keep separate from Likert primary",
      reliability_note="Same items as GHQ Likert; alpha reported under Likert construct", priority="secondary"),
    C("SF-12 mental component summary", "mental health", "indresp", ["sf12mcs_dv"],
      response_scale="Norm-based continuous MCS, higher=better mental functioning",
      scale_scoring="Use released sf12mcs_dv algorithm", reliability_note="Composite scoring; internal consistency is not the preferred reliability estimand"),
    C("Overall life satisfaction", "subjective well-being", "indresp", ["sclfsato"],
      response_scale="1 completely dissatisfied to 7 completely satisfied", scale_scoring="Single item, retain 1-7"),
    C("Health satisfaction", "subjective well-being", "indresp", ["sclfsat1"],
      response_scale="1 completely dissatisfied to 7 completely satisfied", scale_scoring="Single item, retain 1-7", priority="secondary"),
    C("Income satisfaction", "subjective well-being", "indresp", ["sclfsat2"],
      response_scale="1 completely dissatisfied to 7 completely satisfied", scale_scoring="Single item, retain 1-7", priority="secondary"),
    C("Leisure-time satisfaction", "subjective well-being", "indresp", ["sclfsat7"],
      response_scale="1 completely dissatisfied to 7 completely satisfied", scale_scoring="Single item, retain 1-7", priority="secondary"),
    C("Short Warwick-Edinburgh mental well-being", "positive mental well-being", "indresp", ["swemwbs_dv"], items=WEMWBS_ITEMS,
      number_of_items=7, response_scale="Seven items, 1 none of the time to 5 all of the time; released transformed score",
      scale_scoring="Use released swemwbs_dv; higher=better well-being", reliability_note="Cronbach alpha recomputed by wave"),
    C("UCLA three-item loneliness", "subjective social disconnection", "indresp", LONELY_ITEMS, score_root=None, items=LONELY_ITEMS,
      number_of_items=3, response_scale="1 hardly ever/never, 2 some of the time, 3 often",
      scale_scoring="Mean of all 3 items, higher=more loneliness", reliability_note="Cronbach alpha recomputed by wave"),
    C("Perceived support from co-resident family", "social resources", "indresp", ["famsup"],
      response_scale="Single perceived-support item; inspect wave value labels before modeling", scale_scoring="Retain released coding", priority="secondary"),
    C("Friend support/strain scale", "social resources", "indresp", FRIEND_SUPPORT, score_root=None, items=FRIEND_SUPPORT,
      number_of_items=6, response_scale="1 a lot to 4 not at all",
      reverse_items=["scfrely", "scfopenup", "scfundstnd"],
      scale_scoring="Reverse positive-support items; average all 6 so higher=more support/less strain",
      reliability_note="Cronbach alpha recomputed by wave", same_wording="Yes in waves 2,5,11,14; long rotation gap",
      comparability="Item labels/scales align; four irregularly spaced waves"),
    C("Family support/strain scale", "social resources", "indresp", FAMILY_SUPPORT, score_root=None, items=FAMILY_SUPPORT,
      number_of_items=4, response_scale="1 a lot to 4 not at all", reverse_items=["scrrely", "scrundstnd"],
      scale_scoring="Reverse positive-support items; average all 4 so higher=more support/less strain",
      reliability_note="Cronbach alpha recomputed by wave", same_wording="Yes in waves 2,5,11,14; long rotation gap",
      comparability="Item labels/scales align; four irregularly spaced waves"),
    C("Partner support/strain scale", "relationship psychology", "indresp", PARTNER_SUPPORT, score_root=None, items=PARTNER_SUPPORT,
      number_of_items=4, response_scale="1 a lot to 4 not at all", reverse_items=["scprely", "scpundstnd"],
      scale_scoring="Reverse positive-support items; average all 4 so higher=more support/less strain",
      reliability_note="Cronbach alpha recomputed by wave", same_wording="Yes in waves 2,5,11,14; partnered respondents only",
      comparability="Item labels/scales align; four irregularly spaced waves"),
    C("Dyadic Adjustment Scale relationship satisfaction", "relationship psychology", "indresp", ["scdassat_dv"],
      number_of_items=8, response_scale="Released DAS satisfaction subscale",
      scale_scoring="Use released scdassat_dv", reliability_note="Released validated subscale; item-specific response scales preclude naive alpha",
      same_wording="Core subscale repeats in odd waves", comparability="Strong except relationship composition changes are substantive"),
    C("Dyadic Adjustment Scale relationship cohesion", "relationship psychology", "indresp", ["scdascoh_dv"],
      number_of_items=3, response_scale="Released DAS cohesion subscale",
      scale_scoring="Use released scdascoh_dv", reliability_note="Released validated subscale",
      same_wording="Available in waves 1,5,7,9,11,13,15; wave 3 derived score absent", priority="secondary"),
    C("Relationship happiness", "relationship psychology", "indresp", ["screlhappy"],
      response_scale="1 extremely unhappy to 7 perfect", scale_scoring="Single item, retain 1-7"),
    C("Current co-resident partnership", "relationship state/exposure", "indresp", ["ppid"], custom="partner_pointer",
      response_scale="Positive pidp=current co-resident partner; negative codes=no usable pointer",
      scale_scoring="Binary pointer availability plus exact partner pidp for dyadic linkage", proxy_allowed="No for psychological analyses",
      reliability_note="Administrative/derived link; reciprocal pointer audited separately"),
    C("Job satisfaction", "work psychology", "indresp", ["jbsat"],
      response_scale="1 completely dissatisfied to 7 completely satisfied", scale_scoring="Single item, employed respondents"),
    C("Perceived job security", "work psychology", "indresp", ["jbsec"],
      response_scale="Likelihood of losing job in next 12 months: 1 very likely to 4 very unlikely",
      scale_scoring="Retain 1-4; higher means greater perceived security (lower expected loss)",
      same_wording="Rotating even waves", comparability="Questionnaire wording and four response options align across audited even waves"),
    C("Job-related well-being: anxiety", "work psychology", "indresp", ["jwbs1_dv"],
      response_scale="Released 0 most anxious to 15 least anxious", scale_scoring="Use released subscale; higher=less anxiety",
      reliability_note="Released multi-item subscale", same_wording="Rotating even waves"),
    C("Job-related well-being: depression", "work psychology", "indresp", ["jwbs2_dv"],
      response_scale="Released subscale, higher=less depressed", scale_scoring="Use released subscale",
      reliability_note="Released multi-item subscale", same_wording="Rotating even waves"),
    C("Subjective current financial strain", "economic stress appraisal", "indresp", ["finnow"],
      response_scale="1 living comfortably to 5 finding it very difficult", scale_scoring="Retain 1-5; higher=more strain"),
    C("Expected financial change", "economic stress appraisal", "indresp", ["finfut"],
      response_scale="1 better off, 2 worse off, 3 about the same", scale_scoring="Recode to ordered worse/same/better only if preregistered", priority="secondary"),
    C("Long-standing illness/disability", "health adversity exposure", "indresp", ["health"],
      response_scale="1 yes, 2 no", scale_scoring="Binary; onset is no at t-1 to yes at t", reliability_note="Event/state indicator", priority="exposure"),
    C("Informal caregiving", "family/health adversity exposure", "indresp", ["aidhh", "aidxhh"], custom="binary_any_yes",
      response_scale="Each item 1 yes, 2 no (inside household; non-resident)", scale_scoring="Caregiver if either item=1; non-caregiver only if both=2",
      reliability_note="Event/state indicator", priority="exposure"),
    C("Caregiving intensity", "family/health adversity exposure", "indresp", ["aidhrs"],
      response_scale="Hours per week spent caring", scale_scoring="Continuous hours; inspect top-code and zero/inapplicable coding",
      reliability_note="Behavioral exposure", priority="exposure"),
    C("Neighbourhood cohesion", "community/social psychology", "indresp", ["nbrsnci_dv"],
      number_of_items=8, response_scale="Released 1 lowest to 5 highest cohesion", scale_scoring="Use released short Buckner score",
      reliability_note="Official labels report alpha .86 (wave 3) and .88 (other waves)"),
    C("Organisational membership", "social participation", "indresp", ["org"],
      response_scale="1 member of at least one listed organisation, 2 no", scale_scoring="Binary membership",
      same_wording="Repeated in waves 3,6,9,12", reliability_note="Behavioral indicator"),
    C("Harassment/unsafe-place exposure", "discrimination/social threat", "indresp", ["attacked_dv", "avoided_dv", "insulted_dv", "unsafe_dv"], custom="binary_any_yes",
      response_scale="Four last-12-month binary indicators", scale_scoring="Any of four=1; retain components for robustness",
      reliability_note="Formative exposure index; alpha inappropriate", same_wording="Repeated in waves 1,3,5,7,9,11"),
    C("Number of close friends", "objective social network", "indresp", ["closenum"],
      response_scale="Count", scale_scoring="Retain count; inspect top-code", reliability_note="Behavioral/network indicator", priority="secondary"),
    C("Regular internet use", "digital behavior", "indresp", ["netuse"],
      response_scale="Wave 1: 1 no access to 7 daily; later appearances: 1 yes, 2 no",
      scale_scoring="Do not pool: wording/coding and routed subsamples change across waves",
      reliability_note="Behavioral indicator", priority="hard-stop", notes="Variable exists in metadata across waves but later values are mostly structural inapplicables; not a comparable 15-wave construct"),
    C("Big Five traits", "personality", "indresp", ["big5a_dv", "big5c_dv", "big5e_dv", "big5n_dv", "big5o_dv"], custom="all_items",
      number_of_items=15, response_scale="Five released trait scores from 15 items", scale_scoring="Use five separate released traits",
      reliability_note="Trait-specific scales", priority="hard-stop", notes="Only wave 3: unsuitable for within-person trait-state analysis"),
    C("General self-efficacy", "self-regulation/personality", "indresp", [f"se{i}" for i in range(1, 11)], score_root=None, items=[f"se{i}" for i in range(1, 11)],
      number_of_items=10, response_scale="General Self-Efficacy items", scale_scoring="Score only after item-direction audit",
      reliability_note="Alpha calculable but only one wave", priority="hard-stop", notes="Only wave 5"),
    C("Risk preference", "decision/personality", "indresp", ["scriska"],
      response_scale="Prepared to take risks", scale_scoring="Single item", priority="hard-stop", notes="Only wave 1"),
    C("Generalised trust", "social psychology", "indresp", ["sctrust"],
      response_scale="Trustworthiness of others", scale_scoring="Single item", priority="hard-stop", notes="Only wave 1"),
    C("Sleep quality battery", "health behavior", "indresp", ["schrs_slph", "schrs_slpm", "sctslp_30m", "sctslp_wak", "scslp_qual"], custom="all_items",
      number_of_items=5, response_scale="Sleep hours and difficulty/quality items", scale_scoring="Do not label PSQI without validated scoring audit",
      reliability_note="Heterogeneous formative indicators", priority="hard-stop", notes="Only wave 1"),
    C("Youth SDQ total difficulties", "developmental mental health", "youth", ["ypsdqtd_dv"],
      number_of_items=20, response_scale="Released Strengths and Difficulties Questionnaire total 0-40",
      scale_scoring="Use released ypsdqtd_dv; higher=more difficulties",
      reliability_note="Multidomain total; report subscales rather than relying on alpha", same_wording="Odd waves 1-15"),
    C("Youth global happiness", "developmental well-being", "youth", ["yphlf"],
      response_scale="1 completely happy to 7 not at all happy", scale_scoring="Reverse if presenting higher=better", reliability_note="Single item"),
    C("Youth close-friend count", "developmental social network", "youth", ["ypnpal"],
      response_scale="Number of close friends", scale_scoring="Retain count; inspect top-code", reliability_note="Network indicator"),
    C("Youth family support", "developmental social resources", "youth", ["ypfamsup"],
      response_scale="Perceived family support", scale_scoring="Retain released coding", reliability_note="Single item", same_wording="Odd waves"),
    C("Youth loneliness", "developmental social disconnection", "youth", ["yplonely"],
      response_scale="How often feels lonely", scale_scoring="Retain released coding", reliability_note="Single item", same_wording="Waves 12-15 only"),
    C("Youth self-esteem", "developmental self-concept", "youth", SELF_ESTEEM, score_root=None, items=SELF_ESTEEM,
      number_of_items=8, response_scale="1 strongly agree to 4 strongly disagree",
      reverse_items=["ypesta", "ypestc", "ypestj", "ypestk"],
      scale_scoring="Reverse positive items, average all 8 so higher=better self-esteem",
      reliability_note="Cronbach alpha recomputed by wave", same_wording="Even waves 2-14"),
]

EVENT_ROOTS = [
    "jbstat", "mastat_dv", "marstat_dv", "ppid", "health", "aidhh", "aidxhh", "finnow",
    "fimnnet_dv", "nchild_dv", "qfhigh", "hhsize", "mvyr", "mvmnth", "intdatd_dv", "intdatm_dv", "intdaty_dv",
    "scghq1_dv", "age_dv", "sex_dv", "scflag_dv",
]


def alpha_complete(items: pd.DataFrame) -> tuple[float, int]:
    complete = items.dropna()
    if len(complete) < 100 or complete.shape[1] < 2:
        return np.nan, len(complete)
    variances = complete.var(axis=0, ddof=1)
    total_var = complete.sum(axis=1).var(ddof=1)
    if not np.isfinite(total_var) or total_var <= 0:
        return np.nan, len(complete)
    k = complete.shape[1]
    return float(k / (k - 1) * (1 - variances.sum() / total_var)), len(complete)


def valid_numeric(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    return x.where(x >= 0)


def load_dictionary() -> pd.DataFrame:
    return pd.read_csv(TABLES / "ukhls_variable_dictionary.csv")


def availability_map(dictionary: pd.DataFrame) -> dict[tuple[str, int], dict[str, str]]:
    x = dictionary[(dictionary.study == "UKHLS") & dictionary.wave.astype(str).str.fullmatch(r"\d+")].copy()
    x["wave"] = x.wave.astype(int)
    out = {}
    for (family, wave), group in x.groupby(["dataset_family", "wave"]):
        out[(family, wave)] = dict(zip(group.normalized_name, group.variable_name))
    return out


def construct_available(c: dict, roots_here: set[str]) -> bool:
    if c["custom"] in {"binary_any_yes", "all_items"}:
        return all(root in roots_here for root in c["roots"])
    if c["score_root"]:
        return c["score_root"] in roots_here
    return all(root in roots_here for root in c["items"])


def build_required_roots(avail: dict) -> dict[tuple[str, int], set[str]]:
    required: dict[tuple[str, int], set[str]] = {}
    for wave in range(1, 16):
        for family in ("indresp", "youth"):
            roots_here = set(avail.get((family, wave), {}))
            use = {"pidp"}
            if family == "indresp":
                use.add("ivfio")
                use.update(root for root in EVENT_ROOTS if root in roots_here)
            for c in CONSTRUCTS:
                if c["dataset_family"] != family or not construct_available(c, roots_here):
                    continue
                use.update(root for root in (c["roots"] + c["items"] + ([c["score_root"]] if c["score_root"] else [])) if root in roots_here)
            required[(family, wave)] = use & roots_here
    return required


def load_wave_data(avail: dict, required: dict) -> dict[tuple[str, int], pd.DataFrame]:
    cache = {}
    for (family, wave), roots in required.items():
        prefix = PREFIXES[wave - 1]
        columns = [avail[(family, wave)][root] for root in sorted(roots)]
        data = pd.read_stata(UKHLS / f"{prefix}_{family}.dta", columns=columns, convert_categoricals=False)
        rename = {avail[(family, wave)][root]: root for root in roots}
        data = data.rename(columns=rename)
        data.insert(1, "wave", wave)
        cache[(family, wave)] = data
        print(f"Loaded construct fields: {family} wave {wave} ({len(columns)} vars)", flush=True)
    return cache


def score_construct(data: pd.DataFrame, c: dict) -> tuple[pd.Series, pd.Series, pd.DataFrame | None]:
    roots = c["roots"]
    available = [root for root in roots if root in data]
    if c["dataset_family"] == "indresp":
        respondent_mask = data.ivfio.isin(SELF_CODES)
    else:
        respondent_mask = pd.Series(True, index=data.index)
    if c["custom"] == "binary_any_yes":
        raw = data[available].apply(pd.to_numeric, errors="coerce")
        score = pd.Series(np.nan, index=data.index)
        score[(raw == 1).any(axis=1)] = 1
        # Source yes/no items use 1/2; released harassment dummies use 1/0.
        zero_code = 0 if raw.eq(0).any().any() else 2
        score[(raw == zero_code).all(axis=1)] = 0
        eligible = respondent_mask & raw.isin([0, 1, 2]).any(axis=1)
        return score.where(respondent_mask), eligible, None
    if c["custom"] == "partner_pointer":
        raw = pd.to_numeric(data["ppid"], errors="coerce")
        score = pd.Series(0.0, index=data.index)
        score[raw > 0] = 1
        return score.where(respondent_mask), respondent_mask, None
    if c["custom"] == "all_items":
        raw = data[available].apply(pd.to_numeric, errors="coerce")
        valid = raw.where(raw >= 0)
        score = valid.mean(axis=1).where(valid.notna().all(axis=1))
        eligible = respondent_mask & (~raw.eq(-8).all(axis=1))
        return score.where(respondent_mask), eligible, valid.where(respondent_mask)
    if c["score_root"] and c["score_root"] in data:
        raw = pd.to_numeric(data[c["score_root"]], errors="coerce")
        score = raw.where(raw >= 0).where(respondent_mask)
        eligible = respondent_mask & raw.ne(-8) & raw.ne(-7)
        items = None
        if c["items"] and all(item in data for item in c["items"]):
            items = data[c["items"]].apply(valid_numeric)
            for item in c["reverse_items"]:
                if item in items:
                    items[item] = 5 - items[item]
            items = items.where(respondent_mask)
        return score, eligible, items
    items = data[c["items"]].apply(valid_numeric)
    for item in c["reverse_items"]:
        if item in items:
            items[item] = 5 - items[item]
    score = items.mean(axis=1).where(items.notna().all(axis=1)).where(respondent_mask)
    raw = data[c["items"]].apply(pd.to_numeric, errors="coerce")
    eligible = respondent_mask & (~raw.eq(-8).all(axis=1))
    return score, eligible, items.where(respondent_mask)


def build_construct_matrix(dictionary: pd.DataFrame, avail: dict, cache: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    score_frames = []
    for c in CONSTRUCTS:
        c_scores = []
        waves = []
        for wave in range(1, 16):
            roots_here = set(avail.get((c["dataset_family"], wave), {}))
            if not construct_available(c, roots_here):
                continue
            waves.append(wave)
            data = cache[(c["dataset_family"], wave)]
            score, eligible, items = score_construct(data, c)
            usable = score.notna()
            alpha_requested = "Cronbach alpha recomputed" in c["reliability_note"] or "Alpha calculable" in c["reliability_note"]
            alpha, alpha_n = alpha_complete(items) if items is not None and alpha_requested else (np.nan, 0)
            exact_roots = []
            for root in dict.fromkeys(c["roots"] + c["items"] + ([c["score_root"]] if c["score_root"] else [])):
                if root in avail[(c["dataset_family"], wave)]:
                    exact_roots.append(avail[(c["dataset_family"], wave)][root])
            labels = dictionary[
                (dictionary.study == "UKHLS")
                & (dictionary.dataset_family == c["dataset_family"])
                & (dictionary.wave.astype(str) == str(wave))
                & (dictionary.normalized_name.isin(dict.fromkeys(c["items"] or c["roots"])))
            ].variable_label.dropna().drop_duplicates().tolist()
            person_scores = pd.DataFrame({"pidp": data.pidp, "wave": wave, "construct_name": c["construct_name"], "score": score})
            person_scores = person_scores[person_scores.score.notna()]
            c_scores.append(person_scores)
            rows.append({
                "construct_name": c["construct_name"],
                "domain": c["domain"],
                "wave": wave,
                "wave_availability": "",  # filled after all waves are known
                "exact_variable_names": "|".join(exact_roots),
                "question_wording_or_official_labels": " | ".join(labels),
                "response_scale": c["response_scale"],
                "respondent": c["respondent"],
                "number_of_items": c["number_of_items"],
                "scale_scoring": c["scale_scoring"],
                "reliability_alpha": alpha,
                "reliability_complete_n": alpha_n,
                "reliability_note": c["reliability_note"],
                "self_response_or_youth_rows": int(data.ivfio.isin(SELF_CODES).sum()) if c["dataset_family"] == "indresp" else len(data),
                "eligible_nonstructural_n": int(eligible.sum()),
                "usable_n": int(usable.sum()),
                "missing_pct_among_eligible": 100 * (1 - usable.sum() / eligible.sum()) if eligible.sum() else np.nan,
                "same_wording_across_waves": c["same_wording"],
                "measurement_comparability": c["comparability"],
                "proxy_allowed": c["proxy_allowed"],
                "priority": c["priority"],
                "notes": c["notes"],
            })
        if not c_scores:
            continue
        all_scores = pd.concat(c_scores, ignore_index=True)
        counts = all_scores.groupby("pidp").wave.nunique()
        availability = ",".join(map(str, waves))
        usable_waves = [row["wave"] for row in rows if row["construct_name"] == c["construct_name"] and row["usable_n"] > 0]
        effective_availability = ",".join(map(str, usable_waves))
        for row in rows:
            if row["construct_name"] == c["construct_name"]:
                row["wave_availability"] = availability
                row["effective_wave_availability"] = effective_availability
                row["panel_n_2plus_waves"] = int((counts >= 2).sum())
                row["panel_n_3plus_waves"] = int((counts >= 3).sum())
                row["panel_n_all_effectively_available_waves"] = int((counts >= len(usable_waves)).sum()) if usable_waves else 0
        score_frames.append(all_scores)
    matrix = pd.DataFrame(rows)
    scores = pd.concat(score_frames, ignore_index=True)
    return matrix, scores


def adult_core(cache: dict) -> pd.DataFrame:
    frames = []
    for wave in range(1, 16):
        data = cache[("indresp", wave)].copy()
        data = data[data.ivfio.isin(SELF_CODES)].copy()
        data["wave"] = wave
        frames.append(data)
    core = pd.concat(frames, ignore_index=True, sort=False)
    valid_date = core.intdaty_dv.between(2008, 2030) & core.intdatm_dv.between(1, 12) & core.intdatd_dv.between(1, 31)
    core["interview_date"] = pd.NaT
    core.loc[valid_date, "interview_date"] = pd.to_datetime(dict(
        year=core.loc[valid_date, "intdaty_dv"], month=core.loc[valid_date, "intdatm_dv"], day=core.loc[valid_date, "intdatd_dv"]
    ), errors="coerce")
    core["caregiver"] = np.nan
    core.loc[(core.aidhh == 1) | (core.aidxhh == 1), "caregiver"] = 1
    core.loc[(core.aidhh == 2) & (core.aidxhh == 2), "caregiver"] = 0
    core["partnered"] = (pd.to_numeric(core.ppid, errors="coerce") > 0).astype(float)
    core.loc[core.ppid.isna(), "partnered"] = np.nan
    core = core.sort_values(["pidp", "wave"])
    return core


def transition_events(core: pd.DataFrame) -> pd.DataFrame:
    g = core.groupby("pidp", sort=False)
    previous = g.shift(1)
    consecutive = core.wave.sub(previous.wave).eq(1)
    actual_gap = (core.interview_date - previous.interview_date).dt.days / 30.4375

    def status_event(name, mask, sources, definition, precision, risk):
        events = core.loc[consecutive & mask].copy()
        gaps = actual_gap.loc[events.index]
        return {
            "event_name": name,
            "operational_definition": definition,
            "source_variables": sources,
            "usable_wave_transitions": "1-2 through 14-15",
            "estimated_events": len(events),
            "median_months_between_measurements": round(float(gaps.median()), 2) if len(gaps) else np.nan,
            "temporal_precision": precision,
            "event_centered_feasibility": "High" if len(events) >= 1000 else "Moderate" if len(events) >= 300 else "Low",
            "biggest_risk": risk,
        }

    current_employed = core.jbstat.isin([1, 2])
    prev_employed = previous.jbstat.isin([1, 2])
    current_unemployed = core.jbstat.eq(3)
    prev_unemployed = previous.jbstat.eq(3)
    current_retired = core.jbstat.eq(4)
    prev_retired = previous.jbstat.eq(4)
    current_health = core.health
    prev_health = previous.health
    current_care = core.caregiver
    prev_care = previous.caregiver
    current_partnered = core.partnered.eq(1)
    prev_partnered = previous.partnered.eq(1)
    current_unpartnered = core.partnered.eq(0)
    prev_unpartnered = previous.partnered.eq(0)
    events = [
        status_event("Job loss into unemployment", prev_employed & current_unemployed, "w_jbstat", "Paid/self-employed at t-1 and unemployed at t", "Interval-censored between interview dates", "Anticipation and re-employment can occur inside interval"),
        status_event("Re-employment", prev_unemployed & current_employed, "w_jbstat", "Unemployed at t-1 and paid/self-employed at t", "Interval-censored between interview dates", "Job spell may start months before current interview"),
        status_event("Retirement onset", ~prev_retired & current_retired & previous.jbstat.gt(0), "w_jbstat; retirement histories where available", "Not retired at t-1 and retired at t", "Interval-censored; exact retirement month is not repeated in every wave", "Voluntary/involuntary status not consistently available"),
        status_event("Long-standing illness/disability onset", prev_health.eq(2) & current_health.eq(1), "w_health", "No long-standing illness/disability at t-1; yes at t", "Interval-censored between interviews", "Self-report concept can fluctuate; diagnosis onset may predate report"),
        status_event("Long-standing illness/disability remission/reporting exit", prev_health.eq(1) & current_health.eq(2), "w_health", "Yes at t-1; no at t", "Interval-censored between interviews", "May reflect reporting change rather than recovery"),
        status_event("Caregiving onset", prev_care.eq(0) & current_care.eq(1), "w_aidhh; w_aidxhh", "Cares for nobody at t-1; cares for co-resident or non-resident at t", "Interval-censored between interviews", "Care recipient/intensity can change; role start date unavailable"),
        status_event("Caregiving exit", prev_care.eq(1) & current_care.eq(0), "w_aidhh; w_aidxhh", "Caregiver at t-1; non-caregiver at t", "Interval-censored between interviews", "Exit mechanism (recovery, institutionalisation, bereavement) not identified directly"),
        status_event("Relationship formation", prev_unpartnered & current_partnered, "w_ppid; w_mastat_dv", "No co-resident partner pointer at t-1; positive current partner pidp at t", "Interval-censored; relationship may begin before co-residence", "Conflates cohabitation with relationship start"),
        status_event("Separation/divorce", prev_partnered & current_unpartnered & core.mastat_dv.isin([4, 5, 7, 8]), "w_ppid; w_mastat_dv", "Co-resident partner at t-1; none at t and separated/divorced status", "Interval-censored between interviews", "Legal and residential transitions can be asynchronous"),
        status_event("Widowhood", prev_partnered & current_unpartnered & core.mastat_dv.isin([6, 9]), "w_ppid; w_mastat_dv; partner death flag", "Co-resident partner at t-1; no partner at t and widowed/surviving partner status", "Interval-censored; partner death linkage can sharpen date", "Some deaths and interviews may be close in time"),
        status_event("Partner change", prev_partnered & current_partnered & core.ppid.ne(previous.ppid), "w_ppid", "Positive partner pidp at both interviews but pidp changes", "Change known between interviews", "May include linkage correction; require reciprocal pointer validation"),
        status_event("Acute subjective financial strain", previous.finnow.isin([1, 2, 3]) & core.finnow.isin([4, 5]), "w_finnow", "Comfortable/alright/getting-by at t-1; quite/very difficult at t", "Annual appraisal transition", "Subjective appraisal overlaps psychologically with outcome"),
        status_event("Large net personal-income drop", previous.fimnnet_dv.gt(0) & core.fimnnet_dv.gt(0) & core.fimnnet_dv.lt(previous.fimnnet_dv * 0.70), "w_fimnnet_dv", "Nominal net monthly personal income falls >=30%", "Observed annual income snapshots", "Requires inflation adjustment and handling volatile income"),
        status_event("Own child enters household", previous.nchild_dv.ge(0) & core.nchild_dv.gt(previous.nchild_dv), "w_nchild_dv; newborn files", "Number of own children in household increases", "Between interviews; newborn history can sharpen", "Includes step/returning children unless newborn file confirms birth"),
        status_event("Own child leaves household", previous.nchild_dv.gt(core.nchild_dv) & core.nchild_dv.ge(0), "w_nchild_dv; family matrix", "Number of own children in household decreases", "Between interviews", "May reflect custody/household reconfiguration, not developmental leaving-home"),
    ]
    # Address move based on reported month/year falling after prior and no later than current interview.
    move_date = pd.to_datetime(dict(year=core.mvyr.where(core.mvyr > 1900), month=core.mvmnth.where(core.mvmnth.between(1, 12)), day=1), errors="coerce")
    moved = move_date.gt(previous.interview_date) & move_date.le(core.interview_date)
    events.append(status_event("Residential move", moved, "w_mvyr; w_mvmnth; interview dates", "Reported move-to-current-address date falls between consecutive interviews", "Month precision where reported", "Recall/carry-forward and multiple moves within interval"))
    return pd.DataFrame(events)


def relationship_matrix() -> pd.DataFrame:
    ds = pd.read_csv(TABLES / "ukhls_dataset_inventory.csv")
    linkage = pd.read_csv(TABLES / "ukhls_relationship_linkage.csv")
    couples = pd.read_csv(PROCESSED / "ukhls_couple_wave_linkage.csv.gz")
    pc = pd.read_csv(PROCESSED / "ukhls_parent_child_wave_linkage.csv.gz")
    ego = ds[(ds.study == "UKHLS") & (ds.dataset_family == "egoalt")]
    rows = [
        {"relationship_type": "Co-resident partner/couple", "source": "w_indall w_ppid", "waves": "1-15", "link_rows_or_pairs": len(couples), "quality": "Near-perfect reciprocal pointer; see wave metrics", "time_information": "Wave-specific current co-residence", "main_use": "Couple-wave actor-partner models; partner change"},
        {"relationship_type": "All within-household ego-alter dyads", "source": "w_egoalt", "waves": "1-15", "link_rows_or_pairs": int(ego.rows.sum()), "quality": "Unique on pidp-apidp; edited relationship variable available", "time_information": "Wave-specific", "main_use": "Sibling/parent/other co-resident comparisons"},
        {"relationship_type": "Natural parent-child, co-resident", "source": "w_indall fnpid/mnpid", "waves": "1-15", "link_rows_or_pairs": len(pc), "quality": "Pointers target enumerated household members", "time_information": "Wave-specific co-residence", "main_use": "Parent-child/youth transmission"},
        {"relationship_type": "Cross-wave family relationships", "source": "xhhrel", "waves": "Static compilation through wave 15", "link_rows_or_pairs": int(linkage.loc[linkage.scope == "xhhrel_static", "natural_parent_child_links"].iloc[0]), "quality": "Official error flags documented; static relationships", "time_information": "No relationship start/end dates", "main_use": "Recover non-resident parents/children/siblings, then merge outcomes by pidp"},
        {"relationship_type": "Youth-to-adult identity transition", "source": "pidp in w_youth and later w_indresp", "waves": "1-15", "link_rows_or_pairs": int(pd.read_csv(TABLES / "ukhls_person_flow.csv").youth_to_adult_interview_from_prior_wave.fillna(0).sum()), "quality": "Same pidp; adjacent-wave transitions counted", "time_information": "Wave/interview dates", "main_use": "Developmental transition and intergenerational follow-up"},
    ]
    return pd.DataFrame(rows)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    dictionary = load_dictionary()
    avail = availability_map(dictionary)
    required = build_required_roots(avail)
    cache = load_wave_data(avail, required)
    matrix, scores = build_construct_matrix(dictionary, avail, cache)
    core = adult_core(cache)
    events = transition_events(core)
    relationships = relationship_matrix()

    matrix.to_csv(TABLES / "ukhls_construct_wave_matrix.csv", index=False, encoding="utf-8-sig")
    events.to_csv(TABLES / "ukhls_event_matrix.csv", index=False, encoding="utf-8-sig")
    relationships.to_csv(TABLES / "ukhls_relationship_matrix.csv", index=False, encoding="utf-8-sig")
    scores.to_csv(PROCESSED / "ukhls_construct_scores_long.csv.gz", index=False, compression="gzip")
    core_out = core.copy()
    core_out["interview_date"] = core_out.interview_date.dt.strftime("%Y-%m-%d")
    core_out.to_csv(PROCESSED / "ukhls_adult_core_panel.csv.gz", index=False, compression="gzip")
    print(f"Construct-wave rows={len(matrix)}; score rows={len(scores):,}; events={len(events)}")


if __name__ == "__main__":
    main()
