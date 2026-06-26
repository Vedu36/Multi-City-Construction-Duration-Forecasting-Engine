# -*- coding: utf-8 -*-
"""
run_inference_v13.py
====================
Batch inference companion to construction_pipeline_v13.py.

Key differences vs run_inference_fixed.py
------------------------------------------
1. CITY-BASED model routing — loads per-city models; falls back to global.
2. Clean date handling — 1970-01-01 → NaT everywhere.
3. Categorical cleanup — UPPERCASE + strip on all text columns.
4. Delay imputation — city-average when either date is missing (not zero).
5. ConstructionCreditDays encoding — NULL=-1, 0=0, >0=log1p(value).
6. GarageDrop ordinal — No Drop=0, <1 Drop=1, 1-4 Drop=2, >4 Drop=3.
7. CommunityName cyclic sin/cos via frequency rank (fitted on training data,
   approximated at inference from community history rank).
8. SelectionsDelay — only for trim/interior; if either date is null → 0.
9. No DivisionName one-hot dummies in feature set.
10. Rolling averages — CityRollingAvg10 + CommunityRollingAvg10 (last 10).

Usage
-----
python run_inference_v13.py --fact   Ongoing_Actual.csv --stage  "Ongoing _Stage.csv" --models ./output_v13 --out  predictions_v13.csv [--no-log-space]
"""

import argparse
import json
import math
import warnings
from datetime import date, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# Column rename tables  (must match training notebook exactly)
# ──────────────────────────────────────────────────────────────────────────────
FACT_EXPECTED_COLS = [
    "JobNumber","JobReferenceNumber","State","County","City",
    "CommunityName","CommunityId",
    "Plan","Elevation","Plansqft","Lotsqft","LotType",
    "CornerLotYN","Swing","JobSwing","GarageDrop",
    "Section","Block","Lot","DivisionName",
    "ConstructionCreditDays","StageOfConstruction","StageOfConstructionId",
    "PreReleaseDate","ReleaseDate","EstimatedReleaseDate",
    "EstimatedConstructionDate","ActualConstructionDate","ActualStartDate",
    "EstimatedCompletionDate","ActualCompletionDate",
    "FoundationStartDate","FrameDate","CorniceDate","MechanicalDate",
    "SheetrockDate","TrimDate","InteriorDate",
    "CityPermitRequestedDate","CityPermitReceiveddDate",
    "CountyPermitRequesteddDate","CountyPermitReceiveddDate",
    "FoundationSentDate","FoundationReceivedDate",
    "HVACManualJDSrequestedDate","HVACManualJDSReceivedDate",
    "SelectionsRequestedDate","SelectionsReceivedDate",
    "EstimatedEscrowCloseDate","ActualEscrowCloseDate",
    "BasePrice","SalesPrice","IsRental",
    "BusinessProjectedCloseDate","PossibleCloseDate",
    "ZipCode","MarketCode","MarketName",
    "LotPremium","NegotiatingAllowance","RealtorBonus",
    "AnyAdditionalMasonryRequirement","AnyCustomRequirement","AnyCOPRequirement",
    "InventoryStatus","UnitStatus",
    "PlumbingRequestedDate","PlumbingReceivedDate",
    "SitePlanRequestedDate","SitePlanReceivedDate",
    "FoundationRevisionRequestedDate","FoundationRevisionReceivedDate",
    "EngineeringSentDate","EngineeringReceivedDate",
    "DeveloperRequestedDate","DeveloperReceivedDate",
    "ThermalPlanRequestedDate","ThermalPlanReceivedDate",
    "AtticVentingPlanRequestedDate","AtticVentingPlanReceivedDate",
    "EnergyREZCheckRequestedDate","EnergyREZCheckReceivedDate",
    "TreePermitRequestedDate","TreePermitReceivedDate",
    "CityPermitDenieddDate","CountyPermitDeniedDate",
    "PanelDate","PanelOnHoldDate","PanelOffHoldDate",
    "NoRexRequestedDate","NoRexReceivedDate",
]

STAGE_EXPECTED_COLS = [
    "JobNumber","WarehouseId","DivisionName","WarehouseName","StreetAddress",
    "ReleasedCompleteDate",
    "ActualFoundationDate","ActualFrameDate","ActualCorniceDate",
    "ActualMechanicalsDate","ActualSheetrockDate","ActualTrimDate",
    "ActualInteriorDate","ActualFinalDate","ActualCloseDate",
    "FourMonthAvgFoundationDate","FourMonthAvgFrameDate",
    "FourMonthAvgCorniceDate","FourMonthAvgMechanicalsDate",
    "FourMonthAvgSheetrockDate","FourMonthAvgTrimDate",
    "FourMonthAvgInteriorDate","FourMonthAvgFinalDate","FourMonthAvgCloseDate",
    "FoundationDuration","FrameDuration","CorniceDuration",
    "MechanicalsDuration","SheetrockDuration","TrimDuration",
    "InteriorDuration","FinalDuration","CloseDuration",
    "FoundationDurationAvgProj","FoundationDurationAvgDiv",
    "FrameDurationAvgProj","FrameDurationAvgDiv",
    "CorniceDurationAvgProj","CorniceDurationAvgDiv",
    "MechanicalsDurationAvgProj","MechanicalsDurationAvgDiv",
    "SheetrockDurationAvgProj","SheetrockDurationAvgDiv",
    "TrimDurationAvgProj","TrimDurationAvgDiv",
    "InteriorDurationAvgProj","InteriorDurationAvgDiv",
    "FinalDurationAvgProj","FinalDurationAvgDiv",
    "CloseDurationAvgProj","CloseDurationAvgDiv",
    "CurrentStage","LastCompletedStage","LastCompletedStageDate",
]

FACT_DATE_COLS = [
    "FoundationStartDate","FrameDate","CorniceDate","MechanicalDate",
    "SheetrockDate","TrimDate","InteriorDate","ActualCompletionDate",
    "CityPermitRequestedDate","CityPermitReceiveddDate",
    "CountyPermitRequesteddDate","CountyPermitReceiveddDate",
    "FoundationSentDate","FoundationReceivedDate",
    "PreReleaseDate","ReleaseDate","EstimatedReleaseDate",
    "EstimatedConstructionDate","ActualConstructionDate","ActualStartDate",
    "EstimatedCompletionDate","BusinessProjectedCloseDate",
    "SelectionsRequestedDate","SelectionsReceivedDate",
    "PlumbingRequestedDate","PlumbingReceivedDate",
    "SitePlanRequestedDate","SitePlanReceivedDate",
    "FoundationRevisionRequestedDate","FoundationRevisionReceivedDate",
    "EngineeringSentDate","EngineeringReceivedDate",
    "DeveloperRequestedDate","DeveloperReceivedDate",
    "ThermalPlanRequestedDate","ThermalPlanReceivedDate",
    "AtticVentingPlanRequestedDate","AtticVentingPlanReceivedDate",
    "EnergyREZCheckRequestedDate","EnergyREZCheckReceivedDate",
    "TreePermitRequestedDate","TreePermitReceivedDate",
    "CityPermitDenieddDate","CountyPermitDeniedDate",
    "PanelDate","PanelOnHoldDate","PanelOffHoldDate",
    "NoRexRequestedDate","NoRexReceivedDate",
    "HVACManualJDSrequestedDate","HVACManualJDSReceivedDate",
]

STAGE_DATE_COLS = [
    "FourMonthAvgFoundationDate","FourMonthAvgFrameDate",
    "FourMonthAvgCorniceDate","FourMonthAvgMechanicalsDate",
    "FourMonthAvgSheetrockDate","FourMonthAvgTrimDate",
    "FourMonthAvgInteriorDate","FourMonthAvgFinalDate","FourMonthAvgCloseDate",
    "ActualFoundationDate","ActualFrameDate","ActualCorniceDate",
    "ActualMechanicalsDate","ActualSheetrockDate","ActualTrimDate",
    "ActualInteriorDate","ActualFinalDate","ActualCloseDate",
]

CAT_COLS_FACT = [
    "City","CommunityName","DivisionName","GarageDrop",
    "Swing","JobSwing","Plan","LotType","InventoryStatus",
    "UnitStatus","MarketName","State","County",
]

_EPOCH = pd.Timestamp("1970-01-01")

# ──────────────────────────────────────────────────────────────────────────────
# Stage config
# ──────────────────────────────────────────────────────────────────────────────
STAGE_ORDER = ["foundation","frame","cornice","mechanicals","sheetrock","trim","interior"]

STAGE_TO_START_COL = {
    "foundation":  "FoundationStartDate",
    "frame":       "FrameDate",
    "cornice":     "CorniceDate",
    "mechanicals": "MechanicalDate",
    "sheetrock":   "SheetrockDate",
    "trim":        "TrimDate",
    "interior":    "InteriorDate",
}
STAGE_TO_END_COL = {
    "foundation":  "FrameDate",
    "frame":       "CorniceDate",
    "cornice":     "MechanicalDate",
    "mechanicals": "SheetrockDate",
    "sheetrock":   "TrimDate",
    "trim":        "InteriorDate",
    "interior":    "ActualCompletionDate",
}
STAGE_TO_4MO_DATE = {
    "foundation":  "FourMonthAvgFrameDate",
    "frame":       "FourMonthAvgCorniceDate",
    "cornice":     "FourMonthAvgMechanicalsDate",
    "mechanicals": "FourMonthAvgSheetrockDate",
    "sheetrock":   "FourMonthAvgTrimDate",
    "trim":        "FourMonthAvgInteriorDate",
    "interior":    "FourMonthAvgFinalDate",
}
STAGE_LABEL = {
    "foundation":"Foundation","frame":"Frame","cornice":"Cornice",
    "mechanicals":"Mechanicals","sheetrock":"Sheetrock","trim":"Trim",
    "interior":"Interior",
}

PRIOR_STAGE_DUR_COLS = {
    "foundation":  [],
    "frame":       ["FoundationDuration"],
    "cornice":     ["FoundationDuration","FrameDuration"],
    "mechanicals": ["FoundationDuration","FrameDuration","CorniceDuration"],
    "sheetrock":   ["FoundationDuration","FrameDuration","CorniceDuration","MechanicalsDuration"],
    "trim":        ["FoundationDuration","FrameDuration","CorniceDuration","MechanicalsDuration","SheetrockDuration"],
    "interior":    ["FoundationDuration","FrameDuration","CorniceDuration","MechanicalsDuration","SheetrockDuration","TrimDuration"],
}

SC_MAP = {
    "FoundationDuration":  ("FoundationStartDate","FrameDate"),
    "FrameDuration":       ("FrameDate","CorniceDate"),
    "CorniceDuration":     ("CorniceDate","MechanicalDate"),
    "MechanicalsDuration": ("MechanicalDate","SheetrockDate"),
    "SheetrockDuration":   ("SheetrockDate","TrimDate"),
    "TrimDuration":        ("TrimDate","InteriorDate"),
}
PRIOR_NAME_MAP = {
    "FoundationDuration":"PriorFoundationDur",
    "FrameDuration":"PriorFrameDur",
    "CorniceDuration":"PriorCorniceDur",
    "MechanicalsDuration":"PriorMechanicalsDur",
    "SheetrockDuration":"PriorSheetrockDur",
    "TrimDuration":"PriorTrimDur",
}

SELECTIONS_STAGES = {"trim","interior"}


# ──────────────────────────────────────────────────────────────────────────────
# Holiday helpers  (identical to training notebook)
# ──────────────────────────────────────────────────────────────────────────────
def _easter(year):
    a=year%19; b=year//100; c=year%100; d=b//4; e=b%4
    f=(b+8)//25; g=(b-f+1)//3; h=(19*a+b-d-g+15)%30
    i=c//4; k=c%4; l=(32+2*e+2*i-h-k)%7; m=(a+11*h+22*l)//451
    return date(year,(h+l-7*m+114)//31,((h+l-7*m+114)%31)+1)


def _company_holidays(year):
    h = set()
    h.add(date(year,1,1))
    jan=pd.date_range(f"{year}-01-01",f"{year}-01-31",freq="W-MON"); h.add(jan[2].date())
    h.add(_easter(year)-timedelta(days=2))
    may=pd.date_range(f"{year}-05-01",f"{year}-05-31",freq="W-MON"); h.add(may[-1].date())
    h.add(date(year,7,4))
    sep=pd.date_range(f"{year}-09-01",f"{year}-09-30",freq="W-MON"); h.add(sep[0].date())
    nov=pd.date_range(f"{year}-11-01",f"{year}-11-30",freq="W-THU"); tg=nov[3].date()
    h.add(tg); h.add(tg+timedelta(days=1))
    for d_ in range(9): h.add(date(year,12,24)+timedelta(days=d_))
    return h


_HOLIDAYS         = {y: _company_holidays(y) for y in range(2023, 2035)}
_ALL_HOLIDAY_DATES = set()
for _hset in _HOLIDAYS.values():
    _ALL_HOLIDAY_DATES.update(_hset)


def _holidays_in_window(start, end):
    if pd.isna(start) or pd.isna(end): return 0
    s=pd.to_datetime(start).date(); e=pd.to_datetime(end).date()
    if e < s: return 0
    return sum(1 for y in range(s.year,e.year+1)
               for h in _HOLIDAYS.get(y,[]) if s<=h<=e)


def _workdays_in_window(start, end):
    if pd.isna(start) or pd.isna(end): return 0
    s=pd.to_datetime(start).date(); e=pd.to_datetime(end).date()
    if e < s: return 0
    return int(np.busday_count(s, e+timedelta(days=1)))


def _weekends_in_window(start, end):
    if pd.isna(start) or pd.isna(end): return 0
    return max(0,(pd.to_datetime(end)-pd.to_datetime(start)).days+1
               -_workdays_in_window(start,end))


def _days_to_next_holiday(dt):
    if pd.isna(dt): return 30
    d_=pd.to_datetime(dt).date()
    for i in range(1,60):
        if (d_+timedelta(days=i)) in _ALL_HOLIDAY_DATES: return i
    return 60


def _holidays_in_next_n(dt, n=30):
    if pd.isna(dt): return 0
    d_=pd.to_datetime(dt).date()
    return sum(1 for i in range(1,n+1)
               if (d_+timedelta(days=i)) in _ALL_HOLIDAY_DATES)


def _is_thanksgiving_week(dt):
    if pd.isna(dt): return 0
    d_=pd.to_datetime(dt).date()
    if d_.month != 11: return 0
    nov=pd.date_range(f"{d_.year}-11-01",f"{d_.year}-11-30",freq="W-THU")
    tg=nov[3].date(); ws=tg-timedelta(days=tg.weekday()); we=ws+timedelta(days=6)
    return int(ws<=d_<=we)


def _is_christmas_week(dt):
    if pd.isna(dt): return 0
    d_=pd.to_datetime(dt).date()
    return int(d_ in _ALL_HOLIDAY_DATES and (d_.month==12 or (d_.month==1 and d_.day==1)))


def _days_to_year_end(dt):
    if pd.isna(dt): return 180
    d_=pd.to_datetime(dt).date()
    return max(0,(date(d_.year,12,31)-d_).days)


# ──────────────────────────────────────────────────────────────────────────────
# Data loaders
# ──────────────────────────────────────────────────────────────────────────────
def _drop_header_row(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) > 0 and df.iloc[0].tolist() == list(df.columns):
        df = df.iloc[1:].reset_index(drop=True)
    return df


def _clean_dates(df: pd.DataFrame, date_cols: list) -> pd.DataFrame:
    """Parse dates, coerce errors, and replace 1970-01-01 with NaT."""
    for col in date_cols:
        if col in df.columns:
            s = pd.to_datetime(df[col], errors="coerce")
            s[s == _EPOCH] = pd.NaT
            df[col] = s
    return df


def _clean_categoricals(df: pd.DataFrame, cat_cols: list) -> pd.DataFrame:
    """UPPERCASE + strip all categorical text columns."""
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str).str.strip().str.upper()
    return df


def load_fact(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, low_memory=False)
    df = _drop_header_row(df)
    if len(df.columns) == len(FACT_EXPECTED_COLS):
        df.columns = FACT_EXPECTED_COLS
        print(f"  load_fact: canonical rename ({len(df.columns)} cols)")
    else:
        print(f"  [WARN] load_fact: {len(df.columns)} cols vs expected "
              f"{len(FACT_EXPECTED_COLS)} – using file's own names")
    df = _clean_dates(df, FACT_DATE_COLS)
    df = _clean_categoricals(df, CAT_COLS_FACT)
    if "JobNumber" not in df.columns:
        raise ValueError(f"'JobNumber' missing. First 8 cols: {list(df.columns[:8])}")
    df["JobNumber"] = df["JobNumber"].astype(str).str.strip()
    if "JobReferenceNumber" in df.columns:
        df["JobReferenceNumber"] = df["JobReferenceNumber"].astype(str).str.strip()
    return df


def load_stage(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, low_memory=False)
    df = _drop_header_row(df)
    if len(df.columns) == len(STAGE_EXPECTED_COLS):
        df.columns = STAGE_EXPECTED_COLS
        print(f"  load_stage: canonical rename ({len(df.columns)} cols)")
    else:
        print(f"  [WARN] load_stage: {len(df.columns)} cols vs expected "
              f"{len(STAGE_EXPECTED_COLS)} – using file's own names")
    df = _clean_dates(df, STAGE_DATE_COLS)
    df["JobNumber"] = df["JobNumber"].astype(str).str.strip()
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Community / City history index
# ──────────────────────────────────────────────────────────────────────────────
def build_history(fact_df: pd.DataFrame) -> dict:
    """
    Returns nested dict:
      history[city][community_id][stage] = [dur1, dur2, ...]   (past durations)
      history[city]["_total"]            = [total_dur, ...]
      history["_global"][stage]          = city-level average delay for each delay col

    Also builds city-level average delay for imputation:
      delay_city_avg[city][delay_col] = float
    """
    dur_map = {
        "foundation":  ("FoundationStartDate","FrameDate"),
        "frame":       ("FrameDate","CorniceDate"),
        "cornice":     ("CorniceDate","MechanicalDate"),
        "mechanicals": ("MechanicalDate","SheetrockDate"),
        "sheetrock":   ("SheetrockDate","TrimDate"),
        "trim":        ("TrimDate","InteriorDate"),
        "interior":    ("InteriorDate","ActualCompletionDate"),
    }
    delay_pairs = [
        ("NoRexRequestedDate","NoRexReceivedDate","NoRexDelayDays"),
        ("TreePermitRequestedDate","TreePermitReceivedDate","TreePermitDelayDays"),
        ("EnergyREZCheckRequestedDate","EnergyREZCheckReceivedDate","EnergyREZCheckDelayDays"),
        ("AtticVentingPlanRequestedDate","AtticVentingPlanReceivedDate","AtticVentingDelayDays"),
        ("ThermalPlanRequestedDate","ThermalPlanReceivedDate","ThermalPlanDelayDays"),
        ("DeveloperRequestedDate","DeveloperReceivedDate","DeveloperDelayDays"),
        ("EngineeringSentDate","EngineeringReceivedDate","EngineeringDelayDays"),
        ("FoundationRevisionRequestedDate","FoundationRevisionReceivedDate","FoundationRevisionDelayDays"),
        ("SitePlanRequestedDate","SitePlanReceivedDate","SitePlanDelayDays"),
        ("HVACManualJDSrequestedDate","HVACManualJDSReceivedDate","HVACManualJDSDelayDays"),
        ("PlumbingRequestedDate","PlumbingReceivedDate","PlumbingDelayDays"),
    ]

    city_history    = {}   # city -> {community -> {stage -> [durs]}, "_total" -> [...]}
    delay_city_acc  = {}   # city -> {delay_col -> [values]}
    global_delay_acc= {}   # delay_col -> [values]

    for _, row in fact_df.iterrows():
        city   = str(row.get("City","")).strip().upper() or "_UNKNOWN"
        comm   = str(row.get("CommunityId","")) or "_UNKNOWN"

        if city not in city_history:
            city_history[city] = {"_total": []}
        if comm not in city_history[city]:
            city_history[city][comm] = {s: [] for s in STAGE_ORDER}

        # Stage durations
        for stage, (sc, ec) in dur_map.items():
            s_dt = pd.to_datetime(row.get(sc), errors="coerce")
            e_dt = pd.to_datetime(row.get(ec), errors="coerce")
            if pd.notna(s_dt) and pd.notna(e_dt) and s_dt != _EPOCH and e_dt != _EPOCH:
                d = (e_dt - s_dt).days
                if d > 0:
                    city_history[city][comm][stage].append(float(d))

        # Total duration
        fs = pd.to_datetime(row.get("FoundationStartDate"), errors="coerce")
        ac = pd.to_datetime(row.get("ActualCompletionDate"), errors="coerce")
        if pd.notna(fs) and pd.notna(ac) and fs != _EPOCH and ac != _EPOCH:
            td = (ac - fs).days
            if td > 0:
                city_history[city]["_total"].append(float(td))

        # Delay columns (city-level accumulator for imputation)
        if city not in delay_city_acc:
            delay_city_acc[city] = {out: [] for _,_,out in delay_pairs}
        for req, rec, out in delay_pairs:
            r_ = pd.to_datetime(row.get(req), errors="coerce")
            c_ = pd.to_datetime(row.get(rec), errors="coerce")
            if pd.notna(r_) and pd.notna(c_) and r_ != _EPOCH and c_ != _EPOCH:
                d_ = (c_ - r_).days
                if d_ >= 0:
                    delay_city_acc[city][out].append(float(d_))
                    global_delay_acc.setdefault(out, []).append(float(d_))

    # Compute city-average delays
    delay_city_avg = {}
    global_delay_avg = {k: float(np.mean(v)) if v else 0.0
                        for k, v in global_delay_acc.items()}
    for city, d_acc in delay_city_acc.items():
        delay_city_avg[city] = {
            k: (float(np.mean(v)) if v else global_delay_avg.get(k, 0.0))
            for k, v in d_acc.items()
        }

    # Compute community frequency ranks for cyclic encoding
    comm_freq: dict[str, int] = {}
    for city_d in city_history.values():
        for comm, stages in city_d.items():
            if comm == "_total":
                continue
            total = sum(len(v) for v in stages.values())
            comm_freq[comm] = comm_freq.get(comm, 0) + total

    # Rank: most-frequent = rank 0
    sorted_comms = sorted(comm_freq.keys(), key=lambda x: -comm_freq.get(x, 0))
    comm_rank    = {c: i for i, c in enumerate(sorted_comms)}
    n_comms      = max(len(comm_rank), 1)

    return {
        "city":            city_history,
        "delay_city_avg":  delay_city_avg,
        "global_delay_avg":global_delay_avg,
        "comm_rank":       comm_rank,
        "n_comms":         n_comms,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Model bundle
# ──────────────────────────────────────────────────────────────────────────────
class V13Predictor:
    """
    Loads output_v13/ and exposes predict_stage().

    Expected files:
      model_{stage}.pkl                  global stage model
      model_{stage}_{CITY_SAFE}.pkl      per-city stage models
      stage_features.json
      encoder_bundle.pkl
      outlier_bounds.json
      city_map.json
      model_types.json
    """

    def __init__(self, model_dir: str, log_space: bool = True):
        self.model_dir = Path(model_dir)
        self.log_space = log_space

        self.encoder_bundle  = joblib.load(self.model_dir / "encoder_bundle.pkl")
        self.stage_features  = json.loads(
            (self.model_dir / "stage_features.json").read_text())
        self.outlier_bounds  = json.loads(
            (self.model_dir / "outlier_bounds.json").read_text())
        self.city_map        = json.loads(
            (self.model_dir / "city_map.json").read_text())

        mt_fp = self.model_dir / "model_types.json"
        self.model_types = json.loads(mt_fp.read_text()) if mt_fp.exists() else {}

        # Load stage models
        self.global_models = {}
        self.city_models   = {}   # {stage: {city_safe: model}}
        for stage in STAGE_ORDER:
            gp = self.model_dir / f"model_{stage}.pkl"
            if gp.exists():
                self.global_models[stage] = joblib.load(gp)
            self.city_models[stage] = {}
            for p in sorted(self.model_dir.glob(f"model_{stage}_*.pkl")):
                city_safe = p.stem[len(f"model_{stage}_"):]
                self.city_models[stage][city_safe] = joblib.load(p)

        print(f"Model bundle loaded from '{model_dir}'")
        for stage in STAGE_ORDER:
            n = len(self.city_models.get(stage, {}))
            found = "OK" if stage in self.global_models else "MISSING"
            print(f"  [{stage:<14}] global={found}  city_models={n}")

    @staticmethod
    def _safe_city(city: str) -> str:
        return city.replace(" ","_").replace("/","_")

    def _get_model(self, stage: str, model_city: str):
        safe = self._safe_city(model_city)
        return (self.city_models[stage].get(safe)
                or self.global_models.get(stage))

    def _predict_raw(self, model, X: pd.DataFrame) -> np.ndarray:
        preds = model.predict(X)
        if self.log_space:
            preds = np.expm1(preds)
        return np.clip(preds, 1.0, None)

    # ── Feature builder ────────────────────────────────────────────────────────
    def _build_row(self, raw: dict, stage: str,
                   history: dict, working_raw: dict) -> pd.DataFrame:
        """
        Build a single-row feature DataFrame for one stage prediction.
        raw         : original job row (immutable)
        working_raw : mutable copy used for chaining (may contain ML-predicted dates)
        history     : output of build_history()
        """
        eb          = self.encoder_bundle
        feat_list   = self.stage_features.get(stage, [])
        start_col   = STAGE_TO_START_COL[stage]
        start_dt    = pd.to_datetime(working_raw.get(start_col), errors="coerce")
        city        = str(raw.get("City","")).strip().upper() or "_UNKNOWN"
        model_city  = self.city_map.get(city, city)
        comm_id     = str(raw.get("CommunityId","")) or "_UNKNOWN"

        row = {}

        # ── Calendar ──────────────────────────────────────────────────────────
        if pd.notna(start_dt):
            m = start_dt.month
            row.update({
                "StartMonth":      m,
                "StartQuarter":    start_dt.quarter,
                "StartDayOfWeek":  start_dt.dayofweek,
                "StartYear":       start_dt.year,
                "StartWeekOfYear": start_dt.isocalendar()[1],
                "IsSummer":   int(m in [6,7,8]),
                "IsWinter":   int(m in [12,1,2]),
                "IsSpring":   int(m in [3,4,5]),
                "IsFall":     int(m in [9,10,11]),
                "IsHolidayMonth": int(m in [12,1]),
                "IsQ4":       int(m in [10,11,12]),
                "IsNovDec":   int(m in [11,12]),
            })
            if m in [3,4,5]:      senc=0; s0=pd.Timestamp(start_dt.year,3,1)
            elif m in [6,7,8]:    senc=1; s0=pd.Timestamp(start_dt.year,6,1)
            elif m in [9,10,11]:  senc=2; s0=pd.Timestamp(start_dt.year,9,1)
            else:
                senc=3
                s0=(pd.Timestamp(start_dt.year,12,1) if m==12
                    else pd.Timestamp(start_dt.year-1,12,1))
            row["SeasonEnc"]      = senc
            row["DaysIntoSeason"] = max(0,(start_dt - s0).days)
        else:
            for k in ["StartMonth","StartQuarter","StartDayOfWeek","StartYear",
                      "StartWeekOfYear","IsSummer","IsWinter","IsSpring","IsFall",
                      "IsHolidayMonth","IsQ4","IsNovDec","SeasonEnc","DaysIntoSeason"]:
                row[k] = 0

        # ── Holiday proximity ─────────────────────────────────────────────────
        row["DaysToNextHoliday"]  = _days_to_next_holiday(start_dt)
        row["HolidaysNext14d"]    = _holidays_in_next_n(start_dt, 14)
        row["HolidaysNext30d"]    = _holidays_in_next_n(start_dt, 30)
        row["IsThanksgivingWeek"] = _is_thanksgiving_week(start_dt)
        row["IsChristmasWeek"]    = _is_christmas_week(start_dt)
        row["DaysToYearEnd"]      = _days_to_year_end(start_dt)

        # ── Rolling averages from history ─────────────────────────────────────
        city_hist  = history["city"].get(city, {})
        comm_hist  = city_hist.get(comm_id, {})
        stage_city = [d for c in city_hist.values()
                      if isinstance(c, dict)
                      for d in c.get(stage, [])]
        stage_comm = comm_hist.get(stage, [])

        city_rolling  = (float(np.mean(stage_city[-10:])) if len(stage_city) >= 3
                         else float(np.mean(stage_city)) if stage_city else 30.0)
        comm_rolling  = (float(np.mean(stage_comm[-10:])) if len(stage_comm) >= 3
                         else float(np.mean(stage_comm)) if stage_comm
                         else city_rolling)

        row["CityRollingAvg10"]      = city_rolling
        row["CommunityRollingAvg10"] = comm_rolling

        avg_d = max(1, int(round(city_rolling)))
        end_est = (start_dt + pd.Timedelta(days=avg_d)) if pd.notna(start_dt) else None
        row["WorkdaysInAvgWindow"]  = _workdays_in_window(start_dt, end_est)
        row["WeekendsInAvgWindow"]  = _weekends_in_window(start_dt, end_est)
        row["HolidaysInAvgWindow"]  = _holidays_in_window(start_dt, end_est)
        row["NetWorkdaysAvgWindow"] = max(1, row["WorkdaysInAvgWindow"] - row["HolidaysInAvgWindow"])
        row["WeekendRatioAvg"]      = row["WeekendsInAvgWindow"] / (row["WorkdaysInAvgWindow"] + 1)

        # ── Prior stage durations ─────────────────────────────────────────────
        needed = PRIOR_STAGE_DUR_COLS.get(stage, [])
        cum    = 0.0
        for src, name in PRIOR_NAME_MAP.items():
            if src in needed:
                sc_, ec_ = SC_MAP[src]
                s_dt = pd.to_datetime(working_raw.get(sc_), errors="coerce")
                e_dt = pd.to_datetime(working_raw.get(ec_), errors="coerce")
                dur  = max(1.0, (e_dt-s_dt).days) if (pd.notna(s_dt) and pd.notna(e_dt)) else 30.0
                row[name] = dur; cum += dur
            else:
                row[name] = 0.0
        row["PriorStagesCompleted"] = len(needed)
        row["PriorStageCumDays"]    = cum

        # ── SelectionsDelay (trim / interior only; else 0) ───────────────────
        if stage in SELECTIONS_STAGES:
            req_s = pd.to_datetime(raw.get("SelectionsRequestedDate"), errors="coerce")
            rec_s = pd.to_datetime(raw.get("SelectionsReceivedDate"),  errors="coerce")
            if pd.notna(req_s) and pd.notna(rec_s):
                sd = max(0.0, (rec_s - req_s).days)
            else:
                sd = 0.0   # either date missing → 0
            row["SelectionsDelayDays"] = sd
        else:
            row["SelectionsDelayDays"] = 0.0

        # ── Helper for safe numeric reads ────────────────────────────────────
        def _n(key, default=0.0):
            try: return float(raw.get(key) or default)
            except: return default

        def _dt(col):
            v = raw.get(col)
            if v is None: return pd.NaT
            ts = pd.to_datetime(v, errors="coerce")
            return pd.NaT if ts == _EPOCH else ts

        # ── Delay features with city-average imputation ───────────────────────
        delay_city_avg = history["delay_city_avg"].get(city, {})
        global_delay_avg = history.get("global_delay_avg", {})

        def _delay_imputed(req_col, rec_col, out_key):
            r_ = _dt(req_col); c_ = _dt(rec_col)
            if pd.notna(r_) and pd.notna(c_):
                return max(0.0, (c_ - r_).days)
            # Either side missing → city average, then global average
            return (delay_city_avg.get(out_key)
                    or global_delay_avg.get(out_key, 0.0))

        row["NoRexDelayDays"]              = _delay_imputed("NoRexRequestedDate","NoRexReceivedDate","NoRexDelayDays")
        row["TreePermitDelayDays"]         = _delay_imputed("TreePermitRequestedDate","TreePermitReceivedDate","TreePermitDelayDays")
        row["EnergyREZCheckDelayDays"]     = _delay_imputed("EnergyREZCheckRequestedDate","EnergyREZCheckReceivedDate","EnergyREZCheckDelayDays")
        row["AtticVentingDelayDays"]       = _delay_imputed("AtticVentingPlanRequestedDate","AtticVentingPlanReceivedDate","AtticVentingDelayDays")
        row["ThermalPlanDelayDays"]        = _delay_imputed("ThermalPlanRequestedDate","ThermalPlanReceivedDate","ThermalPlanDelayDays")
        row["DeveloperDelayDays"]          = _delay_imputed("DeveloperRequestedDate","DeveloperReceivedDate","DeveloperDelayDays")
        row["EngineeringDelayDays"]        = _delay_imputed("EngineeringSentDate","EngineeringReceivedDate","EngineeringDelayDays")
        row["FoundationRevisionDelayDays"] = _delay_imputed("FoundationRevisionRequestedDate","FoundationRevisionReceivedDate","FoundationRevisionDelayDays")
        row["SitePlanDelayDays"]           = _delay_imputed("SitePlanRequestedDate","SitePlanReceivedDate","SitePlanDelayDays")
        row["HVACManualJDSDelayDays"]      = _delay_imputed("HVACManualJDSrequestedDate","HVACManualJDSReceivedDate","HVACManualJDSDelayDays")
        row["PlumbingDelayDays"]           = _delay_imputed("PlumbingRequestedDate","PlumbingReceivedDate","PlumbingDelayDays")

        # ── ConstructionCreditDays encoding ──────────────────────────────────
        ccd_raw = raw.get("ConstructionCreditDays")
        try:
            ccd_val = float(ccd_raw)
        except (TypeError, ValueError):
            ccd_val = None
        if ccd_val is None:
            row["ConstructionCreditEnc"] = -1.0   # NULL → -1
        elif ccd_val == 0:
            row["ConstructionCreditEnc"] = 0.0
        else:
            row["ConstructionCreditEnc"] = float(np.log1p(max(0.0, ccd_val)))

        # ── PlanSqFt ─────────────────────────────────────────────────────────
        fill_psf = float(eb.get("plansqft_fill", 1800))
        row["PlanSqFt"] = max(1.0, _n("Plansqft", fill_psf))

        # ── GarageDrop ordinal ────────────────────────────────────────────────
        gd = str(raw.get("GarageDrop","")).strip().upper()
        if ">4" in gd or "> 4" in gd:     row["GarageDropOrd"] = 3
        elif "1-4" in gd or "1 - 4" in gd: row["GarageDropOrd"] = 2
        elif "<1" in gd or "< 1" in gd:    row["GarageDropOrd"] = 1
        else:                               row["GarageDropOrd"] = 0

        # ── Plan parsing ──────────────────────────────────────────────────────
        plan_str = str(raw.get("Plan","")).strip().upper() or "UNKNOWN"
        if plan_str not in ("UNKNOWN","NAN","") and plan_str[-1].isalpha():
            plan_code = pd.to_numeric(plan_str[:-1], errors="coerce")
            plan_ltr  = plan_str[-1]
        else:
            plan_code = pd.to_numeric(plan_str, errors="coerce")
            plan_ltr  = ""
        plan_code = float(plan_code) if pd.notna(plan_code) else 0.0
        row["PlanCode"] = plan_code

        # ── Elevation ────────────────────────────────────────────────────────
        elev = int(_n("Elevation", 0))
        row["ElevationFamily"]  = elev // 10
        row["ElevationVariant"] = elev % 10

        # ── CornerLot ─────────────────────────────────────────────────────────
        row["CornerLot"] = int(str(raw.get("CornerLotYN","")).strip().upper()
                               in ["Y","YES","1","TRUE"])

        # ── Community cyclic encoding ─────────────────────────────────────────
        comm_rank = history.get("comm_rank", {})
        n_comms   = history.get("n_comms", 1)
        rank_     = comm_rank.get(comm_id, n_comms - 1)
        angle_    = (rank_ / n_comms) * 2 * math.pi
        row["CommunityNameSin"] = math.sin(angle_)
        row["CommunityNameCos"] = math.cos(angle_)

        # ── Ordinal-encoded categoricals ─────────────────────────────────────
        def _enc(enc_key, val_str, col_name):
            enc = eb.get(enc_key)
            if enc is None: return -1
            try:
                return int(enc.transform(
                    pd.DataFrame({col_name: [val_str]})).ravel()[0])
            except Exception:
                return -1

        row["CityEnc"]        = _enc("city",         city,       "City")
        row["ModelcityEnc"]   = _enc("model_city",   model_city, "ModelCity")
        row["SwingEnc"]       = _enc("swing",
                                     str(raw.get("Swing","")).strip().upper(),
                                     "Swing")
        row["PlanletterencEnc"] = _enc("plan_letter", plan_ltr or "UNKNOWN", "PlanLetter")
        row["CommunityEnc"]   = _enc("community",    comm_id,    "CommunityName")

        city_plan_str = city + "_" + str(int(plan_code))
        row["CityPlanEnc"] = _enc("cityplan", city_plan_str, "CityPlan")

        m_str = str(start_dt.month if pd.notna(start_dt) else 1)
        def _seas(m_):
            m_ = int(m_)
            return ("Spring" if m_ in [3,4,5] else
                    "Summer" if m_ in [6,7,8] else
                    "Fall"   if m_ in [9,10,11] else "Winter")

        row["CityMonthEnc"]  = _enc("citymonth",
                                     model_city+"_"+m_str, "CityMonth")
        row["CitySeasonEnc"] = _enc("cityseason",
                                     model_city+"_"+_seas(m_str), "CitySeason")

        # ── City-level leakage-free aggregates ───────────────────────────────
        total_hist = city_hist.get("_total", [])
        row["City_ExpandingMean"] = float(np.mean(total_hist))       if total_hist        else 120.0
        row["City_ExpandingStd"]  = float(np.std(total_hist))        if len(total_hist)>1 else 20.0
        row["City_Velocity"]      = (120.0 / max(1.0, row["City_ExpandingMean"])
                                     if row["City_ExpandingMean"] > 0 else 1.0)
        row["City_Trend"]         = float(np.mean(total_hist[-50:])) if len(total_hist)>=5 else row["City_ExpandingMean"]
        row["City_Lag1Duration"]  = float(total_hist[-1])            if total_hist        else row["City_ExpandingMean"]

        # Fill any remaining expected feature with 0
        for feat in feat_list:
            if feat not in row:
                row[feat] = 0.0

        X = pd.DataFrame([row])
        for f in feat_list:
            if f not in X.columns:
                X[f] = 0.0
        return X[feat_list]

    # ── Public API ─────────────────────────────────────────────────────────────
    def predict_stage(self, raw: dict, stage: str,
                      history: dict, working_raw: dict) -> dict:
        city       = str(raw.get("City","")).strip().upper() or "_UNKNOWN"
        model_city = self.city_map.get(city, city)
        model      = self._get_model(stage, model_city)
        if model is None:
            return {"predicted_days": float("nan"), "projected_completion": None}

        X      = self._build_row(raw, stage, history, working_raw)
        days   = float(self._predict_raw(model, X)[0])
        start  = pd.to_datetime(working_raw.get(STAGE_TO_START_COL[stage]),
                                errors="coerce")
        comp   = (str((start + pd.Timedelta(days=int(round(days)))).date())
                  if pd.notna(start) else None)
        return {"predicted_days": days, "projected_completion": comp}


# ──────────────────────────────────────────────────────────────────────────────
# Per-job prediction (stage chaining)
# ──────────────────────────────────────────────────────────────────────────────
def predict_job(row: pd.Series, predictor: V13Predictor,
                history: dict) -> dict:
    job_num = str(row.get("JobNumber",""))
    raw     = row.to_dict()

    # Ensure dates are proper Timestamps in raw (not strings from CSV)
    for col in FACT_DATE_COLS:
        val = raw.get(col)
        if val is not None and pd.notna(val):
            try:
                ts = pd.Timestamp(val)
                raw[col] = pd.NaT if ts == _EPOCH else ts
            except Exception:
                raw[col] = pd.NaT

    out         = {"JobNumber": job_num}
    working_raw = dict(raw)   # mutable copy for chaining

    for stage in STAGE_ORDER:
        start_col  = STAGE_TO_START_COL[stage]
        end_col    = STAGE_TO_END_COL[stage]
        actual_end = pd.to_datetime(raw.get(end_col), errors="coerce")
        start_date = pd.to_datetime(working_raw.get(start_col), errors="coerce")

        if pd.isna(start_date):
            out[f"ML_{STAGE_LABEL[stage]}Date"]  = pd.NaT
            out[f"ML_{STAGE_LABEL[stage]}_Days"] = float("nan")
            continue

        try:
            res     = predictor.predict_stage(raw, stage, history, working_raw)
            ml_days = res["predicted_days"]
            ml_date = (pd.Timestamp(res["projected_completion"])
                       if res["projected_completion"] else pd.NaT)
        except Exception as e:
            print(f"  [ERROR] {stage} job={job_num}: {e}")
            ml_days, ml_date = float("nan"), pd.NaT

        out[f"ML_{STAGE_LABEL[stage]}Date"]  = ml_date
        out[f"ML_{STAGE_LABEL[stage]}_Days"] = ml_days

        # Chain: actual end date takes priority over ML prediction
        if pd.notna(actual_end):
            working_raw[end_col] = actual_end
        elif pd.notna(ml_date):
            working_raw[end_col] = ml_date

    return out


# ──────────────────────────────────────────────────────────────────────────────
# Build comparison output
# ──────────────────────────────────────────────────────────────────────────────
def _fmt(dt) -> str:
    try:
        ts = pd.to_datetime(dt, errors="coerce")
        return str(ts.date()) if pd.notna(ts) else ""
    except Exception:
        return ""


def _rnd(v) -> str:
    try:
        f = float(v)
        return str(round(f, 1)) if not math.isnan(f) else ""
    except Exception:
        return ""


def build_comparison(fact_df: pd.DataFrame, stage_df: pd.DataFrame,
                     predictions: list) -> pd.DataFrame:
    pred_df = pd.DataFrame(predictions)
    pred_df["JobNumber"] = pred_df["JobNumber"].astype(str).str.strip()

    id_cols = ["JobNumber","DivisionName","City","CommunityId","CommunityName",
               "ReleaseDate","FoundationStartDate","ActualCompletionDate",
               "FrameDate","CorniceDate","MechanicalDate",
               "SheetrockDate","TrimDate","InteriorDate"]
    if "JobReferenceNumber" in fact_df.columns:
        id_cols.insert(1, "JobReferenceNumber")
    id_cols   = [c for c in id_cols if c in fact_df.columns]

    fact_sub  = fact_df[id_cols].copy()
    fact_sub["JobNumber"] = fact_sub["JobNumber"].astype(str).str.strip()

    fourmo_cols = ["JobNumber"] + [c for c in STAGE_TO_4MO_DATE.values()
                                   if c in stage_df.columns]
    stage_sub = (stage_df[fourmo_cols]
                 .drop_duplicates("JobNumber").copy())
    stage_sub["JobNumber"] = stage_sub["JobNumber"].astype(str).str.strip()

    df = pred_df.merge(fact_sub,  on="JobNumber", how="left")
    df = df.merge(stage_sub,      on="JobNumber", how="left")

    rows_out = []
    for _, r in df.iterrows():
        row_out = {
            "JobNumber":           r.get("JobNumber",""),
            "JobReferenceNumber":  r.get("JobReferenceNumber",""),
            "DivisionName":        r.get("DivisionName",""),
            "City":                r.get("City",""),
            "CommunityName":       r.get("CommunityName",""),
            "FoundationStartDate": _fmt(r.get("FoundationStartDate")),
        }

        for stage in STAGE_ORDER:
            lbl     = STAGE_LABEL[stage]
            end_col = STAGE_TO_END_COL[stage]
            mo4_col = STAGE_TO_4MO_DATE.get(stage)

            actual  = pd.to_datetime(r.get(end_col),          errors="coerce")
            ml      = pd.to_datetime(r.get(f"ML_{lbl}Date"),  errors="coerce")
            mo4     = (pd.to_datetime(r.get(mo4_col), errors="coerce")
                       if mo4_col else pd.NaT)
            ml_days = r.get(f"ML_{lbl}_Days", float("nan"))

            row_out[f"Actual_{lbl}Date"]               = _fmt(actual)
            row_out[f"ML_{lbl}Date"]                   = _fmt(ml)
            row_out[f"ML_{lbl}_PredictedDays"]         = _rnd(ml_days)
            row_out[f"FourMonthAvg_{lbl}Date"]         = _fmt(mo4)
            row_out[f"ML_vs_Actual_{lbl}_days"]        = (
                int((ml - actual).days) if pd.notna(actual) and pd.notna(ml) else "")
            row_out[f"FourMonth_vs_Actual_{lbl}_days"] = (
                int((mo4 - actual).days) if pd.notna(actual) and pd.notna(mo4) else "")

        rows_out.append(row_out)

    return pd.DataFrame(rows_out)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="V13 batch ML inference")
    parser.add_argument("--fact",         required=True,  help="Ongoing fact CSV")
    parser.add_argument("--stage",        required=True,  help="Ongoing stage CSV")
    parser.add_argument("--models",       required=True,  help="output_v13 directory")
    parser.add_argument("--out",          default="predictions_v13.csv")
    parser.add_argument("--jobs",         nargs="*",      help="Filter to specific JobNumbers")
    parser.add_argument("--cutoff",       default=None,   help="Only jobs >= this FoundationStartDate")
    parser.add_argument("--no-log-space", action="store_true")
    args      = parser.parse_args()
    log_space = not args.no_log_space

    print(f"Loading data...  log_space={log_space}")
    fact_df  = load_fact(args.fact)
    stage_df = load_stage(args.stage)
    print(f"  Fact:  {len(fact_df):,} rows")
    print(f"  Stage: {len(stage_df):,} rows")

    if args.jobs:
        fact_df = fact_df[fact_df["JobNumber"].isin(set(args.jobs))]
        print(f"  -> Filtered to {len(fact_df)} specified jobs")

    if args.cutoff:
        fact_df = fact_df[
            pd.to_datetime(fact_df["FoundationStartDate"], errors="coerce")
            >= pd.Timestamp(args.cutoff)]
        print(f"  -> After cutoff {args.cutoff}: {len(fact_df):,}")

    runnable = fact_df[fact_df["FoundationStartDate"].notna()].copy()
    print(f"  -> Runnable (FoundationStartDate present): {len(runnable):,}")
    if len(runnable) == 0:
        print("ERROR: 0 runnable jobs. Check that FoundationStartDate is populated.")
        return

    print("\nBuilding city/community history index...")
    full_fact    = load_fact(args.fact)
    history      = build_history(full_fact)
    print(f"  {len(history['city']):,} cities indexed  |  "
          f"{len(history['comm_rank']):,} communities ranked")

    print(f"\nLoading model bundle from: {args.models}")
    predictor = V13Predictor(args.models, log_space=log_space)

    print(f"\nRunning inference for {len(runnable):,} jobs...")
    predictions, errors = [], []
    for i, (_, row) in enumerate(runnable.iterrows()):
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(runnable)} ...")
        try:
            predictions.append(predict_job(row, predictor, history))
        except Exception as e:
            errors.append({"JobNumber": str(row.get("JobNumber","?")), "error": str(e)})
            predictions.append({"JobNumber": str(row.get("JobNumber",""))})

    print(f"  Done. {len(predictions)} predictions | {len(errors)} errors")
    for err in errors[:5]:
        print(f"    X Job {err['JobNumber']}: {err['error']}")

    print("\nBuilding output CSV...")
    result_df = build_comparison(runnable, stage_df, predictions)
    result_df.to_csv(args.out, index=False)
    print(f"Saved {len(result_df):,} rows -> {args.out}")

    # ── Per-stage accuracy summary ────────────────────────────────────────────
    print("\n-- Per-Stage Accuracy (vs Actuals) ----------------------------------")
    header = f"  {'Stage':<14}  {'n':>4}  {'MAE':>7}  {'Bias':>8}  {'<=7d':>6}  {'<=14d':>6}"
    print(header)
    print("  " + "-"*62)
    for stage in STAGE_ORDER:
        lbl  = STAGE_LABEL[stage]
        col  = f"ML_vs_Actual_{lbl}_days"
        vals = pd.to_numeric(
            result_df.get(col, pd.Series(dtype=float)), errors="coerce").dropna()
        if len(vals) > 0:
            print(f"  {lbl:<14}  {len(vals):>4}  "
                  f"{vals.abs().mean():>6.1f}d  "
                  f"{vals.mean():>+7.1f}d  "
                  f"{(vals.abs()<=7).mean()*100:>5.1f}%  "
                  f"{(vals.abs()<=14).mean()*100:>5.1f}%")
    print("---------------------------------------------------------------------")


if __name__ == "__main__":
    main()
