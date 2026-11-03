"""Stage 3: the default model, from database rows to a cost-based decision.

Reading      only from the view model_input (sql/credit/20_views.sql), in chunks, through
             SQLAlchemy; the notebook never reads a CSV.
Features     `add_features` is row by row and learns nothing, so it is the same on training
             data and on a new file. It runs inside the scikit-learn Pipeline.
Boundary     everything that learns from data (medians, scales, category lists, the model)
             is a step of one Pipeline, fitted on training rows only (D65). A Pipeline fitted
             on a fold never sees that fold's validation rows.
Decision     the model gives a probability; the decision "reject" is probability >= threshold,
             where the threshold is the one with the lowest total cost on out-of-fold
             predictions (D67), not 0.5.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sqlalchemy import text
from sqlalchemy.engine import Connection

from data_factory.credit_contract import TARGET

SQL_DIR = Path(__file__).resolve().parents[2] / "sql" / "credit"
READ_SQL = SQL_DIR / "read_model_input.sql"

BASE_NUMERIC = ["revolving_utilization", "age", "past_due_30_59", "past_due_60_89", "past_due_90",
                "debt_ratio", "monthly_income", "open_credit_lines", "real_estate_loans", "dependents"]
CATEGORICAL = ["past_due_code"]          # 'none', '96', '98': a code, not a number (D63)
# Engineered features (D66); each one tests one business hypothesis, see add_features.
NEW_FEATURES = ["total_past_due", "monthly_debt", "income_per_person", "unsecured_lines",
                "debt_ratio_known_income"]
MODEL_KINDS = ("logreg", "hgb")
MISSING_POLICIES = ("indicator", "median")   # D65: median + "was missing" flag, or median only
AGE_BANDS = [0, 30, 45, 60, 200]
AGE_LABELS = ["<30", "30-44", "45-59", "60+"]


# ---------------------------------------------------------------- reading
def read_model_input(connection: Connection, schema: str, chunk_rows: int,
                     source: str = "train") -> tuple[pd.DataFrame, dict]:
    """All rows of one source from the view, in chunks; returns (frame, read statistics)."""
    connection.execute(text("SELECT set_config('search_path', :path, false)"), {"path": schema})
    started = time.perf_counter()
    chunks = list(pd.read_sql_query(text(READ_SQL.read_text("utf-8")), connection,
                                    params={"source": source}, chunksize=chunk_rows))
    frame = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    stats = {"rows": len(frame), "chunks": len(chunks), "seconds": round(time.perf_counter() - started, 2)}
    for column in BASE_NUMERIC:
        frame[column] = frame[column].astype("float64")
    return frame, stats


# ---------------------------------------------------------------- features
def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Row-by-row features; every input is known when the application is made.

    total_past_due           more late payments of any length -> more risk
    monthly_debt             the debt burden in money, not only as a share of income
    income_per_person        the same income feeds more people -> less room to repay
    unsecured_lines          credit lines without a house behind them carry more risk
    debt_ratio_known_income  DebtRatio means "debt / income" only when the income is known;
                             without it the source stores the raw debt (D64), so missing
    """
    out = frame.copy()
    income_known = out["monthly_income"].notna() & (out["monthly_income"] > 0)
    out["total_past_due"] = out[["past_due_30_59", "past_due_60_89", "past_due_90"]].sum(axis=1, min_count=3)
    out["monthly_debt"] = (out["debt_ratio"] * out["monthly_income"]).where(income_known)
    out["income_per_person"] = out["monthly_income"] / (out["dependents"].fillna(0) + 1)
    out["unsecured_lines"] = (out["open_credit_lines"] - out["real_estate_loans"]).clip(lower=0)
    out["debt_ratio_known_income"] = out["debt_ratio"].where(income_known)
    return out


def age_band(age: pd.Series) -> pd.Series:
    return pd.cut(age, AGE_BANDS, labels=AGE_LABELS, right=False).astype("string").fillna("unknown")


def leaky_age_band_rate(frame: pd.DataFrame) -> pd.Series:
    """AUDIT ONLY, never in the model: the default rate of the applicant's age band computed
    on ALL rows, labels of evaluation rows included. It is here to measure how much a feature
    that has seen the answers inflates the validation numbers (leakage audit, D68)."""
    band = age_band(frame["age"])
    return band.map(frame.groupby(band)[TARGET].mean()).astype("float64")


# ---------------------------------------------------------------- model
def build_model(kind: str, seed: int, missing: str = "indicator",
                drop: tuple[str, ...] = (), extra: tuple[str, ...] = ()) -> Pipeline:
    """features -> preparation -> classifier, as one Pipeline (fitted on training rows only).

    logreg  a linear model on weighted sums of the inputs: it needs scaled inputs (the
            penalty and the solver treat all coefficients alike) and a log of the long-tailed
            amounts, so a few huge incomes do not dominate (D69).
    hgb     trees split on thresholds, so any monotone rescaling changes nothing: no scaling.
    """
    if kind not in MODEL_KINDS or missing not in MISSING_POLICIES:
        raise ValueError(f"unknown model {kind!r} or missing-value policy {missing!r}")
    numeric = [c for c in BASE_NUMERIC + NEW_FEATURES if c not in drop] + list(extra)
    imputer = SimpleImputer(strategy="median", add_indicator=(missing == "indicator"))
    if kind == "logreg":
        steps = [("impute", imputer), ("log", FunctionTransformer(_signed_log1p)), ("scale", StandardScaler())]
        classifier = LogisticRegression(max_iter=2000, C=1.0)
    else:
        steps = [("impute", imputer)]
        classifier = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, max_leaf_nodes=31,
                                                    min_samples_leaf=100, l2_regularization=1.0,
                                                    early_stopping=False, random_state=seed)
    prepare = ColumnTransformer([
        ("numeric", Pipeline(steps), numeric),
        ("code", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ], verbose_feature_names_out=False)
    return Pipeline([("features", FunctionTransformer(add_features)), ("prepare", prepare),
                     ("model", classifier)])


def _signed_log1p(values):
    return np.sign(values) * np.log1p(np.abs(values))


# ---------------------------------------------------------------- cost and threshold
def decision_cost(y_true, reject, cost_fn: float, cost_fp: float) -> float:
    """Total cost: an approved defaulter costs cost_fn, a rejected good applicant cost_fp."""
    y_true, reject = np.asarray(y_true).astype(bool), np.asarray(reject).astype(bool)
    return float(cost_fn * np.sum(y_true & ~reject) + cost_fp * np.sum(~y_true & reject))


def per_1000(cost: float, applicants: int) -> float:
    return 1000.0 * cost / applicants


def best_threshold(y_true, proba, cost_fn: float, cost_fp: float) -> tuple[float, float]:
    """(threshold, total cost) minimising the cost of "reject if proba >= threshold".

    Every distinct probability is tried (plus "reject nobody"), in one pass over the sorted
    scores. Ties are resolved towards the higher threshold (fewer rejections)."""
    y = np.asarray(y_true).astype(bool)
    p = np.asarray(proba, dtype=float)
    order = np.argsort(-p, kind="stable")
    p, y = p[order], y[order]
    last_of_value = np.r_[p[1:] != p[:-1], True]          # cut after the last row of each value
    caught_bad = np.cumsum(y)[last_of_value]
    rejected_good = np.cumsum(~y)[last_of_value]
    costs = np.r_[cost_fn * y.sum(), cost_fn * (y.sum() - caught_bad) + cost_fp * rejected_good]
    thresholds = np.r_[np.inf, p[last_of_value]]
    best = int(np.argmin(costs))                           # first minimum = highest threshold
    return float(thresholds[best]), float(costs[best])


def bayes_threshold(cost_fn: float, cost_fp: float) -> float:
    """The threshold for perfectly calibrated probabilities: reject when p * cost_fn > (1 - p) * cost_fp."""
    return cost_fp / (cost_fp + cost_fn)


# ---------------------------------------------------------------- evaluation
def split_holdout(frame: pd.DataFrame, test_share: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified hold-out set, chosen once from the row order of the view (D61)."""
    train, test = train_test_split(frame, test_size=test_share, stratify=frame[TARGET], random_state=seed)
    return train.sort_values("applicant_id"), test.sort_values("applicant_id")


@dataclass
class CvResult:
    oof: np.ndarray          # out-of-fold probability of every training row
    folds: pd.DataFrame      # one row per fold: ranking metrics and fold-optimal cost
    fold_of_row: np.ndarray


def cross_validate(make_model, frame: pd.DataFrame, folds: int, seed: int, cost_fn: float,
                   cost_fp: float, prepare=None) -> CvResult:
    """Stratified k-fold: each fold's Pipeline is fitted on the other folds only.
    `prepare(train_part, valid_part)` may add columns computed from the training part."""
    y = frame[TARGET].to_numpy()
    oof = np.zeros(len(frame))
    fold_of_row = np.zeros(len(frame), dtype=int)
    rows = []
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for number, (fit_idx, val_idx) in enumerate(splitter.split(frame, y), start=1):
        fit_part, val_part = frame.iloc[fit_idx], frame.iloc[val_idx]
        if prepare is not None:
            fit_part, val_part = prepare(fit_part, val_part)
        model = make_model().fit(fit_part, y[fit_idx])
        proba = model.predict_proba(val_part)[:, 1]
        oof[val_idx], fold_of_row[val_idx] = proba, number
        threshold, cost = best_threshold(y[val_idx], proba, cost_fn, cost_fp)
        rows.append({"fold": number, "rows": len(val_idx), "roc_auc": roc_auc_score(y[val_idx], proba),
                     "pr_auc": average_precision_score(y[val_idx], proba),
                     "own_threshold": threshold, "own_cost_per_1000": per_1000(cost, len(val_idx))})
    return CvResult(oof=oof, folds=pd.DataFrame(rows), fold_of_row=fold_of_row)


def costs_at(y_true, proba, groups, threshold: float, cost_fn: float, cost_fp: float) -> pd.Series:
    """Cost per 1000 applicants of each group (fold) at one fixed threshold."""
    frame = pd.DataFrame({"y": np.asarray(y_true), "reject": np.asarray(proba) >= threshold, "g": groups})
    return frame.groupby("g").apply(lambda part: per_1000(decision_cost(part["y"], part["reject"], cost_fn, cost_fp),
                                                          len(part)), include_groups=False)


def classification_report(y_true, proba, threshold: float, cost_fn: float, cost_fp: float) -> dict:
    y = np.asarray(y_true)
    reject = np.asarray(proba) >= threshold
    tn, fp, fn, tp = confusion_matrix(y, reject, labels=[0, 1]).ravel()
    return {"applicants": len(y), "threshold": threshold, "tn": int(tn), "fp": int(fp), "fn": int(fn),
            "tp": int(tp), "precision": precision_score(y, reject, zero_division=0),
            "recall": recall_score(y, reject), "f1": f1_score(y, reject),
            "roc_auc": roc_auc_score(y, proba), "pr_auc": average_precision_score(y, proba),
            "accuracy": accuracy_score(y, reject),
            "cost_per_1000": per_1000(decision_cost(y, reject, cost_fn, cost_fp), len(y))}


# ---------------------------------------------------------------- saving and prediction
def save_model(path: Path, model: Pipeline, threshold: float, info: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": model, "threshold": threshold, "sklearn": sklearn.__version__, **info}, path)


def load_model(path: Path) -> dict:
    return joblib.load(path)


def predict_frame(saved: dict, frame: pd.DataFrame) -> pd.DataFrame:
    """Probability and decision for decoded rows (the columns of model_input)."""
    proba = saved["pipeline"].predict_proba(frame)[:, 1]
    return pd.DataFrame({"applicant_id": frame["applicant_id"].to_numpy(), "probability": proba,
                         "decision": np.where(proba >= saved["threshold"], "reject", "approve")})
