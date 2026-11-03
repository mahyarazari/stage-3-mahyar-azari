"""Stage 3: credit contract, database view, features, preparation, cost, threshold and prediction."""
import dataclasses
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from psycopg import sql

from data_factory import credit_model as cm
from data_factory.config import load_config
from data_factory.credit_contract import TARGET, credit_tables
from data_factory.credit_predict import predict_file
from data_factory.table_validation import validate_tables
from tests.db_helpers import PROJECT_CONFIG, TemporarySchemaMixin, needs_database

HEADER = ("Unnamed: 0,SeriousDlqin2yrs,RevolvingUtilizationOfUnsecuredLines,age,NumberOfTime30-59DaysPastDueNotWorse,"
          "DebtRatio,MonthlyIncome,NumberOfOpenCreditLinesAndLoans,NumberOfTimes90DaysLate,"
          "NumberRealEstateLoansOrLines,NumberOfTime60-89DaysPastDueNotWorse,NumberOfDependents\n")
ROWS = ("1,1,0.766126609,45,2,0.802982129,9120,13,0,6,0,2\n"
        "2,0,7.96E-05,0,98,1500,NA,4,98,0,98,NA\n"         # code 98, age 0, income missing
        "3,0,50708,30,0,0.5,4000,5,0,1,0,0\n"             # impossible utilization
        "4,0,0.2,-3,0,0.5,4000,5,0,1,0,0\n")              # negative age: quarantined


def applicants(**columns) -> pd.DataFrame:
    base = {"applicant_id": [1, 2, 3], TARGET: [1, 0, 0], "revolving_utilization": [0.5, 0.1, np.nan],
            "age": [45.0, np.nan, 30.0], "past_due_30_59": [2.0, np.nan, 0.0], "past_due_60_89": [1.0, np.nan, 0.0],
            "past_due_90": [0.0, np.nan, 0.0], "past_due_code": ["none", "98", "none"],
            "debt_ratio": [0.5, 1500.0, 0.25], "monthly_income": [4000.0, np.nan, 2000.0],
            "open_credit_lines": [10.0, 4.0, 1.0], "real_estate_loans": [2.0, 0.0, 3.0],
            "dependents": [3.0, np.nan, 0.0]}
    base.update(columns)
    return pd.DataFrame(base)


class ContractTests(unittest.TestCase):
    def test_codes_are_kept_and_unreadable_rows_quarantined(self):
        folder = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, folder)
        (folder / "cs-training.csv").write_text(HEADER + ROWS, encoding="utf-8")
        stats = validate_tables(credit_tables(("NA",)), folder, folder / "work", folder / "q", 2, 1.0, {}, "credit")
        self.assertEqual((stats["valid_records"], stats["quarantined_records"]), (3, 1))
        accepted = pd.read_csv(folder / "work" / "applicants.csv", dtype=str, keep_default_na=False)
        self.assertEqual(accepted.loc[1, "past_due_90"], "98")          # a code is not an error
        self.assertEqual(accepted.loc[1, "monthly_income"], "")         # NA -> missing


class FeatureTests(unittest.TestCase):
    def test_engineered_features(self):
        out = cm.add_features(applicants())
        self.assertEqual(out["total_past_due"].tolist()[0], 3.0)
        self.assertTrue(np.isnan(out["total_past_due"][1]))              # a code is not a count
        self.assertEqual(out["monthly_debt"][0], 2000.0)
        self.assertTrue(np.isnan(out["monthly_debt"][1]))                # no income, no burden
        self.assertEqual(out["income_per_person"][0], 1000.0)
        self.assertEqual(out["unsecured_lines"].tolist(), [8.0, 4.0, 0.0])
        self.assertTrue(np.isnan(out["debt_ratio_known_income"][1]))
        self.assertNotIn("total_past_due", applicants().columns)         # input is not changed


class PreparationTests(unittest.TestCase):
    def test_missing_values_and_unseen_codes_are_handled(self):
        train = pd.concat([applicants()] * 20, ignore_index=True)
        train[TARGET] = [1, 0] * 30
        new = applicants(past_due_code=["96", "96", "96"], monthly_income=[np.nan] * 3, age=[np.nan] * 3)
        for kind in cm.MODEL_KINDS:
            for missing in cm.MISSING_POLICIES:
                model = cm.build_model(kind, seed=1, missing=missing).fit(train, train[TARGET])
                proba = model.predict_proba(new)[:, 1]
                self.assertTrue(np.all((proba >= 0) & (proba <= 1)), (kind, missing))

    def test_statistics_come_from_training_rows_only(self):
        train = applicants(monthly_income=[1000.0, np.nan, 3000.0])
        test = applicants(monthly_income=[10**6, 10**6, 10**6])
        model = cm.build_model("logreg", seed=1).fit(train, train[TARGET])
        imputer = model.named_steps["prepare"].named_transformers_["numeric"].named_steps["impute"]
        names = model.named_steps["prepare"].transformers_[0][2]
        self.assertEqual(imputer.statistics_[names.index("monthly_income")], 2000.0)
        model.predict_proba(test)                                        # predicting does not refit
        self.assertEqual(imputer.statistics_[names.index("monthly_income")], 2000.0)


class CostTests(unittest.TestCase):
    def test_cost_counts_each_error_with_its_price(self):
        y = [1, 1, 0, 0, 0]
        reject = [1, 0, 1, 0, 0]                                         # 1 TP, 1 FN, 1 FP, 2 TN
        self.assertEqual(cm.decision_cost(y, reject, 12, 1), 13.0)
        self.assertEqual(cm.per_1000(13.0, 5), 2600.0)

    def test_best_threshold_matches_brute_force(self):
        rng = np.random.default_rng(0)
        y = rng.random(400) < 0.1
        proba = np.round(np.clip(y * 0.2 + rng.random(400) * 0.5, 0, 1), 2)
        threshold, cost = cm.best_threshold(y, proba, 12, 1)
        candidates = np.r_[np.unique(proba), np.inf]
        brute = min(cm.decision_cost(y, proba >= t, 12, 1) for t in candidates)
        self.assertEqual(cost, brute)
        self.assertEqual(cm.decision_cost(y, proba >= threshold, 12, 1), cost)
        self.assertAlmostEqual(cm.bayes_threshold(12, 1), 1 / 13)

    def test_holdout_split_is_stratified_and_repeatable(self):
        frame = pd.DataFrame({"applicant_id": range(1000), TARGET: [1] * 70 + [0] * 930})
        a_train, a_test = cm.split_holdout(frame, 0.2, 7)
        b_train, b_test = cm.split_holdout(frame, 0.2, 7)
        self.assertEqual(a_test["applicant_id"].tolist(), b_test["applicant_id"].tolist())
        self.assertEqual((len(a_test), int(a_test[TARGET].sum())), (200, 14))
        self.assertFalse(set(a_train["applicant_id"]) & set(a_test["applicant_id"]))


@needs_database
class DatabaseTests(TemporarySchemaMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        config = load_config(PROJECT_CONFIG)
        self.folder = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.folder)
        raw = self.folder / "raw" / "credit"
        raw.mkdir(parents=True)
        (raw / "cs-training.csv").write_text(HEADER + ROWS, encoding="utf-8")
        paths = dataclasses.replace(config.paths, raw=self.folder / "raw", work=self.folder / "work",
                                    quarantine=self.folder / "q", curated=self.folder / "curated")
        credit = dataclasses.replace(config.credit, files={},
                                     load=dataclasses.replace(config.credit.load, schema=self.schema),
                                     model=dataclasses.replace(config.credit.model,
                                                               model_file=self.folder / "model.joblib"))
        self.config = dataclasses.replace(config, paths=paths, credit=credit, max_quarantine_ratio=1.0)
        from data_factory import credit_pipeline
        from data_factory.pipeline import RunContext
        context = RunContext(config=self.config)
        credit_pipeline.validate(context)
        credit_pipeline.load(context)

    def test_view_decodes_codes_and_impossible_values(self):
        rows = self.query("SELECT applicant_id, age, past_due_90, past_due_code, revolving_utilization "
                          "FROM model_input ORDER BY applicant_id")
        self.assertEqual([tuple(r) for r in rows], [(1, 45, 0, "none", 0.766126609),
                                                    (2, None, None, "98", 7.96e-05),
                                                    (3, 30, 0, "none", None)])

    def test_prediction_path_on_a_new_file(self):
        with self.engine.connect() as connection:
            frame, _ = cm.read_model_input(connection, self.schema, 100)
        frame = pd.concat([frame] * 10, ignore_index=True)
        frame[TARGET] = [1, 0] * 15
        cm.save_model(self.config.credit.model.model_file, cm.build_model("hgb", seed=1).fit(frame, frame[TARGET]),
                      0.5, {})
        new = self.folder / "new.csv"
        new.write_text(HEADER.replace("SeriousDlqin2yrs,", "") +
                       "11,0.3,52,0,0.2,5000,7,0,1,0,1\n12,0.9,23,98,0.1,NA,2,98,0,98,0\n", encoding="utf-8")
        result = predict_file(self.config, new, self.folder / "out.csv")
        self.assertEqual(result["applicant_id"].tolist(), [11, 12])
        self.assertTrue(set(result["decision"]) <= {"approve", "reject"})
        self.assertEqual(self.query("SELECT count(*) FROM scoring_batch")[0][0], 0)   # rolled back
        self.assertTrue((self.folder / "out.csv").exists())


if __name__ == "__main__":
    unittest.main()
