"""Data contract for the credit file (stage 3, Kaggle "Give Me Some Credit").

One table, checked by the same engine as Olist (`table_validation.py`). The contract
rejects only values that cannot be read: a negative count, a word in a number column, a
label other than 0/1. Values that can be read but mean something special are real
answers of the source, not broken records, so they are kept here:
  96 / 98 in the three past-due counts   codes of the source system, not counts (D63)
  age 0, revolving utilization > 10      impossible values (D64)
They are decoded in the database view `model_input` (sql/credit/20_views.sql).
The first column of the file has no name; it is the applicant's row number.
"""
from __future__ import annotations

from data_factory.olist_contract import Column, Kind, TableSpec

LABELS = ("0", "1")
TARGET = "serious_dlq_2yrs"


def credit_tables(missing_tokens: tuple[str, ...]) -> tuple[TableSpec, ...]:
    return (TableSpec("applicants", "cs-training.csv", (
        Column("Unnamed: 0", "applicant_id", Kind.COUNT, True),
        Column("SeriousDlqin2yrs", TARGET, Kind.CHOICE, True, LABELS),
        Column("RevolvingUtilizationOfUnsecuredLines", "revolving_utilization", Kind.DECIMAL, True),
        Column("age", "age", Kind.COUNT, True),
        Column("NumberOfTime30-59DaysPastDueNotWorse", "past_due_30_59", Kind.COUNT, True),
        Column("DebtRatio", "debt_ratio", Kind.DECIMAL, True),
        Column("MonthlyIncome", "monthly_income", Kind.DECIMAL),
        Column("NumberOfOpenCreditLinesAndLoans", "open_credit_lines", Kind.COUNT, True),
        Column("NumberOfTimes90DaysLate", "past_due_90", Kind.COUNT, True),
        Column("NumberRealEstateLoansOrLines", "real_estate_loans", Kind.COUNT, True),
        Column("NumberOfTime60-89DaysPastDueNotWorse", "past_due_60_89", Kind.COUNT, True),
        Column("NumberOfDependents", "dependents", Kind.COUNT),
    ), key=("applicant_id",), missing_tokens=missing_tokens),)
