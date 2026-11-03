"""Score a new file with the schema of cs-training.csv (the label column may be empty or absent).

The new rows take the same road as the training rows, so no rule is written twice:
  1. the same contract (credit_contract) checks them; bad rows go to a quarantine file;
  2. inside one transaction they are written to credit.scoring_batch and read back through
     the same view model_input (same decoding of codes and impossible values);
  3. the transaction is rolled back: the database is left exactly as it was;
  4. the saved Pipeline (features + preparation fitted on training rows) and the saved
     threshold turn them into probabilities and decisions.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import pandas as pd

from data_factory.config import Config
from data_factory.credit_contract import TARGET, credit_tables
from data_factory.credit_model import load_model, predict_frame, read_model_input
from data_factory.db import database
from data_factory.exceptions import MissingInputError
from data_factory.table_validation import validate_tables
from data_factory.tables import write_csv

SCORING_TABLE = "scoring_batch"


def scoring_spec(config: Config, file_name: str):
    spec = credit_tables(config.credit.missing_tokens)[0]
    columns = tuple(c for c in spec.columns if c.target != TARGET)
    return dataclasses.replace(spec, table=SCORING_TABLE, file=file_name, columns=columns)


def predict_file(config: Config, input_path: Path, output_path: Path) -> pd.DataFrame:
    input_path = Path(input_path).resolve()
    if not input_path.exists():
        raise MissingInputError(f"{input_path} not found")
    if not config.credit.model.model_file.exists():
        raise MissingInputError(f"{config.credit.model.model_file} not found: train the model first (notebook)")
    work = config.paths.work / "credit" / "scoring"
    validate_tables((scoring_spec(config, input_path.name),), input_path.parent, work, work,
                    config.clean.chunk_rows, config.max_quarantine_ratio, {}, "credit scoring")
    rows = pd.read_csv(work / f"{SCORING_TABLE}.csv")
    saved = load_model(config.credit.model.model_file)
    with database(config.database) as engine, engine.connect() as connection:
        transaction = connection.begin()
        try:
            rows.to_sql(SCORING_TABLE, connection, schema=config.credit.load.schema, if_exists="append",
                        index=False)
            decoded, _ = read_model_input(connection, config.credit.load.schema, config.clean.chunk_rows,
                                          source="score")
        finally:
            transaction.rollback()                      # nothing stays in the database
    result = predict_frame(saved, decoded)
    write_csv(Path(output_path), result)
    return result

