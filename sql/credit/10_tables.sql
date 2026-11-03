-- Stage 3: credit applicants (Kaggle "Give Me Some Credit"), loaded by the factory.
-- Recreated on every load; load_runs is a log and is never dropped.
-- Types: counts smallint (largest value 98); ratios and income double precision (they are
-- measurements, not money, and the source writes some in scientific notation: 7.96E-05).
-- Values are stored as received: codes and impossible values are decoded in the view
-- model_input, so this table stays a faithful copy of the source (D62).
DROP TABLE IF EXISTS applicants, scoring_batch, quarantine CASCADE;

CREATE TABLE applicants (
    applicant_id          integer          PRIMARY KEY,
    serious_dlq_2yrs      smallint         NOT NULL CHECK (serious_dlq_2yrs IN (0, 1)),
    revolving_utilization double precision NOT NULL CHECK (revolving_utilization >= 0),
    age                   smallint         NOT NULL CHECK (age >= 0),
    past_due_30_59        smallint         NOT NULL CHECK (past_due_30_59 >= 0),
    debt_ratio            double precision NOT NULL CHECK (debt_ratio >= 0),
    monthly_income        double precision          CHECK (monthly_income >= 0),
    open_credit_lines     smallint         NOT NULL CHECK (open_credit_lines >= 0),
    past_due_90           smallint         NOT NULL CHECK (past_due_90 >= 0),
    real_estate_loans     smallint         NOT NULL CHECK (real_estate_loans >= 0),
    past_due_60_89        smallint         NOT NULL CHECK (past_due_60_89 >= 0),
    dependents            smallint                  CHECK (dependents >= 0)
);

-- New applicants to score (prediction path): the same columns without the label. The
-- prediction command fills it inside a transaction that it rolls back, so for every
-- other session it is always empty.
CREATE TABLE scoring_batch (LIKE applicants INCLUDING CONSTRAINTS);
ALTER TABLE scoring_batch DROP COLUMN serious_dlq_2yrs;

CREATE TABLE quarantine (
    source_table text    NOT NULL,
    source_row   integer NOT NULL,          -- data row number in the source file (1 = first)
    record       jsonb   NOT NULL,          -- raw values as received
    reason_codes text[]  NOT NULL CHECK (cardinality(reason_codes) > 0),
    PRIMARY KEY (source_table, source_row)
);

CREATE TABLE IF NOT EXISTS load_runs (
    load_id        text      PRIMARY KEY,   -- run_id of the run manifest
    loaded_at_utc  timestamp NOT NULL,
    snapshot       text      NOT NULL,
    config_sha256  text      NOT NULL CHECK (config_sha256 ~ '^[0-9a-f]{64}$'),
    git_commit     text      NOT NULL,
    files          jsonb     NOT NULL
);
