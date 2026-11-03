-- model_input: the one place the model reads applicants from, for training and prediction.
-- Only row-by-row rules that need no statistics live here (D62): they are facts about the
-- source, the same for every reader, and they cannot carry information from one applicant
-- to another, so they cannot leak. Anything learnt from data (medians, scales, category
-- lists) is fitted in Python on the training part only.
--   * 96 and 98 in the past-due counts are codes of the source system, not counts: the
--     counts become missing and past_due_code keeps the code (D63).
--   * age 0 is impossible for a borrower: it becomes missing (D64).
--   * revolving utilization above 10 (a balance ten times the credit limit) is a data
--     error, not a ratio: it becomes missing (D64).
--   * source tells training rows (label known) from a scoring batch (label null).
CREATE OR REPLACE VIEW model_input AS
WITH all_rows AS (
    SELECT 'train' AS source, applicant_id, serious_dlq_2yrs, revolving_utilization, age,
           past_due_30_59, debt_ratio, monthly_income, open_credit_lines, past_due_90,
           real_estate_loans, past_due_60_89, dependents
    FROM applicants
    UNION ALL
    SELECT 'score', applicant_id, NULL, revolving_utilization, age,
           past_due_30_59, debt_ratio, monthly_income, open_credit_lines, past_due_90,
           real_estate_loans, past_due_60_89, dependents
    FROM scoring_batch
)
SELECT source,
       applicant_id,
       serious_dlq_2yrs,
       CASE WHEN revolving_utilization <= 10 THEN revolving_utilization END AS revolving_utilization,
       NULLIF(age, 0)                                                     AS age,
       CASE WHEN past_due_30_59 < 96 THEN past_due_30_59 END              AS past_due_30_59,
       CASE WHEN past_due_60_89 < 96 THEN past_due_60_89 END              AS past_due_60_89,
       CASE WHEN past_due_90 < 96 THEN past_due_90 END                    AS past_due_90,
       CASE WHEN greatest(past_due_30_59, past_due_60_89, past_due_90) >= 96
            THEN greatest(past_due_30_59, past_due_60_89, past_due_90)::text
            ELSE 'none' END                                               AS past_due_code,
       debt_ratio,
       monthly_income,
       open_credit_lines,
       real_estate_loans,
       dependents
FROM all_rows;
