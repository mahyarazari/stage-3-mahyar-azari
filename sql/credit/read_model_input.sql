-- The model's only way in: decoded applicants from the view, in a fixed order (deterministic splits).
SELECT applicant_id, serious_dlq_2yrs, revolving_utilization, age, past_due_30_59, past_due_60_89,
       past_due_90, past_due_code, debt_ratio, monthly_income, open_credit_lines, real_estate_loans,
       dependents
FROM model_input
WHERE source = :source
ORDER BY applicant_id
