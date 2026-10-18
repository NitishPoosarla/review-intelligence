-- 07-ai-review-intelligence — label balance & corpus stats in SQL.
-- Loads: data/processed/reviews.csv -> `reviews` (text, label, corpus)
--        outputs/ai_classifications_sample.csv -> `ai_sample` (text, label, pred)

-- Class balance per corpus (eval gate: majority-class floor)
SELECT corpus,
       COUNT(*) AS n,
       ROUND(AVG(label) * 100, 1) AS positive_share_pct
FROM reviews GROUP BY corpus ORDER BY corpus;

-- Review length by label (feature-peek before modelling)
SELECT label, ROUND(AVG(ARRAY_LENGTH(SPLIT(text, ' ')) - 1), 1) AS avg_words,
       COUNT(*) AS n
FROM reviews GROUP BY label;

-- AI agreement: accuracy of the sampled zero-shot run (recomputable in SQL)
SELECT
  COUNT(*) AS sampled,
  ROUND(AVG(CAST(pred = label AS INT64)) * 100, 1)          AS ai_accuracy_pct,
  ROUND(AVG(CAST(pred = 1 AS INT64)) * 100, 1)              AS ai_positive_rate_pct,
  ROUND(AVG(CAST(label = 1 AS INT64)) * 100, 1)             AS truth_positive_rate_pct
FROM ai_sample;
