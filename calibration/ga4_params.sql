-- Calibration pass: pull the parameters that anchor the simulation grids.
-- Dataset: bigquery-public-data.ga4_obfuscated_sample_ecommerce
-- Coverage: events_20201101 .. events_20210131
--
-- Run each block separately and record bytes scanned. Nothing here should
-- exceed a few GB; if it does, narrow the _TABLE_SUFFIX range first.

-- ===========================================================================
-- Block 1: sessions per user, baseline session conversion, session-count
--          dispersion, and the intraclass correlation of conversion.
-- ===========================================================================
WITH sessions AS (
  SELECT
    user_pseudo_id,
    (SELECT value.int_value FROM UNNEST(event_params)
     WHERE key = 'ga_session_id')                        AS session_id,
    MAX(IF(event_name = 'purchase', 1, 0))               AS converted
  FROM `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*`
  WHERE _TABLE_SUFFIX BETWEEN '20201101' AND '20210131'
  GROUP BY user_pseudo_id, session_id
  HAVING session_id IS NOT NULL
),
per_user AS (
  SELECT
    user_pseudo_id,
    COUNT(*)        AS n_sessions,
    SUM(converted)  AS n_converted
  FROM sessions
  GROUP BY user_pseudo_id
),
agg AS (
  SELECT
    COUNT(*)                                   AS n_users,
    SUM(n_sessions)                            AS total_sessions,
    SUM(n_converted)                           AS total_conversions,
    AVG(n_sessions)                            AS sessions_per_user_mean,
    VAR_SAMP(n_sessions)                       AS sessions_per_user_var,
    -- within-cluster mean square, for the method-of-moments ICC
    SUM(n_converted * (1 - n_converted / n_sessions))
      / (SUM(n_sessions) - COUNT(*))           AS msw
  FROM per_user
  WHERE n_sessions > 0
)
SELECT
  n_users,
  total_sessions,
  sessions_per_user_mean,
  SAFE_DIVIDE(sessions_per_user_var, sessions_per_user_mean)
    AS sessions_per_user_dispersion,
  SAFE_DIVIDE(total_conversions, total_sessions)
    AS baseline_session_conversion,
  1 - SAFE_DIVIDE(
        msw,
        SAFE_DIVIDE(total_conversions, total_sessions)
          * (1 - SAFE_DIVIDE(total_conversions, total_sessions))
      ) AS session_conversion_icc
FROM agg;

-- ===========================================================================
-- Block 2: daily users, for the M1 arrival stream and the MDE calculation.
-- ===========================================================================
SELECT
  AVG(daily_users) AS daily_users
FROM (
  SELECT
    event_date,
    APPROX_COUNT_DISTINCT(user_pseudo_id) AS daily_users
  FROM `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*`
  WHERE _TABLE_SUFFIX BETWEEN '20201101' AND '20210131'
  GROUP BY event_date
);

-- ===========================================================================
-- Block 3: the CUPED covariate correlation.
--
-- Split the window into a pre period and a post period, compute a per-user
-- engagement measure in each, and correlate. This is the number that decides
-- whether CUPED is worth anything on data like this: the literature quotes
-- rho ~ 0.7, and if the observed value is materially lower that is itself a
-- README line.
--
-- Users with no pre-period activity are EXCLUDED here by the inner join,
-- which is exactly the population M2's no_preperiod_share sweep is about.
-- Block 3b measures how large that population is.
-- ===========================================================================
WITH pre AS (
  SELECT user_pseudo_id, COUNT(*) AS pre_events
  FROM `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*`
  WHERE _TABLE_SUFFIX BETWEEN '20201101' AND '20201215'
  GROUP BY user_pseudo_id
),
post AS (
  SELECT user_pseudo_id, COUNT(*) AS post_events
  FROM `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*`
  WHERE _TABLE_SUFFIX BETWEEN '20201216' AND '20210131'
  GROUP BY user_pseudo_id
)
SELECT
  COUNT(*)                                  AS n_users_both_periods,
  CORR(pre.pre_events, post.post_events)    AS cuped_pre_post_correlation
FROM pre JOIN post USING (user_pseudo_id);

-- ===========================================================================
-- Block 3b: share of post-period users with no pre-period history.
-- Feeds M2's no_preperiod_share grid with an observed value.
-- ===========================================================================
WITH pre AS (
  SELECT DISTINCT user_pseudo_id
  FROM `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*`
  WHERE _TABLE_SUFFIX BETWEEN '20201101' AND '20201215'
),
post AS (
  SELECT DISTINCT user_pseudo_id
  FROM `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*`
  WHERE _TABLE_SUFFIX BETWEEN '20201216' AND '20210131'
)
SELECT
  COUNT(*)                                          AS n_post_users,
  COUNTIF(pre.user_pseudo_id IS NULL)               AS n_without_preperiod,
  SAFE_DIVIDE(COUNTIF(pre.user_pseudo_id IS NULL), COUNT(*))
    AS no_preperiod_share
FROM post LEFT JOIN pre USING (user_pseudo_id);

-- ===========================================================================
-- Block 4: ICC ESTIMATOR OF RECORD -- size-stratified moment estimator.
--
-- Supersedes the ANOVA estimator in Block 1, which is biased downward when
-- cluster size correlates with cluster rate. It does here, steeply: per-session
-- purchase rate runs 0.0048 at n=1 to 0.042 at n=8, and the ANOVA estimator
-- consequently reported -0.0395 for a metric whose true ICC is +0.039.
--
-- Holding n fixed removes the confound. For exchangeable binary sessions,
--     Var(s) = n p (1-p) [1 + (n-1) rho]
-- so per stratum:
--     p   = mean(s) / n
--     rho = (var(s) / (n p (1-p)) - 1) / (n - 1)
-- Pool across strata weighted by 1 / SE^2, SE ~= sqrt(2 / (n (n-1) k)).
-- ===========================================================================
WITH sessions AS (
  SELECT
    user_pseudo_id,
    (SELECT value.int_value FROM UNNEST(event_params)
     WHERE key = 'ga_session_id')            AS session_id,
    MAX(IF(event_name = 'purchase', 1, 0))    AS purch,
    MAX(IF(event_name = 'add_to_cart', 1, 0)) AS atc
  FROM `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*`
  WHERE _TABLE_SUFFIX BETWEEN '20201101' AND '20210131'
  GROUP BY user_pseudo_id, session_id
  HAVING session_id IS NOT NULL
),
per_user AS (
  SELECT user_pseudo_id, COUNT(*) AS n, SUM(purch) AS sp, SUM(atc) AS sa
  FROM sessions GROUP BY user_pseudo_id
)
SELECT n, COUNT(*) AS k_users,
       AVG(sp) AS mean_purch, VAR_SAMP(sp) AS var_purch,
       AVG(sa) AS mean_atc,   VAR_SAMP(sa) AS var_atc
FROM per_user
WHERE n BETWEEN 2 AND 4
GROUP BY n ORDER BY n;

-- ===========================================================================
-- Block 5: DIAGNOSTIC -- per-session conversion rate by session count.
-- This is the query that identified the ANOVA bias. Run it for any metric
-- before trusting a pooled ICC: a rate that varies with n means the pooled
-- estimate is confounded.
-- ===========================================================================
WITH sessions AS (
  SELECT
    user_pseudo_id,
    (SELECT value.int_value FROM UNNEST(event_params)
     WHERE key = 'ga_session_id')          AS session_id,
    MAX(IF(event_name = 'purchase', 1, 0)) AS converted
  FROM `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*`
  WHERE _TABLE_SUFFIX BETWEEN '20201101' AND '20210131'
  GROUP BY user_pseudo_id, session_id
  HAVING session_id IS NOT NULL
),
per_user AS (
  SELECT user_pseudo_id, COUNT(*) AS n, SUM(converted) AS s
  FROM sessions GROUP BY user_pseudo_id
)
SELECT LEAST(n, 8) AS n_sessions, COUNT(*) AS users,
       SUM(s) / SUM(n) AS rate_per_session
FROM per_user GROUP BY 1 ORDER BY 1;
