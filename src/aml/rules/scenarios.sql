-- Rule baseline (PLAN.md §6 M1): seven scenarios, one numeric severity per transaction.
--
-- As-of rule (PLAN.md §4): a transaction in minute m sees only events in minutes [m - W, m - 1].
-- Every window is a RANGE frame on the integer `minute` that ends at 1 PRECEDING, so events in
-- the same minute (in either direction) and later are never visible. A transaction's own fields
-- (amount, format, its own round / in-band / high-risk flags) are allowed.
--
-- Input: a view `tx` over the transactions table and a table `r_high_risk_formats(fmt)`, both set
-- up by aml.rules.sql_baseline. `${name}` placeholders are filled there with validated numbers
-- only. Statements are separated by "-- step: <name>" lines and run in file order; the last step
-- is the SELECT that returns the severities.

-- step: base
CREATE OR REPLACE TEMP TABLE r_tx AS
WITH t AS (
    SELECT
        row_id, rank, minute, day, split, src, dst, amount_usd, payment_format,
        src = dst AS self_loop,
        -- Whole cents: windowed sums are then exact integers, whatever the summation order.
        CAST(round(amount_usd * 100) AS BIGINT) AS usd_c,
        -- Same definition as aml.features.tx_features.round_amount_expr (half away from zero).
        CAST(round(amount_paid * 100) AS BIGINT) AS paid_c
    FROM tx
)
SELECT
    row_id, rank, minute, day, split, src, dst, self_loop, usd_c,
    paid_c > 0 AND paid_c % ${round_cents} = 0 AS is_round,
    amount_usd >= ${band_low_usd} AND amount_usd < ${band_high_usd} AS in_band,
    payment_format IN (SELECT fmt FROM r_high_risk_formats) AS high_risk
FROM t;

-- step: hubs
-- Hubs: accounts whose train-period degree (in + out edge count, as in the EDA) exceeds the hub
-- cap. They are excluded as round-trip intermediates and, where the config asks for it
-- (`exclude_hub_senders`), segmented out of sender-keyed scenarios: a transaction-monitoring
-- system treats known high-volume accounts separately. Fixed from train, so no look-ahead.
CREATE OR REPLACE TEMP TABLE r_hubs AS
WITH e AS (SELECT src, dst FROM r_tx WHERE split = 'train')
SELECT acct
FROM (SELECT src AS acct FROM e UNION ALL SELECT dst AS acct FROM e)
GROUP BY acct
HAVING count(*) > ${hub_cap};

-- step: sender
-- Scenarios keyed on the sender u: fan-out velocity and the three bursts. A segmented-out hub
-- sender gets severity 0.
CREATE OR REPLACE TEMP TABLE r_sender AS
SELECT
    row_id,
    CASE WHEN ${fan_out_excl_hubs} = 1 AND src_hub THEN 0
        ELSE count(DISTINCT dst) OVER w_fan_out END AS fan_out_velocity,
    CASE WHEN in_band AND NOT (${structuring_excl_hubs} = 1 AND src_hub)
        THEN 1 + count(*) FILTER (WHERE in_band) OVER w_structuring ELSE 0 END AS structuring,
    CASE WHEN is_round AND NOT (${round_excl_hubs} = 1 AND src_hub)
        THEN 1 + count(*) FILTER (WHERE is_round) OVER w_round ELSE 0 END AS round_amount_burst,
    CASE WHEN high_risk AND NOT (${high_risk_excl_hubs} = 1 AND src_hub)
        THEN 1 + count(*) FILTER (WHERE high_risk) OVER w_high_risk ELSE 0
    END AS high_risk_format_burst
FROM (
    SELECT r.*, h.acct IS NOT NULL AS src_hub
    FROM r_tx r LEFT JOIN r_hubs h ON h.acct = r.src
)
WINDOW
    w_fan_out AS (PARTITION BY src ORDER BY minute
        RANGE BETWEEN ${fan_out_window} PRECEDING AND 1 PRECEDING),
    w_structuring AS (PARTITION BY src ORDER BY minute
        RANGE BETWEEN ${structuring_window} PRECEDING AND 1 PRECEDING),
    w_round AS (PARTITION BY src ORDER BY minute
        RANGE BETWEEN ${round_window} PRECEDING AND 1 PRECEDING),
    w_high_risk AS (PARTITION BY src ORDER BY minute
        RANGE BETWEEN ${high_risk_window} PRECEDING AND 1 PRECEDING);

-- step: receiver
-- Fan-in velocity: distinct senders into the receiver v.
CREATE OR REPLACE TEMP TABLE r_receiver AS
SELECT
    row_id,
    count(DISTINCT src) OVER (PARTITION BY dst ORDER BY minute
        RANGE BETWEEN ${fan_in_window} PRECEDING AND 1 PRECEDING) AS fan_in_velocity
FROM r_tx;

-- step: pass_through
-- Rapid pass-through: the sender's windowed inflow (self-loops excluded on both sides) is read
-- from one stream per account holding its inflow events and its outgoing (query) rows.
CREATE OR REPLACE TEMP TABLE r_pass AS
WITH s AS (
    SELECT dst AS acct, minute, usd_c AS inflow_c, NULL::BIGINT AS row_id, 0::BIGINT AS usd_c
    FROM r_tx WHERE NOT self_loop
    UNION ALL
    SELECT src, minute, 0, row_id, usd_c
    FROM r_tx WHERE NOT self_loop
),
w AS (
    SELECT
        row_id, usd_c,
        sum(inflow_c) OVER (PARTITION BY acct ORDER BY minute
            RANGE BETWEEN ${pass_through_window} PRECEDING AND 1 PRECEDING) AS inflow_c
    FROM s
    QUALIFY row_id IS NOT NULL
)
SELECT
    row_id,
    CASE WHEN inflow_c > 0
        THEN greatest(0.0, 1.0 - abs(CAST(usd_c AS DOUBLE) / CAST(inflow_c AS DOUBLE) - 1.0))
        ELSE 0.0
    END AS rapid_pass_through
FROM w;

-- step: round_trip_edges
-- Round trip works on non-self-loop edges. r_closed holds the (a, b) pairs some transaction
-- b -> a closes (a path a -> ... -> b is only useful for those); r_hubs (step hubs) holds the
-- accounts excluded as intermediates.
CREATE OR REPLACE TEMP TABLE r_edges AS
SELECT row_id, src, dst, minute FROM r_tx WHERE NOT self_loop;
CREATE OR REPLACE TEMP TABLE r_closed AS
SELECT DISTINCT dst AS a, src AS b FROM r_edges;

-- step: round_trip_paths
-- Candidate 2-edge paths a -> w -> b, materialised once: w is not a hub, t1 <= t2 <= t1 + H,
-- and (a, b) is a closed pair. The per-hop window is a band join on time buckets of width H:
-- each first hop is offered to its own bucket and the next one, so every pair with
-- 0 <= t2 - t1 <= H meets exactly once and the join never pairs hops far apart in time.
CREATE OR REPLACE TEMP TABLE r_paths AS
WITH first_hop AS (
    SELECT e.src AS a, e.dst AS w, e.minute AS t1, e.minute // ${hop_bucket} + k.k AS bucket
    FROM (SELECT * FROM r_edges e ANTI JOIN r_hubs h ON h.acct = e.dst) e
    CROSS JOIN (VALUES (0), (1)) k(k)
),
second_hop AS (
    SELECT e.src AS w, e.dst AS b, e.minute AS t2, e.minute // ${hop_bucket} AS bucket
    FROM r_edges e ANTI JOIN r_hubs h ON h.acct = e.src
)
SELECT f.a, s.b, f.t1, s.t2
FROM first_hop f
JOIN second_hop s ON s.w = f.w AND s.bucket = f.bucket
SEMI JOIN r_closed c ON c.a = f.a AND c.b = s.b
WHERE f.t1 <= s.t2 AND s.t2 - f.t1 <= ${hop_window} AND f.a <> s.b;

-- step: round_trip
-- For a target u -> v at minute m, count the earlier paths v -> u (key a = v, b = u) with every
-- edge in [m - W, m - 1]. A path counts iff t_last <= m - 1 and t_first >= m - W. Because
-- t_last <= t_first + H and H <= W, any path with t_first <= m - W - 1 also has t_last <= m - 1,
-- so the count is #(t_last <= m - 1) - #(t_first <= m - W - 1): two cumulative window sums over
-- one stream per pair, with no join of targets against paths. A 2-hop path is the single edge
-- v -> u (t_first = t_last).
CREATE OR REPLACE TEMP TABLE r_round_trip AS
WITH s AS (
    SELECT a, b, t2 AS minute, 1 AS n_last, 0 AS n_first, NULL::BIGINT AS row_id FROM r_paths
    UNION ALL
    SELECT a, b, t1, 0, 1, NULL FROM r_paths
    UNION ALL
    SELECT e.src, e.dst, e.minute, 1, 1, NULL
    FROM r_edges e SEMI JOIN r_closed c ON c.a = e.src AND c.b = e.dst
    UNION ALL
    SELECT dst, src, minute, 0, 0, row_id FROM r_edges
),
w AS (
    SELECT
        row_id,
        coalesce(sum(n_last) OVER (PARTITION BY a, b ORDER BY minute
            RANGE BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING), 0)
        - coalesce(sum(n_first) OVER (PARTITION BY a, b ORDER BY minute
            RANGE BETWEEN UNBOUNDED PRECEDING AND ${round_trip_window_plus_1} PRECEDING), 0)
            AS n_paths
    FROM s
    QUALIFY row_id IS NOT NULL
)
SELECT row_id, least(n_paths, ${max_round_trip_paths}) AS round_trip
FROM w;

-- step: stats
CREATE OR REPLACE TEMP TABLE r_stats AS
SELECT
    (SELECT count(*) FROM r_tx) AS rows,
    (SELECT count(*) FROM r_hubs) AS hub_accounts,
    (SELECT count(*) FROM r_closed) AS closed_pairs,
    (SELECT count(*) FROM r_paths) AS round_trip_candidate_paths;

-- step: severities
SELECT
    t.row_id,
    t.split,
    t.day,
    CAST(rc.fan_in_velocity AS DOUBLE) AS fan_in_velocity,
    CAST(sd.fan_out_velocity AS DOUBLE) AS fan_out_velocity,
    CAST(coalesce(p.rapid_pass_through, 0.0) AS DOUBLE) AS rapid_pass_through,
    CAST(coalesce(rt.round_trip, 0) AS DOUBLE) AS round_trip,
    CAST(sd.structuring AS DOUBLE) AS structuring,
    CAST(sd.round_amount_burst AS DOUBLE) AS round_amount_burst,
    CAST(sd.high_risk_format_burst AS DOUBLE) AS high_risk_format_burst
FROM r_tx t
JOIN r_sender sd USING (row_id)
JOIN r_receiver rc USING (row_id)
LEFT JOIN r_pass p USING (row_id)
LEFT JOIN r_round_trip rt USING (row_id)
ORDER BY t.rank;
