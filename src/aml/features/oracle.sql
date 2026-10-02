-- DuckDB feature oracle (M2 spec §9.3): the engine's windowed features recomputed by set-based
-- SQL, written from the spec and never from the engine code.
--
-- As-of rule (PLAN.md §4): an event in minute m sees only events in minutes [m - W, m - 1]. Every
-- window is a RANGE frame on the integer `minute` that ends at 1 PRECEDING, so events of the same
-- minute (in either rank direction) and later are never visible; "all earlier" frames are
-- UNBOUNDED PRECEDING AND 1 PRECEDING. A feature keyed on the event's own sender (out side) or
-- receiver (in side) is read at the event's own row of that partition. A feature keyed on the
-- other end (v's out-stream, u's in-stream, the reverse pair) is read at a query row: a copy of the
-- event placed in that partition whose data columns are all NULL, so it adds nothing to any
-- aggregate and only reads its frame (count(col), sum, avg, var_pop and max ignore NULLs).
--
-- Input: view `o_tx(row_id, rank, minute, src, dst, amount_usd, amount_paid, payment_format, l)`,
-- l = log1p(amount_usd) computed by polars, and table `o_fmt(fmt, code)` = the train
-- payment-format vocabulary. `${name}` placeholders are filled by aml.features.oracle with
-- validated numbers; `${fmt_counts}` is a generated list of per-format counts built from integer
-- vocabulary indices only. Statements are separated by "-- step: <name>" lines and run in file
-- order; the last step is the SELECT that returns one row per event in rank order.

-- step: events
-- Cents as M1 rounds them (half away from zero on the double), the flags the RULE group counts,
-- the format code (-1 = not in the train vocabulary) and the pair's first minute: an event is
-- new-pair iff no event of its pair has an earlier minute.
CREATE OR REPLACE TEMP TABLE o_ev AS
WITH t AS (
    SELECT
        row_id, rank, minute, src, dst, l, amount_usd, payment_format,
        CAST(round(amount_usd * 100) AS BIGINT) AS usd_c,
        CAST(round(amount_paid * 100) AS BIGINT) AS paid_c
    FROM o_tx
),
p AS (SELECT src, dst, min(minute) AS first_min FROM t GROUP BY src, dst)
SELECT
    t.row_id, t.rank, t.minute, t.src, t.dst, t.usd_c, t.l,
    t.src = t.dst AS self_loop,
    t.amount_usd >= ${band_low_usd} AND t.amount_usd < ${band_high_usd} AS in_band,
    t.paid_c > 0 AND t.paid_c % ${round_cents} = 0 AS is_round,
    coalesce(f.code, -1) AS fmt,
    t.minute = p.first_min AS is_new
FROM t
JOIN p ON p.src = t.src AND p.dst = t.dst
LEFT JOIN o_fmt f ON f.fmt = t.payment_format;

-- step: ports
-- Port of a pair (Multi-GNN, causal): the number of the sender's receivers whose first minute is
-- strictly earlier than this pair's first minute (pairs first seen in the same minute share a
-- port); the same for the receiver's senders. Self-loop pairs count as both.
CREATE OR REPLACE TEMP TABLE o_ports AS
WITH p AS (SELECT src, dst, min(minute) AS first_min FROM o_ev GROUP BY src, dst)
SELECT
    src, dst,
    count(*) OVER (PARTITION BY src ORDER BY first_min
        RANGE BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS port_out,
    count(*) OVER (PARTITION BY dst ORDER BY first_min
        RANGE BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS port_in
FROM p;

-- step: u_out
-- E_W(u, out), read at the event's own row of its sender's partition.
CREATE OR REPLACE TEMP TABLE o_u_out AS
SELECT
    row_id,
    count(*) OVER ws AS u_out_cnt_s,
    count(DISTINCT dst) OVER ws AS u_out_uniq_s,
    coalesce(CAST(sum(usd_c) OVER ws AS BIGINT), 0) AS u_out_sum_s,
    avg(l) OVER ws AS u_out_mean_s,
    var_pop(l) OVER ws AS u_out_var_s,
    avg(l * l) OVER ws AS u_out_m2_s,
    max(l) OVER ws AS u_out_max_s,
    count(*) OVER wl AS u_out_cnt_l,
    count(DISTINCT dst) OVER wl AS u_out_uniq_l,
    coalesce(CAST(sum(usd_c) OVER wl AS BIGINT), 0) AS u_out_sum_l,
    avg(l) OVER wl AS u_out_mean_l,
    var_pop(l) OVER wl AS u_out_var_l,
    avg(l * l) OVER wl AS u_out_m2_l,
    count(*) FILTER (WHERE in_band) OVER ws AS u_out_inband_s,
    count(*) FILTER (WHERE is_round) OVER ws AS u_out_round_s,
    count(*) FILTER (WHERE is_new) OVER ws AS u_out_newcp_s,
    ${fmt_counts}
    max(minute) OVER wall AS u_out_last
FROM o_ev
WINDOW
    ws AS (PARTITION BY src ORDER BY minute RANGE BETWEEN ${short} PRECEDING AND 1 PRECEDING),
    wl AS (PARTITION BY src ORDER BY minute RANGE BETWEEN ${long} PRECEDING AND 1 PRECEDING),
    wall AS (PARTITION BY src ORDER BY minute
        RANGE BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING);

-- step: v_in
-- E_W(v, in), read at the event's own row of its receiver's partition; the same-format count
-- partitions the receiver's events by format code as well (0 for an unknown format).
CREATE OR REPLACE TEMP TABLE o_v_in AS
SELECT
    row_id,
    count(*) OVER ws AS v_in_cnt_s,
    count(DISTINCT src) OVER ws AS v_in_uniq_s,
    coalesce(CAST(sum(usd_c) OVER ws AS BIGINT), 0) AS v_in_sum_s,
    avg(l) OVER ws AS v_in_mean_s,
    var_pop(l) OVER ws AS v_in_var_s,
    avg(l * l) OVER ws AS v_in_m2_s,
    max(l) OVER ws AS v_in_max_s,
    count(*) OVER wl AS v_in_cnt_l,
    count(DISTINCT src) OVER wl AS v_in_uniq_l,
    coalesce(CAST(sum(usd_c) OVER wl AS BIGINT), 0) AS v_in_sum_l,
    avg(l) OVER wl AS v_in_mean_l,
    var_pop(l) OVER wl AS v_in_var_l,
    avg(l * l) OVER wl AS v_in_m2_l,
    CASE WHEN fmt < 0 THEN 0 ELSE count(*) OVER wf END AS v_in_same_fmt_s,
    max(minute) OVER wall AS v_in_last
FROM o_ev
WINDOW
    ws AS (PARTITION BY dst ORDER BY minute RANGE BETWEEN ${short} PRECEDING AND 1 PRECEDING),
    wl AS (PARTITION BY dst ORDER BY minute RANGE BETWEEN ${long} PRECEDING AND 1 PRECEDING),
    wf AS (PARTITION BY dst, fmt ORDER BY minute
        RANGE BETWEEN ${short} PRECEDING AND 1 PRECEDING),
    wall AS (PARTITION BY dst ORDER BY minute
        RANGE BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING);

-- step: v_out
-- E_W(v, out): one query row per event in its receiver's out-partition.
CREATE OR REPLACE TEMP TABLE o_v_out AS
WITH s AS (
    SELECT src AS acct, minute, dst AS cp, usd_c, minute AS t, NULL::BIGINT AS qid FROM o_ev
    UNION ALL
    SELECT dst, minute, NULL, NULL, NULL, row_id FROM o_ev
)
SELECT
    qid AS row_id,
    count(cp) OVER ws AS v_out_cnt_s,
    count(DISTINCT cp) OVER ws AS v_out_uniq_s,
    coalesce(CAST(sum(usd_c) OVER ws AS BIGINT), 0) AS v_out_sum_s,
    count(cp) OVER wl AS v_out_cnt_l,
    count(DISTINCT cp) OVER wl AS v_out_uniq_l,
    coalesce(CAST(sum(usd_c) OVER wl AS BIGINT), 0) AS v_out_sum_l,
    max(t) OVER wall AS v_out_last
FROM s
WINDOW
    ws AS (PARTITION BY acct ORDER BY minute RANGE BETWEEN ${short} PRECEDING AND 1 PRECEDING),
    wl AS (PARTITION BY acct ORDER BY minute RANGE BETWEEN ${long} PRECEDING AND 1 PRECEDING),
    wall AS (PARTITION BY acct ORDER BY minute
        RANGE BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
QUALIFY qid IS NOT NULL;

-- step: u_in
-- E_W(u, in) and the pass-through inflow (non-self-loop edges into u over the rule's window):
-- one query row per event in its sender's in-partition.
CREATE OR REPLACE TEMP TABLE o_u_in AS
WITH s AS (
    SELECT
        dst AS acct, minute, src AS cp, usd_c,
        CASE WHEN NOT self_loop THEN usd_c END AS nsl_c,
        minute AS t, NULL::BIGINT AS qid
    FROM o_ev
    UNION ALL
    SELECT src, minute, NULL, NULL, NULL, NULL, row_id FROM o_ev
)
SELECT
    qid AS row_id,
    count(cp) OVER ws AS u_in_cnt_s,
    count(DISTINCT cp) OVER ws AS u_in_uniq_s,
    coalesce(CAST(sum(usd_c) OVER ws AS BIGINT), 0) AS u_in_sum_s,
    count(cp) OVER wl AS u_in_cnt_l,
    count(DISTINCT cp) OVER wl AS u_in_uniq_l,
    coalesce(CAST(sum(usd_c) OVER wl AS BIGINT), 0) AS u_in_sum_l,
    coalesce(CAST(sum(nsl_c) OVER wpt AS BIGINT), 0) AS inflow_c,
    max(t) OVER wall AS u_in_last
FROM s
WINDOW
    ws AS (PARTITION BY acct ORDER BY minute RANGE BETWEEN ${short} PRECEDING AND 1 PRECEDING),
    wl AS (PARTITION BY acct ORDER BY minute RANGE BETWEEN ${long} PRECEDING AND 1 PRECEDING),
    wpt AS (PARTITION BY acct ORDER BY minute
        RANGE BETWEEN ${pass_through} PRECEDING AND 1 PRECEDING),
    wall AS (PARTITION BY acct ORDER BY minute
        RANGE BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
QUALIFY qid IS NOT NULL;

-- step: pair
-- Earlier u -> v events: counts per feature window and the last earlier minute.
CREATE OR REPLACE TEMP TABLE o_pair AS
SELECT
    row_id,
    count(*) OVER ws AS pair_cnt_s,
    count(*) OVER wl AS pair_cnt_l,
    max(minute) OVER wall AS pair_last
FROM o_ev
WINDOW
    ws AS (PARTITION BY src, dst ORDER BY minute
        RANGE BETWEEN ${short} PRECEDING AND 1 PRECEDING),
    wl AS (PARTITION BY src, dst ORDER BY minute
        RANGE BETWEEN ${long} PRECEDING AND 1 PRECEDING),
    wall AS (PARTITION BY src, dst ORDER BY minute
        RANGE BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING);

-- step: rev_pair
-- Earlier v -> u events (the 2-edge cycles closing on u -> v over the round-trip window, and the
-- reverse pair's last minute): one query row per event in the reverse pair's partition.
CREATE OR REPLACE TEMP TABLE o_rev AS
WITH s AS (
    SELECT src AS a, dst AS b, minute, minute AS t, NULL::BIGINT AS qid FROM o_ev
    UNION ALL
    SELECT dst, src, minute, NULL, row_id FROM o_ev
)
SELECT
    qid AS row_id,
    count(t) OVER wrt AS rev_cnt_rt,
    max(t) OVER wall AS rev_last
FROM s
WINDOW
    wrt AS (PARTITION BY a, b ORDER BY minute
        RANGE BETWEEN ${round_trip} PRECEDING AND 1 PRECEDING),
    wall AS (PARTITION BY a, b ORDER BY minute
        RANGE BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
QUALIFY qid IS NOT NULL;

-- step: features
SELECT
    e.row_id, e.rank, e.minute, e.src, e.dst, e.usd_c, e.l, e.self_loop, e.in_band, e.fmt,
    e.is_new, po.port_out, po.port_in,
    uo.* EXCLUDE (row_id),
    vi.* EXCLUDE (row_id),
    vo.* EXCLUDE (row_id),
    ui.* EXCLUDE (row_id),
    pr.* EXCLUDE (row_id),
    rv.* EXCLUDE (row_id)
FROM o_ev e
JOIN o_ports po ON po.src = e.src AND po.dst = e.dst
JOIN o_u_out uo ON uo.row_id = e.row_id
JOIN o_v_in vi ON vi.row_id = e.row_id
JOIN o_v_out vo ON vo.row_id = e.row_id
JOIN o_u_in ui ON ui.row_id = e.row_id
JOIN o_pair pr ON pr.row_id = e.row_id
JOIN o_rev rv ON rv.row_id = e.row_id
ORDER BY e.rank;
