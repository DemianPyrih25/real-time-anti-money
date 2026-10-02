"""Engine golden test (M2 spec §9.2): the first 2,000 fixture events reproduce the recorded output.

Compared under the tolerance classes on every machine: exact features, severities and the int tail
bit-equal, ulp features within MACHINE_ULPS ulps, mean/std features within their bound (m2 from
the recorded mean and std). Bit-exactness is not asserted across machines because the fixture's
amount_usd itself varies in the last bits between machines (see MACHINE_ULPS); bit-exact output on
one machine is covered by the determinism and restart tests. A changed spec hash or ENGINE_VERSION
fails with the regeneration command: intended semantic changes regenerate it.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from aml.features.spec import ENGINE_VERSION, window_tag
from tests.fixtures.engine_frames import require_engine, run_engine
from tests.fixtures.engine_ref import moment_ok, same_bits
from tests.golden.regen import (
    GOLDEN_PATH,
    N_EVENTS,
    REGEN_CMD,
    decode,
    encode,
    golden_inputs,
    read_golden,
)

# Slack for `ulp` features between machines. Measured against the golden (recorded on Linux):
# log1p of the same exact value differs by at most 1 ulp between libms (ports, cent sums).
# Features read straight from amount_usd (log_amount_usd, the window maxima) differ by up to 7 ulps
# on Windows and also between two Linux machines, because amount_usd carries FX rates fitted by
# least squares, whose last bits depend on the CPU's BLAS kernels (e.g. AVX2 vs AVX-512).
# Exact-class features matched bit for bit in every case.
MACHINE_ULPS = 16


def within_ulps(got: float, want: float, n: int) -> bool:
    if math.isnan(got) or math.isnan(want):
        return same_bits(got, want)
    return abs(got - want) <= n * math.ulp(max(abs(got), abs(want)))


def golden_mismatches(path: Path, prepared, rules_cfg) -> list[str]:
    """Differences between the engine on the prepared fixture and the golden file at `path`."""
    header, cols, data = read_golden(path)
    spec, tx = golden_inputs(prepared, rules_cfg)
    if (
        header.get("engine_version") != str(ENGINE_VERSION)
        or header.get("spec_hash") != spec.spec_hash()
    ):
        pytest.fail(
            f"the golden file was written for engine_version {header.get('engine_version')} / "
            f"spec_hash {header.get('spec_hash')}; this code is engine_version {ENGINE_VERSION} "
            f"/ spec_hash {spec.spec_hash()}. For an intended change of engine semantics bump "
            f"ENGINE_VERSION, regenerate with `{REGEN_CMD}` and review the diff."
        )
    assert cols == ["row_id", "rank", *spec.row_layout]
    assert len(data) == N_EVENTS == tx.height
    rows = run_engine(tx, spec)
    n_float = spec.i_inflow
    bad: list[str] = []
    for rec, row, rid in zip(data, rows, tx["row_id"].to_list(), strict=True):
        if int(rec[0]) != rid:
            bad.append(f"row order: golden row_id {rec[0]} != {rid}")
            continue
        want = [decode(s, k < n_float) for k, s in enumerate(rec[2:])]
        for k, name in enumerate(spec.row_layout):
            g, w = row[k], want[k]
            if k >= n_float:
                ok = type(g) is int and g == w
            elif k >= spec.n_features or spec.features[k].tol == "exact":
                ok = same_bits(g, w)
            elif spec.features[k].tol == "ulp":
                ok = within_ulps(g, w, MACHINE_ULPS)
            else:
                fd = spec.features[k]
                t = window_tag(fd.window)
                mean = want[spec.feature_index[f"{fd.side}_{fd.direction}_mean_{t}"]]
                std = want[spec.feature_index[f"{fd.side}_{fd.direction}_std_{t}"]]
                m2 = 0.0 if math.isnan(mean) else std * std + mean * mean
                ok = moment_ok(fd.tol, g, w, m2)
            if not ok:
                bad.append(f"row {rid}: {name} {g!r} != golden {w!r}")
    return bad


def test_engine_matches_golden(prepared, rules_cfg):
    if not GOLDEN_PATH.exists():
        pytest.skip(
            f"golden file {GOLDEN_PATH.name} is absent: it is generated at integration with "
            f"`{REGEN_CMD}` (then committed)"
        )
    require_engine()
    bad = golden_mismatches(GOLDEN_PATH, prepared, rules_cfg)
    assert not bad, bad[:20]


def test_golden_encoding_round_trips_bits():
    for x in (0.0, -0.0, 1.0, 2.0, 0.1, math.pi, 1e-300, 5e-324, 1e308, math.inf):
        assert same_bits(decode(encode(x), True), x)
    assert math.isnan(decode(encode(math.nan), True))
    assert decode(encode(7), False) == 7 and encode(2.0) == "0x1p+1"
