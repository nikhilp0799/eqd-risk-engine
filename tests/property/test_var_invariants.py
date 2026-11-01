"""Property-based invariants for the historical-simulation risk measures.

README Step 16's testing pyramid names "VaR <= ES" as an invariant that must
always hold.
"""

from __future__ import annotations

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from eqdrisk.risk.var import var_es

pnls = st.lists(st.floats(min_value=-1e7, max_value=1e7, allow_nan=False), min_size=1, max_size=600)
confidences = st.floats(min_value=0.5, max_value=0.999)


@given(pnls, confidences)
def test_es_is_never_below_var(pnl, confidence):
    var, es = var_es(np.array(pnl), confidence)
    assert es >= var - 1e-6 * max(1.0, abs(var))


@given(pnls, confidences)
def test_var_is_one_of_the_scenario_losses(pnl, confidence):
    var, _ = var_es(np.array(pnl), confidence)
    assert -var in pnl


@given(pnls, st.floats(min_value=0.5, max_value=0.98))
def test_higher_confidence_never_lowers_var(pnl, confidence):
    lower, _ = var_es(np.array(pnl), confidence)
    higher, _ = var_es(np.array(pnl), confidence + 0.01)
    assert higher >= lower
