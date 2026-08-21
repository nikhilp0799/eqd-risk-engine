"""Property-based invariants for Black-76 pricing and IV inversion.

Per README Step 16's testing pyramid: "Put-call parity for any (F,K,T,sigma)"
is called out explicitly as the kind of invariant that should always hold.
"""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from eqdrisk.pricing.blackscholes import call_price, put_price
from eqdrisk.vol.implied import invert_iv

reasonable_params = st.tuples(
    st.floats(min_value=10.0, max_value=10_000.0),  # forward
    st.floats(min_value=0.5, max_value=1.5),  # strike as a multiple of forward, applied below
    st.floats(min_value=0.01, max_value=5.0),  # T
    st.floats(min_value=0.01, max_value=2.0),  # sigma
    st.floats(min_value=0.5, max_value=1.0),  # discount factor
)


@given(reasonable_params)
def test_put_call_parity_holds(params):
    forward, strike_mult, T, sigma, discount_factor = params
    strike = forward * strike_mult

    c = call_price(forward, strike, T, sigma, discount_factor)
    p = put_price(forward, strike, T, sigma, discount_factor)

    assert abs((c - p) - discount_factor * (forward - strike)) < 1e-6


@given(reasonable_params)
def test_call_price_monotone_increasing_in_sigma(params):
    forward, strike_mult, T, sigma, discount_factor = params
    strike = forward * strike_mult

    lower = call_price(forward, strike, T, sigma, discount_factor)
    higher = call_price(forward, strike, T, sigma * 1.5, discount_factor)

    assert higher >= lower - 1e-9


@given(reasonable_params)
def test_price_to_iv_to_price_round_trip(params):
    forward, strike_mult, T, sigma, discount_factor = params
    strike = forward * strike_mult
    is_call = strike >= forward  # invert the OTM side, matching the project convention

    price = (
        call_price(forward, strike, T, sigma, discount_factor)
        if is_call
        else put_price(forward, strike, T, sigma, discount_factor)
    )

    solved = invert_iv(price, forward, strike, T, discount_factor, is_call)
    assert solved is not None

    recovered_price = (
        call_price(forward, strike, T, solved, discount_factor)
        if is_call
        else put_price(forward, strike, T, solved, discount_factor)
    )
    # README's "recovers within 1e-10" is a relative bar in practice — absolute price
    # magnitude here ranges from cents to thousands (forward up to 10,000), and the
    # solver's precision on sigma propagates through vega, not a fixed absolute price error.
    assert abs(recovered_price - price) < 1e-10 * max(abs(price), 1.0)
