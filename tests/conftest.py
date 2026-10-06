"""Shared pytest helpers.

`run_or_skip(fn, *args, **kwargs)` calls `fn` and turns a NotImplementedError
from the starter's TODO stubs into a *skip*. So a contract test sits idle
(reported as "not implemented yet") until someone fills in that function,
then switches on automatically and must pass from then on.
"""

from __future__ import annotations

import pytest


def _run_or_skip(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except NotImplementedError as exc:
        pytest.skip(f"not implemented yet: {getattr(fn, '__qualname__', fn)} ({exc})")


@pytest.fixture
def run_or_skip():
    return _run_or_skip
