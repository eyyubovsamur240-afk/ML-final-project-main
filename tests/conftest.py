"""Shared pytest helpers.

`run_or_skip(fn, *args, **kwargs)` calls `fn` and turns the starter's TODO
stubs (`raise NotImplementedError("TODO: ...")`) into a *skip*. So a contract
test sits idle (reported as "not implemented yet") until someone fills in that
function, then switches on automatically and must pass from then on. Any other
NotImplementedError is a real failure.
"""

from __future__ import annotations

import pytest


def _run_or_skip(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except NotImplementedError as exc:
        if not str(exc).startswith("TODO"):
            raise  # a deliberate NotImplementedError in finished code is a real failure
        pytest.skip(f"not implemented yet: {getattr(fn, '__qualname__', fn)} ({exc})")


@pytest.fixture
def run_or_skip():
    return _run_or_skip
