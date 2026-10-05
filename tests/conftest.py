import inspect

import pytest


@pytest.fixture
def anyio_backend():
    return "asyncio"


def pytest_collection_modifyitems(items):
    """Run `async def` tests on asyncio via the anyio plugin (bundled with httpx/FastAPI)."""
    for item in items:
        if inspect.iscoroutinefunction(getattr(item, "function", None)):
            item.add_marker(pytest.mark.anyio)
