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


@pytest.fixture(scope="session")
def valkey_url(tmp_path_factory):
    """A real Valkey/Redis server for tests that need Lua or exact Redis semantics (skipped if absent)."""
    import shutil
    import socket
    import subprocess
    import time

    binary = shutil.which("valkey-server") or shutil.which("redis-server")
    if not binary:
        pytest.skip("valkey-server/redis-server not installed")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    workdir = tmp_path_factory.mktemp("valkey")
    proc = subprocess.Popen([binary, "--port", str(port), "--save", "", "--appendonly", "no", "--dir", str(workdir)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  # fmt: skip
    for _ in range(100):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                break
        time.sleep(0.05)
    yield f"redis://127.0.0.1:{port}/0"
    proc.terminate()
    proc.wait(timeout=5)
