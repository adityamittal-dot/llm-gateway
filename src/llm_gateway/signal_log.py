"""Buffered Parquet writer for per-response signal rows.

Rows are flushed as part files under <dir>/date=YYYY-MM-DD/ so a run can be read back
with `pandas.read_parquet(dir)` without coordinating writers.
"""

import asyncio
import datetime as dt
import logging
import os
import uuid
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from llm_gateway.signals import ResponseSignals

log = logging.getLogger(__name__)

# Explicit schema so part files always concatenate, even when a column is all-null in one batch.
SCHEMA = pa.schema(
    [
        ("ts", pa.float64()),
        ("request_id", pa.string()),
        ("tenant", pa.string()),
        ("provider", pa.string()),
        ("model", pa.string()),
        ("stream", pa.bool_()),
        ("status_code", pa.int32()),
        ("latency_s", pa.float64()),
        ("ttft_s", pa.float64()),
        ("tokens_per_s", pa.float64()),
        ("prompt_tokens", pa.int64()),
        ("completion_tokens", pa.int64()),
        ("cached_tokens", pa.int64()),
        ("finish_reason", pa.string()),
        ("output_chars", pa.int64()),
        ("empty", pa.bool_()),
        ("refusal", pa.bool_()),
        ("repetition", pa.float64()),
        ("tools_offered", pa.bool_()),
        ("tool_calls", pa.int64()),
        ("tool_call_valid", pa.bool_()),
        ("prompt_hash", pa.string()),
        ("regenerate", pa.bool_()),
        ("truth_fault", pa.string()),
        ("extra", pa.string()),
    ]
)


class SignalLog:
    def __init__(
        self, directory: str | os.PathLike | None, flush_rows: int = 200, flush_interval_s: float = 10.0
    ):
        self.directory = Path(directory) if directory else None
        self.flush_rows = flush_rows
        self.flush_interval_s = flush_interval_s
        self._rows: list[dict] = []
        self._task: asyncio.Task | None = None

    @property
    def enabled(self) -> bool:
        return self.directory is not None

    def append(self, signals: ResponseSignals) -> None:
        if not self.enabled:
            return
        self._rows.append(signals.row())
        if len(self._rows) >= self.flush_rows:
            self.flush()

    def flush(self) -> Path | None:
        if not self.enabled or not self._rows:
            return None
        rows, self._rows = self._rows, []
        day = dt.datetime.fromtimestamp(rows[0]["ts"], dt.UTC).strftime("%Y-%m-%d")
        target = self.directory / f"date={day}"
        target.mkdir(parents=True, exist_ok=True)
        path = target / f"part-{dt.datetime.now(dt.UTC):%H%M%S}-{uuid.uuid4().hex[:8]}.parquet"
        pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), path)
        return path

    async def _periodic_flush(self) -> None:
        while True:
            await asyncio.sleep(self.flush_interval_s)
            try:
                self.flush()
            except Exception:  # a broken disk must never take the gateway down
                log.exception("signal log flush failed")

    def start(self) -> None:
        if self.enabled and self._task is None:
            self._task = asyncio.create_task(self._periodic_flush())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            self._task = None
        self.flush()
