import pytest

from llm_gateway.research import workloads
from llm_gateway.research.record import CONDITIONS, interleave, stable_coin

needs_data = pytest.mark.skipif(
    not (workloads.DATA / "gsm8k_test.parquet").exists(), reason="run scripts/get_datasets.sh first"
)


def test_gsm8k_answer_checking():
    assert workloads.gsm8k_correct("6 * 3 = 18\nAnswer: 18", "18")
    assert workloads.gsm8k_correct("Answer: $1,250", "1250")
    assert workloads.gsm8k_correct("so the total is 42", "42")  # falls back to the last number
    assert not workloads.gsm8k_correct("Answer: 17", "18")
    assert not workloads.gsm8k_correct("no idea", "18")


def test_bfcl_python_types_become_json_schema():
    schema = {"type": "dict", "properties": {"x": {"type": "float"}, "pts": {"type": "tuple", "items": {}}}}
    assert workloads._json_schema(schema) == {
        "type": "object",
        "properties": {"x": {"type": "number"}, "pts": {"type": "array", "items": {}}},
    }


def test_interleave_and_stream_choice_are_deterministic():
    items = [workloads.Item("t", str(i), {}) for i in range(20)]
    assert [i.item_id for i in interleave(items, 3)] == [i.item_id for i in interleave(items, 3)]
    assert stable_coin("gsm8k-1") == stable_coin("gsm8k-1")


def test_every_condition_has_one_fault_at_most_and_known_tenants():
    for cond in CONDITIONS.values():
        assert len(cond["faults"]) <= 1
        assert set(cond["shifted"]) <= set(workloads.TENANTS)


@needs_data
def test_tenants_load_with_system_prompts_and_shift_changes_them():
    items = workloads.load(["math", "chat", "code", "tools"], 3)
    assert len(items) == 12
    assert all(i.body["messages"][0]["role"] == "system" for i in items)
    assert all("tools" in i.body for i in items if i.tenant == "tools")
    plain = workloads.chat_items(1)[0].body["messages"][0]["content"]
    assert workloads.chat_items(1, shifted=True)[0].body["messages"][0]["content"] != plain
