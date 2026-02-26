#!/usr/bin/env python3
"""Unit tests for the refactored utility modules."""

import json
import os
import tempfile

import argparse

from inference.data_utils import (
    load_jsonl, save_jsonl, strip_thinking, get_text_from_sample,
    get_solution_text, detect_format, create_refinement_prompt, load_dataset,
    apply_chat_template, apply_chat_template_with_prefix,
)
from inference.verify_utils import compute_pass_at_k, DEFAULT_VERIFY_TIMEOUT
from inference.dp_utils import shard_data, SampleState, save_intermediate_results
from inference.args_utils import (
    add_common_args, add_dp_args, add_refinement_args,
    post_process_args, validate_args,
)


def test_load_save_jsonl():
    """Test round-trip JSONL load/save."""
    data = [{"a": 1, "b": "hello"}, {"a": 2, "b": "world"}]
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        tmp_path = f.name
        for item in data:
            f.write(json.dumps(item) + '\n')

    try:
        loaded = load_jsonl(tmp_path)
        assert loaded == data, f"Expected {data}, got {loaded}"

        # Test save
        save_path = tmp_path + ".out"
        save_jsonl(data, save_path)
        reloaded = load_jsonl(save_path)
        assert reloaded == data, f"Expected {data}, got {reloaded}"

        # Test append mode
        save_jsonl([{"a": 3}], save_path, mode='a')
        final = load_jsonl(save_path)
        assert len(final) == 3
        assert final[2] == {"a": 3}
        os.unlink(save_path)
    finally:
        os.unlink(tmp_path)

    print("  test_load_save_jsonl: PASS")


def test_strip_thinking():
    """Test strip_thinking function."""
    # No think tags
    assert strip_thinking("plain text") == "plain text"

    # Normal think block
    assert strip_thinking("<think>reasoning</think>Answer") == "Answer"

    # Think block with no closing tag (model ran out of tokens)
    assert strip_thinking("<think>reasoning without end") == ""

    # Whitespace handling
    assert strip_thinking("<think>foo</think>  bar  ") == "bar"

    print("  test_strip_thinking: PASS")


def test_get_text_from_sample():
    """Test get_text_from_sample for verify_solutions style extraction."""
    # Rounds with current_round_generation
    sample = {
        "rounds": [
            {"current_round_generation": "round 1"},
            {"current_round_generation": "round 2 with \\boxed{42}"}
        ],
        "generation": "fallback"
    }
    text, source = get_text_from_sample(sample)
    assert text == "round 2 with \\boxed{42}", f"Got: {text}"
    assert source == "last_round_generation"

    # No rounds, fall back to generation
    sample2 = {"generation": "direct gen"}
    text2, source2 = get_text_from_sample(sample2)
    assert text2 == "direct gen"
    assert source2 == "generation"

    # Empty rounds
    sample3 = {"rounds": [], "full_assistant_message": "full msg"}
    text3, source3 = get_text_from_sample(sample3)
    assert text3 == "full msg"
    assert source3 == "full_assistant_message"

    print("  test_get_text_from_sample: PASS")


def test_get_solution_text():
    """Test get_solution_text for grading scripts style extraction."""
    # Rounds with refined context
    sample = {
        "rounds": [{"current_round_generation": "final gen"}],
        "final_refined_context": "summary context",
    }
    text = get_solution_text(sample)
    assert text == "summary context\n\nfinal gen", f"Got: {text}"

    # Rounds without refined context
    sample2 = {
        "rounds": [{"current_round_generation": "only gen"}],
        "final_refined_context": "",
    }
    text2 = get_solution_text(sample2)
    assert text2 == "only gen"

    # No rounds, fall back to generation
    sample3 = {"generation": "baseline gen"}
    text3 = get_solution_text(sample3)
    assert text3 == "baseline gen"

    print("  test_get_solution_text: PASS")


def test_detect_format():
    """Test detect_format."""
    assert detect_format([{"samples": []}]) == "refinement"
    assert detect_format([{"generation": "x"}]) == "baseline"
    assert detect_format([]) == "unknown"
    assert detect_format([{"other": "x"}]) == "unknown"

    print("  test_detect_format: PASS")


def test_create_refinement_prompt():
    """Test create_refinement_prompt generates expected content."""
    prompt = create_refinement_prompt("problem text", "partial gen", preserve_answer=True)
    assert "problem text" in prompt
    assert "partial gen" in prompt
    assert "PRESERVE IT" in prompt

    prompt2 = create_refinement_prompt("problem text", "partial gen", preserve_answer=False)
    assert "problem text" in prompt2
    assert "NEVER include any final answer" in prompt2

    print("  test_create_refinement_prompt: PASS")


def test_compute_pass_at_k():
    """Test compute_pass_at_k."""
    # All correct
    assert compute_pass_at_k(10, 10, 1) == 1.0
    # None correct
    assert compute_pass_at_k(10, 0, 1) == 0.0
    # n < k
    assert compute_pass_at_k(2, 1, 5) == 1.0
    assert compute_pass_at_k(2, 0, 5) == 0.0
    # Known values
    # pass@1 with n=4, c=1: 1 - (3/4) = 0.25
    assert abs(compute_pass_at_k(4, 1, 1) - 0.25) < 1e-9
    # pass@2 with n=4, c=2: 1 - (2/4 * 1/3) = 1 - 1/6 = 5/6
    assert abs(compute_pass_at_k(4, 2, 2) - (5/6)) < 1e-9

    print("  test_compute_pass_at_k: PASS")


def test_shard_data():
    """Test shard_data round-robin sharding."""
    data = [{"id": i} for i in range(10)]

    shards = shard_data(data, 3)
    assert len(shards) == 3
    # Round-robin: 0->s0, 1->s1, 2->s2, 3->s0, 4->s1, 5->s2, ...
    assert len(shards[0]) == 4  # items 0, 3, 6, 9
    assert len(shards[1]) == 3  # items 1, 4, 7
    assert len(shards[2]) == 3  # items 2, 5, 8
    assert shards[0][0]["id"] == 0
    assert shards[0][1]["id"] == 3
    assert shards[1][0]["id"] == 1

    # Single shard
    shards1 = shard_data(data, 1)
    assert len(shards1) == 1
    assert len(shards1[0]) == 10

    print("  test_shard_data: PASS")


def test_load_dataset_hmmt():
    """Test load_dataset for hmmt format."""
    data = [{"prompt": "Q1", "answer": "42"}, {"prompt": "Q2", "answer": "7"}]
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        tmp_path = f.name
        for item in data:
            f.write(json.dumps(item) + '\n')
    try:
        loaded = load_dataset("hmmt", input_file=tmp_path)
        assert loaded == data
    finally:
        os.unlink(tmp_path)

    print("  test_load_dataset_hmmt: PASS")


def test_add_common_args():
    """Test add_common_args adds expected arguments."""
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    args = parser.parse_args(["--dataset", "imobench", "--num_tokens", "1024"])
    assert args.dataset == "imobench"
    assert args.num_tokens == 1024
    assert args.temperature == 0.7
    assert args.top_p == 0.9
    assert args.top_k == -1
    assert args.num_samples == 1

    print("  test_add_common_args: PASS")


def test_add_dp_args():
    """Test add_dp_args adds expected arguments."""
    parser = argparse.ArgumentParser()
    add_dp_args(parser)
    args = parser.parse_args([])
    assert args.num_gpus is None
    assert args.max_model_len is None
    assert args.gpu_memory_utilization == 0.95
    assert args.tensor_parallel_size == 1

    print("  test_add_dp_args: PASS")


def test_add_refinement_args():
    """Test add_refinement_args adds expected arguments."""
    parser = argparse.ArgumentParser()
    add_refinement_args(parser)
    args = parser.parse_args([])
    assert args.preserve_answer == True
    assert args.strip_answer == False
    assert args.max_refinement_tokens is None
    assert args.strip_thinking_from_refinement == True
    assert args.strip_thinking_from_generation == True

    print("  test_add_refinement_args: PASS")


def test_post_process_args():
    """Test post_process_args applies shared logic."""
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    add_dp_args(parser)
    add_refinement_args(parser)

    # Test strip_answer sets preserve_answer=False
    args = parser.parse_args(["--dataset", "imobench", "--num_tokens", "1024", "--strip_answer"])
    post_process_args(args)
    assert args.preserve_answer == False

    # Test keep_thinking overrides
    args2 = parser.parse_args(["--dataset", "imobench", "--num_tokens", "1024",
                                "--keep_thinking_in_refinement", "--keep_thinking_in_generation"])
    post_process_args(args2)
    assert args2.strip_thinking_from_refinement == False
    assert args2.strip_thinking_from_generation == False

    # Test max_refinement_tokens default
    args3 = parser.parse_args(["--dataset", "imobench", "--num_tokens", "1024"])
    post_process_args(args3)
    assert args3.max_refinement_tokens == 16384

    # Test max_model_len default calculation
    assert args3.max_model_len == min(1024 + 16384 + 8192, 131072)

    print("  test_post_process_args: PASS")


def test_validate_args():
    """Test validate_args catches missing input_file for hmmt."""
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    args = parser.parse_args(["--dataset", "hmmt", "--num_tokens", "1024"])
    try:
        validate_args(args, parser)
        assert False, "Should have raised SystemExit"
    except SystemExit:
        pass  # Expected

    # imobench should pass without input_file
    args2 = parser.parse_args(["--dataset", "imobench", "--num_tokens", "1024"])
    validate_args(args2, parser)  # Should not raise

    print("  test_validate_args: PASS")


def test_sample_state():
    """Test SampleState dataclass."""
    state = SampleState(q_idx=0, s_idx=1, original_prompt="test")
    assert state.q_idx == 0
    assert state.s_idx == 1
    assert state.original_prompt == "test"
    assert state.conversation_turns == []
    assert state.rounds_data == []
    assert state.is_done == False
    assert state.done_reason == ""
    assert state.num_tool_calls == 0
    assert state.last_refined_context == ""

    # Test mutability
    state.rounds_data.append({"round": 1})
    state.is_done = True
    state.done_reason = "completed"
    assert len(state.rounds_data) == 1
    assert state.is_done == True

    print("  test_sample_state: PASS")


def test_save_intermediate_results():
    """Test save_intermediate_results with SampleState."""
    valid_items = [{"prompt": "Q1", "problem_id": "p1"}]
    original_prompts = ["Q1"]
    state = SampleState(q_idx=0, s_idx=0, original_prompt="Q1")
    state.rounds_data = [{"round": 1, "current_round_generation": "gen1"}]
    state.last_refined_context = "refined"
    state.num_tool_calls = 1
    state.done_reason = "completed"
    all_states = [[state]]

    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = os.path.join(tmpdir, "test_output.jsonl")
        save_intermediate_results(0, valid_items, original_prompts,
                                   all_states, 1, 0, output_file)

        intermediate_file = os.path.join(tmpdir, "test_output_intermediate_gpu0.jsonl")
        assert os.path.exists(intermediate_file)
        data = load_jsonl(intermediate_file)
        assert len(data) == 1
        assert data[0]["original_prompt"] == "Q1"
        assert len(data[0]["samples"]) == 1
        assert data[0]["samples"][0]["num_tool_calls"] == 1

    print("  test_save_intermediate_results: PASS")


def test_verify_batch():
    """Test verify_batch with actual math-verify."""
    from inference.verify_utils import verify_batch

    items = [
        ("42", "The answer is \\boxed{42}"),
        ("7", "I think the answer is \\boxed{7}"),
        ("100", "The result is \\boxed{99}"),
    ]
    results = verify_batch(items, timeout=5, num_workers=2)
    assert len(results) == 3
    # First two should be correct
    assert results[0][0] == True, f"Expected correct, got {results[0]}"
    assert results[1][0] == True, f"Expected correct, got {results[1]}"
    # Third should be incorrect
    assert results[2][0] == False, f"Expected incorrect, got {results[2]}"

    print("  test_verify_batch: PASS")


if __name__ == "__main__":
    print("Running unit tests for refactored utility modules...")
    print()

    test_load_save_jsonl()
    test_strip_thinking()
    test_get_text_from_sample()
    test_get_solution_text()
    test_detect_format()
    test_create_refinement_prompt()
    test_compute_pass_at_k()
    test_shard_data()
    test_load_dataset_hmmt()
    test_add_common_args()
    test_add_dp_args()
    test_add_refinement_args()
    test_post_process_args()
    test_validate_args()
    test_sample_state()
    test_save_intermediate_results()
    test_verify_batch()

    print()
    print("All tests passed!")
