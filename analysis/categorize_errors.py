import json
import re
from collections import defaultdict

filepath = "/scratch/dkhasha1/tli104/outputs/hmmt25_inference/output_t8192_n16_r3_Qwen3-4B-Instruct-2507.jsonl"

def extract_boxed(text):
    matches = re.findall(r"\\boxed\{([^}]+)\}", text)
    return matches

wrong_cases = []

with open(filepath) as f:
    for line_num, line in enumerate(f, 1):
        data = json.loads(line)
        ground_truth = data["answer"]
        problem_idx = data.get("problem_idx", line_num)
        problem = data['original_problem']

        for i, sample in enumerate(data["samples"]):
            r1_gen = sample["rounds"][0]["current_round_generation"]
            r1_ref = sample["rounds"][0]["refined_context"]

            r1_gen_answers = extract_boxed(r1_gen)
            r1_answer = r1_gen_answers[-1] if r1_gen_answers else None

            if r1_answer and r1_answer != ground_truth:
                wrong_cases.append({
                    "problem_idx": problem_idx,
                    "sample_idx": i,
                    "problem": problem,
                    "ground_truth": ground_truth,
                    "r1_answer": r1_answer,
                    "r1_gen": r1_gen
                })

# Group by problem
by_problem = defaultdict(list)
for case in wrong_cases:
    by_problem[case["problem_idx"]].append(case)

print(f"Total Round 1 wrong answers: {len(wrong_cases)}")
print(f"Problems with wrong answers: {len(by_problem)}")
print("=" * 80)

# For each problem, analyze the error pattern
for prob_idx in sorted(by_problem.keys()):
    cases = by_problem[prob_idx]
    gt = cases[0]["ground_truth"]
    problem = cases[0]["problem"]

    print(f"\n{'='*80}")
    print(f"PROBLEM {prob_idx}: {len(cases)} wrong samples out of 16")
    print(f"{'='*80}")
    print(f"Problem: {problem[:200]}...")
    print(f"Ground Truth: {gt}")

    # Show distribution of wrong answers
    answer_dist = defaultdict(int)
    for c in cases:
        answer_dist[c["r1_answer"]] += 1

    print(f"\nWrong answer distribution:")
    for ans, count in sorted(answer_dist.items(), key=lambda x: -x[1]):
        print(f"  {ans}: {count} samples")

    # Analyze one case in detail - look for error patterns
    case = cases[0]
    gen = case["r1_gen"]

    # Look for common error indicators
    error_indicators = {
        "calculation_error": ["wait", "let me recalculate", "mistake", "error", "wrong", "incorrect"],
        "uncertainty": ["not sure", "perhaps", "maybe", "I think", "assume", "guess"],
        "incomplete": ["...", "to be continued", "left as exercise"],
        "contradiction": ["but this contradicts", "however", "this is impossible"],
        "gave_up": ["I cannot", "unable to", "don't know", "stuck"],
    }

    found_patterns = []
    gen_lower = gen.lower()
    for pattern_type, keywords in error_indicators.items():
        for kw in keywords:
            if kw.lower() in gen_lower:
                found_patterns.append(pattern_type)
                break

    print(f"\nError indicators found: {set(found_patterns) if found_patterns else 'None detected'}")

    # Show key excerpt
    print(f"\n--- Excerpt from Sample {case['sample_idx']} (last 800 chars) ---")
    print(gen[-800:] if len(gen) > 800 else gen)
