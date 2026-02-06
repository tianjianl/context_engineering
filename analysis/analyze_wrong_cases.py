import json
import re

filepath = "/scratch/dkhasha1/tli104/outputs/hmmt25_inference/output_t8192_n16_r3_Qwen3-4B-Instruct-2507.jsonl"

def extract_boxed(text):
    matches = re.findall(r"\\boxed\{([^}]+)\}", text)
    return matches

wrong_to_wrong_cases = []

with open(filepath) as f:
    for line_num, line in enumerate(f, 1):
        data = json.loads(line)
        ground_truth = data["answer"]
        problem_idx = data.get("problem_idx", line_num)
        problem = data['original_problem']

        for i, sample in enumerate(data["samples"]):
            answers_by_round = []

            for r in sample["rounds"]:
                gen = r["current_round_generation"]
                ref = r["refined_context"]

                gen_answers = extract_boxed(gen)
                ref_answers = extract_boxed(ref)

                final_in_gen = gen_answers[-1] if gen_answers else None
                final_in_ref = ref_answers[-1] if ref_answers else None

                answers_by_round.append({
                    "round": r["round"],
                    "gen_answer": final_in_gen,
                    "ref_answer": final_in_ref,
                    "generation": gen,
                    "refined": ref
                })

            all_answers = []
            for r in answers_by_round:
                if r["gen_answer"]:
                    all_answers.append((f"R{r['round']}-gen", r["gen_answer"], r["generation"]))
                if r["ref_answer"]:
                    all_answers.append((f"R{r['round']}-ref", r["ref_answer"], r["refined"]))

            unique_answers = []
            for loc, ans, text in all_answers:
                if not unique_answers or unique_answers[-1][1] != ans:
                    unique_answers.append((loc, ans, text))

            answer_values = [a[1] for a in unique_answers]
            has_correction = len(set(answer_values)) > 1

            if has_correction:
                first_ans = unique_answers[0][1]
                last_ans = unique_answers[-1][1]
                first_correct = first_ans == ground_truth
                last_correct = last_ans == ground_truth

                if not first_correct and not last_correct:
                    wrong_to_wrong_cases.append({
                        "problem_idx": problem_idx,
                        "sample_idx": i,
                        "problem": problem,
                        "ground_truth": ground_truth,
                        "first_answer": first_ans,
                        "last_answer": last_ans,
                        "first_round_gen": answers_by_round[0]["generation"],
                        "first_round_ref": answers_by_round[0]["refined"],
                        "progression": [(loc, ans) for loc, ans, _ in unique_answers]
                    })

print(f"Total WRONG -> WRONG cases: {len(wrong_to_wrong_cases)}")
print("=" * 80)

# Group by problem
from collections import defaultdict
by_problem = defaultdict(list)
for case in wrong_to_wrong_cases:
    by_problem[case["problem_idx"]].append(case)

print(f"\nProblems with WRONG->WRONG cases: {len(by_problem)}")
for prob_idx in sorted(by_problem.keys()):
    cases = by_problem[prob_idx]
    print(f"\n{'='*80}")
    print(f"PROBLEM {prob_idx} ({len(cases)} cases)")
    print(f"{'='*80}")
    print(f"Problem: {cases[0]['problem']}")
    print(f"Ground Truth: {cases[0]['ground_truth']}")

    # Show first case in detail
    case = cases[0]
    print(f"\n--- Sample {case['sample_idx']} ---")
    print(f"Progression: {' -> '.join([f'{loc}:{ans}' for loc, ans in case['progression']])}")

    gen = case["first_round_gen"]
    if gen:
        print(f"\nRound 1 generation length: {len(gen)} chars")
        # Show the reasoning near the answer
        print("\n--- Last 1500 chars of Round 1 generation ---")
        print(gen[-1500:] if len(gen) > 1500 else gen)
