import json
import re
import sys

filepath = sys.argv[1] if len(sys.argv) > 1 else "/scratch/dkhasha1/tli104/outputs/hmmt25_inference/output_t8192_n16_r3_Qwen3-4B-Instruct-2507.jsonl"

with open(filepath) as f:
    data = json.loads(f.readline())

ground_truth = data["answer"]
print(f"Ground truth answer: {ground_truth}")
print(f"Problem: {data['original_problem']}")
print("=" * 80)

def extract_boxed(text):
    matches = re.findall(r"\\boxed\{([^}]+)\}", text)
    return matches

corrections_found = []

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
            "ref_answer": final_in_ref
        })

    all_answers = []
    for r in answers_by_round:
        if r["gen_answer"]:
            all_answers.append((f"R{r['round']}-gen", r["gen_answer"]))
        if r["ref_answer"]:
            all_answers.append((f"R{r['round']}-ref", r["ref_answer"]))

    unique_answers = []
    for loc, ans in all_answers:
        if not unique_answers or unique_answers[-1][1] != ans:
            unique_answers.append((loc, ans))

    answer_values = [a[1] for a in unique_answers]
    has_correction = len(set(answer_values)) > 1

    final_answer = unique_answers[-1][1] if unique_answers else "None"
    is_correct = final_answer == ground_truth

    status = "✓" if is_correct else "✗"
    correction_mark = " [CORRECTED]" if has_correction else ""

    print(f"Sample {i}: {status} Final: {final_answer}{correction_mark}")
    if has_correction:
        progression = " -> ".join([f"{loc}:{ans}" for loc, ans in unique_answers])
        print(f"    Progression: {progression}")

        # Check if correction was beneficial
        first_ans = unique_answers[0][1]
        last_ans = unique_answers[-1][1]
        first_correct = first_ans == ground_truth
        last_correct = last_ans == ground_truth

        if not first_correct and last_correct:
            corrections_found.append((i, "WRONG -> RIGHT", progression))
        elif first_correct and not last_correct:
            corrections_found.append((i, "RIGHT -> WRONG", progression))
        else:
            corrections_found.append((i, "WRONG -> WRONG (different)", progression))

print("\n" + "=" * 80)
print("SUMMARY OF CORRECTIONS:")
print("=" * 80)

if corrections_found:
    for sample_idx, correction_type, progression in corrections_found:
        print(f"  Sample {sample_idx}: {correction_type}")
        print(f"    {progression}")
else:
    print("  No corrections found across rounds.")
