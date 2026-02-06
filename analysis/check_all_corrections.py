import json
import re
import sys

filepath = sys.argv[1] if len(sys.argv) > 1 else "/scratch/dkhasha1/tli104/outputs/hmmt25_inference/output_t8192_n16_r3_Qwen3-4B-Instruct-2507.jsonl"

def extract_boxed(text):
    matches = re.findall(r"\\boxed\{([^}]+)\}", text)
    return matches

all_corrections = {
    "WRONG -> RIGHT": [],
    "RIGHT -> WRONG": [],
    "WRONG -> WRONG (different)": []
}

total_samples = 0
total_with_corrections = 0

with open(filepath) as f:
    for line_num, line in enumerate(f, 1):
        data = json.loads(line)
        ground_truth = data["answer"]
        problem_idx = data.get("problem_idx", line_num)
        problem_preview = data['original_problem'][:60] + "..."

        for i, sample in enumerate(data["samples"]):
            total_samples += 1
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

            if has_correction:
                total_with_corrections += 1
                first_ans = unique_answers[0][1]
                last_ans = unique_answers[-1][1]
                first_correct = first_ans == ground_truth
                last_correct = last_ans == ground_truth
                progression = " -> ".join([f"{loc}:{ans}" for loc, ans in unique_answers])

                entry = {
                    "problem_idx": problem_idx,
                    "sample_idx": i,
                    "ground_truth": ground_truth,
                    "progression": progression,
                    "problem": problem_preview
                }

                if not first_correct and last_correct:
                    all_corrections["WRONG -> RIGHT"].append(entry)
                elif first_correct and not last_correct:
                    all_corrections["RIGHT -> WRONG"].append(entry)
                else:
                    all_corrections["WRONG -> WRONG (different)"].append(entry)

print(f"Total samples analyzed: {total_samples}")
print(f"Samples with answer changes: {total_with_corrections}")
print("=" * 80)

for correction_type, entries in all_corrections.items():
    print(f"\n{correction_type}: {len(entries)} cases")
    print("-" * 40)
    for entry in entries[:10]:  # Show first 10
        print(f"  Problem {entry['problem_idx']}, Sample {entry['sample_idx']}")
        print(f"    GT: {entry['ground_truth']}")
        print(f"    {entry['progression']}")
    if len(entries) > 10:
        print(f"  ... and {len(entries) - 10} more")
