import json
import re
from collections import defaultdict

filepath = "/scratch/dkhasha1/tli104/outputs/hmmt25_inference/output_t8192_n16_r3_Qwen3-4B-Instruct-2507.jsonl"

def extract_boxed(text):
    matches = re.findall(r"\\boxed\{([^}]+)\}", text)
    return matches

# Categorize errors by looking at patterns in the reasoning
error_categories = {
    "gave_up_guessed": [],      # Explicitly uncertain, guessed
    "missed_cases": [],          # Incomplete case analysis
    "wrong_formula": [],         # Applied wrong theorem/formula
    "hallucinated_known": [],    # Claimed it's a "known result"
    "calculation_error": [],     # Arithmetic/algebra mistakes
    "misread_problem": [],       # Misunderstood problem statement
}

with open(filepath) as f:
    for line_num, line in enumerate(f, 1):
        data = json.loads(line)
        ground_truth = data["answer"]
        problem_idx = data.get("problem_idx", line_num)
        problem = data['original_problem']

        for i, sample in enumerate(data["samples"]):
            r1_gen = sample["rounds"][0]["current_round_generation"]
            r1_gen_answers = extract_boxed(r1_gen)
            r1_answer = r1_gen_answers[-1] if r1_gen_answers else None

            if r1_answer and r1_answer != ground_truth:
                gen_lower = r1_gen.lower()

                entry = {
                    "problem_idx": problem_idx,
                    "sample_idx": i,
                    "gt": ground_truth,
                    "got": r1_answer,
                }

                # Categorize
                if any(p in gen_lower for p in ["not confident", "i will guess", "go with that", "i'll box", "i think the intended"]):
                    error_categories["gave_up_guessed"].append(entry)
                elif "known" in gen_lower and any(p in gen_lower for p in ["known result", "known problem", "after research"]):
                    error_categories["hallucinated_known"].append(entry)
                elif any(p in gen_lower for p in ["no others", "only two", "only one", "that's all"]) and "wait" in gen_lower:
                    error_categories["missed_cases"].append(entry)
                elif any(p in gen_lower for p in ["but this contradicts", "this is impossible", "something is wrong"]):
                    error_categories["wrong_formula"].append(entry)
                else:
                    error_categories["calculation_error"].append(entry)

# Print summary
print("=" * 80)
print("ERROR CATEGORY SUMMARY")
print("=" * 80)

for cat, entries in error_categories.items():
    if entries:
        # Group by problem
        by_prob = defaultdict(list)
        for e in entries:
            by_prob[e["problem_idx"]].append(e)

        print(f"\n{cat.upper().replace('_', ' ')}: {len(entries)} cases across {len(by_prob)} problems")
        print("-" * 60)
        for prob_idx in sorted(by_prob.keys())[:5]:  # Show first 5 problems
            samples = by_prob[prob_idx]
            print(f"  Problem {prob_idx}: {len(samples)} samples")
            print(f"    GT: {samples[0]['gt']}, Got: {samples[0]['got']}")
