import json
import re

def extract_boxed(text):
    matches = re.findall(r"\\boxed\{([^}]+)\}", text)
    return matches

with open("/scratch/dkhasha1/tli104/outputs/hmmt25_inference/output_t8192_n16_r3_Qwen3-4B-Instruct-2507.jsonl") as f:
    for i, line in enumerate(f, 1):
        data = json.loads(line)
        if data.get("problem_idx") == 4:
            print(f"Problem: {data['original_problem']}")
            print(f"Ground Truth: {data['answer']}")
            print("=" * 70)

            sample = data["samples"][2]  # Sample 2
            for r in sample["rounds"]:
                print(f"\n{'='*20} ROUND {r['round']} {'='*20}")
                gen = r["current_round_generation"]
                ref = r["refined_context"]

                if gen:
                    boxed = extract_boxed(gen)
                    print(f"Generation length: {len(gen)} chars")
                    print(f"Boxed answers found: {boxed}")
                    # Show last part with the answer
                    print(f"\n...last 800 chars of generation:")
                    print(gen[-800:] if len(gen) > 800 else gen)
                else:
                    print("Generation: (empty)")

                print(f"\nRefined context boxed: {extract_boxed(ref)}")
            break
