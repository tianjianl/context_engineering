# Refinement Prompt Comparison: Old (RC) vs New (RC-Verify)

## Key Differences

| Aspect | Old RC Prompt | New RC-Verify Prompt |
|--------|--------------|---------------------|
| **Goal** | Summarize what happened | Critically evaluate and guide next attempt |
| **Stance toward solution** | Neutral reporter | Active verifier/critic |
| **Error handling** | Summarize the partial solution as-is | Flag errors, incorrect assumptions, unjustified leaps |
| **New reasoning** | **Explicitly forbidden** ("Do not under any circumstances add any additional reasoning") | **Encouraged** (verify steps, suggest fixes, propose alternatives) |
| **Forward-looking** | No guidance for next attempt | Concrete strategies for improvement |
| **Answer handling** | Not addressed | Assess correctness against problem constraints |
| **Approach diversity** | Not addressed | Suggest abandoning bad approaches, trying alternatives |

## Old RC Prompt (Summary-Only)

```
Your task is to write a summary of the current candidate solution.

The new summary you generate should possess the following characteristics:
- It should provide a detailed overview of what occurred in the current candidate solution.
- It should summarize the current candidate solution in light of any previous summaries.
- It should be no more than two paragraphs long.
- It should be written in the first person.

IMPORTANT: Do not under any circumstances add any additional reasoning not
contained in the latest reasoning step. Your task is only to summarize what
is given to you.
```

## New RC-Verify Prompt (Verify + Improve)

```
Your task is to write a critical summary that will help guide the next
solution attempt. The summary should analyze what was tried and identify
how to improve.

The summary you generate should possess the following characteristics:
- It should verify the correctness of key steps. Flag any errors, incorrect
  assumptions, or unjustified leaps in logic.
- It should note which approaches were tried and whether they seem promising
  or should be abandoned.
- It should preserve important intermediate results and correct calculations.
- If a final answer was reached, assess whether it is likely correct.
- It should suggest concrete strategies for the next attempt:
  - If error found → explain where it went wrong and how to fix it.
  - If stuck → suggest alternative problem-solving strategy.
  - If incomplete → indicate what remains and how to proceed.
  - If correct → suggest verification via different method.
- It should be no more than two paragraphs long.
- It should be written in the first person.
```

## Why This Should Help

1. **Error correction loop**: The old prompt propagates errors silently by summarizing them. The new prompt catches errors and tells the model to fix them in the next round.

2. **Strategic diversity**: Rather than repeatedly trying the same failing approach, the new prompt can redirect the model to try algebraic vs. geometric methods, substitution vs. induction, etc.

3. **Verification signal**: By explicitly checking answers against constraints, the model can detect and correct wrong answers instead of carrying them forward.

4. **Guided continuation**: Instead of just knowing "where we left off," the next generation knows *what to do differently*, making each round more productive.

## Usage

```bash
# Old RC prompt
python inference/context_refinement_dp.py --rc --rounds 12 ...

# New RC-Verify prompt
python inference/context_refinement_dp.py --rc_verify --rounds 12 ...
```
