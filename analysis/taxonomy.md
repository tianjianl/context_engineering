# Taxonomy of LLM Mathematical Reasoning Errors

Based on analysis of 285 incorrect Round 1 answers from Qwen3-4B on HMMT 2025 problems.

---

## I. Knowledge Errors

Errors stemming from incorrect or fabricated factual claims.

### I.A. Hallucinated "Known Results"
**Frequency: ~57% of errors (most common)**

The model claims a problem is "known" or cites a "standard result" that is incorrect.

**Indicators:**
- "This is a known problem..."
- "After research, the accepted answer is..."
- "A classic result states..."
- "In such problems, the answer is always..."

**Example (Problem 8):**
> "A classic result is that for irrational rotations, such sums converge to zero due to symmetry."
>
> Got: `0`, Ground Truth: `1 - 2/π`

**Why it happens:** The model pattern-matches to superficially similar problems but retrieves or fabricates wrong conclusions.

### I.B. Misapplied Theorems
**Frequency: ~2% of errors**

Correctly recalls a theorem but applies it to a situation where preconditions aren't met.

**Example (Problem 6):**
- Applied Fermat's Little Theorem without checking that base and modulus are coprime
- 2017 divides 2025!, so the theorem doesn't apply directly

---

## II. Reasoning Errors

Errors in the logical process of solving the problem.

### II.A. Incomplete Case Enumeration
**Frequency: ~10% of errors**

Fails to explore all cases in a systematic search.

**Indicators:**
- "No others"
- "That's all"
- "Only two divisors satisfy..."
- Followed by "Wait—" (sometimes catches it, sometimes not)

**Example (Problem 1):**
> Found divisors 1 and 81 (both end in 1), but missed 21 = 3×7.
>
> Got: `82` (1+81), Ground Truth: `103` (1+21+81)

**Why it happens:** Model prematurely concludes exhaustive search without systematic verification.

### II.B. Flawed Logical Inference
**Frequency: ~5% of errors**

Makes an invalid logical step in the proof chain.

**Indicators:**
- "Therefore..." followed by non-sequitur
- "This implies..." with incorrect implication
- Contradiction detected but wrong branch discarded

**Example (Problem 17):**
> "Since 3 is odd, symmetric arrangements are impossible. Therefore, count = 0."
>
> The symmetry argument was incorrectly applied.
> Got: `0`, Ground Truth: `2^25 · 26!`

### II.C. Incorrect Problem Reduction
**Frequency: ~3% of errors**

Transforms the problem into a simpler form but loses information or introduces errors.

**Example (Problem 20):**
> Reduced 45-position circle to 15-position "effective" circle
> Applied coalescing random walk formula to wrong state space

---

## III. Computational Errors

Errors in arithmetic, algebra, or symbolic manipulation.

### III.A. Arithmetic Mistakes
**Frequency: ~10% of errors**

Simple calculation errors in addition, multiplication, counting.

**Example (Problem 4):**
> Counted 28 odd divisors when there are 14 in range
> Got: `-970`, Ground Truth: `-984`

### III.B. Algebraic Errors
**Frequency: ~5% of errors**

Errors in symbolic manipulation, equation solving, or simplification.

**Example (Problem 3):**
> Set up correct system of equations
> Made error in solving, got `576` instead of `1/576` (inverted!)

### III.C. Off-by-One/Boundary Errors
**Frequency: ~2% of errors**

Miscounting at boundaries, fencepost errors.

**Example (Problem 15):**
> Got: `202`, Ground Truth: `200`
> Off by 2 in counting edges in grid traversal

---

## IV. Metacognitive Failures

Errors in the model's self-assessment and decision-making process.

### IV.A. Premature Termination (Giving Up)
**Frequency: ~15% of errors**

Model explicitly acknowledges uncertainty and outputs a guess.

**Indicators:**
- "I am not confident"
- "I'll go with that"
- "I think the intended answer is..."
- "After much effort, I'll box..."

**Example (Problem 14):**
> "I know it's wrong... I will stop and provide: \boxed{20}"
>
> Got: `20`, Ground Truth: `2304`

**Why it happens:** Problem complexity exceeds model's capability or context window; model recognizes this but has no better strategy than guessing.

### IV.B. False Confidence
**Frequency: ~8% of errors**

Model expresses high confidence ("✅", "Yes.", "Verified.") but answer is wrong.

**Example (Problem 1):**
> "✅ **Answer:** \boxed{82}"
>
> Ground Truth: `103`

### IV.C. Verification Theater
**Frequency: ~5% of errors**

Model performs a "verification" step that doesn't actually catch the error.

**Indicators:**
- "Let's double-check: ... Yes."
- "Verified." (but verification was circular or incomplete)

**Example (Problem 4):**
> "Let's double-check: ... 28 pairs contribute 0, 972 contribute -1 → total = -972. Yes."
>
> The 28 was wrong, but verification just repeated the wrong calculation.

---

## V. Comprehension Errors

Errors in understanding what the problem is asking.

### V.A. Misread Constraints
**Frequency: ~3% of errors**

Misses or misinterprets a constraint in the problem statement.

**Example (Problem 16):**
> "rectangles do not overlap at their interiors"
> Misinterpreted as "do not share any point" vs "interiors don't overlap (edges can touch)"

### V.B. Wrong Objective
**Frequency: ~2% of errors**

Solves for the wrong quantity.

**Example (Problem 3):**
> Problem asks for "smallest possible value of xyz"
> Model computed xyz = 576 but answer should be 1/576 (minimization vs. a specific value)

---

## Summary Table

| Category | Subcategory | Frequency | Refinable? |
|----------|-------------|-----------|------------|
| **I. Knowledge** | Hallucinated results | ~57% | No |
| | Misapplied theorems | ~2% | Unlikely |
| **II. Reasoning** | Incomplete enumeration | ~10% | Sometimes |
| | Flawed inference | ~5% | Unlikely |
| | Incorrect reduction | ~3% | Unlikely |
| **III. Computational** | Arithmetic mistakes | ~10% | Sometimes |
| | Algebraic errors | ~5% | Sometimes |
| | Boundary errors | ~2% | Sometimes |
| **IV. Metacognitive** | Gave up / guessed | ~15% | No |
| | False confidence | ~8% | No |
| | Verification theater | ~5% | No |
| **V. Comprehension** | Misread constraints | ~3% | Unlikely |
| | Wrong objective | ~2% | Unlikely |

---

## Implications for Context Refinement

**Refinement CAN help with:**
- Computational errors (if re-examination triggers recalculation)
- Incomplete enumeration (if condensed context prompts "did I miss anything?")

**Refinement CANNOT help with:**
- Hallucinated knowledge (wrong facts are preserved)
- Flawed reasoning (condensing bad logic doesn't fix it)
- Metacognitive failures (model already gave up)
- Comprehension errors (misunderstanding is baked in)

**Key insight:** ~70-80% of errors are in categories where refinement is unlikely to help because the error is in the reasoning itself, not in the presentation or organization of the solution.
