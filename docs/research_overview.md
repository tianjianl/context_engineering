# Learning to Manage Context: LM-Call Tools for Non-Autoregressive Reasoning

## Big Picture

Autoregressive generation has no backspace. Once a model commits 2000 tokens to a proof by induction, it cannot abandon that strategy and try contradiction instead. Humans reason non-autoregressively — we cross out, rewrite, summarize, and redirect. We want to give LMs this capability by equipping them with tools that are themselves LM calls: summarize, refine, delete, and compact their own reasoning context.

The overarching goal is a model that learns to call these context-management tools at the right times — compressing when the context is bloated, redirecting when the strategy is wrong, and leaving good reasoning alone.

## Current Focus

Before giving the model multiple tools, we first need to show that a single "summarize + refine" tool works. This tool takes the model's reasoning so far, produces a compressed summary that flags what worked and what didn't, and the model continues from the cleaned context.

Success criterion: a model with the tool outperforms the same model without it.

## What Worked

**Fixed-schedule tool calling improves accuracy on average.** Forcing the model to summarize+refine every N steps (the "RC reimpl" approach) outperforms single-pass generation on competition math benchmarks. The tool provides genuine value — the summary concentrates useful signal and the refine step can redirect failed strategies.

## What Didn't Work

**Fixed-schedule calling is unreliable.** It sometimes reverts correct solutions. Example: a model finds the right answer on round 2, but the forced summarization at round 7 loses critical detail and the answer regresses. The tool is useful but dangerous — calling it at the wrong time causes regressions. A fixed schedule cannot distinguish "I'm stuck, compress and retry" from "I have a good answer, don't touch it."

**Autonomous tool calling fails without training.** When given the option to call the tool voluntarily, current models call it reflexively — 93% of tool calls are empty (no reasoning before calling), and the call rate is ~74% regardless of problem difficulty. The model treats the tool as a reflex, not a strategic decision. 51% of samples get stuck in a loop of repeated summarizations with 3.9% accuracy.

**Prompt engineering doesn't fix meta-cognition.** Forcing reasoning before the first tool call, cleaning the prompt format, and other interventions reduce reflexive calling but don't produce strategic tool use. The model still lacks the ability to assess its own reasoning state.

## Key Hypothesis

LMs are robust to local errors (arithmetic corruption in correct reasoning traces causes only 0.3-3pp accuracy drop — models recompute locally). But they are likely fragile to strategic errors — committing to the wrong proof technique, the wrong decomposition, the wrong approach. We don't have a clean way to inject strategic errors synthetically, but the pattern is consistent: models fail by going deep down wrong paths, not by making arithmetic mistakes.

The summarize+refine tool's value is not fixing local errors. It is providing an exit ramp from bad strategic choices — a structured moment where the model can step back and redirect.

## Possible Paper Framing

Contributions, in order:

1. **Formalize the distinction between fixed scaffolds and model-controlled context management.** Fixed scaffolds (call the tool every N steps) vs learned policies (model decides when to call). These are fundamentally different — one is a pipeline, the other is a capability.

2. **Show fixed-schedule tool calling helps on average but hurts on specific instances.** The tool improves aggregate accuracy, but regressions on individual problems reveal that *when* you call matters as much as *whether* you call. Bad timing destroys correct solutions.

3. **Show autonomous calling fails and characterize why.** Without training, models lack meta-cognition about their own reasoning state. They call reflexively, not strategically. The failure mode is well-characterized: empty calls, difficulty-independent call rates, Groundhog Day loops.

4. **Show that optimal calling points exist and are predictable.** This is the critical piece. Using per-round correctness data from fixed-schedule runs, construct an oracle policy that calls the tool only when it helps. The gap between oracle and fixed-schedule accuracy establishes a ceiling — there is significant headroom for a learned policy. Optionally, show that a lightweight classifier/probe on reasoning states can predict helpful vs harmful tool calls above chance, demonstrating the signal is learnable.

Without (4), the story is "autonomous doesn't work." With (4), the story becomes "the signal exists, the ceiling is high, and this is a tractable learning problem."

## Oracle Ceiling Analysis

Using per-round correctness data from fixed-schedule runs (RC reimpl, 12 steps, 16 samples per problem), we simulate an oracle policy that knows the optimal step to stop at for each sample.

### Qwen3-4B-Instruct-2507 on IMOBench (400 problems)

**Per-step pass@1 progression:**

| Step | pass@1 |
|------|--------|
| 1 | 32.6% |
| 4 | 37.2% |
| 8 | 38.8% |
| 12 | 40.0% |

**Oracle vs fixed schedule:**

| Policy | pass@1 | Majority Vote |
|--------|--------|---------------|
| No tool (step 1) | 32.6% | 23.2% |
| Fixed 12 steps | 40.0% | 27.8% |
| Oracle (best step per sample) | **45.9%** | **31.5%** |

**The ceiling is substantial.** The oracle gains 5.9pp over fixed schedule (45.9% vs 40.0%). Of the total headroom between no-tool and oracle (13.3pp), fixed schedule captures only 56% — the remaining 44% is lost to bad timing.

**Efficiency:** The early-stopping oracle achieves 45.9% while stopping at step 2.2 on average, saving 82% of compute (9.8 fewer tool calls per sample).

**Regressions are the mechanism.** Across all steps: 1009 correct→wrong transitions vs 1486 wrong→correct. 40% of correctness-changing transitions are regressions caused by the tool. An oracle that avoids just these regressions captures the full ceiling.

### Qwen3-30B-A3B-Instruct-2507 on IMOBench (400 problems)

**Baseline**: 45.2% pass@1 (t32768, n16, temp=0.7)

**Fixed-schedule tool calling also improves accuracy at 30B scale.** Using the RC reimpl approach (rc_user: summary embedded in user prompt, fixed tool call every round), accuracy improves steadily:

| Step | rt2048 | rt4096 | rt8192 |
|------|--------|--------|--------|
| 1    | 45.5%  | 45.1%  | 45.7%  |
| 2    | 48.7%  | 48.9%  | —      |
| 3    | 50.4%  | 50.3%  | 50.3%  |
| 5    | 52.5%  | —      | —      |

Best result: **52.5% pass@1** (+7.0pp over step 1) with rt2048 after 5 steps. The gain is consistent across per-round token limits.

**Scaling comparison.** Both 4B and 30B benefit from fixed-schedule tool calling: 4B gains +7.4pp over 12 steps (32.6% → 40.0%), 30B gains +7.0pp over 5 steps (45.5% → 52.5%). The 30B model achieves comparable gains in fewer steps.

## Current State and Next Steps

**The oracle ceiling is real and large (contribution 4 is viable).** The 4B oracle analysis shows 5.9pp headroom over fixed schedule. The 30B fixed-schedule results confirm the tool consistently helps at scale — the remaining question is whether a learned policy can capture the oracle headroom by optimizing *when* to call.

**Next steps:**
1. Run oracle ceiling analysis on 30B data to quantify headroom for a learned policy.
2. Investigate whether a lightweight probe on reasoning state features (generation length, answer stability, repetition) can predict helpful vs harmful tool calls.
3. Decide whether to pursue RL training (262K reasoning states ready, SLIME infrastructure set up) or focus on the probe-based learnability argument.

**Jack's comments:**
1. Can try more powerful model better trained to use tools (Qwen3.5, Minimax)
2. Try search agent setting where agent reflect on whether they need to remove useless search results from their context
3. Agree that experimenting on oracle and showing that oracle leads to improvement is a good goal
4. Potential method after paper framing step (4): Context Compaction Companion (or buddy; CCB): use the oracle data to train a small classifier that determines when it's best to call the context compaction tool (either that classifier calls the tool directly or give a suggestion to main model)
5. In the paper framing "**Formalize the distinction between fixed scaffolds and model-controlled context management.**", can emphasize aspects on "self-controlled context management" "model meta-cognition on context" (relates to CoT controllability!)

**Open risks:**
- RL rollouts are expensive (2-3x single-turn RL due to sequential generation → summarization → re-generation).
- Meta-cognition may not be learnable at small model scales — strong prompted models already fail at strategic tool use.
