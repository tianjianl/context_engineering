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

**Fixed-schedule calling is catastrophic at 30B scale.** Unlike the 4B model where fixed-schedule tool calling improves accuracy monotonically (32.6% → 40.0%), the 30B model's accuracy crashes after the first forced summarization and never recovers. The pattern is consistent across all per-round token limits:

| Step | rc_fix rt2048 | rc_fix rt4096 | rc_fix rt8192 |
|------|---------------|---------------|---------------|
| 1    | 37.7%         | 38.4%         | 38.7%         |
| 2    | 18.8%         | 19.4%         | 19.2%         |
| 4    | 17.4%         | 17.3%         | 18.2%         |
| 8    | 16.5%         | 16.5%         | 16.4%         |
| 12   | 17.0%         | 16.4%         | 17.0%         |

Step 1 accuracy (~38%) is below baseline (~45%) because the per-round token limit constrains the initial generation. Regressions dominate: ~8000 correct→wrong vs ~6700 wrong→correct transitions across 12 steps.

**Oracle ceiling under fixed schedule:** ~46.3% pass@1 across all rt values, achieved almost entirely by stopping at step 1 (avg oracle stop step ~7.2). The oracle-fixed gap is massive (+29pp) but only because the fixed schedule is so destructive — the oracle gains just +8pp over no-tool.

**Model-controlled calling works well.** When the 30B model decides when to call the tool (rc_user), accuracy improves steadily:

| Rounds | rc_user rt2048 | rc_user rt4096 | rc_user rt8192 |
|--------|----------------|----------------|----------------|
| 1      | 45.5%          | 45.1%          | 45.7%          |
| 2      | 48.7%          | 48.9%          | —              |
| 3      | 50.4%          | 50.3%          | 50.3%          |
| 5      | 52.5%          | —              | —              |

Best result: **52.5% pass@1** (+7.0pp over round 1) with rc_user rt2048 after 5 rounds.

**30B vs 4B: opposite responses to fixed schedules.** This is the strongest evidence that model-controlled tool calling matters. The 4B model lacks meta-cognition for strategic tool use but benefits from forced calling — its reasoning chains are short enough that summarization preserves critical detail. The 30B model's reasoning is complex enough that naive forced summarization is destructive, yet when given control, the model's voluntary tool use actually improves accuracy. The capability to benefit from the tool exists; the fixed schedule just applies it at the wrong times.

## Current State and Next Steps

**The 30B results strengthen the paper narrative.** The 4B oracle ceiling showed headroom for a learned policy. The 30B results go further: fixed schedules that help small models are catastrophic at scale, while model-controlled calling already captures substantial gains. This reframes the problem from "can we learn when to call?" to "why does scale unlock better tool-use meta-cognition, and can we push it further?"

**Next steps:**
1. Investigate whether a lightweight probe on reasoning state features (generation length, answer stability, repetition) can predict helpful vs harmful tool calls on 4B data.
2. Characterize what the 30B model does differently in rc_user: does it call the tool less often? At different points? With different reasoning before the call?
3. Decide whether to pursue RL training (262K reasoning states ready, SLIME infrastructure set up) or focus on the probe-based learnability argument.

**Open risks:**
- RL rollouts are expensive (2-3x single-turn RL due to sequential generation → summarization → re-generation).
- The 30B rc_user success may be partially due to temperature/prompt differences (temp=1.0 vs 0.7 for rc_fix) rather than pure meta-cognition — needs controlled ablation.
