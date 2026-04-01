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

## Current State and Next Steps

**Immediate priority: establish the oracle ceiling (contribution 4).** Per-round generations from fixed-schedule runs (RC user, RC reimpl) already exist. Grading each round independently and simulating an oracle policy requires no new experiments — just analysis of existing data.

**Then:** Decide whether to pursue RL training (262K reasoning states ready, SLIME infrastructure set up) or a probe-based approach to demonstrate learnability of tool-call timing.

**Open risks:**
- Oracle ceiling may be small (fixed-schedule timing may already be close to optimal).
- RL rollouts are expensive (2-3x single-turn RL due to sequential generation → summarization → re-generation).
- Meta-cognition may not be learnable at small model scales — strong prompted models already fail at strategic tool use.
