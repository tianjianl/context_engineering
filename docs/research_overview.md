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

## Current State and Next Steps

The infrastructure for RL training exists: 262K extracted reasoning states (problem + summary-so-far at each round boundary), SLIME training framework, configurable tool calls, correctness-based reward. The model would learn from outcome signal whether calling the tool at a given reasoning state leads to better answers.

This has not launched yet for two reasons:

1. **Rollout cost.** Each RL episode with tool calling requires sequential generation → summarization → re-generation. This is 2-3x the cost of standard single-turn RL.

2. **Feasibility uncertainty.** If strong prompted models can't use the tool strategically, it's unclear whether RL can teach this meta-cognitive capability to smaller models. The autonomous tool calling experiments suggest the gap is large.

The immediate priority is completing the autonomous tool calling evaluation (baseline vs fixed-schedule vs various prompt interventions on Qwen3.5-9B) to quantify exactly how much the prompting gap is, then deciding whether to commit to the RL training.
