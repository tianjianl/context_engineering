# BrowseComp-Plus MiniMax-M2.5 — All Experiments

All runs use MiniMax-M2.5 via direct API (api.minimax.io/v1), BM25 corpus search (Tevatron/browsecomp-plus), and MiniMax-M2.5 self-grading.

**Column legend:**
- **N**: number of questions evaluated
- **Acc**: accuracy (correct / N)
- **Sums**: average summarizations per problem
- **Srch**: average web searches per problem
- **Rnds**: average rounds per problem
- **Prompt (K)**: average prompt tokens per problem across all agent rounds (thousands). Each round re-sends the full conversation, so this reflects cumulative context cost, not unique tokens.
- **Comp (K)**: average completion tokens per problem across all agent rounds (thousands)

Token counts cover agent rounds only. Separate summarizer LLM calls (used by auto-sum, sum-tool v2/v5, checkpoint no-KV, selective v1) are not included. KV-reuse summarizer calls (checkpoint KV, sum-inline, selective v2 KV) are included since they extend the agent conversation.

| # | Acc | N | Sums | Srch | Rnds | Prompt (K) | Comp (K) | Description |
|---|-----|---|------|------|------|------------|----------|-------------|
| 1 | **64.7%** | 150 | 0.0 | 7.8 | 9.5 | 835 | 5.1 | No context mgmt. af10, sr10, r15. t=1.0, tp0.95, tk40 (MiniMax recommended sampling). |
| 2 | **64.7%** | 150 | 0.0 | 12.8 | 14.4 | 1088 | 7.2 | No context mgmt. Auto-fetch 5, 25 rounds (long context). t=1.0. |
| 3 | **64.0%** | 150 | 6.3 | 9.5 | 14.0 | 437 | 7.0 | Checkpoint-free KV: every-round nudge + KV reuse summarization. Free summarize rounds. af10, sr10, r15. t=1.0, tp0.95, tk40. Summarizer t=0.0 (deterministic). |
| 4 | **64.0%** | 150 | 3.4 | 10.0 | 13.3 | 623 | 6.1 | Selective v2 + nudge@100K: v2 prompt (mentions summarize tool casually). Summarize tool always available but model rarely uses unprompted. [CONTEXT ALERT] nudge only at >=100K tokens. Max 4 sums. Separate LLM call (no KV). Free summarize rounds. af10, sr10, r15. t=1.0, tp0.95, tk40. Summarizer t=0.0. |
| 5 | **62.4%** | 830 | 0.0 | 10.7 | 12.5 | 807 | 6.1 | Auto-sum v5 @30%: threshold-triggered at 300K (30% of 1M). Full 830q, af20, r20. t=1.0. Summarizer t=0.0. |
| 6 | **61.3%** | 830 | 0.0 | 10.8 | 12.5 | 821 | 6.1 | No context mgmt. Full 830q dataset, af20, r20. t=1.0. |
| 7 | **60.7%** | 150 | 0.0 | 11.5 | 13.2 | 879 | 6.1 | No context mgmt. af30, r20 (extreme context stress, ~500K+ tokens). t=1.0. |
| 8 | **60.0%** | 150 | 0.0 | 11.2 | 13.0 | 492 | 5.5 | No context mgmt. Auto-fetch top 3 docs per search. t=1.0, r15. |
| 9 | **59.4%** | 830 | 0.9 | 11.6 | 14.1 | 659 | 6.3 | Sum-tool v5: hierarchical evidence/interpretation split. Full 830q, af20, r20. t=1.0. |
| 10 | **59.3%** | 150 | 1.5 | 8.9 | 11.3 | 666 | 5.6 | Selective v1 @100K: starts as baseline (no summarize tool). At 100K tokens, summarize_context tool injected + [CONTEXT ALERT] nudge. Max 4 summarizations. Separate LLM call (no KV reuse). af10, sr10, r15. t=1.0, tp0.95, tk40. Summarizer t=0.0. |
| 11 | **58.7%** | 150 | 1.7 | 9.4 | 11.2 | 359 | 5.1 | Auto-sum v5 @30%: threshold-triggered (at 58982 tokens = 30% of 196K context). Separate LLM call with v5 hierarchical prompt. No summarize tool for model. af10, sr10, r15. t=1.0. Summarizer t=0.0. |
| 12 | **58.7%** | 150 | 0.0 | 7.7 | 9.3 | 789 | 4.9 | No context mgmt. af10, sr10, r15 (context stress test). t=1.0. |
| 13 | **58.7%** | 150 | 0.0 | 12.3 | 14.0 | 964 | 6.5 | No context mgmt. af20, r20 (heavy context stress). t=1.0. |
| 14 | **58.7%** | 150 | 0.8 | 11.4 | 13.8 | 679 | 6.1 | Sum-tool v5: hierarchical evidence/interpretation split. af30, r20. t=1.0. |
| 15 | **58.0%** | 150 | 0.7 | 8.6 | 10.4 | 761 | 5.3 | Selective v1 @150K: same as @100K but threshold at 150K tokens. Max 4 summarizations. Separate LLM call. af10, sr10, r15. t=1.0, tp0.95, tk40. Summarizer t=0.0. |
| 16 | **58.0%** | 150 | 0.8 | 12.6 | 14.9 | 795 | 6.3 | Sum-tool v5: hierarchical evidence/interpretation split. af20, r20. t=1.0. |
| 17 | **57.3%** | 150 | 0.0 | 11.0 | 12.8 | 843 | 6.2 | Auto-sum v5 @50%: threshold-triggered at 500K (50% of 1M). af30, r20. t=1.0. Summarizer t=0.0. |
| 18 | **57.3%** | 150 | 6.3 | 9.7 | 14.2 | 434 | 7.1 | Checkpoint-free KV + stochastic summarizer: same as checkpoint-free but summarizer also uses t=1.0/tp0.95/tk40 instead of t=0.0. af10, sr10, r15. |
| 19 | **57.1%** | 154 | 0.0 | 11.1 | 12.9 | 487 | 5.4 | No context mgmt. 400q significance run (first 400 from dataset), af3. t=1.0, r15. |
| 20 | **56.7%** | 150 | 0.8 | 8.6 | 10.9 | 743 | 5.7 | Sum-tool v5: model-triggered, separate LLM generates hierarchical summary splitting Evidence (search log, entities, quotes) from Interpretation (constraint status, hypotheses, strategy). af10, sr10, r15. t=1.0. |
| 21 | **56.0%** | 150 | 0.0 | 12.1 | 13.9 | 234 | 5.2 | No context mgmt. Auto-fetch top 1 doc per search. t=1.0, r15. |
| 22 | **55.3%** | 150 | 1.7 | 9.4 | 11.1 | 354 | 5.0 | Auto-sum v5 @30%: same threshold-triggered summarization. af10, sr10, r15. t=1.0, tp0.95, tk40. Summarizer t=1.0/tp0.95/tk40 (stochastic). |
| 23 | **55.3%** | 150 | 0.0 | 11.5 | 13.2 | 501 | 5.6 | No context mgmt. Same as af3 run 1 — variance check (2nd run). t=1.0, r15. |
| 24 | **55.3%** | 150 | 5.0 | 8.0 | 11.7 | 352 | 5.8 | Checkpoint KV: every round, [CHECKPOINT] user msg nudges model to decide whether to summarize. When model calls summarize_context, V5 prompt appended inline (KV reuse), then conversation truncated. Summarize rounds consume budget. af10, sr10, r15. t=0.6 (default). Summarizer t=0.0. |
| 25 | **55.3%** | 150 | 2.3 | 10.9 | 14.9 | 210 | 5.0 | Sum-tool v1: model-triggered narrative summary. af3. t=1.0, r15. Summarize rounds consume budget. |
| 26 | **54.8%** | 157 | 2.2 | 10.9 | 14.8 | 210 | 5.0 | Sum-tool v1: 400q significance run, af3. t=1.0, r15. |
| 27 | **54.0%** | 150 | 6.2 | 9.7 | 14.1 | 458 | 6.9 | Checkpoint-free KV: every-round nudge + KV reuse summarization. Summarize rounds are FREE (don't consume budget). af10, sr10, r15. t=1.0. Summarizer t=0.0. |
| 28 | **54.0%** | 150 | 5.0 | 17.2 | 24.0 | 436 | 8.0 | Sum-tool v1: model-triggered narrative summary. af5, r25 (long context). t=1.0. Summarize rounds consume budget. |
| 29 | **54.0%** | 150 | 2.8 | 10.2 | 14.7 | 269 | 5.8 | Sum-tool v3: same 6-section model-written summary. af5. t=1.0, r15. |
| 30 | **54.0%** | 150 | 1.2 | 8.3 | 11.1 | 559 | 5.3 | Sum-tool v6: softer prompt than v5. System prompt mentions summarize casually ('Call when you have partial findings to consolidate or feel stuck'). Model decides when. af10, sr10, r15. t=1.0. |
| 31 | **52.7%** | 150 | 1.5 | 11.4 | 14.8 | 129 | 4.7 | Sum-tool v1: model calls summarize_context to write narrative summary (Key Findings, Promising Leads, Failed Approaches, Hypothesis, Next Steps). Conversation replaced with summary. af1. t=1.0, r15. Summarize rounds consume budget. |
| 32 | **52.0%** | 150 | 5.2 | 7.9 | 11.8 | 383 | 5.9 | Checkpoint no-KV: same every-round nudge, but summarization uses separate LLM call (no KV reuse). Summarize rounds consume budget. af10, sr10, r15. t=0.6 (default). Summarizer t=0.0. |
| 33 | **51.3%** | 150 | 2.9 | 13.0 | 17.7 | 241 | 5.9 | Sum-v1-fixed: same as sum-tool v1 but summarize-only rounds are FREE (don't count toward round budget). af3. t=1.0, r15. |
| 34 | **50.7%** | 150 | 4.1 | 9.6 | 13.4 | 203 | 5.1 | Sum-tool v2: model-triggered, separate LLM generates structured search log (like auto-sum v2). Tool results TRUNCATED to 1500 chars, thinking to 300 chars before summarization. af3. t=1.0, r15. |
| 35 | **49.3%** | 150 | 4.7 | 9.7 | 13.8 | 209 | 5.3 | Sum-tool v2-fixed: same as v2 but passes FULL untruncated conversation to summarizer (v2 truncated 98.4% of results). af3. t=1.0, r15. |
| 36 | **47.3%** | 150 | 0.0 | 12.5 | 14.6 | 99 | 4.8 | No context mgmt. Model-driven search/fetch only (no auto-fetch). t=1.0. |
| 37 | **46.0%** | 150 | 3.7 | 10.1 | 13.7 | 214 | 5.1 | Sum-inline: when model calls summarize_context, summarize prompt appended as user msg to existing conversation (KV cache reuse). Model generates summary inline, then conversation truncated. af3. t=1.0, r15. |
| 38 | **44.7%** | 150 | 2.3 | 10.8 | 14.8 | 209 | 5.1 | Sum-tool v1: same as af3 run 1 — variance check (2nd run). af3. t=1.0, r15. |
| 39 | **42.7%** | 150 | 2.1 | 10.7 | 14.6 | 213 | 5.2 | Sum-tool v3: model-triggered, model writes 6-section summary (Search Log + Key Findings + Promising Leads + Dead Ends + Hypothesis + Next Steps). Combines v1 narrative + v2 search log. af3. t=1.0, r15. |
| 40 | **20.0%** | 150 | 13.7 | 13.4 | 14.7 | 18 | 4.9 | Auto-sum v2: after EVERY round, separate LLM compresses conversation into structured search log (one line per search/fetch with query, result, relevance tag). Key Facts only, no hypotheses. af3. t=1.0, r15. |
