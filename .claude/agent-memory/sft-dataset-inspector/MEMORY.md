# SFT Dataset Inspector Memory

## Dataset Format Conventions (Dogtooth Tool SFT)

### ShareGPT Format
- Keys: `from` (role) and `value` (content), NOT `role`/`content`
- Roles used: `human`, `gpt`, `observation` (never `function_call`)
- Tool calls are embedded inline in `gpt` turns as `<tool_call>\n{JSON}\n</tool_call>` at the end
- This matches Qwen's native tool call format exactly
- Top-level fields: `conversations`, `tools` (JSON string), `system` (string)

### LLaMA-Factory Processing
- `SharegptDatasetConverter` maps: human->user, gpt->assistant, observation->observation, function_call->function
- Since data uses `gpt` (not `function_call`), `FunctionFormatter` is NEVER invoked
- Tool call tags pass through as literal text in assistant content -- this is correct for Qwen
- Odd turn count conversations are SILENTLY DROPPED (not even a warning in some cases)
- `format_observation` wraps content in `<tool_response>...</tool_response>` tags
- `format_tools` generates Qwen-style tool prompt appended to system message

### Known Issues
- **Odd turn count**: 3/722 in best, 8/4584 in all dataset -- these get silently dropped
- **Short final GPT turns**: 144/722 (best), 901/4584 (all) -- just boxed answers < 100 chars
- **Heavy prompt duplication**: best_722 has only 220 unique prompts; all_4584 has only 398 unique prompts
- **Redundant tool description**: system prompt describes tool in NL + tools field adds formal Qwen tool prompt
- cutoff_len=32768 in training configs; longest examples are ~127K chars which may get truncated

### Dataset Sizes
- `dogtooth_filtered_best_722.json`: 722 examples (answer_changed only)
- `dogtooth_filtered_all_4584.json`: 4584 examples (answer_changed + no_initial_answer)
- `tool_sft_train.json`: 11140 examples (full unfiltered)
- `tool_sft_val.json`: validation split

### Key File Paths
- Dataset config: `/weka/home/tli104/principia/LLaMA-Factory/data/dataset_info.json`
- Template code: `/weka/home/tli104/principia/LLaMA-Factory/src/llamafactory/data/template.py`
- Converter code: `/weka/home/tli104/principia/LLaMA-Factory/src/llamafactory/data/converter.py`
- Tool utils: `/weka/home/tli104/principia/LLaMA-Factory/src/llamafactory/data/tool_utils.py`
- Formatter: `/weka/home/tli104/principia/LLaMA-Factory/src/llamafactory/data/formatter.py`
