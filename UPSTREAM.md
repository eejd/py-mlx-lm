# Work carried on `local` that is not in ml-explore/mlx-lm

`local` = upstream release tag v0.32.2 plus the upstream commits and `topic/*`
branches below, merged with `--no-ff`. `main` mirrors `ml-explore/mlx-lm` `main`
(fast-forward only). Each topic branch is one change on top of v0.32.2.

| Topic branch | What | Upstream | Original branch (PR head, kept) |
|---|---|---|---|
| `topic/qwen3-coder-bare-function` | recover Qwen3-Coder tool calls written without the opening `<tool_call>` marker | not submitted (eejd/py-mlx-lm#3) | `fix/qwen3-coder-bare-function` |
| `topic/mistral-tool-parser-from-vocab` | infer the mistral tool parser from the `[TOOL_CALLS]` vocab token | ml-explore/mlx-lm#1969, closed (withdrawn 2026-10-16); eejd/py-mlx-lm#1 | `fix/mistral-tool-parser-from-vocab` |
| `topic/batch-kv-cache-fast-mask` | skip the attention mask in batch caches with no padding | ml-explore/mlx-lm#1970, closed (withdrawn 2026-10-16); eejd/py-mlx-lm#2 | `fix/batch-kv-cache-fast-mask` |
| `topic/llama3-tool-parser` | tool parser for Llama 3.1+ JSON calls (`<\|python_tag\|>{...}` and bare JSON); `ToolCallFormatter` can return a parser's non-call text as content | not submitted: upstream candidate only, after the quantbench + agentbench soak (eejd/py-mlx-lm#5) | - |
| `topic/server-ignore-eos` | `ignore_eos` request option on `/v1/completions` and `/v1/chat/completions` (EOS tokens do not end the request; `max_tokens` and `stop` still do; default off), for fixed-output load tests | not submitted (eejd/py-mlx-lm#12) | - |
| `topic/bpe-detokenizer-finalize` | decode pending BPE bytes safely and isolate batched request detokenizer errors | not submitted (eejd/py-mlx-lm#24) | - |
| `topic/server-load-errors-actionable` | return actionable model and tokenizer load errors without killing the generation thread | not submitted (eejd/py-mlx-lm#21, #22, #25) | - |

Not on `local` (not carried by any port): `fix/gpt-oss-harmony-channel-parsing`
(ml-explore/mlx-lm#1867, open; eejd/py-mlx-lm#4). Its branch is still the upstream PR's head.

The `fix/*` branches are the heads of the upstream PRs above and are left untouched; the
`topic/*` branches are their content rebased onto `main` (v0.32.2).

## Merge of upstream 4cfc647

The merge includes the 12 upstream commits through `4cfc647` (Flint support).
Upstream's TextMachine refactor replaced tuple-state `split`/`step` with
`TextStateMachine.reader().feed()` and `Delta`. The server now applies the
Qwen3-Coder bare-call filter to each content segment around state transitions,
retains the Llama 3 formatter's non-call content fallback, and preserves
`ignore_eos` independently for batched and single generation on the new API.
The token stop-sequence helper remains for the batch-generation API and its
tests. Mistral vocabulary detection and the batch KV-cache fast-mask path remain
carried unchanged.

The setuptools-scm `git describe` command excludes the fork's `-local*` tags:
`git describe` finds the fork's own newest tag first, and names such as
`v0.32.2-local05` are not PEP 440 versions. This was already so at `local05`
(setuptools-scm and the dynamic version were in its `pyproject.toml`); only git
checkouts are affected, since the MacPorts port builds from the GitHub archive
and sets `SETUPTOOLS_SCM_PRETEND_VERSION`.

Finish reason with tool calls (`v0.32.2-local07`): a response that ends with
`stop` becomes `tool_calls` only when at least one call parsed
(`ToolCallFormatter.count`, upstream's rule since ab7f704; the fork's earlier
`made_call` rule, which also counted a call that was dropped as unparseable, is
gone). A dropped call is not a call: the client gets `stop` and no tool calls
instead of `tool_calls` next to an empty list. A response cut by `max_tokens`
keeps `length`. This matches vLLM's `resolve_finish_reason`. mlx-lm's server has
no `tool_choice` handling, so OpenAI's forced-named-tool case (`stop`) does not
arise. Tests: `ToolCallFormatter` counts, and handler-level
`test_a_call_that_fails_to_parse_does_not_make_a_tool_call_finish` and
`test_a_call_cut_by_max_tokens_stays_length`.

Packaging note: upstream now requires `pillow` (dc2012b) and `mlx>=0.32.3`
(e7d21f4). The MacPorts port `py-mlx-lm-local` needs `pillow` in `depends_lib`.

Test status of this merge (offline, mlx 0.32.4+local03, cached models): the full
suite fails the same 23 tests and errors on the same collection error as pristine
upstream `4cfc647` (numerical assertions in test_generate/test_models, uncached
models); the merged tree passes 451 tests against upstream's 357, the difference
being the carried topics' tests.

## Tags


`v0.32.2-local01` (superseded: its qwen3-coder port raised `UnboundLocalError` for any request with tools to a bare-call parser); `v0.32.2-local02` and `v0.32.2-local03` (tests only over local02; the flush now runs per segment, with handler-level tests; local03 is superseded by local04); `v0.32.2-local04` (local03 + `topic/llama3-tool-parser`; superseded by local05; **behaviour change for callers of `mlx_lm.server.ToolCallFormatter`:** it returns `(tool_calls, content)` instead of a list); `v0.32.2-local05` (local04 + `topic/server-ignore-eos`; new `ignore_eos` request option, default off); `v0.32.2-local06` (merge of upstream 4cfc647; superseded by local07); `v0.32.2-local07` (not yet cut: local06 + `ToolCallFormatter.count` finish-reason rule, `made_call` removed).
